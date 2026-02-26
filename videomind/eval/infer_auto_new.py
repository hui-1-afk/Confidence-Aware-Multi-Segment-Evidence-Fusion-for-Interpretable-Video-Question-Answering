# Copyright (c) 2025 Ye Liu. Licensed under the BSD-3-Clause License.

import argparse
import copy
import json
from contextlib import nullcontext
import math
import numpy as np
import nncore
import torch # 确保 torch 已导入

from videomind.constants import GROUNDER_PROMPT, PLANNER_PROMPT, VERIFIER_PROMPT
from videomind.dataset.hybrid import DATASETS
from videomind.dataset.utils import process_vision_info
from videomind.model.builder import build_model
from videomind.utils.io import get_duration, load_subtitle
from videomind.utils.parser import parse_query, parse_span


# ==================== 辅助函数：权重计算与答案选择 ====================

def calculate_weights(conf1, conf2, T=1.0):
    """
    计算两个时刻的 softmax 风格权重。
    """
    conf1 = np.float64(conf1)
    conf2 = np.float64(conf2)
    
    c1_T = conf1 / T
    c2_T = conf2 / T

    exp_c1_T = np.exp(c1_T)
    exp_c2_T = np.exp(c2_T)
    
    denominator = exp_c1_T + exp_c2_T
    
    if denominator == 0:
        return (0.5, 0.5)

    w1 = exp_c1_T / denominator
    w2 = exp_c2_T / denominator
    
    return w1.item(), w2.item() 

def select_final_answer(ans1, ans2, w1, w2, moment1, moment2):
    """
    根据答案一致性或权重选择最终答案和对应时刻。
    """
    moment1 = moment1.tolist() if isinstance(moment1, torch.Tensor) else moment1
    moment2 = moment2.tolist() if isinstance(moment2, torch.Tensor) else moment2

    if ans1.strip() == ans2.strip():
        # 答案相同，选择权重最高的时刻作为最终时刻
        final_answer = ans1
        final_moment = moment1 if w1 >= w2 else moment2 
    else:
        # 答案不同，选择权重较高的答案和时刻
        if w1 > w2:
            final_answer = ans1
            final_moment = moment1
        else:
            final_answer = ans2
            final_moment = moment2
            
    return final_answer, final_moment

# ==================== 答案生成封装函数 (已修改为接收动态参数) ====================

def generate_answer_for_moment(moment, anno, prompt_str, video_path, duration, args, processor, model, device, dataset_cls, adapter_state, video_max_frames, video_fps):
    """Generates an answer from the Answerer model for a given video moment, using dynamic video parameters."""
    
    if hasattr(dataset_cls, 'MIN_RATIO'):
        min_len = duration * dataset_cls.MIN_RATIO
    else:
        min_len = getattr(dataset_cls, 'MIN_LEN', 32) 
    
    min_len_param = max(16, min_len) 
    
    s, e = parse_span(moment, duration, min_len_param)
    
    final_prompt = prompt_str

    if args.use_subtitle and 'subtitle_path' in anno and nncore.is_file(anno['subtitle_path']):
        subs = load_subtitle(anno['subtitle_path'])
        subs_text = [f'{round(max(0, a - s), 1)}s - {round(b - s, 1)}s, {t}\n' for a, b, t in subs if a < e and b > s]
        subs_text = ''.join(subs_text[:100])
        final_prompt = f'You are given a video with {round(e - s, 1)} seconds long.\nSubtitles:\n{subs_text}' + prompt_str

    messages = [{
        'role':
        'user',
        'content': [{
            'type': 'video',
            'video': video_path,
            'num_threads': args.num_threads,
            'video_start': s,
            'video_end': e,
            'min_pixels': getattr(dataset_cls, 'MIN_PIXELS', 128) * 28 * 28,
            'max_pixels': getattr(dataset_cls, 'MAX_PIXELS', 256) * 28 * 28,
            'max_frames': video_max_frames,
            'fps': video_fps
        }, {
            'type': 'text',
            'text': final_prompt
        }]
    }]

    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    text += 'Best Option: (' if args.style == 'mcq' else ''
    
    images, videos = process_vision_info(messages)
    data = processor(text=[text], images=images, videos=videos, return_tensors='pt')
    data = data.to(device)

    if adapter_state['answerer']:
        model.base_model.disable_adapter_layers()
        model.base_model.enable_adapter_layers()
        model.set_adapter('answerer')
        context = nullcontext
    else:
        context = model.disable_adapter 

    with context():
        output_ids = model.generate(
            **data,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            repetition_penalty=None,
            max_new_tokens=256)

    assert data.input_ids.size(0) == output_ids.size(0) == 1
    output_ids = output_ids[0, data.input_ids.size(1):]
    if output_ids[-1] == processor.tokenizer.eos_token_id:
        output_ids = output_ids[:-1]
    response = processor.decode(output_ids, clean_up_tokenization_spaces=False)
    
    return response, [s, e]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset')
    parser.add_argument('--pred_path')
    parser.add_argument('--model_gnd_path')
    parser.add_argument('--model_ver_path')
    parser.add_argument('--model_pla_path')
    parser.add_argument('--model_ans_path')
    parser.add_argument('--split', default='test', choices=['train', 'valid', 'test'])
    parser.add_argument('--style', default='mcq', choices=['mcq', 'options', 'direct'])
    parser.add_argument('--use_subtitle', action='store_true')
    parser.add_argument('--auto_rephrasing', action='store_true')
    parser.add_argument('--auto_planning', action='store_true')
    parser.add_argument('--num_threads', type=int, default=1)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--chunk', type=int, default=1)
    parser.add_argument('--index', type=int, default=0)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    if args.chunk > 1:
        pred_path = nncore.join(args.pred_path, f'output_{args.index}.json')
    else:
        pred_path = nncore.join(args.pred_path, 'output.json')

    adapter_state = dict(planner=False, verifier=False, answerer=False)

    model, processor = build_model(args.model_gnd_path, device=args.device)
    device = next(model.parameters()).device

    if args.model_pla_path is not None:
        adapter_path = nncore.join(args.model_pla_path, 'planner')
        if nncore.is_dir(adapter_path):
            model.load_adapter(adapter_path, adapter_name='planner')
            adapter_state['planner'] = True

    if args.model_ver_path is not None:
        adapter_path = nncore.join(args.model_ver_path, 'verifier')
        if nncore.is_dir(adapter_path):
            model.load_adapter(adapter_path, adapter_name='verifier')
            adapter_state['verifier'] = True

    if args.model_ans_path is not None:
        adapter_path = nncore.join(args.model_ans_path, 'answerer')
        if nncore.is_dir(adapter_path):
            model.load_adapter(adapter_path, adapter_name='answerer')
            adapter_state['answerer'] = True

    dataset_cls = DATASETS.get(args.dataset)

    annos = dataset_cls.load_annos(split=args.split)
    annos = [annos[i::args.chunk] for i in range(args.chunk)][args.index]

    dumps = []
    
    # --- OOM 内存管理和重试机制设置 ---
    MAX_RETRIES = 3 # 最大重试次数
    # Grounder/Planner 的初始帧率/最大帧数 (基准值 1.0 FPS / 150 帧)
    RETRY_FPS_VALUES = [1.0, 0.5, 0.25] 
    RETRY_MAX_FRAMES_GND = [150, 100, 75]
    
    
    for i in nncore.ProgressBar(range(len(annos))):
        
        # --- 每个样本的重试循环 ---
        sample_processed_successfully = False
        
        for retry_count in range(MAX_RETRIES):
            
            # 1. 设置当前视频参数 (降级)
            current_fps = RETRY_FPS_VALUES[retry_count]
            current_max_frames = RETRY_MAX_FRAMES_GND[retry_count]
            
            # Planner/Grounder 的 FPS 和 Max Frames 乘数
            fps_multiplier = current_fps / 1.0 
            max_frames_multiplier = current_max_frames / 150 
            
            try:
                # 重置样本数据和状态
                anno = copy.deepcopy(annos[i])
                dump = copy.deepcopy(annos[i])

                video_path, duration, span = anno['video_path'], anno.get('duration'), anno.get('span')

                if duration is None:
                    duration = get_duration(video_path, num_threads=args.num_threads)
                    dump['duration'] = duration

                do_answering = all(k in anno for k in ('question', 'options'))

                if do_answering:
                    question, options= anno['question'], anno['options']

                    if args.style in ('mcq', 'options'):
                        prompt = question + '\nOptions:'
                        for idx, opt in enumerate(options):
                            prompt += f"\n({chr(ord('A') + idx)}) {opt[0].upper() + opt[1:]}"
                        prompt += '\nPlease only give the best option.'
                    else:
                        prompt = question

                else:
                    question = anno['query']

                do_grounding = True
                query = question
                dump['agents'] = []

                # --- 规划器 (Planner) ---
                if adapter_state['planner'] and (args.auto_rephrasing or args.auto_planning):
                    dump['agents'].append('planner')

                    messages = [{
                        'role':
                        'user',
                        'content': [{
                            'type': 'video',
                            'video': video_path,
                            'num_threads': args.num_threads,
                            'min_pixels': 36 * 28 * 28,
                            'max_pixels': 64 * 28 * 28,
                            'max_frames': int(100 * max_frames_multiplier), # 动态帧数
                            'fps': current_fps                           # 动态帧率
                        }, {
                            'type': 'text',
                            'text': PLANNER_PROMPT.format(question)
                        }]
                    }]

                    text = processor.apply_chat_template(messages, add_generation_prompt=True)
                    images, videos = process_vision_info(messages)
                    data = processor(text=[text], images=images, videos=videos, return_tensors='pt')
                    data = data.to(device)

                    model.base_model.disable_adapter_layers()
                    model.base_model.enable_adapter_layers()
                    model.set_adapter('planner')

                    output_ids = model.generate(
                        **data,
                        do_sample=False,
                        temperature=None,
                        top_p=None,
                        top_k=None,
                        repetition_penalty=None,
                        max_new_tokens=256)

                    assert data.input_ids.size(0) == output_ids.size(0) == 1
                    output_ids = output_ids[0, data.input_ids.size(1):]
                    if output_ids[-1] == processor.tokenizer.eos_token_id:
                        output_ids = output_ids[:-1]
                    response = processor.decode(output_ids, clean_up_tokenization_spaces=False)

                    dump['planner_response'] = response

                    try:
                        parsed = json.loads(response)
                        action = parsed[0] if isinstance(parsed, list) else parsed
                        if args.auto_rephrasing and action['type'].lower() == 'grounder' and action['value']:
                            query = action['value']
                            dump['planner_parsed_query'] = query
                        elif args.auto_planning and action['type'].lower() == 'answerer':
                            do_grounding = False
                    except Exception:
                        pass

                # --- 定位器 (Grounder) ---
                if do_grounding:
                    dump['agents'].append('grounder')

                    query = parse_query(query)

                    messages = [{
                        'role':
                        'user',
                        'content': [{
                            'type': 'video',
                            'video': video_path,
                            'num_threads': args.num_threads,
                            'min_pixels': 36 * 28 * 28,
                            'max_pixels': 64 * 28 * 28,
                            'max_frames': current_max_frames, # 动态帧数
                            'fps': current_fps              # 动态帧率
                        }, {
                            'type': 'text',
                            'text': GROUNDER_PROMPT.format(query)
                        }]
                    }]

                    text = processor.apply_chat_template(messages, add_generation_prompt=True)
                    images, videos = process_vision_info(messages)
                    data = processor(text=[text], images=images, videos=videos, return_tensors='pt')
                    data = data.to(device)

                    model.base_model.disable_adapter_layers()
                    model.base_model.enable_adapter_layers()
                    model.set_adapter('grounder')

                    output_ids = model.generate(
                        **data,
                        do_sample=False,
                        temperature=None,
                        top_p=None,
                        top_k=None,
                        repetition_penalty=None,
                        max_new_tokens=256)

                    assert data.input_ids.size(0) == output_ids.size(0) == 1
                    output_ids = output_ids[0, data.input_ids.size(1):]
                    if output_ids[-1] == processor.tokenizer.eos_token_id:
                        output_ids = output_ids[:-1]
                    response = processor.decode(output_ids, clean_up_tokenization_spaces=False)

                    dump['grounder_response'] = response
                    dump['grounder_success'] = len(model.reg) > 0

                    if dump['grounder_success']:
                        blob = model.reg[0].cpu().float()
                        pred, conf = blob[:, :2] * duration, blob[:, -1].tolist()
                        pred = pred.clamp(min=0, max=duration)
                        unit = getattr(dataset_cls, 'UNIT', 0.001)
                        pred = torch.round(pred / unit).long() * unit
                        inds = (pred[:, 1] - pred[:, 0] < 0).nonzero()[:, 0]
                        pred[inds] = pred[inds].roll(1)
                        pred = pred.tolist()
                    else:
                        if adapter_state['verifier']:
                            pred = [[i * duration / 6, (i + 2) * duration / 6] for i in range(5)]
                            conf = [0] * 5
                        else:
                            pred = [[0, duration]]
                            conf = [0]

                    dump['pred'] = pred
                    dump['conf'] = conf
                
                # --- 验证器 (Verifier) ---
                probs = [] 
                if do_grounding and adapter_state['verifier'] and len(pred) > 1:
                    dump['agents'].append('verifier')

                    max_frames_ver_base = 64
                    fps_ver_base = 2.0
                    
                    # 动态调整 Verifier 的帧率和最大帧数
                    new_fps_ver = min(fps_ver_base, current_fps * 2.0) 
                    new_max_frames_ver = int(max_frames_ver_base * (new_fps_ver / fps_ver_base))

                    for cand in pred[:5]:
                        s0, e0 = parse_span(cand, duration, 2)
                        offset = (e0 - s0) / 2
                        s1, e1 = parse_span([s0 - offset, e0 + offset], duration)

                        s = (s0 - s1) / (e1 - s1)
                        e = (e0 - s1) / (e1 - s1)

                        messages = [{
                            'role':
                            'user',
                            'content': [{
                                'type': 'video',
                                'video': video_path,
                                'num_threads': args.num_threads,
                                'video_start': s1,
                                'video_end': e1,
                                'min_pixels': 36 * 28 * 28,
                                'max_pixels': 64 * 28 * 28,
                                'max_frames': new_max_frames_ver, # 动态帧数
                                'fps': new_fps_ver                 # 动态帧率
                            }, {
                                'type': 'text',
                                'text': VERIFIER_PROMPT.format(question)
                            }]
                        }]

                        text = processor.apply_chat_template(messages, add_generation_prompt=True)
                        images, videos = process_vision_info(messages)
                        data = processor(text=[text], images=images, videos=videos, return_tensors='pt')

                        video_grid_thw = data['video_grid_thw'][0]
                        num_frames, window = int(video_grid_thw[0]), int(video_grid_thw[1] * video_grid_thw[2] / 4)
                        assert num_frames * window * 4 == data['pixel_values_videos'].size(0)

                        pos_s, pos_e = round(s * num_frames), round(e * num_frames)
                        pos_s, pos_e = min(max(0, pos_s), num_frames), min(max(0, pos_e), num_frames)
                        assert pos_s <= pos_e, (num_frames, s, e)

                        base_idx = torch.nonzero(data['input_ids'][0] == model.config.vision_start_token_id).item()
                        pos_s, pos_e = pos_s * window + base_idx + 1, pos_e * window + base_idx + 2

                        input_ids = data['input_ids'][0].tolist()
                        input_ids.insert(pos_s, model.config.seg_s_token_id)
                        input_ids.insert(pos_e, model.config.seg_e_token_id)
                        data['input_ids'] = torch.LongTensor([input_ids])
                        data['attention_mask'] = torch.ones_like(data['input_ids'])

                        data = data.to(device)

                        model.base_model.disable_adapter_layers()
                        model.base_model.enable_adapter_layers()
                        model.set_adapter('verifier')

                        with torch.inference_mode():
                            logits = model(**data).logits[0, -1].softmax(dim=-1)

                        score = (logits[9454] - logits[2753]).sigmoid().item()
                        probs.append(score)

                    ranks = torch.Tensor(probs).argsort(descending=True).tolist()

                    pred = [pred[idx] for idx in ranks]
                    conf = [conf[idx] for idx in ranks]
                    probs = [probs[idx] for idx in ranks]

                    dump['probs'] = probs
                    dump['ranks'] = ranks
                    dump['pred_ori'] = dump['pred']
                    dump['conf_ori'] = dump['conf']
                    dump['pred'] = pred
                    dump['conf'] = conf

                # --- 回答器 (Answerer) ---
                if do_answering:
                    dump['agents'].append('answerer')

                    if args.style in ('mcq', 'options'):
                        prompt_for_answerer = question + '\nOptions:'
                        for idx, opt in enumerate(options):
                            prompt_for_answerer += f"\n({chr(ord('A') + idx)}) {opt[0].upper() + opt[1:]}"
                        prompt_for_answerer += '\nPlease only give the best option.'
                    else:
                        prompt_for_answerer = question
                    
                    final_response = ''
                    selected = None
                    s, e = None, None
                    
                    # 动态调整 Answerer 的帧率和最大帧数
                    max_frames_ans_base = getattr(dataset_cls, 'MAX_FRAMES', 32)
                    fps_ans_base = getattr(dataset_cls, 'FPS', 2.0)
                    new_fps_ans = min(fps_ans_base, current_fps * 2.0) 
                    new_max_frames_ans = int(max_frames_ans_base * (new_fps_ans / fps_ans_base))

                    # 1. Top-2 答案融合逻辑
                    if 'pred' in dump and len(dump['pred']) >= 2 and adapter_state['verifier'] and 'probs' in dump:
                        
                        moment1, moment2 = dump['pred'][0], dump['pred'][1]
                        conf1, conf2 = dump['probs'][0], dump['probs'][1]
                        
                        # 生成 ans1 和 ans2
                        ans1, moment1_cropped = generate_answer_for_moment(
                            moment1, anno, prompt_for_answerer, video_path, duration, args, processor, model, device, dataset_cls, adapter_state, new_max_frames_ans, new_fps_ans)
                        
                        ans2, moment2_cropped = generate_answer_for_moment(
                            moment2, anno, prompt_for_answerer, video_path, duration, args, processor, model, device, dataset_cls, adapter_state, new_max_frames_ans, new_fps_ans)

                        # 计算权重并选择最终答案/时刻
                        w1, w2 = calculate_weights(conf1, conf2, T=1.0)
                        final_answer, final_moment = select_final_answer(ans1, ans2, w1, w2, moment1, moment2)
                        
                        # 确定用于 Dump 的最终结果
                        if final_moment == moment1:
                            s, e = moment1_cropped
                            final_response = ans1
                        else:
                            s, e = moment2_cropped
                            final_response = ans2
                        
                        selected = final_moment
                        
                        # 写入 Dump
                        dump['ans1_response'] = ans1
                        dump['ans2_response'] = ans2
                        dump['moment1'] = moment1
                        dump['moment2'] = moment2
                        dump['weight1'] = w1
                        dump['weight2'] = w2
                        dump['final_moment'] = selected
                        
                    # 2. 回退到原始逻辑：单个时刻或完整视频模式
                    else:
                        selected = pred[0] if 'pred' in dump else [0, duration]
                        
                        final_response, selected_cropped = generate_answer_for_moment(
                            selected, anno, prompt_for_answerer, video_path, duration, args, processor, model, device, dataset_cls, adapter_state, new_max_frames_ans, new_fps_ans)
                        s, e = selected_cropped
                        
                    dump['answerer_response'] = final_response
                    dump['response'] = final_response
                
                # 成功处理，跳出重试循环
                sample_processed_successfully = True
                break

            except torch.cuda.OutOfMemoryError as e:
                # OOM 错误处理
                torch.cuda.empty_cache() # 清理缓存
                # if retry_count < MAX_RETRIES - 1:
                #     # 继续下一次重试
                # else:
                #     # 达到最大重试次数
                pass
            
            except Exception as e:
                # 其它错误直接重新抛出
                raise e

        # 处理所有重试后的最终结果
        if sample_processed_successfully:
            dumps.append(dump)
        else:
            # 所有重试均失败，追加原始样本并标记错误
            dump = copy.deepcopy(annos[i])
            dump['error'] = "All retries for OOM failed"
            dumps.append(dump)


    nncore.dump(dumps, pred_path)