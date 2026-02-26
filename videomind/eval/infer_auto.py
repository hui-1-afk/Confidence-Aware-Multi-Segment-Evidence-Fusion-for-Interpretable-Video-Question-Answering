# Copyright (c) 2025 Ye Liu. Licensed under the BSD-3-Clause License.

import argparse
import copy
import json
import gc
import os
import traceback
from contextlib import nullcontext

import nncore
import torch

from videomind.constants import GROUNDER_PROMPT, PLANNER_PROMPT, VERIFIER_PROMPT
from videomind.dataset.hybrid import DATASETS
from videomind.dataset.utils import process_vision_info
from videomind.model.builder import build_model
from videomind.utils.io import get_duration, load_subtitle
from videomind.utils.parser import parse_query, parse_span


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
    # 新增参数：最大重试次数
    parser.add_argument('--max_retries', type=int, default=5, help='Max retries for OOM errors')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    # 确定输出路径
    if args.chunk > 1:
        pred_path = nncore.join(args.pred_path, f'output_{args.index}.json')
    else:
        pred_path = nncore.join(args.pred_path, 'output.json')

    # print(f'Dataset: {args.dataset}({args.split}) Chunk: {args.chunk} Index: {args.index} Output Path: {pred_path}')

    # 初始化 Adapter 状态
    adapter_state = dict(planner=False, verifier=False, answerer=False)

    print('Initializing role *grounder*')
    model, processor = build_model(args.model_gnd_path, device=args.device)
    device = next(model.parameters()).device

    if args.model_pla_path is not None:
        adapter_path = nncore.join(args.model_pla_path, 'planner')
        if nncore.is_dir(adapter_path):
            print('Initializing role *planner*')
            model.load_adapter(adapter_path, adapter_name='planner')
            adapter_state['planner'] = True

    if args.model_ver_path is not None:
        adapter_path = nncore.join(args.model_ver_path, 'verifier')
        if nncore.is_dir(adapter_path):
            print('Initializing role *verifier*')
            model.load_adapter(adapter_path, adapter_name='verifier')
            adapter_state['verifier'] = True

    if args.model_ans_path is not None:
        adapter_path = nncore.join(args.model_ans_path, 'answerer')
        if nncore.is_dir(adapter_path):
            print('Initializing role *answerer*')
            model.load_adapter(adapter_path, adapter_name='answerer')
            adapter_state['answerer'] = True

    dataset_cls = DATASETS.get(args.dataset)

    # 加载数据
    annos = dataset_cls.load_annos(split=args.split)
    annos = [annos[i::args.chunk] for i in range(args.chunk)][args.index]

    # ================= [新增功能 1：断点续传] =================
    dumps = []
    processed_count = 0
    if nncore.is_file(pred_path):
        print(f"Found existing output file: {pred_path}")
        try:
            dumps = nncore.load(pred_path)
            processed_count = len(dumps)
            print(f"Resuming from index {processed_count}...")
        except Exception as e:
            print(f"Failed to load existing file: {e}. Starting from scratch.")
            dumps = []
            processed_count = 0
    # ========================================================

    for i in nncore.ProgressBar(range(len(annos))):
        
        # 跳过已经处理过的数据
        if i < processed_count:
            continue

        anno = copy.deepcopy(annos[i])
        

        # ================= [新增功能 2：OOM 重试循环] =================
        retry_cnt = 0
        success = False
        dump = None 

        # === [修改点 1] 初始化动态参数 ===
        # 默认最大帧数 (你可以根据你的显卡显存调整这个初始值，比如 100 或 64)
        current_max_frames = 100 
        # 默认最大像素 (同理)
        current_max_pixels = 64 * 28 * 28

        while retry_cnt <= args.max_retries:
            try:
                # 每次重试前，重置 dump 对象
                dump = copy.deepcopy(annos[i])
                
                video_path, duration, span = anno['video_path'], anno.get('duration'), anno.get('span')

                if duration is None:
                    duration = get_duration(video_path, num_threads=args.num_threads)
                    dump['duration'] = duration

                # print()
                # print(video_path)

                do_answering = all(k in anno for k in ('question', 'options'))

                if do_answering:
                    question, options, ans = anno['question'], anno['options'], anno['ans']

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

                # ------------------- Planner -------------------
                if adapter_state['planner'] and (args.auto_rephrasing or args.auto_planning):
                    dump['agents'].append('planner')

                    messages = [{
                        'role': 'user',
                        'content': [{
                            'type': 'video',
                            'video': video_path,
                            'num_threads': args.num_threads,
                            'min_pixels': 36 * 28 * 28,
                            'max_pixels': current_max_pixels,
                            'max_frames': current_max_frames,
                            'fps': 1.0
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
                        max_new_tokens=256)

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
                        print('WARNING: Failed to parse planner response')
                    
                    # 及时释放内存
                    del data, images, videos, output_ids
                    torch.cuda.empty_cache()

                # ------------------- Grounder -------------------
                if do_grounding:
                    dump['agents'].append('grounder')
                    query = parse_query(query)

                    messages = [{
                        'role': 'user',
                        'content': [{
                            'type': 'video',
                            'video': video_path,
                            'num_threads': args.num_threads,
                            'min_pixels': 36 * 28 * 28,
                            'max_pixels': current_max_pixels,
                            'max_frames': current_max_frames,
                            'fps': 1.0
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
                        max_new_tokens=256)

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
                        print('WARNING: Failed to parse grounder response')
                        if adapter_state['verifier']:
                            pred = [[i * duration / 6, (i + 2) * duration / 6] for i in range(5)]
                            conf = [0] * 5
                        else:
                            pred = [[0, duration]]
                            conf = [0]

                    dump['pred'] = pred
                    dump['conf'] = conf
                    
                    # 及时释放内存
                    del data, images, videos, output_ids
                    torch.cuda.empty_cache()

                # ------------------- Verifier -------------------
                if do_grounding and adapter_state['verifier'] and len(pred) > 1:
                    dump['agents'].append('verifier')

                    probs = []
                    for cand in pred[:5]:
                        s0, e0 = parse_span(cand, duration, 2)
                        offset = (e0 - s0) / 2
                        s1, e1 = parse_span([s0 - offset, e0 + offset], duration)
                        s = (s0 - s1) / (e1 - s1)
                        e = (e0 - s1) / (e1 - s1)

                        messages = [{
                            'role': 'user',
                            'content': [{
                                'type': 'video',
                                'video': video_path,
                                'num_threads': args.num_threads,
                                'video_start': s1,
                                'video_end': e1,
                                'min_pixels': 36 * 28 * 28,
                                'max_pixels': current_max_pixels,
                                'max_frames': current_max_frames,
                                'fps': 2.0
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
                        pos_s, pos_e = round(s * num_frames), round(e * num_frames)
                        pos_s, pos_e = min(max(0, pos_s), num_frames), min(max(0, pos_e), num_frames)
                        
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
                        
                        # 循环内释放
                        del data, images, videos
                        torch.cuda.empty_cache()

                    ranks = torch.Tensor(probs).argsort(descending=True).tolist()
                    pred = [pred[idx] for idx in ranks]
                    conf = [conf[idx] for idx in ranks]

                    dump['probs'] = probs
                    dump['ranks'] = ranks
                    dump['pred_ori'] = dump['pred']
                    dump['conf_ori'] = dump['conf']
                    dump['pred'] = pred
                    dump['conf'] = conf

                # ------------------- Answerer -------------------
                if do_answering:
                    dump['agents'].append('answerer')
                    selected = pred[0] if 'pred' in dump else [0, duration]

                    if hasattr(dataset_cls, 'MIN_RATIO'):
                        min_len = duration * dataset_cls.MIN_RATIO
                    else:
                        min_len = getattr(dataset_cls, 'MIN_LEN', 32)

                    s, e = parse_span(selected, duration, min_len)

                    if args.use_subtitle and 'subtitle_path' in anno and nncore.is_file(anno['subtitle_path']):
                        subs = load_subtitle(anno['subtitle_path'])
                        subs = [f'{round(a - s, 1)}s - {round(b - s, 1)}s, {t}\n' for a, b, t in subs if a >= s and b <= e]
                        subs = ''.join(subs[:100])
                        prompt = f'You are given a video with {round(e - s, 1)} seconds long.\nSubtitles:\n{subs}' + prompt

                    messages = [{
                        'role': 'user',
                        'content': [{
                            'type': 'video',
                            'video': video_path,
                            'num_threads': args.num_threads,
                            'video_start': s,
                            'video_end': e,
                            'min_pixels': getattr(dataset_cls, 'MIN_PIXELS', 128) * 28 * 28,
                            'max_pixels': getattr(dataset_cls, 'MAX_PIXELS', 256) * 28 * 28,
                            'max_frames': getattr(dataset_cls, 'MAX_FRAMES', 32),
                            'fps': getattr(dataset_cls, 'FPS', 2.0)
                        }, {
                            'type': 'text',
                            'text': prompt
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
                            max_new_tokens=256)

                    output_ids = output_ids[0, data.input_ids.size(1):]
                    if output_ids[-1] == processor.tokenizer.eos_token_id:
                        output_ids = output_ids[:-1]
                    response = processor.decode(output_ids, clean_up_tokenization_spaces=False)

                    dump['answerer_response'] = response
                    dump['response'] = response
                    
                    del data, images, videos, output_ids
                    torch.cuda.empty_cache()

                # 如果代码能走到这里，说明推理成功，跳出重试循环
                success = True
                break
            
            except torch.OutOfMemoryError:
                retry_cnt += 1
                print(f"\n[Warning] CUDA OOM on: {anno['video_path']}. Retry {retry_cnt}/{args.max_retries}")
                
                # === [修改点 3] 核心：OOM 后主动降级配置 ===
                print(f"Downgrading config: Frames {current_max_frames} -> {int(current_max_frames * 0.7)}")
                current_max_frames = int(current_max_frames * 0.7)  # 每次砍掉 30% 的帧数
                current_max_pixels = int(current_max_pixels * 0.8) # 也可以选择降低分辨率
                
                # 防止降得太离谱，设个底线
                if current_max_frames < 8:
                    print("[Error] Config too low to proceed. Skipping.")
                    dump['error'] = 'OOM_Even_With_Low_Config'
                    success = True # 强行标记为处理完，跳过
                    break

                print("Cleaning up cache and retrying...")
                
                # 显式删除当前循环内可能占用的变量
                if 'data' in locals(): del data
                if 'images' in locals(): del images
                if 'videos' in locals(): del videos
                if 'output_ids' in locals(): del output_ids
                
                torch.cuda.ipc_collect()
                gc.collect()
                torch.cuda.empty_cache()
                
                if retry_cnt > args.max_retries:
                    print(f"[Error] Max retries reached for video {anno['video_path']}. Skipping.")
                    dump = copy.deepcopy(annos[i])
                    dump['error'] = 'CUDA OOM Max Retries'
                    dump['response'] = 'ERROR'
                    success = True # 标记为“已处理”（虽然是失败的），以便写入文件并继续下一个
                    break
            
            except Exception as e:
                print(f"\n[Error] Unknown error on {anno['video_path']}: {e}")
                traceback.print_exc()
                dump = copy.deepcopy(annos[i])
                dump['error'] = str(e)
                success = True
                break

        # ================= [新增功能 3：实时保存] =================
        if success and dump is not None:
            dumps.append(dump)
            nncore.dump(dumps, pred_path)
        
        # 每一轮循环结束，再次强制清理
        gc.collect()
        torch.cuda.empty_cache()