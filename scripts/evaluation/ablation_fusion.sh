#!/bin/bash

set -e

# 设置显卡
export CUDA_VISIBLE_DEVICES=1,6
export PYTHONPATH="./:$PYTHONPATH"

dataset=$1
split=${2:-"test"}

model_gnd_path="model_zoo/VideoMind-2B"
model_ver_path="model_zoo/VideoMind-2B"
model_pla_path="model_zoo/VideoMind-2B"

# 定义消融模式列表
MODES=("softmax" "top1" "equal" "hard")

for mode in "${MODES[@]}"; do
    # 为每个模式创建独立的输出路径
    pred_path="outputs_ablation/${dataset}_${split}_${mode}"
    echo -e "\e[1;33m[Running Ablation]\e[0m Mode: \e[1;32m$mode\e[0m | Dataset: $dataset"

    IFS="," read -ra GPULIST <<< "${CUDA_VISIBLE_DEVICES:-0}"
    CHUNKS=${#GPULIST[@]}

    # 并行推理
    for IDX in $(seq 0 $((CHUNKS-1))); do
        CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python videomind/eval/infer_ablation.py \
            --dataset $dataset \
            --split $split \
            --pred_path $pred_path \
            --model_gnd_path $model_gnd_path \
            --model_ver_path $model_ver_path \
            --model_pla_path $model_pla_path \
            --fusion_mode $mode \
            --chunk $CHUNKS \
            --index $IDX &
    done

    wait

    # 运行评估脚本并记录结果
    echo -e "\e[1;36m[Evaluating]\e[0m Mode: $mode"
    python videomind/eval/eval_auto.py $pred_path --dataset $dataset >> ablation_results.txt
    echo "------------------------------------------------" >> ablation_results.txt
done

echo "消融实验全部完成！结果已保存在 ablation_results.txt"