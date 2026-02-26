#!/bin/bash

# ==============================================================================
# HM-RAG 模块级消融实验 (Module-level Ablation)
# 涵盖: Vanilla, w/o Planner, w/o Verifier, Full (Ours)
# ==============================================================================

set -e

# --- 1. 环境配置 ---
export CUDA_VISIBLE_DEVICES=1,6  # 使用你的显卡编号
export PYTHONPATH="./:$PYTHONPATH"

dataset=$1
split=${2:-"test"}

# 基础模型路径
MODEL_BASE="model_zoo/VideoMind-2B"

# 获取 GPU 列表
IFS="," read -ra GPULIST <<< "${CUDA_VISIBLE_DEVICES:-0}"
CHUNKS=${#GPULIST[@]}

# 结果汇总文件
RESULT_LOG="module_ablation_${dataset}.txt"
echo "=== Module Ablation Study: $dataset ===" > $RESULT_LOG

# --- 2. 定义四个对照组任务 ---
# 格式: "任务名;是否跳过定位;Planner路径;Verifier路径"
TASKS=(
    "Vanilla;true;none;none"         # 1. 纯回答器：全视频推理，无任何 Agent
    "wo_Planner;false;none;$MODEL_BASE" # 2. 无 Planner：直接定位 + 验证
    "wo_Verifier;false;$MODEL_BASE;none" # 3. 无 Verifier：规划 + 定位，直接回答
    "Full_Ours;false;$MODEL_BASE;$MODEL_BASE" # 4. 完整版：HM-RAG 全流程
)

# --- 3. 循环执行任务 ---
for task in "${TASKS[@]}"; do
    IFS=";" read -ra CONF <<< "$task"
    NAME=${CONF[0]}
    SKIP_GND=${CONF[1]}
    PLA_PATH=${CONF[2]}
    VER_PATH=${CONF[3]}

    PRED_PATH="outputs_ablation/${dataset}_${NAME}"
    echo -e "\n\e[1;33m[Running]\e[0m \e[1;32m$NAME\e[0m"

    # 构建基础参数
    ARGS="--dataset $dataset --split $split --pred_path $PRED_PATH --model_gnd_path $MODEL_BASE --model_ans_path $MODEL_BASE"

    # 根据配置添加模块开关
    if [ "$SKIP_GND" = "true" ]; then
        ARGS="$ARGS --no_grounding"
    fi

    if [ "$PLA_PATH" != "none" ]; then
        ARGS="$ARGS --model_pla_path $PLA_PATH --auto_rephrasing --auto_planning"
    fi

    if [ "$VER_PATH" != "none" ]; then
        ARGS="$ARGS --model_ver_path $VER_PATH"
    fi

    # --- 多卡并行推理 ---
    for IDX in $(seq 0 $((CHUNKS-1))); do
        GPU_ID=${GPULIST[$IDX]}
        CUDA_VISIBLE_DEVICES=$GPU_ID python videomind/eval/infer_ab_modules.py \
            $ARGS \
            --chunk $CHUNKS \
            --index $IDX &
    done

    wait # 等待推理结束

    # --- 评估并记录 ---
    echo "Evaluating $NAME..."
    echo -e "\n>> $NAME Result:" >> $RESULT_LOG
    python videomind/eval/eval_auto.py $PRED_PATH --dataset $dataset >> $RESULT_LOG
    echo "------------------------------------------------" >> $RESULT_LOG
done

echo -e "\n\e[1;36m[Done]\e[0m 实验结果已汇总至: $RESULT_LOG"
