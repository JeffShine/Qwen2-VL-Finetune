#!/bin/bash

# 设置环境变量
export PYTHONPATH=src:$PYTHONPATH
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"
export CUDA_VISIBLE_DEVICES=6,7

# 模型参数
MODEL_BASE="Qwen/Qwen2.5-VL-3B-Instruct"   # 基础模型名称
DEVICE="cuda"

MODEL_PATH="output/fft_qwen25_3b_rec_merged/checkpoint-3600"  # 微调后的模型路径
OUTPUT_PATH="output_eval/fft_qwen25_3b_rec_merged-3600.json"        # 输出结果路径

# 数据集参数
DATASET_PATH="data/AS-V2/merged_rec_data.json"    # 评估数据集路径
IMAGE_FOLDER="data/image"         # 图片文件夹路径

# 运行评估脚本
python src/evaluating/eval.py \
    --model-path "$MODEL_PATH" \
    --model-base "$MODEL_BASE" \
    --device "$DEVICE" \
    --dataset-path "$DATASET_PATH" \
    --image-folder "$IMAGE_FOLDER" \
    --output-path "$OUTPUT_PATH"

echo "评估完成，结果已保存到 $OUTPUT_PATH"