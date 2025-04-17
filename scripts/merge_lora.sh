#!/bin/bash

MODEL_NAME="Qwen/Qwen2.5-VL-3B-Instruct"
# MODEL_NAME="Qwen/Qwen2-VL-2B-Instruct"

export PYTHONPATH=src:$PYTHONPATH

python src/merge_lora_weights.py \
    --model-path /home/jinchenhui/ustc/Qwen2-VL-Finetune/output/lora_qwen25_3b_rec_22k \
    --model-base $MODEL_NAME  \
    --save-model-path /home/jinchenhui/ustc/Qwen2-VL-Finetune/output/lora-merged_qwen25_3b_rec_22k \
    --safe-serialization

python src/merge_lora_weights.py \
    --model-path /home/jinchenhui/ustc/Qwen2-VL-Finetune/output/lora_qwen25_3b_rec_63k \
    --model-base $MODEL_NAME  \
    --save-model-path /home/jinchenhui/ustc/Qwen2-VL-Finetune/output/lora-merged_qwen25_3b_rec_63k \
    --safe-serialization

python src/merge_lora_weights.py \
    --model-path /home/jinchenhui/ustc/Qwen2-VL-Finetune/output/lora_qwen25_3b_rec_merged \
    --model-base $MODEL_NAME  \
    --save-model-path /home/jinchenhui/ustc/Qwen2-VL-Finetune/output/lora-merged_qwen25_3b_rec_merged \
    --safe-serialization