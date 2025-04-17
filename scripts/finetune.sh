#!/bin/bash

# You can use 2B instead of 7B
# MODEL_NAME="Qwen/Qwen2-VL-7B-Instruct"
# MODEL_NAME="Qwen/Qwen2-VL-2B-Instruct"
# MODEL_NAME="Qwen/Qwen2.5-VL-3B-Instruct"
MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"

export PYTHONPATH=src:$PYTHONPATH
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"
export CUDA_VISIBLE_DEVICES=4,5,6,7
GLOBAL_BATCH_SIZE=32
BATCH_PER_DEVICE=8
NUM_DEVICES=4
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))

export PROMPT_TEMPLATE='''
Task: Generate a detailed scene graph caption for the given image.

Requirements:

- Use <ref>object_name</ref><box>[x1, y1, x2, y2]</box> to annotate each object and its corresponding bounding box.
- Use <pred>relation</pred><box>[subject_box]</box><box>[object_box]</box> to describe spatial or functional relationships between objects using their bounding boxes.
- Each <pred> relation must be followed by two <box> tags, where the first <box> corresponds to the subject, and the second <box> to the object.
- If an object has multiple bounding boxes (e.g., several <ref>motorcycles</ref>), combine them into a single <box> with a list of box coordinates.
- Be consistent in naming: each object must use the same <ref> tag label across the caption.
- Avoid duplicating <pred> relations for the same object pair unless the relation is different.
- The caption should read as a natural paragraph describing the scene and the relationships between all major objects (e.g., people, vehicles, buildings, nature, etc.).

Example Output Format:
"In the image, there is a <ref>person</ref><box>[[162, 327, 259, 615]]</box>
<pred>standing on</pred><box>[[162, 327, 259, 615]]</box><box>[[103, 459, 804, 831]]</box> the <ref>pavement</ref><box>[[103, 459, 804, 831]]</box>
in front of a <ref>building</ref><box>[[0, 165, 999, 725]]</box> that appears to be a shop. There are several <ref>motorcycles</ref><box>[[529, 373, 617, 467], [517, 439, 637, 589], [717, 445, 999, 831]]</box>
<pred>parked on</pred><box>[[529, 373, 617, 467], [517, 439, 637, 589], [717, 445, 999, 831]]</box><box>[[103, 459, 804, 831]]</box> the pavement..."
'''
deepspeed src/training/train.py \
    --use_liger True \
    --deepspeed scripts/zero3_offload.json \
    --model_id $MODEL_NAME \
    --data_path data/AS-V2/rec_detailed_description_42k.json \
    --image_folder data/image \
    --remove_unused_columns False \
    --freeze_vision_tower False \
    --freeze_llm False \
    --tune_merger True \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --output_dir output/fft_7b_rec_conversation_42k_finetune \
    --num_train_epochs 4 \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --image_min_pixels $((256 * 28 * 28)) \
    --image_max_pixels $((1280 * 28 * 28)) \
    --learning_rate 1e-5 \
    --merger_lr 1e-5 \
    --vision_lr 2e-6 \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --report_to tensorboard \
    --lazy_preprocess True \
    --save_strategy "steps" \
    --save_steps 400 \
    --save_total_limit 10 \
    --dataloader_num_workers 4