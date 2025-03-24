
RUN_NAME=$1
DATA_PATH=$2
MODEL_ARGS=${@:3}
PREV_STAGE_CHECKPOINT=${PREV_STAGE_CHECKPOINT:-"Qwen/Qwen2-0.5B-Instruct"}


echo "MID_RUN_NAME: ${RUN_NAME}"
echo "MODEL ARGS: ${MODEL_ARGS}"

deepspeed --master_port 12345 \
    slimm/train/train.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path $PREV_STAGE_CHECKPOINT \
    --data_path $DATA_PATH \
    --image_folder data \
    --video_folder data \
    --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model" \
    --group_by_modality_length True \
    --bf16 True \
    --run_name $RUN_NAME \
    --output_dir checkpoints/$RUN_NAME \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 10240 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to none \
    --frames_upbound 16 \
    --force_sample True \
    --dataloader_drop_last True \
    $MODEL_ARGS


exit 0;
