# we recommend 8 GPUs for each job

BASE_CKPT=Qwen/Qwen2-VL-2B-Instruct
DATA_PATH=scripts/data/llava_ov_si.yaml
MODEL_NAME=SliMM-DeepStackE-Qwen2VL-2B
CKPT=checkpoints/$MODEL_NAME

bash scripts/slimm/job_template.sh $MODEL_NAME $DATA_PATH  \
    --model_name_or_path $BASE_CKPT \
    --per_device_train_batch_size 4 --gradient_accumulation_steps 4 --max_num_vistoken 1280 \
    --model_max_length 4096 \
    --custom_visual_model menglc/DFN-Huge-Qwen2VL-7B \
    --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model" \
    --use_deepstack --deepstack_type efficient
