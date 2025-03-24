# we recommend 8 GPUs for each job

DATA_PATH=scripts/data/lcs_558k_pt.yaml
MODEL_NAME=SliMM-Qwen2-0.5B/stage1

CKPT=checkpoints/$MODEL_NAME

bash scripts/slimm/job_template.sh $MODEL_NAME $DATA_PATH  \
    --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
    --model_max_length 4096 \
    --per_device_train_batch_size 32 --gradient_accumulation_steps 1 --max_num_vistoken 1024 \
    --model_name_or_path $BASE_CKPT \
    --custom_visual_model menglc/DFN-Huge-Qwen2VL-7B \
    --mm_tunable_parts="mm_mlp_adapter" --learning_rate 1e-3 


PREV_CKPT=$CKPT
DATA_PATH=scripts/data/llava_ov_mid.yaml
MODEL_NAME=SliMM-Qwen2-0.5B/stage1.5
CKPT=checkpoints/$MODEL_NAME

bash scripts/slimm/job_template.sh $MODEL_NAME $DATA_PATH  \
    --model_name_or_path $PREV_CKPT \
    --per_device_train_batch_size 32 --gradient_accumulation_steps 1 --max_num_vistoken 1024 \
    --model_max_length 4096 \
    --custom_visual_model menglc/DFN-Huge-Qwen2VL-7B \
    --mm_tunable_parts="mm_language_model,mm_mlp_adapter" \
    --mm_projector_lr 5e-3 \
    --learning_rate 2e-5 


PREV_CKPT=$CKPT
DATA_PATH=scripts/data/llava_ov_si.yaml
MODEL_NAME=SliMM-Qwen2-0.5B/
CKPT=checkpoints/$MODEL_NAME

bash scripts/slimm/job_template.sh $MODEL_NAME $DATA_PATH  \
    --model_name_or_path $PREV_CKPT \
    --per_device_train_batch_size 8 --gradient_accumulation_steps 2 --max_num_vistoken 1024 \
    --model_max_length 4096 \
    --custom_visual_model menglc/DFN-Huge-Qwen2VL-7B \
    --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model" \
    --mm_vision_tower_lr=1e-6 \
    --mm_projector_lr 5e-3 \
    --learning_rate 1e-5 
