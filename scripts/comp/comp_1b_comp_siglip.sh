## Stage-I
DATA_PATH=scripts/data/Stage-I.yaml
MODEL_NAME=CoMP-MM-1B-CoMP-SigLIP/stage1
CKPT=checkpoints/$MODEL_NAME

bash scripts/slimm/job_template.sh $MODEL_NAME $DATA_PATH  \
    --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
    --model_max_length 4096 \
    --per_device_train_batch_size 32 --gradient_accumulation_steps 1 --max_num_vistoken 1024 \
    --custom_visual_model SliMM-X/CoMP-SigLIP-So400M \
    --mm_tunable_parts="mm_mlp_adapter" --learning_rate 1e-3 \
    --use_rope 1 \
    --use_alignment_loss 0 \


## Stage-II
PREV_CKPT=$CKPT
DATA_PATH=scripts/data/Stage-II.yaml
MODEL_NAME=CoMP-MM-1B-CoMP-SigLIP/stage2
CKPT=checkpoints/$MODEL_NAME

bash scripts/slimm/job_template.sh $MODEL_NAME $DATA_PATH  \
    --model_name_or_path $PREV_CKPT \
    --per_device_train_batch_size 16 --gradient_accumulation_steps 2 --max_num_vistoken 3000 \
    --model_max_length 8000 \
    --custom_visual_model SliMM-X/CoMP-SigLIP-So400M \
    --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model,mm_lm_head" \
    --mm_vision_tower_lr=1e-4 \
    --mm_projector_lr 5e-3 \
    --learning_rate 2e-5 \
    --use_rope 1 \
    --use_alignment_loss 0 \

## Stage-III
PREV_CKPT=$CKPT
DATA_PATH=scripts/data/Stage-III.yaml
MODEL_NAME=CoMP-MM-1B-CoMP-SigLIP/
export CKPT=checkpoints/$MODEL_NAME

bash scripts/slimm/job_template.sh $MODEL_NAME $DATA_PATH  \
    --model_name_or_path $PREV_CKPT \
    --per_device_train_batch_size 16 --gradient_accumulation_steps 1 --max_num_vistoken 3000 \
    --model_max_length 4096 \
    --custom_visual_model SliMM-X/CoMP-SigLIP-So400M \
    --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model,mm_lm_head" \
    --mm_vision_tower_lr=5e-6 \
    --learning_rate 1e-5 \
    --use_rope 1 \
    --use_alignment_loss 0 \