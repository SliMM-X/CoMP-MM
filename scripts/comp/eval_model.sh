MODEL=${MODEL:-'llava'}
NGPUS=${NGPUS:-'8'}
PORT=${PORT:-'29500'}
MODEL_ARGS=${@}
echo $MODEL $CKPT
echo $eval_datasets 

accelerate launch --num_processes=$NGPUS --main_process_port=$PORT \
    -m lmms_eval \
    --model $MODEL \
    --model_args pretrained=${CKPT},${MODEL_ARGS} \
    --tasks $eval_datasets \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix outputs \
    --output_path $CKPT/logs