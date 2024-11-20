MODEL_PATH="/root/autodl-tmp/models/Qwen2.5-Coder-32B-Instruct"
API_KEY="EMPTY"

vllm serve $MODEL_PATH \
    --api-key $API_KEY \
    --tensor_parallel_size 1 \
    --max-model-len 8192