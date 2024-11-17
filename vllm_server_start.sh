MODEL_PATH="/root/autodl-tmp/projects/models/Qwen2.5-Coder-32B-Instruct"
API_KEY="EMPTY"

vllm serve $MODEL_PATH \
    --api-key $API_KEY \
    --tensor_parallel_size 2 \