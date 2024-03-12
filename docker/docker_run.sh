docker run \
    --env WANDB_API_KEY \
    --user $(id -u):$(id -g) \
    --gpus all \
    --volume $(pwd):/home/user/workspace \
    --volume ~/.cache/huggingface:/home/user/.cache/huggingface \
    --name scx \
    -it scllm