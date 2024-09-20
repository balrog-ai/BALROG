docker kill icl-bench-c
docker rm icl-bench-c
docker run \
    --env WANDB_API_KEY \
    --user $(id -u):$(id -g) \
    --gpus all \
    --volume $(pwd):/home/user/workspace \
    --name icl-bench-c \
    -it icl-bench