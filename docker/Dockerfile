FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3-pip \
    vim \
    tmux \
    git

# mainly for wandb
RUN ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /app

# RUN pip install --upgrade jumanji
RUN pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
RUN pip install --upgrade \
    optax \
    flax \
    chex \
    distrax \
    matplotlib \
    notebook \
    pytest \
    wandb \
    transformers \
    orbax-checkpoint \
    moviepy \
    imageio