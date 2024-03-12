FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ARG UID=1000
ARG GID=1000

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    vim \
    tmux \
    git \
    cmake \
    build-essential \
    libbz2-dev \
    flex \
    bison

# mainly for wandb
RUN ln -s /usr/bin/python3 /usr/bin/python

RUN useradd -m -u ${UID} -g ${GID} user

USER user
WORKDIR /home/user/workspace

# RUN pip install --upgrade jumanji
RUN pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip install --upgrade \
    matplotlib \
    notebook \
    pytest \
    wandb \
    transformers \
    accelerate \
    moviepy \
    imageio \
    nle-language-wrapper