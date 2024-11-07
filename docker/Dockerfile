FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ARG UID=1000
ARG GID=1000
ARG GIT_ACCESS_TOKEN=""

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    python3-venv \
    unzip \
    vim \
    tmux \
    git \
    cmake \
    build-essential \
    libbz2-dev \
    flex \
    bison \
    libglib2.0-dev \
    libgl1-mesa-glx \
    ninja-build \
    curl \
    wget

# mainly for wandb
RUN ln -s /usr/bin/python3 /usr/bin/python

RUN useradd -m -u ${UID} -g ${GID} user

USER user
RUN pip install --upgrade pip

RUN pip install wandb pytest hydra-core requests
RUN pip install openai anthropic google-generativeai

pip install gym==0.23
RUN pip install textworld
RUN pip install crafter
RUN pip install balrog-nle
RUN pip install git+https://github.com/nacloos/baba-is-ai.git
RUN pip install git+https://github.com/BartekCupial/Minigrid.git
RUN pip install git+https://github.com/facebookresearch/minihack.git
RUN pip install .
RUN python post_install.py 

RUN echo "alias wandb='/home/user/.local/bin/wandb'" >> /home/user/.bashrc
RUN echo "alias pytest='/home/user/.local/bin/pytest'" >> /home/user/.bashrc

WORKDIR /home/user/workspace
