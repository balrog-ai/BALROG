# FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
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
    libgl1-mesa-glx

# mainly for wandb
RUN ln -s /usr/bin/python3 /usr/bin/python

RUN useradd -m -u ${UID} -g ${GID} user

# dirty hack
RUN mkdir -p /home/user/.cache && chown -R ${UID}:${GID} /home/user/.cache

USER user
# ENV PATH="/home/user/.local/bin:${PATH}"
RUN pip install --upgrade pip

RUN pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
RUN pip install --upgrade optax flax chex distrax gymnax craftax
RUN pip install --upgrade torch
RUN pip install --upgrade transformers datasets accelerate trl peft bitsandbytes
RUN pip install --upgrade matplotlib notebook pytest wandb moviepy imageio
RUN pip install --upgrade vllm
RUN pip install --upgrade nltk jsonlines opencv-python

WORKDIR /home/user
RUN git clone https://dramacow:${GIT_ACCESS_TOKEN}@github.com/DramaCow/jaxued
WORKDIR /home/user/jaxued
RUN pip install .

WORKDIR /home/user
RUN git clone https://dramacow:${GIT_ACCESS_TOKEN}@github.com/DavidePaglieri/nle-language-wrapper.git --recursive
WORKDIR /home/user/nle-language-wrapper
RUN pip install ".[dev]"
# RUN rm -r /home/user/nle-language-wrapper

RUN pip install envpool

WORKDIR /home/user/workspace
