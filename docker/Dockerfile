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
    curl

# mainly for wandb
RUN ln -s /usr/bin/python3 /usr/bin/python

RUN useradd -m -u ${UID} -g ${GID} user

USER user
RUN pip install --upgrade pip

RUN pip install wandb pytest hydra-core
RUN pip install openai anthropic google-generativeai replicate

# Where we put all the disgusting shit
RUN mkdir /home/user/external
WORKDIR /home/user/external

RUN pip install git+https://github.com/flowersteam/Grounding_LLMs_with_online_RL.git@31847fb8c1db58bc8f595bea8f8cbaf26fb79dc2#subdirectory=babyai-text
# TODO: this really needs to be fixed in babyai - I'll just hack it for now...
# RUN pip install git+https://github.com/flowersteam/Grounding_LLMs_with_online_RL.git@31847fb8c1db58bc8f595bea8f8cbaf26fb79dc2#subdirectory=babyai-text/babyai
RUN git clone https://github.com/flowersteam/Grounding_LLMs_with_online_RL.git
WORKDIR Grounding_LLMs_with_online_RL/babyai-text/babyai
RUN git fetch --depth 1 origin 31847fb8c1db58bc8f595bea8f8cbaf26fb79dc2
RUN git checkout 31847fb8c1db58bc8f595bea8f8cbaf26fb79dc2
RUN sed -i -e '/from setuptools import setup/a from setuptools import find_packages' -e 's/packages=\[.*\]/packages=find_packages(include=\[\"babyai*\"\])/' setup.py
RUN pip install .
RUN pip install git+https://github.com/flowersteam/Grounding_LLMs_with_online_RL.git@31847fb8c1db58bc8f595bea8f8cbaf26fb79dc2#subdirectory=babyai-text/gym-minigrid

RUN pip install git+https://github.com/nacloos/baba-is-ai.git
RUN pip install gym==0.26.2

# RUN pip install git+https://github.com/BartekCupial/nle.git@099cad98e05faa605d002d1bea8cf36c4dac79b8
# RUN pip install git+https://${GIT_ACCESS_TOKEN}@github.com/DavidePaglieri/nle-language-wrapper.git@021491de4251ceffafcd5d50fef7c28708d65d7c
# RUN pip install git+https://github.com/Farama-Foundation/Minigrid.git
# RUN pip install git+https://github.com/facebookresearch/minihack
# RUN pip install textworld
# RUN pip install craftax


RUN echo "alias wandb='/home/user/.local/bin/wandb'" >> /home/user/.bashrc
RUN echo "alias pytest='/home/user/.local/bin/pytest'" >> /home/user/.bashrc

WORKDIR /home/user/workspace
