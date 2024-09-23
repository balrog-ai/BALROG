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

RUN pip install wandb pytest hydra-core
RUN pip install openai anthropic google-generativeai replicate

# Where we put all the disgusting shit
RUN mkdir /home/user/external
WORKDIR /home/user/external

RUN pip install textworld

# RUN pip install git+https://${GIT_ACCESS_TOKEN}@github.com/DavidePaglieri/nle-language-wrapper.git@021491de4251ceffafcd5d50fef7c28708d65d7c
RUN pip install nle-language-wrapper

RUN pip install git+https://github.com/nacloos/baba-is-ai.git

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

# RUN pip install git+https://github.com/facebookresearch/minihack
WORKDIR /home/user/external
RUN git clone https://github.com/facebookresearch/minihack.git
WORKDIR minihack
RUN pip install -e .
RUN python minihack/scripts/download_boxoban_levels.py 

RUN pip install git+https://github.com/BartekCupial/nle.git@099cad98e05faa605d002d1bea8cf36c4dac79b8

RUN echo "alias wandb='/home/user/.local/bin/wandb'" >> /home/user/.bashrc
RUN echo "alias pytest='/home/user/.local/bin/pytest'" >> /home/user/.bashrc

# # EXTRA DATA HERE
WORKDIR /home/user/workspace
RUN curl -L -o tw-games.zip 'https://drive.google.com/uc?export=download&id=1aeT-45-OBxiHzD9Xn99E5OvC86XmqhzA'
RUN unzip tw-games.zip

# So we are looking at the correct version of minihack
ENV PYTHONPATH=/home/user/external/minihack:${PYTHONPATH}

RUN pip install crafter

WORKDIR /home/user/workspace
