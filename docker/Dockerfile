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
    bison \
    libglib2.0-dev \
    libgl1-mesa-glx

# mainly for wandb
RUN ln -s /usr/bin/python3 /usr/bin/python

RUN useradd -m -u ${UID} -g ${GID} user

USER user
WORKDIR /home/user/workspace

# Jax and asscociated libraries
RUN pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
RUN pip install --upgrade optax flax chex distrax gymnax craftax

# Torch and associated libraries
RUN pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Huggingface and associated libraries
RUN pip install --upgrade transformers accelerate nle-language-wrapper

# Common libraries
RUN pip install --upgrade matplotlib notebook pytest wandb moviepy imageio

# Install unnecessary bullshit
RUN pip install opencv-python pandas datasets numba nltk jsonlines h5py seaborn
