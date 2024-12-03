FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    python3-venv \
    unzip \
    vim \
    tmux \
    git \
    build-essential \
    libglib2.0-dev \
    ninja-build \
    curl \
    wget

ARG PYTHON_VERSION=3.10
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /opt/conda_setup

RUN curl -o miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x miniconda.sh && \
    ./miniconda.sh -b -p /opt/conda && \
    /opt/conda/bin/conda install -y python=$PYTHON_VERSION && \
    /opt/conda/bin/conda clean -ya
ENV PATH=/opt/conda/bin:$PATH

RUN python -m pip install --upgrade pip ipython ipdb

COPY . /opt/balrog/

WORKDIR /opt/balrog

RUN pip install '.'
RUN balrog-post-install

CMD ["/bin/bash"]
