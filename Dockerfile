FROM nvidia/cuda:12.0.1-cudnn8-runtime-ubuntu22.04

# Ensure no installs try to launch interactive screen
ARG DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Update packages and install python3.9 and other dependencies
RUN apt-get update -y && \
    apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get install -y python3.9 python3.9-dev python3-pip python3.9-venv python3-dev python3-opencv swig ffmpeg git unzip wget libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.9 10 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install mini conda
# Install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p ~/miniconda3 && \
    rm ~/miniconda.sh

# Set up the environment
ENV PATH /root/miniconda3/bin:$PATH

# # Update conda and install basic packages
RUN conda update -n base -c defaults conda

# Create two conda environments
RUN conda create -n baselines210 python=3.10 -y && \
    conda create -n baselines200 python=3.10 -y

# Activate environments and set environment variables
ENV PIP200 /root/miniconda3/envs/baselines200/bin/pip
ENV PIP210 /root/miniconda3/envs/baselines210/bin/pip

# Location of baselines folder
ARG folder=/home/app/baselines

# Set working directory
WORKDIR ${folder}

# Copy all code needed to install dependencies
COPY ./environment_wrappers ./environment_wrappers
COPY ./environments ./environments
COPY ./install_environments ./install_environments
COPY ./systems ./systems
COPY ./utils ./utils
COPY all_experiments.py .
COPY main.py .
COPY requirements.txt .

##########################
# Create first conda env #
##########################
# Dependencies
RUN echo "Installing requirements..."
RUN $PIP210 install --quiet --upgrade pip setuptools wheel
RUN $PIP210 install -r requirements.txt

# MPE
RUN $PIP210 install ./environments/multiagent_particle_envs

# SMAC
ENV SC2PATH /home/app/StarCraftII
RUN ./install_environments/starcraft2.sh
RUN $PIP210 install -r ./install_environments/requirements/smacv1.txt

# MuJoCo
RUN ./install_environments/mujoco210.sh

# MAMuJoCo Requirements
RUN $PIP210 install -r ./install_environments/requirements/mamujoco.txt

# MAMuJoCo 210
RUN $PIP210 install git+https://github.com/schroederdewitt/multiagent_mujoco
ENV SUPPRESS_GR_PROMPT 1

###########################
# Create second conda env #
###########################
# Dependencies
RUN echo "Installing requirements..."
RUN $PIP200 install --quiet --upgrade pip setuptools wheel &&  \
    $PIP200 install -r requirements.txt

# MuJoCo
RUN ./install_environments/mujoco200.sh

# MAMuJoCo Requirements
RUN $PIP200 install -r ./install_environments/requirements/mamujoco.txt
# RUN rm -r /root/miniconda3/envs/baselines200/lib/libstdc++.so.6
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/root/.mujoco/mujoco200/bin
RUN $PIP200 install mujoco-py==2.0.2.5