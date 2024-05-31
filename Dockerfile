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
    python -m venv og-marl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Setup virtual env and path
ENV VIRTUAL_ENV /og-marl
ENV PATH /og-marl/bin:$PATH

# Location of og-marl folder
ARG folder=/home/app/og-marl

# Set working directory
WORKDIR ${folder}

# Copy all code needed to install dependencies
COPY ./install_environments ./install_environments
COPY ./og_marl ./og_marl
COPY ./environments ./environments
COPY setup.py .
COPY ./baselines ./baselines

RUN echo "Installing requirements..."
RUN pip install --quiet --upgrade pip setuptools wheel &&  \
    pip install -e .
RUN pip install -U "jax[cuda12]"
RUN pip install flashbax

# MPE
# COPY ./environments ./environments
# RUN pip install ./environments/multiagent-particle-envs

# SMAC
ENV SC2PATH /home/app/StarCraftII
RUN ./install_environments/smacv1.sh
# RUN ./install_environments/smacv2.sh

# MAMuJoCo
# ENV PYTHONPATH=/home/app/og-marl/environments
# RUN pip install -r ./install_environments/requirements/mamujoco.txt
# ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/root/.mujoco/mujoco200/bin:/usr/lib/nvidia
# ENV SUPPRESS_GR_PROMPT 1
# RUN ./install_environments/mamujoco_old.sh
# RUN pip install mujoco-py==2.0.2.5
