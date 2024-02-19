FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Ensure no installs try to launch interactive screen
ARG DEBIAN_FRONTEND=noninteractive

# Update packages and install python3.9 and other dependencies
RUN apt-get update -y && \
    apt-get install -y software-properties-common git wget unzip && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get install -y python3.9 python3.9-dev python3-pip python3.9-venv && \
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
COPY setup.py .

RUN echo "Installing requirements..."
RUN pip install --quiet --upgrade pip setuptools wheel &&  \
    pip install -e . && \
    pip install flashbax==0.1.0

ENV SC2PATH /home/app/StarCraftII
RUN ./install_environments/smacv1.sh

RUN ./install_environments/mamujoco.sh

# Copy all code
COPY ./examples ./examples
COPY ./baselines ./baselines