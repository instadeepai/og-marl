FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

# Ensure no installs try to launch interactive prompts
ARG DEBIAN_FRONTEND=noninteractive
# Location of og-marl folder
ARG folder=/home/app/og-marl

# Set working directory
WORKDIR ${folder}

# Ensure Python output is sent straight to terminal (no buffering) for real-time logs
ENV PYTHONUNBUFFERED=1

# Install Python 3.12 + venv
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    build-essential \
    software-properties-common && \
    rm -rf /var/lib/apt/lists/*

# Setup virtual environment
RUN python3.12 -m venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN python -m ensurepip && \
    pip install --upgrade pip setuptools wheel build

# Copy requirements for better caching
COPY ./requirements ./requirements
COPY ./install_environments/requirements/mujoco.txt ./requirements
COPY ./install_environments/requirements/mamujoco200.txt ./requirements

# Install dependencies in one layer
RUN echo "Installing og-marl dependencies..."
RUN pip install -r ./requirements/datasets.txt
RUN pip install -r ./requirements/tf2_baselines.txt

# Dowload MuJoCo200
RUN echo "Downloading MuJoCo..."
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    wget \
    unzip \
    libosmesa-dev \ 
    patchelf \
    libgl1-mesa-dev \
    libglfw3 \
    libglfw3-dev \
    libglew-dev \
    patchelf \
    libosmesa6-dev \
    libffi-dev && \
    rm -rf /var/lib/apt/lists/*
RUN pip install cffi

ENV MUJOCOPATH=/root/.mujoco
RUN mkdir -p $MUJOCOPATH \
    && wget https://www.roboti.us/download/mujoco200_linux.zip -O mujoco.zip \
    && unzip -d $MUJOCOPATH mujoco.zip \
    && rm mujoco.zip \
    && mv ${MUJOCOPATH}/mujoco200_linux ${MUJOCOPATH}/mujoco200 \
    && wget https://www.roboti.us/file/mjkey.txt -O mjkey.txt \
    && mv mjkey.txt $MUJOCOPATH
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$MUJOCOPATH/mujoco200/bin

# Install MAMuJoCo Requirements
RUN pip install -r ./requirements/mujoco.txt
RUN pip install -r ./requirements/mamujoco200.txt

# Copy over og_marl code
COPY ./og_marl ./og_marl
COPY ./pyproject.toml ./pyproject.toml

# Install og-marl package
RUN echo "Installing og-marl package..." && \
    pip install -e .

# Install SMAC
ENV SC2PATH=/root/StarCraftII
ENV MAP_DIR="$SC2PATH/Maps/"

RUN echo 'Downloading StarCraftII...' && \
    wget --progress=dot:mega http://blzdistsc2-a.akamaihd.net/Linux/SC2.4.10.zip && \
    unzip -oP iagreetotheeula SC2.4.10.zip && \
    mv StarCraftII $SC2PATH && \
    rm -rf SC2.4.10.zip
    
RUN mkdir -p $MAP_DIR && \
    wget https://github.com/oxwhirl/smacv2/releases/download/maps/SMAC_Maps.zip && \
    unzip SMAC_Maps.zip -d SMAC_Maps
   
RUN mv SMAC_Maps $MAP_DIR && \
    rm -rf SMAC_Maps.zip 

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git && \
    rm -rf /var/lib/apt/lists/*

RUN pip install git+https://github.com/oxwhirl/smac.git
RUN pip install git+https://github.com/oxwhirl/smacv2.git

RUN pip install protobuf==5.28.3
ENV  PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python