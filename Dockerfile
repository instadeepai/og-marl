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


#########################
# Baseline Dependencies #
#########################

RUN echo "Installing requirements..."
RUN pip install --quiet --upgrade pip setuptools wheel
RUN pip install -r requirements.txt

############################
# Environment Dependencies #
############################

# MPE
RUN pip install ./environments/multiagent_particle_envs

# SMAC
ENV SC2PATH /home/app/StarCraftII
RUN ./install_environments/starcraft2.sh
RUN pip install -r ./install_environments/requirements/smacv1.txt




#######################
# Start of MuJoCo 210 #
#######################

# MuJoCo
RUN ./install_environments/mujoco210.sh

# MAMuJoCo Requirements
RUN pip install -r ./install_environments/requirements/mamujoco.txt

# MAMuJoCo 210
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin:/usr/lib/nvidia
RUN pip install git+https://github.com/schroederdewitt/multiagent_mujoco
ENV SUPPRESS_GR_PROMPT 1

#####################
# End of MuJoCo 210 #
#####################




#######################
# Start of MuJoCo 200 #
#######################

# # Dependencies
# RUN echo "Installing requirements..."
# RUN pip install --quiet --upgrade pip setuptools wheel &&  \
#     pip install -r requirements.txt

# # MuJoCo
# RUN ./install_environments/mujoco200.sh

# # MAMuJoCo Requirements
# RUN pip install -r ./install_environments/requirements/mamujoco.txt
# ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/root/.mujoco/mujoco200/bin:/usr/lib/nvidia
# RUN pip install mujoco-py==2.0.2.5

#####################
# End of MuJoCo 200 #
#####################