#!/bin/bash

# Make sure you have these packages installed 
sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf

export MUJOCOPATH=~/.mujoco

mkdir -p $MUJOCOPATH \
    && wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O mujoco.tar.gz \
    && tar -xf mujoco.tar.gz -C $MUJOCOPATH \
    && rm mujoco.tar.gz

# Install MA Mujoco
pip install -r install_environments/requirements/mamujoco.txt

# IMPORTANT!!!!
# You will need to set these environment variables everytime you start a new terminal
# or add them to your .bashrc file
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu/:$MUJOCOPATH/mujoco210/bin:/usr/lib/nvidia
export LD_PRELOAD=$LD_PRELOAD:/usr/lib/x86_64-linux-gnu/libGLEW.so