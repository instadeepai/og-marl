#!/bin/bash

# Make sure you have these packages installed
sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf

export MUJOCOPATH=~/.mujoco

mkdir -p $MUJOCOPATH \
    && wget https://www.roboti.us/download/mujoco200_linux.zip -O mujoco.zip \
    && unzip -d $MUJOCOPATH mujoco.zip \
    && rm mujoco.zip \
    && mv ${MUJOCOPATH}/mujoco200_linux ${MUJOCOPATH}/mujoco200 \
    && wget https://www.roboti.us/file/mjkey.txt -O mjkey.txt \
    && mv mjkey.txt $MUJOCOPATH

# IMPORTANT!!!!
# You will need to set these environment variables
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu/:$MUJOCOPATH/mujoco200/bin:/usr/lib/nvidia
export LD_PRELOAD=$LD_PRELOAD:/usr/lib/x86_64-linux-gnu/libGLEW.so
