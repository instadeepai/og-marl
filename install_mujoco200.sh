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
