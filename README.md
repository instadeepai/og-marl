# PJAP

This code is based on the OG-MARL code: [https://github.com/instadeepai/og-marl](https://github.com/instadeepai/og-marl)

# Setup

1. `conda create --name pjap python=3.10`

Then activate the conda environment.

2. `pip install -r requirements1.txt`

Make sure to install requirements in this order.

3. `pip install -r requirements2.txt`

Make sure you inspect the i`nstall_mamujoco.sh` script first.

4. `bash install_mamujoco.sh`

5. `pip install -r mamujoco_requirements.txt`

# Run

`python main.py --system=maddpg+cql+pjap`

# Tested on Ubuntu 22.04

We also provide a Dockerfile.