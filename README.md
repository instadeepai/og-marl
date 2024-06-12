# Create conda env
`conda create --name baselines200 python=3.9`

`conda activate baselines200`

`pip install -r requirements.txt`

# Install SMAC.

`bash install install_environments/starcraft2.sh`

`pip install -r install_environments/requirements/smacv1.txt`

# MPE

`pip install environments/multiagent_particle_envs/`

# Install MAMuJoCo200

`bash install install_environments/mamujoco200.sh`

`pip install -r install_environments/requirements/mamujoco200.txt`

`export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco200/bin`

`pip install mujoco-py==2.0.2.5`

> MAMuJoCo experiments on CFCQL or OMIGA datasets use an older version of MuJoCo. To run experiments on the OG-MARL datasets,
you will need to install a newer version of MuJoCo. Installing two versions of MuJoCo in the same conda environment can be complicated. We reccomend using seperate conda envs.

# Install MAMuJoCo210 in a new conda env
`conda create --name baselines210 python=3.9`

`conda activate baselines210`

`pip install -r requirements.txt`

`bash install install_environments/mamujoco210.sh`

`pip install -r install_environments/requirements/mamujoco.txt`

`export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin:/usr/lib/nvidia`

`pip install git+https://github.com/schroederdewitt/multiagent_mujoco`


# Troubleshooting

Error: `lib/libstdc++.so.6: version 'GLIBCXX_3.4.30' not found (required by /usr/lib/x86_64-linux-gnu/libLLVM-15.so.1)`

Solution: `rm -r ~/miniconda3/envs/baselines200/lib/libstdc++.so.6`