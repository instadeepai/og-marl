
`conda create --name baselines python=3.9`

`conda activate baselines`

`pip install -r requirements.txt`

For SMAC.

`bash install install_environments/starcraft2.sh`
`pip install -r install_environments/requirements/smacv1.txt`

For MAMuJoCo experiments on OG-MARL datasets

`bash install install_environments/mamujoco210.sh`
`pip install -r install_environments/requirements/mamujoco210.txt`

For MAMuJoCo experiments on CFCQL or OMIGA datasets, you will need to install an older version of MuJoCo.
Installing two versions of MuJoCo in the same Conda environment can be complicated. We reccomend using seperate conda envs.

`bash install install_environments/mamujoco200.sh`

`pip install -r install_environments/requirements/mamujoco200.txt`

`pip install mujoco-py==2.0.2.5`

Error: `lib/libstdc++.so.6: version 'GLIBCXX_3.4.30' not found (required by /usr/lib/x86_64-linux-gnu/libLLVM-15.so.1)`

Solution: `rm -r ~/miniconda3/envs/baselines/lib/libstdc++.so.6`

MPE

`pip install environments/multiagent_particle_envs/`