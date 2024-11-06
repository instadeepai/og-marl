# Instalation
## Create conda env
`conda create --name baselines200 python=3.9`

`conda activate baselines200`

`pip install -r requirements.txt`

## Install SMAC.

`bash install install_environments/starcraft2.sh`

`pip install -r install_environments/requirements/smacv1.txt`

## Install MPE

`pip install environments/multiagent_particle_envs/`

## Install MAMuJoCo200

`bash install install_environments/mamujoco200.sh`

`pip install -r install_environments/requirements/mamujoco200.txt`

`export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco200/bin`

`pip install mujoco-py==2.0.2.5`

> MAMuJoCo experiments on OMAR, CFCQL or OMIGA datasets use an older version of MuJoCo. To run experiments on the OG-MARL datasets,
you will need to install a newer version of MuJoCo. Installing two versions of MuJoCo in the same conda environment can be complicated. We reccomend using seperate conda envs.

## Install MAMuJoCo210 in a new conda env
`conda create --name baselines210 python=3.9`

`conda activate baselines210`

`pip install -r requirements.txt`

`bash install install_environments/mamujoco210.sh`

`pip install -r install_environments/requirements/mamujoco.txt`

`export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin:/usr/lib/nvidia`

`pip install git+https://github.com/schroederdewitt/multiagent_mujoco`


## Troubleshooting

Error: `lib/libstdc++.so.6: version 'GLIBCXX_3.4.30' not found (required by /usr/lib/x86_64-linux-gnu/libLLVM-15.so.1)`

Solution: `rm -r ~/miniconda3/envs/baselines200/lib/libstdc++.so.6`

## Alternative Method: Docker
If you fail to successfully install our baselines in a conda environment, we additionally offer a tested Dockerfile. Simply run the following command to build the docker image.

`docker build -t baselines .`

Once the image is built you can run an interactive docker container as follows and then reproduce the experiments from inside the docker container.

`docker run -it baselines`

# Reproducing Results
We provide a script called `main.py` that can be used to run a specific experiment configuration using command line arguments.

`python main.py --env<env_name> --scenario<scenario_name> --dataset<dataset_name> --system<system_name>`

E.g. `python main.py --env=mamujoco_omiga --scenario=2ant --dataset=Expert --system=maddpg+cql`

Refer to `all_experiments.py` to see all valid experiment configurations. Additionally, you can easilly reproduce all experiments for a specific environment as follows:

`python all_experiments.py --env=<env_name>`

E.g. `python all_experiments.py --env=mamujoco_omiga`

# Visualising Datasets

```py
from utils.offline_dataset import download_and_unzip_vault, analyse_vault

# Dataset from OG-MARL
download_and_unzip_vault("smac_v1", "5m_vs_6m")
analyse_vault("smac_v1/5m_vs_6m.vlt", visualise=True)
```
<img width="723" alt="image" src="https://github.com/instadeepai/og-marl/assets/37700709/175e7b0c-af02-4fc9-a5f9-47f2103bf956">


```py
# Vault converted from OMIGA dataset
download_and_unzip_vault("smac_v1_omiga", "5m_vs_6m")
analyse_vault("smac_v1_omiga/5m_vs_6m.vlt", visualise=True)
```
<img width="732" alt="image" src="https://github.com/instadeepai/og-marl/assets/37700709/fa04627a-810f-487a-ac7e-5fb406a98861">


```py
# Vault converted from CFCQL dataset
download_and_unzip_vault("smac_v1_cfcql", "5m_vs_6m")
analyse_vault("smac_v1_cfcql/5m_vs_6m.vlt", visualise=True)
```
<img width="723" alt="image" src="https://github.com/instadeepai/og-marl/assets/37700709/3d04d4ac-1bb9-4175-b8d3-ac7c6b8e3f06">

