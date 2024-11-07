<p align="center">
    <a href="docs/assets/offline_marl_diagram.jpg">
        <img src="docs/assets/og_marl_logo.png" alt="Offline MARL diagram" width="70%"/>
    </a>
</p>

<h2 align="center">
    <p>Offline Multi-Agent Reinforcement Learning Datasets and Baselines</p>
</h2>
<p align="center">
    <a href="https://www.python.org/doc/versions/">
        <img src="https://img.shields.io/badge/python-3.9-blue" alt="Python Versions">
    </a>
    <a href="https://opensource.org/licenses/Apache-2.0">
        <img src="https://img.shields.io/badge/License-Apache%202.0-orange.svg" alt="License">
    </a>
    <a href="https://arxiv.org/abs/2302.00521">
        <img src="https://img.shields.io/badge/PrePrint-ArXiv-red" alt="ArXiv">
    </a>
    <!-- <a href="https://github.com/instadeepai/og-marl/actions/workflows/tests_linters.yml">
        <img src="https://github.com/instadeepai/og-marl/actions/workflows/tests_linters.yml/badge.svg" alt="Tests and Linters">
    </a> -->
    <!-- <a href="https://github.com/psf/black">
        <img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Black">
    </a> -->
    <!-- <a href="http://mypy-lang.org/">
        <img src="http://www.mypy-lang.org/static/mypy_badge.svg" alt="MyPy">
    </a> -->
</p>

<p align="center">
    <a href="docs/assets/offline_marl_diagram.jpg">
        <img src="docs/assets/offline_marl_diagram.jpg" alt="Offline MARL diagram" width="70%"/>
    </a>
</p>

## Going Off-the-Grid! 🤖 ⚡ 🔌 🔋

Offline MARL holds great promise for real-world applications by utilising static datasets to build decentralised controllers of complex multi-agent systems. However, currently offline MARL lacks a standardised benchmark for measuring meaningful research progress. Off-the-Grid MARL (OG-MARL) fills this gap by providing a diverse suite of datasets with baselines on popular MARL benchmark environments in one place, with a unified API and an easy-to-use set of tools.

OG-MARL forms part of the [InstaDeep](https://www.instadeep.com/) MARL [ecosystem](#see-also-🔎), developed jointly with the open-source
community. To join us in these efforts, reach out, raise issues or just
🌟 to stay up to date with the latest developments! 📢 You can contribute to the conversation around OG-MARL in the [Discussion tab](https://github.com/instadeepai/og-marl/discussions). Please don't hesitate to leave a comment. We will be happy to reply.

> 📢 We recently moved our datasets to Hugging Face. This means that previous download links for the datasets may no longer work. Datasets can now be downloaded directly from [Hugging Face](https://huggingface.co/datasets/InstaDeepAI/og-marl).

## Quickstart 🏎️
Clone this repository.

`git clone https://github.com/instadeepai/og-marl.git`

Install `og-marl` and its requirements. We tested `og-marl` with Python 3.10 and Ubuntu 20.04. Consider using a `conda` virtual environment.

`pip install -e .[tf2_baselines]`

Download environment files. We will use SMACv1 in this example. MAMuJoCo installation instructions are included near the bottom of the README.

`bash install_environments/smacv1.sh`

Download environment requirements.

`pip install -r install_environments/requirements/smacv1.txt`

Train an offline system. In this example we will run Independent Q-Learning with Conservative Q-Learning (iql+cql). The script will automatically download the neccessary dataset if it is not found locally.

`python og_marl/tf2_systems/offline/iql_cql.py task.source=og_marl task.env=smac_v1 task.scenario=3m task.dataset=Good`

You can find all offline systems at `og_marl/tf2_systems/offline/` and they can be run similarly. Be careful, some systems only work on discrete action space environments and vice versa for continuous action space environments. The config files for systems are found at `og_marl/tf2_systems/offline/configs/`. We use [hydra](https://hydra.cc/docs/intro/) for our config management. Config defaults can be overwritten as command line arguments like above.

## Dataset API 🔌

To quickly start working with a dataset you do not even need to install `og-marl`. 
Simply install Flashbax and download a dataset from [Hugging Face](https://huggingface.co/datasets/InstaDeepAI/og-marl). 

`pip install flashbax`

Then you should be able to do something like this.

```
from flashbax.vault import Vault
import jax
import numpy as np

vault = Vault("og_marl/smac_v1/2s3z.vlt", vault_uid="Good")

experience = vault.read().experience

numpy_experience = jax.tree.map(lambda x: np.array(x), experience)
```

We also provide a simple demonstrative notebook of how to use OG-MARL's dataset API here:

[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/instadeepai/og-marl/blob/main/examples/dataset_api_demo.ipynb)

## Datasets 🎥

We have generated datasets on a diverse set of popular MARL environments. A list of currently supported environments is included in the table below. It is well known from the single-agent offline RL literature that the quality of experience in offline datasets can play a large role in the final performance of offline RL algorithms. Therefore in OG-MARL, for each environment and scenario, we include a range of dataset distributions including `Good`, `Medium`, `Poor` and `Replay` datasets in order to benchmark offline MARL algorithms on a range of different dataset qualities. For more information on why we chose to include each environment and its task properties, please read our accompanying [paper](https://arxiv.org/abs/2302.00521).

<img src="docs/assets/hugging_face.png" alt="Hugging Face logo" width="25%"/>

Our datasets are now hosted on Hugging Face for improved accessibility for the community: [https://huggingface.co/datasets/InstaDeepAI/og-marl](https://huggingface.co/datasets/InstaDeepAI/og-marl)

> ⚠️ Some datasets have yet to be converted to the new dataset format (Vault). For available datasets, please refer to `og_marl/vault_utils/download_vault.py` or the Hugging Face datasets repository.

<div class="collage">
  <div class="row" align="center">
    <img src="docs/assets/smacv2.png" alt="SMAC v2" width="16%"/>
    <img src="docs/assets/pistonball.png" alt="Pistonball" width="16%"/>
    <img src="docs/assets/coop_pong.png" alt="Cooperative Pong" width="16%"/>
    <img src="docs/assets/pursuit.png" alt="Pursuit" width="16%"/>
    <img src="docs/assets/kaz.png" alt="Pursuit" width="16%"/>
  </div>
  <div class="row" align="center">
    <img src="docs/assets/flatland.png" alt="Flatland" width="16%"/>
    <img src="docs/assets/mamujoco.png" alt="MAMuJoCo" width="16%"/>
    <img src="docs/assets/city_learn.png" alt="CityLearn" width="16%"/>
    <img src="docs/assets/voltage.png" alt="Voltage Control" width="16%"/>
    <img src="docs/assets/mpe.png" alt="Pursuit" width="16%"/>
  </div>
</div>

<br/>

### Environments and Scenarios in OG-MARL 🗺️

| Environment | Scenario | Agents | Act | Obs | Reward | Types | Repo |
|-----|----|----|-----|-----|----|----|-----|
| 🔫SMAC v1 | 3m <br/> 8m <br/> 2s3z <br/> 5m_vs_6m <br/> 27m_vs_30m <br/> 3s5z_vs_3s6z <br/> 2c_vs_64zg| 3 <br/> 8 <br/> 5 <br/> 5 <br/> 27 <br/> 8 <br/> 2 | Discrete  | Vector   |  Dense | Homog <br/> Homog <br/> Heterog <br/> Homog <br/> Homog <br/> Heterog <br/> Homog |[source](https://github.com/oxwhirl/smac) |
| 💣SMAC v2 | terran_5_vs_5 <br/> zerg_5_vs_5 <br/> terran_10_vs_10 | 5 <br/> 5 <br/> 10 | Discrete | Vector | Dense | Heterog | [source](https://github.com/oxwhirl/smacv2) |
| 🚅Flatland | 3 Trains  <br/> 5 Trains | 3 <br/> 5 | Discrete | Vector | Sparse | Homog | [source](https://flatland.aicrowd.com/intro.html) |
| 🐜MAMuJoCo | 2x3 HalfCheetah <br/> 2x4 Ant <br/> 4x2 Ant | 2 <br/> 2 <br/> 4 | Cont. | Vector | Dense | Heterog <br/> Homog <br/> Homog | [source](https://github.com/schroederdewitt/multiagent_mujoco) |
| 🐻PettingZoo | Pursuit  <br/> Co-op Pong | 8 <br/> 2 | Discrete <br/> Discrete  | Pixels <br/> Pixels | Dense | Homog <br/> Heterog | [source](https://pettingzoo.farama.org/) |

### Datasets from Prior Works 🥇
We recently converted several datasets from prior works to Vaults and benchmarked our baseline algorithms on them. For more information, see our [technical report](https://arxiv.org/abs/2406.09068) on ArXiv.

| Paper | Environment | Scenario | Source |
|-----|----|----|-----|
| [Pan et al. (2022)](https://proceedings.mlr.press/v162/pan22a/pan22a.pdf) | 🐜MAMuJoCo | 2x3 HalfCheetah | [source](https://github.com/ling-pan/OMAR) |
| [Pan et al. (2022)](https://proceedings.mlr.press/v162/pan22a/pan22a.pdf) | 🔴MPE | simple_spread | [source](https://github.com/ling-pan/OMAR) |
| [Shao et al. (2023)](https://openreview.net/forum?id=62zmO4mv8X) | 🔫SMAC v1 | 5m_vs_6m <br/> 2s3z <br/> 3s_vs_5z <br/> 6h_vs_8z | [source](https://github.com/thu-rllab/CFCQL) |
| [Wang et al. (2023)](https://papers.nips.cc/paper_files/paper/2023/hash/a46c84276e3a4249ab7dbf3e069baf7f-Abstract-Conference.html) | 🔫SMAC v1 | 5m_vs_6m <br/> 6h_vs_8z <br/> 2c_vs_64zg <br/> corridor| [source](https://github.com/ZhengYinan-AIR/OMIGA) |
| [Wang et al. (2023)](https://papers.nips.cc/paper_files/paper/2023/hash/a46c84276e3a4249ab7dbf3e069baf7f-Abstract-Conference.html) | 🐜MAMuJoCo | 6x1 HalfCheetah <br/> 3x1 Hopper <br/> 2x4 Ant| [source](https://github.com/ZhengYinan-AIR/OMIGA) |

### Overview All Datasets

```
{"og_marl": {
        "smac_v1": {
            "3m": ["Good", "Medium", "Poor"],
            "8m": ["Good", "Medium", "Poor"],
            "5m_vs_6m": ["Good", "Medium", "Poor"],
            "2s3z": ["Good", "Medium", "Poor"],
            "3s5z_vs_3s6z": ["Good", "Medium", "Poor"],
        },
        "smac_v2": {
            "terran_5_vs_5": ["Replay"],
            "terran_10_vs_10": ["Replay"],
            "zerg_5_vs_5": ["Replay"],
        },
        "mamujoco": {
            "2halfcheetah": ["Good", "Medium", "Poor"]
        },
        "gymnasium_mamujoco": {
            "2ant": ["Replay"],
            "2halfcheetah": ["Replay"],
            "2walker": ["Replay"],
            "3hopper": ["Replay"],
            "4ant": ["Replay"],
            "6halfcheetah": ["Replay"],
        },
    },
    "cfcql": {
        "smac_v1": {
            "6h_vs_8z": ["Expert", "Medium", "Medium-Replay", "Mixed"],
            "3s_vs_5z": ["Expert", "Medium", "Medium-Replay", "Mixed"]
            "5m_vs_6m": ["Expert", "Medium", "Medium-Replay", "Mixed"]
            "2s3z": ["Expert", "Medium", "Medium-Replay", "Mixed"]
        },
    },
    "alberdice": {
        "rware": {
            "small-2ag": ["Expert"],
            "small-4ag": ["Expert"],
            "small-6ag": ["Expert"],
            "tiny-2ag": ["Expert"],
            "tiny-4ag": ["Expert"],
            "tiny-6ag": ["Expert"],
        },
    },
    "omar": {
        "mpe": {
            "simple_spread": ["Expert", "Medium", "Medium-Replay", "Random"]
            "simple_tag": ["Expert", "Medium", "Medium-Replay", "Random"]
            "simple_world": ["Expert", "Medium", "Medium-Replay", "Random"]
        },
        "mamujoco": {
            "2halfcheetah": ["Expert", "Medium", "Medium-Replay", "Random"]
        },
    },
    "omiga": {
        "smac_v1": {
            "2c_vs_64zg": ["Good", "Medium", "Poor"],
            "6h_vs_8z": ["Good", "Medium", "Poor"],
            "5m_vs_6m": ["Good", "Medium", "Poor"],
            "corridor": ["Good", "Medium", "Poor"],
        },
        "mamujoco": {
            "6halfcheetah": ["Expert", "Medium", "Medium-Expert", "Medium-Replay"],
            "2ant": ["Expert", "Medium", "Medium-Expert", "Medium-Replay"],
            "3hopper": ["Expert", "Medium", "Medium-Expert", "Medium-Replay"],
        },
    },
}
```

## Installing MAMuJoCo 🐆

The OG-MARL datasets use the latest version of MuJoCo (210). While the OMIGA and OMAR datasets use an older version (200). They each have different instalation instructions and should be installed in seperate virtual environments.

#### MAMuJoCo 210

`bash install_environments/mujoco210.sh`

`pip install -r install_environments/requirements/mujoco.txt`

`pip install -r install_environments/requirements/mamujoco210.txt`

#### MAMuJoCo 200

`bash install_environments/mujoco200.sh`

`pip install -r install_environments/requirements/mujoco.txt`

`export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco200/bin`

`pip install -r install_environments/requirements/mamujoco200.txt`


## See Also 🔎

**InstaDeep's MARL ecosystem in JAX.** In particular, we suggest users check out the following sister repositories:

* 🦁 [Mava](https://github.com/instadeepai/Mava): a research-friendly codebase for distributed MARL in JAX.
* 🌴 [Jumanji](https://github.com/instadeepai/jumanji): a diverse suite of scalable reinforcement learning environments in JAX.
* 😎 [Matrax](https://github.com/instadeepai/matrax): a collection of matrix games in JAX.
* 🔦 [Flashbax](https://github.com/instadeepai/flashbax): accelerated replay buffers in JAX.
* 📈 [MARL-eval](https://github.com/instadeepai/marl-eval): standardised experiment data aggregation and visualisation for MARL.

**Related.** Other libraries related to accelerated MARL in JAX.

* 🦊 [JaxMARL](https://github.com/flairox/jaxmarl): accelerated MARL environments with baselines in JAX.
* ♟️  [Pgx](https://github.com/sotetsuk/pgx): JAX implementations of classic board games, such as Chess, Go and Shogi.
* 🔼 [Minimax](https://github.com/facebookresearch/minimax/): JAX implementations of autocurricula baselines for RL.

## Citing OG-MARL :pencil2:

If you use OG-MARL Datasets in your work, please cite the library using:

```
@inproceedings{formanek2023ogmarl,
    author = {Formanek, Claude and Jeewa, Asad and Shock, Jonathan and Pretorius, Arnu},
    title = {Off-the-Grid MARL: Datasets and Baselines for Offline Multi-Agent Reinforcement Learning},
    year = {2023},
    publisher = {AAMAS},
    booktitle = {Extended Abstract at the 2023 International Conference on Autonomous Agents and Multiagent Systems},
}
```

<img src="docs/assets/aamas2023.png" alt="AAMAS 2023" width="20%"/>


## Acknowledgements 🙏

The development of this library was supported with Cloud TPUs
from Google's [TPU Research Cloud](https://sites.research.google/trc/about/) (TRC) 🌤.
