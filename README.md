<p align="center">
    <a href="docs/assets/offline_marl_diagram.jpg">
        <img src="docs/assets/og_marl_logo.png" alt="Offline MARL diagram" width="70%"/>
    </a>
</p>

<h2 align="center">
    <p>Off-the-Grid MARL: Offline Multi-Agent Reinforcement Learning Datasets and Baselines</p>
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
    <a href="https://sites.google.com/view/og-marl">
        <img src="https://img.shields.io/badge/Datasets-Download-green" alt="Website">
    </a>
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
🌟 to stay up to date with the latest developments!

## Quickstart 🏎️
Clone this repository.

`git clone https://github.com/instadeepai/og-marl.git`

Install `og-marl` and its dependencies. We tested `og-marl` with Python 3.9. Consider using a `conda` virtual environment.

`pip install -e .`

`pip install flashbax==0.1.0`

Download environment dependencies. We will use SMACv1 in this example.

`bash install_environments/smacv1.sh`

Download a dataset.

`python examples/download_vault.py --env=smac_v1 --scenario=3m`

Run a baseline. In this example we will run MAICQ.

`python baselines/main.py --env=smac_v1 --scenario=3m --dataset=Good --system=maicq`

## Datasets 🎥

We have generated datasets on a diverse set of popular MARL environments. A list of currently supported environments is included in the table below. It is well known from the single-agent offline RL literature that the quality of experience in offline datasets can play a large role in the final performance of offline RL algorithms. Therefore in OG-MARL, for each environment and scenario, we include a range of dataset distributions including `Good`, `Medium`, `Poor` and `Replay` datasets in order to benchmark offline MARL algorithms on a range of different dataset qualities. For more information on why we chose to include each environment and its task properties, please read our accompanying [paper](https://arxiv.org/abs/2302.00521).

<div class="collage">
  <div class="row" align="center">
<!--     <img src="docs/assets/smac.png" alt="SMAC v1" width="16%"/> -->
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

## Dataset Backends 🔌
We are in the process of migrating our datasets from TF Records to Flashbax Vaults. Flashbax Vaults have the advantage of being significantly more flexible than the TF Record Datasets.

### Flashbax Vaults ⚡
| Environment | Scenario | Agents | Act | Obs | Reward | Types | Repo |
|-----|----|----|-----|-----|----|----|-----|
| 🔫SMAC v1 | 3m <br/> 8m <br/> 2s3z <br/> 5m_vs_6m <br/> 27m_vs_30m <br/> 3s5z_vs_3s6z <br/> 2c_vs_64zg| 3 <br/> 8 <br/> 5 <br/> 5 <br/> 27 <br/> 8 <br/> 2 | Discrete  | Vector   |  Dense | Homog <br/> Homog <br/> Heterog <br/> Homog <br/> Homog <br/> Heterog <br/> Homog |[source](https://github.com/oxwhirl/smac) |
| 💣SMAC v2 | terran_5_vs_5 <br/> zerg_5_vs_5 <br/> terran_10_vs_10 | 5 <br/> 5 <br/> 10 | Discrete | Vector | Dense | Heterog | [source](https://github.com/oxwhirl/smacv2) |
| 🚅Flatland | 3 Trains  <br/> 5 Trains | 3 <br/> 5 | Discrete | Vector | Sparse | Homog | [source](https://flatland.aicrowd.com/intro.html) |
| 🐜MAMuJoCo | 2-HalfCheetah <br/> 2-Ant <br/> 4-Ant | 2 <br/> 2 <br/> 4 | Cont. | Vector | Dense | Heterog <br/> Homog <br/> Homog | [source](https://github.com/schroederdewitt/multiagent_mujoco) |


### Legacy Datasets (still to be migrated to Vault) 👴
| Environment | Scenario | Agents | Act | Obs | Reward | Types | Repo |
|-----|----|----|-----|-----|----|----|-----|
| 🐻PettingZoo | Pursuit  <br/> Co-op Pong <br/> PistonBall <br/> KAZ| 8 <br/> 2 <br/> 15 <br/> 2| Discrete <br/> Discrete <br/> Cont. <br/> Discrete | Pixels <br/> Pixels <br/> Pixels <br/> Vector | Dense | Homog <br/> Heterog <br/> Homog <br/> Heterog| [source](https://pettingzoo.farama.org/) |
| 🏙️CityLearn | 2022_all_phases | 17 | Cont. | Vector | Dense | Homog | [source](https://github.com/intelligent-environments-lab/CityLearn) |
| 🔌Voltage Control | case33_3min_final | 6 | Cont. | Vector | Dense | Homog | [source](https://github.com/Future-Power-Networks/MAPDN) |
| 🔴MPE | simple_adversary | 3 | Discrete. | Vector | Dense | Competitive | [source](https://pettingzoo.farama.org/environments/mpe/simple_adversary/) |

## Dataset API

We provide a simple demonstrative notebook of how to use OG-MARL's dataset API here:
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/instadeepai/og-marl/blob/main/examples/dataset_api_demo.ipynb)


### Dataset and Vault Locations
For OG-MARL's systems, we require the following dataset storage structure:

```
examples/
    |_> ...
og_marl/
    |_> ...
vaults/
    |_> smac_v1/
        |_> 3m.vlt/
        |   |_> Good/
        |   |_> Medium/
        |   |_> Poor/
        |_> ...
    |_> smac_v2/
        |_> terran_5_vs_5.vlt/
        |   |_> Good/
        |   |_> Medium/
        |   |_> Poor/
        |_> ...
datasets/
    |_> smac_v1/
        |_> 3m/
        |   |_> Good/
        |   |_> Medium/
        |   |_> Poor/
        |_> ...
    |_> smac_v2/
        |_> terran_5_vs_5/
        |   |_> Good/
        |   |_> Medium/
        |   |_> Poor/
        |_> ...
...
```

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

If you use OG-MARL in your work, please cite the library using:

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
