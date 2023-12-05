<p align="center">
    <picture>
        <source media="(prefers-color-scheme: dark)" srcset="docs/assets/og_marl_logo_short_dm.png">
        <source media="(prefers-color-scheme: light)" srcset="docs/assets/og_marl_logo_short.png">
        <img alt="OG-MARL logo" src="docs/assets/og_marl_logo_short.png", width="50%">
    </picture>
</p>

<h2 align="center">
    <p>Off-the-Grid MARL: Offline Multi-Agent Reinforcement Learning made easy</p>
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

<div align="center">
<h3>

[**Download**](#download) | [**Quickstart**](#quick)

</div>

Offline MARL holds great promise for real-world applications by utilising static datasets to build decentralised controllers of complex multi-agent systems. However, currently offline MARL lacks a standardised benchmark for measuring meaningful research progress. Off-the-Grid MARL (OG-MARL) fills this gap by providing a diverse suite of datasets with baselines on popular MARL benchmark environments in one place, with a unified API and an easy-to-use set of tools.

OG-MARL forms part of the [InstaDeep](https://www.instadeep.com/) MARL [ecosystem](#see-also-🔎), developed jointly with the open-source
community. To join us in these efforts, reach out, raise issues and read our
[contribution guidelines](docs/CONTRIBUTING.md) or just
🌟 to stay up to date with the latest developments!

## Updates 📰
OG-MARL is a research tool that is under active development and therefore evolving quickly. We have several very exciting new features on the roadmap but sometimes when we introduce a new feature we may abruptly change how things work in OG-MARL.
But in the interest of moving quickly, we believe this is an acceptable trade-off and ask our users to kindly be aware of this.

The following is a list of the latest updates to OG-MARL:

✅ We have **removed several cumbersome dependencies** from OG-MARL, including `reverb` and `launchpad`. This means that its significantly easier to install and use OG-MARL.

✅ We added **functionality to pre-load the TF Record datasets into a [Cpprb](https://ymd_h.gitlab.io/cpprb/) replay buffer**. This speeds up the time to sample the replay buffer by several orders of magnitude.

✅ We have implemented our **first set of JAX-based systems in OG-MARL**. Our JAX systems use [Flashbax](https://github.com/instadeepai/flashbax) as the replay buffer backend. Flashbax buffers are completly jit-able, which means that our JAX systems have fully intergrated and jitted training and data sampling.

## Need for Speed 🏎️
We have made our TF2 systems compatible with jit compilation. This combined with our new `cpprb` replay buffers have made our systems significantly faster. Furthermore, our JAX systems with tightly integrated replay sampling and training using Flashbax are even faster. 

**Speed comparison: for each setup, we trained MAICQ on the 8m dataset for 10k training steps and evaluated every 1k training steps for 4 episodes using a batch size of 256.**

<img src="docs/assets/system_speed_comparison.png" alt="OG-MARL Speed Comparison" width="32%"/>
<img src="docs/assets/sample_efficiency.png" alt="Sample Efficiency" width="32%"/>
<img src="docs/assets/performance_profile.png" alt="Performance Profile" width="32%"/>

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

<div align="center">

| Environment | Scenario | Agents | Act | Obs | Reward | Types | Repo
| --- | ---| --- | --- | --- | --- | --- | --- |
| 🔫SMAC v1 | 3m <br/> 8m <br/> 2s3z <br/> 5m_vs_6m <br/> 27m_vs_30m <br/> 3s5z_vs_3s6z <br/> 2c_vs_64zg| 3 <br/> 8 <br/> 5 <br/> 5 <br/> 27 <br/> 8 <br/> 2 | Discrete  | Vector   |  Dense | Homog <br/> Homog <br/> Heterog <br/> Homog <br/> Homog <br/> Heterog <br/> Homog |[source](https://github.com/oxwhirl/smac) |
| 💣SMAC v2 | terran_5_vs_5 <br/> zerg_5_vs_5 <br/> terran_10_vs_10 | 5 <br/> 5 <br/> 10 | Discrete | Vector | Dense | Heterog | [source](https://github.com/oxwhirl/smacv2) |
| 🐻PettingZoo | Pursuit  <br/> Co-op Pong <br/> PistonBall <br/> KAZ| 8 <br/> 2 <br/> 15 <br/> 2| Discrete <br/> Discrete <br/> Cont. <br/> Discrete | Pixels <br/> Pixels <br/> Pixels <br/> Vector | Dense | Homog <br/> Heterog <br/> Homog <br/> Heterog| [source](https://pettingzoo.farama.org/) |
| 🚅Flatland | 3 Trains  <br/> 5 Trains | 3 <br/> 5 | Discrete | Vector | Sparse | Homog | [source](https://flatland.aicrowd.com/intro.html) |
| 🐜MAMuJoCo | 2-HalfCheetah <br/> 2-Ant <br/> 4-Ant | 2 <br/> 2 <br/> 4 | Cont. | Vector | Dense | Heterog <br/> Homog <br/> Homog | [source](https://github.com/schroederdewitt/multiagent_mujoco) |
| 🏙️CityLearn | 2022_all_phases | 17 | Cont. | Vector | Dense | Homog | [source](https://github.com/intelligent-environments-lab/CityLearn) |
| 🔌Voltage Control | case33_3min_final | 6 | Cont. | Vector | Dense | Homog | [source](https://github.com/Future-Power-Networks/MAPDN) |
| 🔴MPE | simple_adversary | 3 | Discrete. | Vector | Dense | Competative | [source](https://pettingzoo.farama.org/environments/mpe/simple_adversary/) |

</div>

<h2 name="quick" id="quick">Quickstart 🏁</h2>

### Instalation 🛠️

To install og-marl run the following command.

`pip install -e .`

To run the JAX based systems include the extra requirements.

`pip install -e .[jax]`

### Environments ⛰️

Depending on the environment you want to use, you should install that environments dependencies. We provide convenient shell scripts for this.

`bash install_environments/<environment_name>.sh`

You should replace `<environment_name>` with the name of the environment you want to install.

Installing several different environments dependencies in the same python virtual environment (or conda environment) may work in some cases but in others, they may have conflicting requirements. So we recommend maintaining a different virtual environment for each environment. For more information, we recommend reading the [detailed installation guide](docs/INSTALL.md).

<h2 name="download" id="download">Downloading Datasets ⏬</h3>

Next you need to download the dataset you want to use and add it to the correct file path. We provided a utility for easily downloading and extracting datasets. Below is an example of how to download the dataset for the "3m" map in SMACv1.

```python
from og_marl.offline_dataset import download_and_unzip_dataset

download_and_unzip_dataset("smac_v1", "3m")
```

Alternativly, go to the OG-MARL website (<https://sites.google.com/view/og-marl>) and download the dataset. Once the zip file is downloaded add it to a directory called `datasets` on the same level as the `og-marl/` directory. The folder structure should look like this:

```
examples/
    |_> ...
og_marl/
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

### Launching Experiments 🚀
We include scripts (`examples/tf2/main.py` and `examples/jax/main.py`) for easily launching experiments.

`python examples/<backend>/main.py --system=<system_name> --env=<env_name> --scenario=<scenario_name>`

`<backend>` should be replaced with either `jax` or `tf2`.

`<system_name>` should be replaced with one of `maicq`, `qmix`, `qmix+cql`, `qmix+bcq`, `idrqn`, `iddpg` etc.

`<env_name>` should be replaced with one of `smac_v1`, `smac_v2`, `mamujoco` etc.

`<scenario_name>` should be replaced with one of `3m`, `8m`, `terran_5_vs_5`, `2halfcheetah` etc.

**Note:** We have not implemented any checks to make sure the combination of `env`, `scenario` and `system` is valid. For example, certain algorithms can only be run on discrete action environments. We hope to implement more guard rails in the future. For now, please refer to the code and the paper for clarification. We are also still in the process of migrating all the experiments to this unified launcher. So if some configuration is not supported yet, please reach out in the issues and we will be happy to help. 

<h2 name="citing" id="citing">Citing OG-MARL ✏️</h2>

If you use OG-MARL in your work, please cite the library using:

```
@inproceedings{formanek2023ogmarl,
    author = {Formanek, Claude and Jeewa, Asad and Shock, Jonathan and Pretorius, Arnu},
    title = {Off-the-Grid MARL: Datasets and Baselines for Offline Multi-Agent Reinforcement Learning},
    year = {2023},
    publisher = {International Foundation for Autonomous Agents and Multiagent Systems},
    booktitle = {Proceedings of the 2023 International Conference on Autonomous Agents and Multiagent Systems},
    keywords = {offline reinforcement learning, multi-agent reinforcement learning, reinforcement learning},
    location = {London, United Kingdom},
    series = {AAMAS '23}
}
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
