<p align="center">
    <a href="docs/assets/og_marl_logo_short.png">
        <img src="docs/assets/og_marl_logo_short.png" alt="OG-MARL logo" width="50%"/>
    </a>
</p>

<h2 align="center">
    <p>Off-the-Grid MARL: Offline Multi-Agent Reinforcement Learning made easy</p>
</h2>
<p align="center">
    <a href="https://www.python.org/doc/versions/">
        <img src="https://upload.wikimedia.org/wikipedia/commons/a/a5/Blue_Python_3.8_Shield_Badge.svg" alt="Python Versions">
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

[**Installation**](#installation-) | [**Quickstart**](#quickstart-)

</div>

Multi-Agent Reinforcement Learning (MARL) is a field of research with a lot of potential for real-world impact, promising robust, decentralised controllers for complex multi-agent systems. However, MARL is typically very sample inefficient, requiring a significant amount of environment interactions. In practice, this can be a significant obstacle to applying MARL in the real world where environment interactions can be slow or dangerous to collect. As a  consequence, MARL research has largely been focused on quick-to-simulate grid-world tasks and video games. 

But recently the field of Offline RL has offered a promising way to overcome these challenges. Offline MARL has the potential to help the field go beyond the current MARL research settings (e.g. grid-worlds and video games) and towards real-world applications.

However, offline MARL is currently an under-researched area that lacks standardised benchmark datasets. In order to help drive this nascent field forward, Off-the-Grid MARL (OG-MARL) fills the gap by providing a diverse suite of datasets with baselines on popular MARL benchmark environments.

Originating in the Research Team at
[InstaDeep](https://www.instadeep.com/), OG-MARL is now developed jointly with the open-source
community. To join us in these efforts, reach out, raise issues and read our
[contribution guidelines](https://github.com/instadeepai/og-marl/blob/main/CONTRIBUTING.md) or just
🌟 to stay up to date with the latest developments!

# Datasets 🎥

We have generated datasets on a diverse set of popular MARL environments. A list of currently supported environments is included in the table below. It is well known from the single-agent offline RL literature that the quality of experience in offline datasets can play a large role in the final performance of offline RL algorithms. Therefore in OG-MARL, for each environment and scenario, we include a range of dataset distributions including `Good`, `Medium`, `Poor` and `Replay` datasets in order to benchmark offline MARL algorithms on a range of different dataset qualities. For more information on why we chose to include each environment and its task properties, please read our accompanying [paper](https://arxiv.org/abs/2302.00521).

<div class="collage">
  <div class="row" align="center">
<!--     <img src="docs/assets/smac.png" alt="SMAC v1" width="16%"/> -->
    <img src="docs/assets/smacv2.png" alt="SMAC v2" width="16%"/>
    <img src="docs/assets/pistonball.png" alt="Pistonball" width="16%"/>
    <img src="docs/assets/coop_pong.png" alt="Cooperative Pong" width="16%"/>
    <img src="docs/assets/pursuit.png" alt="Pursuit" width="16%"/>
  </div>
  <div class="row" align="center">
    <img src="docs/assets/flatland.png" alt="Flatland" width="16%"/>
    <img src="docs/assets/mamujoco.png" alt="MAMuJoCo" width="16%"/>
    <img src="docs/assets/city_learn.png" alt="CityLearn" width="16%"/>
    <img src="docs/assets/voltage.png" alt="Voltage Control" width="16%"/>
  </div>
</div>

<br/>

<div align="center">

| Environment | Scenario | Agents | Act | Obs | Reward | Types | Repo
| --- | ---| --- | --- | --- | --- | --- | --- |
| 🔫SMAC v1 | 3m <br/> 8m <br/> 2s3z <br/> 5m_vs_6m <br/> 27m_vs_30m <br/> 3s5z_vs_3s6z <br/> 2c_vs_64zg| 3 <br/> 8 <br/> 5 <br/> 5 <br/> 27 <br/> 8 <br/> 2 | Discrete  | Vector   |  Dense | Homog <br/> Homog <br/> Heterog <br/> Homog <br/> Homog <br/> Heterog <br/> Homog |[source](https://github.com/oxwhirl/smac) |
| 💣SMAC v2 | terran_5_vs_5 <br/> zerg_5_vs_5 <br/> terran_10_vs_10 | 5 <br/> 5 <br/> 10 | Discrete | Vector | Dense | Heterog | [source](https://github.com/oxwhirl/smacv2) |
| 🐻PettingZoo | Pursuit  <br/> Co-op Pong <br/> PistonBall | 8 <br/> 2 <br/> 15 | Discrete <br/> Discrete <br/> Cont. | Pixels | Dense | Homog <br/> Heterog <br/> Homog | [source](https://pettingzoo.farama.org/) | 
| 🚅Flatland | 3 Trains  <br/> 5 Trains | 3 <br/> 5 | Discrete | Vector | Dense | Homog | [source](https://flatland.aicrowd.com/intro.html) | 
| 🐜MAMuJoCo | 2-HalfCheetah <br/> 2-Ant <br/> 4-Ant | 2 <br/> 2 <br/> 4 | Cont. | Vector | Dense | Heterog <br/> Homog <br/> Homog | [source](https://github.com/schroederdewitt/multiagent_mujoco) | 
| 🏙️CityLearn | 2022_all_phases | 17 | Cont. | Vector | Dense | Homog | [source](https://github.com/intelligent-environments-lab/CityLearn) | 
| 🔌Voltage Control | case33_3min_final | 6 | Cont. | Vector | Dense | Homog | [source](https://github.com/Future-Power-Networks/MAPDN) |

</div>

<h2 name="install" id="install">Installation 🎬</h2>

## Using Conda 🐍
Because we support many different environments, each with their own set of dependencies which are often conflicting, you will need to follow slightly different instalation instruction for each environment. 

To manage the different dependencies, we reccomend using `miniconda` as a python virtual environment manager. Follow these instructions to install `conda`. 

* https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html

🚨 In the near future we will also be releasing Dockerfiles! 🚨

## Installing Datasets and/or Baselines 🀄
There are two options for installing OG-MARL. The first is to install the datasets only. The second is to optionally also install the baseline algorithm implementations.

Installing the datasets only is a lot easier and will likely work on most computer setups. Installing the baselines is a bit more complicated because of some additional dependencies. Unfortunatly baselines are unlikely to work on Windows and Mac because of our dependency on DeepMind's `reverb` and `launchpad` packages. In future we hope to relax these requirements.

### Installing OG-MARL Datasets (minimal requirements) 🧮
Install og-marl with the `datasets` option.

`pip install -e .["datasets"]`

### Installing OG-MARL Baselines 🔱
Install og-marl with the `baselines` option.

`pip install -e .["datasets","baselines"]`

### Installing Environments ⛰️
Depending on the environment you want to use, you should install that environments dependencies. We provide convenient shell scripts for this.

`bash install_environments/<environment_name>.sh`

You should replace `<environment_name>` with the name of the environment you want to install.

Installing several different environments dependencies in the same python virtual environment (or conda environment) may work in some cases but in others, they may have conflicting requirements. So we reccomend maintaining a different virtual environment for each environment.

# Downloading Datasets ⏬

Next you need to download the dataset you want to use and add it to the correct file path. Go to the OG-MARL website (https://sites.google.com/view/og-marl) and download the dataset. Once the zip file is downloaded add it to a directory called `datasets` on the same level as the `og-marl/` directory. The folder structure should look like this:

```
examples/
    |_> ...
og_marl/
    |_> ...
datasets/
    |_> smacv1/
        |_> 3m/
        |   |_> Good/
        |   |_> Medium/
        |   |_> Poor/
        |_> ...
    |_> smacv2/
        |_> terran_5_vs_5/
        |   |_> Good/
        |   |_> Medium/
        |   |_> Poor/
        |_> ...
```

<h2 name="quickstart" id="quickstart">Quickstart ⚡</h2>

```python
from og_marl import SMAC
from og_marl import QMIX
from og_marl import OfflineLogger

# Instantiate environment
env = SMAC("3m")

# Wrap env in offline logger
env = OfflineLogger(env)

# Make multi-agent system
system = QMIX(env)

# Collect data
system.run_online()

# Load dataset
dataset = env.get_dataset("Good")

# Train offline
system.run_offline(dataset)
```

# Tutorials 💯
We provide various examples of how to use OG-MARL. 

## Overview 🗼
In the `examples/` directory we include scripts to load and profile each of our datasets.
* `examples/profile_datasets/profile_smacv1.py`
* `examples/profile_datasets/profile_smacv2.py`
* `examples/profile_datasets/profile_flatland.py`
* `examples/profile_datasets/profile_pettingzoo.py`
* `examples/profile_datasets/profile_mamujoco.py`
* `examples/profile_datasets/profile_city_learn.py`
* `examples/profile_datasets/profile_voltage_control.py`

We also include a quickstart tutorial on how to make your own dataset on a new environment:
* `examples/quickstart/part1_double_cartpole.py`
* `examples/quickstart/part2_generate_dataset.py`
* `examples/quickstart/part3_train_offline_algo.py`

We also include scripts for replicating our benchmarking results:
* `examples/benchmark_mamujoco.py`
* `examples/benchmark_smac.py`

## Profiling Datasets 📊
In order to profile a dataset you will need to install og-marl with the `datasets` option, as well as the corresponding environment. You should then be able to run the dataset profiling script for the environement/scenario you just installed and downloaded the datases. 

`python examples/profile_datasets/profile_<environment_name>.py`

Once again, replace `<environment_name>` with the name of the environment you just installed (e.g. "smacv1", "smacv2" or "mamujoco").

Be patient while it runs. It can take a minute or two to loop through the whole dataset. At the end statistics about the dataset will be printed out and a sample of the dataset will also be printed. A violin plot of the data will also be generated and saved alongside the `og_marl/` directory.

## Dataset Generation Quickstart 🎥
In order to run the datasets generation quickstart tutorial you will need to also install the og-marl with the `datasets` and `baselines` options. You are now ready to run through the quickstart tutorial. Open the file `examples/quickstart/generate_dataset.py` and read the comments throughout to do the tutorial.

## Running Baselines 🏃
We provide scripts to reproduce the MAMuJoCo and SMAC baseline results. Inorder to run them you will need to install OG-MARL with the `datasets` and `baselines` options. Then install the corresponding environment. 

After that you can run the SMAC script as follows:

`python examples/baselines/benchmark_smac.py --algo_name=qmix --dataset_quality=Good --env_name=3m`

    --algo_name [used to change the algorithm you want to run]
    --dataset_quality [is used to change wich dataset type to run]
    --env_name [is used to change the scenario]

You will need to make sure you download the datasets from the OG-MARL website.

https://sites.google.com/view/og-marl

Make sure the unzip the dataset and add it to the path 
`datasets/smac/<env_name>/<dataset_quality>/`

## Code Snippet ✂️
Inorder to run the code snippet you will need to also install the OG-MARL with `baselines`. You will then also need to install SMAC.

Finally, download the "3m" dataset from the website and put it in the apropriate directory as above.

You should then be able to run the code snippet:

`python examples/code_snippet.py`

# Roadmap 🗺️
We are currently working on a large refactor of OG-MARL to get rid of the dependency on reverb and launchpad. This will make the code a lot easier to work with. The current progress on the refactor can be followed on the branch `refactor/remove-reverb-and-launchpad`.

Offline MARL also lends itself well to the new wave of hardware-accelerated research and development in the
field of RL. In the near future we hope to add JAX support to our baselines.

# Troubleshoot ⚙️

We will document common problems encountered while using OG-MARL and their solutions in our [TROUBLESHOOTING](TROUBLESHOOTING.md) document.

<h2 name="citing" id="citing">Citing OG-MARL ✏️</h2>

If you use OG-MARL in your work, please cite the library using:

```
@misc{formanek2023offthegrid,
      title={Off-the-Grid MARL: a Framework for Dataset Generation with Baselines for Cooperative Offline Multi-Agent Reinforcement Learning}, 
      author={Claude Formanek and Asad Jeewa and Jonathan Shock and Arnu Pretorius},
      year={2023},
      eprint={2302.00521},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## See Also 🔎

Other works that form part of InstaDeep's MARL ecosystem.
In particular, we suggest users check out the following sister repositories:

- 🦁 [Mava](https://github.com/instadeepai/Mava) is a research-friendly codebase for fast experimentation of multi-agent reinforcement learning in JAX.
- 🌴 [Jumanji](https://github.com/instadeepai/jumanji) is a diverse suite of scalable reinforcement learning environments in JAX.
- 😎 [Matrax](https://github.com/instadeepai/matrax) is a collection of matrix games in JAX.
