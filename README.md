# Welcome to OG-MARL
Off-The-Grid MARL is a research framework Cooperative Offline Multi-Agent Reinforcement Learning (MARL). Our code includes dataset for offline MARL, implementations of utilities for generating your own datasets, and popular offline MARL algorithms. To get started, follow the instructions in this README which walk you through the instalation and how to run the quickstart tutorials.

# Using Conda
Because we support many different environments, each with their own set of dependencies which are often conflicting, you will need to follow slightly different instalation instruction for each environment. 

To manage the different dependencies, we reccomend using `miniconda` as a python virtual environment manager. Follow these instructions to install `conda`. 

* https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html

We have tested these instalation instructions on Ubuntu. 

# Installing OG-MARL Datasets And Baselines
There are two options for installing OG-MARL. The first is to install the datasets only. The second is to optionally also install the baseline algorithm implementations.

Installing the datasets only is a lot easier and will likely work on most computer setups. Installing the baselines is a bit more complicated because of some additional dependencies. Unfortunatly baselines are unlikely to work on Windows and Mac because of our dependency on DeepMind's `reverb` and `launchpad` packages. In future we hope to relax these requirements.

In future we will also be releasing Dockerfiles.

# Overview
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

# Instructions: Profiling Datasets 
First lets work through loading datasets.

First you should install og-marl with the `datasets` option.

`pip install -e .["datasets"]`

Next, depending on the environment you want to use, you should install that environments dependencies. 

`pip install -r install_environments/requirements/<environment_name>.txt`

You should replace `<environment_name>` with the name of the environment you want to install.

Installing several different environments dependencies in the same python virtual environment (or conda environment) may work in some cases but in others, they may have conflicting requirements. So we reccomend maintaining a different virtual environment for each environment.

Next you need to download the dataset you want to use and add it to the correct file path. Go to the OG-MARL website (https://sites.google.com/view/og-marl) and download the dataset. Once the zip file is downloaded add it to a directory called `datasets` on the same level as the `og-marl/` directory. The folder structure should look like this:

```
og_marl/
    |_> ....
datasets/
    |_> <environment>/
        |_> <scenario 1>/
        |   |_> <dataset_type 1>/
        |   |_> <dataset_type 2>/
        |_> <scenario 2>/
            |_> ...
```

For example, take the SMAC 3m and 8m datasets.

```
og_marl/
    |_> ....
datasets/
    |_> smacv1/
        |_> 3m/
        |   |_> Good/
        |   |_> Medium/
        |   |_> Poor/
        |_> 8m/
            |_> Good/
            |_> Medium/
            |_> Poor/
```

You should now be able to run the dataset profiling script for the environement/scenario you just installed and downloaded the datases. 

`python examples/profile_datasets/profile_<environment_name>.py`

Once again, replace `<environment_name>` with the name of the environment you just installed (e.g. "smacv1", "smacv2" or "mamujoco").

Be patient while it runs. It can take a minute or two to loop through the whole dataset. At the end statistics about the dataset will be printed out and a sample of the dataset will also be printed. A violin plot of the data will also be generated and saved alongside the `og_marl` code

# Instructions: Code Snippet
Inorder to run the code snippet you will need to also install the og-marl baselines. So, create a new conda environment and install og-marl like this:

`pip install -e .["datasets","baselines"]`

You will then also need to install SMAC.

`bash install_environments/smac.sh`

Finally, download the "3m" dataset from the website and put it in the apropriate directory as above.

# Instructions: Baselines
First, create a conda environment for the baselines.

`conda create --name og-marl-baselines python=3.8`

Activate the conda environment.

`conda activate og-marl-baselines`

Now install OG-MARL with baselines.

`pip install -e .["datasets","baselines"]`

You will need to follow slightly different instructions for the `smac benchmark` and `mamujoco benchmark`. We reccomend creating a different `conda` environment for each and then switching between them as is neccessary.

## SMAC Instructions
Inorder to run the SMAC benchmarking script `examples/baselines/benchmark_smac.py` you need to follow all the steps above and then as a final step run the SMAC instalation script:

`bash install_environments/smac.sh`

## MAMuJoCo Instructions
Inorder to run the MAMuJoCo benchmarking script `examples/baselines/benchmark_mamujoco.py` you need to follow all the steps above, and then as a final step run the MAMuJoCo instalation script:

`bash install_environments/mamujoco.sh`

 IMPORTANT!!!! You will need to set these environment variables everytime you start a new terminal or add them to your .bashrc file.

`export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu/:$MUJOCOPATH/mujoco210/bin:/usr/lib/nvidia`

`export LD_PRELOAD=$LD_PRELOAD:/usr/lib/x86_64-linux-gnu/libGLEW.so`

## Running the Benchmarks

Now go see the top of the respective python scripts for how to run the benchmarks.

# Instructions: Dataset Generation Quickstart
Inorder to run the datasets generation quickstart tutorial you will need to also install the og-marl baselines. So, create a new conda environment and install og-marl like this:

`pip install -e .["datasets","baselines"]`

You are now ready to run through the quickstart tutorial. Open the file `examples/quickstart/part1_generate_dataset.py` and read the comments throughout to do the tutorial.

# Troubleshoot

Error:

`ImportError: libpython3.8.so.1.0: cannot open shared object file: No such file or directory`

Solution:

`export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH`