<h2 name="install" id="install">Detailed Installation 🎬</h2>

### Using Conda 🐍

Because we support many different environments, each with their own set of dependencies which are often conflicting, you will need to follow slightly different instalation instruction for each environment.

To manage the different dependencies, we reccomend using `miniconda` as a python virtual environment manager. Follow these instructions to install `conda`.

* <https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html>

🚨 In the near future we will also be releasing Dockerfiles! 🚨

### Installing Datasets and/or Baselines 🀄

There are two options for installing OG-MARL. The first is to install the datasets only. The second is to optionally also install the baseline algorithm implementations.

Installing the datasets only is a lot easier and will likely work on most computer setups. Installing the baselines is a bit more complicated because of some additional dependencies. Unfortunatly baselines are unlikely to work on Windows and Mac because of our dependency on DeepMind's `reverb` and `launchpad` packages. In future we hope to relax these requirements.

#### Installing OG-MARL Datasets (minimal requirements) 🧮

Install og-marl with the `datasets` option.

`pip install -e .["datasets"]`

#### Installing OG-MARL Baselines 🔱

Install og-marl with the `baselines` option.

`pip install -e .["datasets","baselines"]`

#### Installing Environments ⛰️

Depending on the environment you want to use, you should install that environments dependencies. We provide convenient shell scripts for this.

`bash install_environments/<environment_name>.sh`

You should replace `<environment_name>` with the name of the environment you want to install.

Installing several different environments dependencies in the same python virtual environment (or conda environment) may work in some cases but in others, they may have conflicting requirements. So we reccomend maintaining a different virtual environment for each environment.

### Downloading Datasets ⏬

Next you need to download the dataset you want to use and add it to the correct file path. Go to the OG-MARL website (<https://sites.google.com/view/og-marl>) and download the dataset. Once the zip file is downloaded add it to a directory called `datasets` on the same level as the `og-marl/` directory. The folder structure should look like this:

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
