## Tutorials 💯

We provide various examples of how to use OG-MARL.

### Overview 🗼

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

### Profiling Datasets 📊

In order to profile a dataset you will need to install og-marl with the `datasets` option, as well as the corresponding environment. You should then be able to run the dataset profiling script for the environement/scenario you just installed and downloaded the datases.

`python examples/profile_datasets/profile_<environment_name>.py`

Once again, replace `<environment_name>` with the name of the environment you just installed (e.g. "smacv1", "smacv2" or "mamujoco").

Be patient while it runs. It can take a minute or two to loop through the whole dataset. At the end statistics about the dataset will be printed out and a sample of the dataset will also be printed. A violin plot of the data will also be generated and saved alongside the `og_marl/` directory.

### Dataset Generation Quickstart 🎥

In order to run the datasets generation quickstart tutorial you will need to also install the og-marl with the `datasets` and `baselines` options. You are now ready to run through the quickstart tutorial. Open the file `examples/quickstart/generate_dataset.py` and read the comments throughout to do the tutorial.

### Running Baselines 🏃

We provide scripts to reproduce the MAMuJoCo and SMAC baseline results. Inorder to run them you will need to install OG-MARL with the `datasets` and `baselines` options. Then install the corresponding environment.

After that you can run the SMAC script as follows:

`python examples/baselines/benchmark_smac.py --algo_name=qmix --dataset_quality=Good --env_name=3m`

    --algo_name [used to change the algorithm you want to run]
    --dataset_quality [is used to change wich dataset type to run]
    --env_name [is used to change the scenario]

You will need to make sure you download the datasets from the OG-MARL website.

<https://sites.google.com/view/og-marl>

Make sure the unzip the dataset and add it to the path
`datasets/smac/<env_name>/<dataset_quality>/`

## Code Snippet ✂️

Inorder to run the code snippet you will need to also install the OG-MARL with `baselines`. You will then also need to install SMAC.

Finally, download the "3m" dataset from the website and put it in the apropriate directory as above.

You should then be able to run the code snippet:

`python examples/code_snippet.py`

## Troubleshoot ⚙️

We will document common problems encountered while using OG-MARL and their solutions in our [TROUBLESHOOTING](docs/TROUBLESHOOTING.md) document.
