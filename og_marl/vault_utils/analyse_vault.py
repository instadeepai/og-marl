# Copyright 2023 InstaDeep Ltd. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import Dict, List

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns
from chex import Array
from flashbax.vault import Vault
from git import Optional
import numpy as np
from og_marl.vault_utils.download_vault import get_available_uids
import pandas as pd
from tabulate import tabulate


def calculate_returns(
    experience: Dict[str, Array], reward_key: str = "rewards", terminal_key: str = "terminals"
) -> Array:
    """Calculate the returns in a dataset of experience.

    Args:
        experience (Dict[str, Array]): experience coming from an OG-MARL vault.
        reward_key (str, optional):
            Dictionary key of the rewards in the experience data.
            Defaults to "rewards".
        terminal_key (str, optional):
            Dictionary key of the terminals in the experience data.
            Defaults to "terminals".

    Returns:
        Array: jnp array of returns for each episode.
    """
    # Experience is of dimension of (1, T, N, *E)
    # We want all the time data, but just from one agent
    experience_one_agent = jax.tree_map(lambda x: x[0, :, 0, ...], experience)
    rewards = experience_one_agent[reward_key]
    terminals = jnp.array(experience[terminal_key][0].all(axis=-1).squeeze(), dtype=jnp.float32)

    def sum_rewards(terminals: Array, rewards: Array) -> Array:
        def scan_fn(carry: Array, inputs: Array) -> Array:
            terminal, reward = inputs
            new_carry = carry + reward
            new_carry = jax.lax.cond(
                terminal.all(),
                lambda _: reward,  # Reset the cumulative sum if terminal
                lambda _: new_carry,  # Continue accumulating otherwise
                operand=None,
            )
            return new_carry, new_carry

        _, cumulative_rewards = jax.lax.scan(
            scan_fn,
            jnp.zeros_like(terminals[0]),
            (
                # We shift the terminals one timestep rightwards,
                # for our cumulative sum approach to work
                jnp.pad(terminals, ((1, 0)))[:-1],
                rewards,
            ),
        )
        return cumulative_rewards[terminals == 1]

    episode_returns = sum_rewards(terminals, rewards)
    return episode_returns


def get_saco(experience):
    states = experience['infos']["state"]

    num_tot = states.shape[1]

    reshaped_actions = experience["actions"].reshape((*experience["actions"].shape[:2],-1))
    state_pairs = np.concatenate((states,reshaped_actions),axis=-1)

    unique_vals, counts = np.unique(state_pairs,axis=1,return_counts=True)

    count_vals, count_freq = np.unique(counts,return_counts=True)

    saco = unique_vals.shape[1]/num_tot

    return saco, count_vals, count_freq


def analyse_vault(
    vault_name: str,
    vault_uids: Optional[List[str]] = None,
    rel_dir: str = "vaults",
    visualise: bool = False,
) -> Dict[str, Array]:
    """Analyse a vault by computing the returns of each dataset quality.

    Args:
        vault_name (str): Name of vault.
        vault_uids (Optional[List[str]], optional):
            List of UIDs to process.
            Defaults to None, which uses all the subdirectories.
        rel_dir (str, optional): Base location of vaults. Defaults to "vaults".
        visualise (bool, optional):
            Optionally plot a violin distribution of this data.
            Defaults to False.

    Returns:
        Dict[str, Array]: Dictionary of {uid: episode_returns}
    """
    vault_uids = get_available_uids(f"./{rel_dir}/{vault_name}")
    
    all_uid_returns: Dict[str, Array] = {}  # Dictionary to store returns for each UID

    for uid in vault_uids:
        vlt = Vault(vault_name=vault_name, rel_dir=rel_dir, vault_uid=uid)
        exp = vlt.read().experience
        uid_returns = calculate_returns(exp)
        all_uid_returns[uid] = uid_returns

    if visualise:
        sns.set_theme(style="whitegrid")  # Set seaborn theme with a light blue background
        plt.figure(figsize=(8, 6))  # Adjust figsize as needed

        sns.violinplot(data=list(all_uid_returns.values()), inner="point")
        plt.title(f"Violin Distributions of Returns for {vault_name}")
        plt.xlabel("Dataset Quality")
        plt.ylabel("Episode Returns")
        plt.xticks(range(len(all_uid_returns)), list(all_uid_returns.keys()))

        plt.show()

    return all_uid_returns



def full_analysis(
    vault_name: str,
    vault_uids: Optional[List[str]] = None,
    rel_dir: str = "vaults",
    n_bins = 40,
) -> Dict[str, Array]:
    vault_uids = get_available_uids(f"./{rel_dir}/{vault_name}")

    all_returns = {}
    all_count_freq = {}
    all_count_vals = {}

    data_just_values = []

    min_return = 10
    max_return = -1

    for uid in vault_uids:
        vlt = Vault(vault_name=vault_name, rel_dir=rel_dir, vault_uid=uid)
        exp = vlt.read().experience
        n_trans = exp['actions'].shape[1]

        # we get episode returns and num traj from here.
        uid_returns = calculate_returns(exp)

        # we get joint saco and counts from here.
        saco, count_vals, count_freq = get_saco(exp)

        data_just_values.append([uid,np.mean(np.array(uid_returns)),np.std(np.array(uid_returns)),n_trans,len(uid_returns),saco])
        all_returns[uid] = uid_returns
        all_count_freq[uid] = count_freq
        all_count_vals[uid] = count_vals

        min_return = min(min(uid_returns),min_return)
        max_return = max(max(uid_returns),max_return)

    print(tabulate(data_just_values,headers=['Uid','Mean','Stddev','Transitions','Trajectories','Joint SACo']))

    # plot the episode return histograms
    fig, ax = plt.subplots(1,len(vault_uids),figsize=(3*len(vault_uids),3),sharex=True,sharey=True)

    colors = sns.color_palette()

    for i, uid in enumerate(vault_uids):
        counts, bins = np.histogram(all_returns[uid],bins=n_bins,range=(min_return-0.01,max_return+0.01))
        ax[i].stairs(counts, bins,fill=True,color=colors[i])
        ax[i].set_title(uid)
        ax[i].set_xlabel("Episode return")
    ax[0].set_ylabel("Frequency")
    fig.tight_layout()
    plt.show()

    # plot the power law showing count frequencies
    for i, uid in enumerate(vault_uids):
        plt.scatter(np.log(all_count_vals[uid].astype(float)),np.log(all_count_freq[uid].astype(float)),label=uid,color=colors[i])
    plt.title("Frequency of unique pair counts power law")
    plt.xlabel("Count (log base 10)")
    plt.ylabel("Frequency of count (log base 10)")
    plt.legend()
    plt.show()

    return data_just_values