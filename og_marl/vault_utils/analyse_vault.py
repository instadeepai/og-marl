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


def get_saco(rel_dir,vault_name,uid):
    vlt = Vault(rel_dir=rel_dir, vault_name=vault_name, vault_uid=uid)
    all_data = vlt.read()
    offline_data = all_data.experience
    del vlt
    del all_data

    states = offline_data['infos']["state"]

    num_tot = states.shape[1]

    reshaped_actions = offline_data["actions"].reshape((*offline_data["actions"].shape[:2],-1))
    state_pairs = np.concatenate((states,reshaped_actions),axis=-1)

    unique_vals = np.unique(state_pairs,axis=1)

    saco = len(unique_vals)/num_tot

    return saco


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
