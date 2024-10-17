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
import sys
import zipfile
from typing import Dict, List

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import requests  # type: ignore
import seaborn as sns
from chex import Array
from flashbax.vault import Vault
from git import Optional

VAULT_INFO = {
    "smac_v1": {
        "3m": {
            "url": "https://huggingface.co/datasets/InstaDeepAI/og-marl/resolve/main/core/smac_v1/3m.zip"
        },
        "8m": {
            "url": "https://huggingface.co/datasets/InstaDeepAI/og-marl/resolve/main/core/smac_v1/8m.zip"
        },
        "5m_vs_6m": {
            "url": "https://huggingface.co/datasets/InstaDeepAI/og-marl/resolve/main/core/smac_v1/5m_vs_6m.zip"
        },
        "2s3z": {
            "url": "https://huggingface.co/datasets/InstaDeepAI/og-marl/resolve/main/core/smac_v1/2s3z.zip"
        },
        "3s5z_vs_3s6z": {
            "url": "https://huggingface.co/datasets/InstaDeepAI/og-marl/resolve/main/core/smac_v1/3s5z_vs_3s6z.zip"
        },
    },
    "smac_v2": {
        "terran_5_vs_5": {
            "url": "https://huggingface.co/datasets/InstaDeepAI/og-marl/resolve/main/core/smac_v2/terran_5_vs_5.zip"
        },
        "zerg_5_vs_5": {
            "url": "https://huggingface.co/datasets/InstaDeepAI/og-marl/resolve/main/core/smac_v2/zerg_5_vs_5.zip"
        },
    },
    "mamujoco": {
        "2halfcheetah": {
            "url": "https://huggingface.co/datasets/InstaDeepAI/og-marl/resolve/main/core/mamujoco/2halfcheetah.zip"
        },
    },
}


def download_and_unzip_vault(
    env_name: str,
    scenario_name: str,
    dataset_base_dir: str = "./vaults",
) -> None:
    if check_directory_exists_and_not_empty(f"{dataset_base_dir}/{env_name}/{scenario_name}.vlt"):
        print(f"Vault '{dataset_base_dir}/{env_name}/{scenario_name}' already exists.")
        return

    dataset_download_url = VAULT_INFO[env_name][scenario_name]["url"]

    os.makedirs(f"{dataset_base_dir}/tmp/", exist_ok=True)
    os.makedirs(f"{dataset_base_dir}/{env_name}/", exist_ok=True)

    zip_file_path = f"{dataset_base_dir}/tmp/tmp_dataset.zip"

    extraction_path = f"{dataset_base_dir}/{env_name}"

    response = requests.get(dataset_download_url, stream=True)
    total_length = response.headers.get("content-length")

    with open(zip_file_path, "wb") as file:
        if total_length is None:  # no content length header
            file.write(response.content)
        else:
            dl = 0
            total_length = int(total_length)  # type: ignore
            for data in response.iter_content(chunk_size=4096):
                dl += len(data)
                file.write(data)
                done = int(50 * dl / total_length)  # type: ignore
                sys.stdout.write("\r[%s%s]" % ("=" * done, " " * (50 - done)))
                sys.stdout.flush()

    # Step 2: Unzip the file
    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        zip_ref.extractall(extraction_path)

    # Optionally, delete the zip file after extraction
    os.remove(zip_file_path)


def check_directory_exists_and_not_empty(path: str) -> bool:
    # Check if the directory exists
    if os.path.exists(path) and os.path.isdir(path):
        # Check if the directory is not empty
        if not os.listdir(path):  # This will return an empty list if the directory is empty
            return False  # Directory exists but is empty
        else:
            return True  # Directory exists and is not empty
    else:
        return False  # Directory does not exist


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
    terminals = jnp.array(experience["terminals"][0].all(axis=-1).squeeze(), dtype=jnp.float32)

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
    vault_uids = sorted(
        next(os.walk(os.path.join(rel_dir, vault_name)))[1],
        reverse=True,
    )
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
