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
from pathlib import Path
from typing import Any, Dict, List

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import requests
import seaborn as sns
import tensorflow as tf
import tree
from chex import Array
from flashbax.vault import Vault
from git import Optional
from tensorflow import DType

from og_marl.environments.base import BaseEnvironment

VAULT_INFO = {
    "smac_v1": {
        "3m": {"url": "https://s3.kao.instadeep.io/offline-marl-dataset/vaults/3m.zip"},
        "8m": {"url": "https://s3.kao.instadeep.io/offline-marl-dataset/vaults/8m.zip"},
        "5m_vs_6m": {"url": "https://s3.kao.instadeep.io/offline-marl-dataset/vaults/5m_vs_6m.zip"},
        "2s3z": {"url": "https://s3.kao.instadeep.io/offline-marl-dataset/vaults/2s3z.zip"},
        "3s5z_vs_3s6z": {
            "url": "https://s3.kao.instadeep.io/offline-marl-dataset/vaults/3s5z_vs_3s6z.zip"
        },
        "2c_vs_64zg": {
            "url": "https://s3.kao.instadeep.io/offline-marl-dataset/vaults/2c_vs_64zg.zip"
        },
    },
    "smac_v2": {
        "terran_5_vs_5": {
            "url": "https://s3.kao.instadeep.io/offline-marl-dataset/vaults/terran_5_vs_5.zip"
        },
    },
    "mamujoco": {
        "2ant": {"url": "https://s3.kao.instadeep.io/offline-marl-dataset/vaults/2ant.zip"},
        "2halfcheetah": {
            "url": "https://s3.kao.instadeep.io/offline-marl-dataset/vaults/2halfcheetah.zip"
        },
        "4ant": {"url": "https://s3.kao.instadeep.io/offline-marl-dataset/vaults/4ant.zip"},
    },
}

DATASET_INFO = {
    "smac_v1": {
        "3m": {"url": "https://tinyurl.com/3m-dataset", "sequence_length": 20, "period": 10},
        "8m": {"url": "https://tinyurl.com/8m-dataset", "sequence_length": 20, "period": 10},
        "5m_vs_6m": {
            "url": "https://tinyurl.com/5m-vs-6m-dataset",
            "sequence_length": 20,
            "period": 10,
        },
        "2s3z": {"url": "https://tinyurl.com/2s3z-dataset", "sequence_length": 20, "period": 10},
        "3s5z_vs_3s6z": {
            "url": "https://tinyurl.com/3s5z-vs-3s6z-dataset3",
            "sequence_length": 20,
            "period": 10,
        },
        "2c_vs_64zg": {
            "url": "https://tinyurl.com/2c-vs-64zg-dataset",
            "sequence_length": 20,
            "period": 10,
        },
        "27m_vs_30m": {
            "url": "https://tinyurl.com/27m-vs-30m-dataset",
            "sequence_length": 20,
            "period": 10,
        },
    },
    "smac_v2": {
        "terran_5_vs_5": {
            "url": "https://tinyurl.com/terran-5-vs-5-dataset",
            "sequence_length": 20,
            "period": 10,
        },
        "zerg_5_vs_5": {
            "url": "https://tinyurl.com/zerg-5-vs-5-dataset",
            "sequence_length": 20,
            "period": 10,
        },
        "terran_10_vs_10": {
            "url": "https://tinyurl.com/terran-10-vs-10-dataset",
            "sequence_length": 20,
            "period": 10,
        },
    },
    "flatland": {
        "3trains": {
            "url": "https://tinyurl.com/3trains-dataset",
            "sequence_length": 20,  # TODO
            "period": 10,
        },
        "5trains": {
            "url": "https://tinyurl.com/5trains-dataset",
            "sequence_length": 20,  # TODO
            "period": 10,
        },
    },
    "mamujoco": {
        "2halfcheetah": {
            "url": "https://tinyurl.com/2halfcheetah-dataset",
            "sequence_length": 20,
            "period": 10,
        },
        "2ant": {"url": "https://tinyurl.com/2ant-dataset", "sequence_length": 20, "period": 10},
        "4ant": {"url": "https://tinyurl.com/4ant-dataset", "sequence_length": 20, "period": 10},
    },
    "voltage_control": {
        "case33_3min_final": {
            "url": "https://tinyurl.com/case33-3min-final-dataset",
            "sequence_length": 20,
            "period": 10,
        },
    },
}


def get_schema_dtypes(environment: BaseEnvironment) -> Dict[str, DType]:
    act_type = list(environment.action_spaces.values())[0].dtype
    schema = {}
    for agent in environment.possible_agents:
        schema[agent + "_observations"] = tf.float32
        schema[agent + "_legal_actions"] = tf.float32
        schema[agent + "_actions"] = act_type
        schema[agent + "_rewards"] = tf.float32
        schema[agent + "_discounts"] = tf.float32

    ## Extras
    # Zero-padding mask
    schema["zero_padding_mask"] = tf.float32

    # Env state
    schema["env_state"] = tf.float32

    # Episode return
    schema["episode_return"] = tf.float32

    return schema


class OfflineMARLDataset:
    def __init__(
        self,
        environment: BaseEnvironment,
        env_name: str,
        scenario_name: str,
        dataset_type: str,
        base_dataset_dir: str = "./datasets",
    ):
        self._environment = environment
        self._schema = get_schema_dtypes(environment)
        self._agents = environment.possible_agents

        path_to_dataset = f"{base_dataset_dir}/{env_name}/{scenario_name}/{dataset_type}"

        file_path = Path(path_to_dataset)
        sub_dir_to_idx = {}
        idx = 0
        for subdir in os.listdir(file_path):
            if file_path.joinpath(subdir).is_dir():
                sub_dir_to_idx[subdir] = idx
                idx += 1

        def get_fname_idx(file_name: str) -> int:
            dir_idx = sub_dir_to_idx[file_name.split("/")[-2]] * 1000
            return dir_idx + int(file_name.split("log_")[-1].split(".")[0])

        filenames = [str(file_name) for file_name in file_path.glob("**/*.tfrecord")]
        filenames = sorted(filenames, key=get_fname_idx)

        filename_dataset = tf.data.Dataset.from_tensor_slices(filenames)
        self.raw_dataset = filename_dataset.flat_map(
            lambda x: tf.data.TFRecordDataset(x, compression_type="GZIP").map(self._decode_fn)
        )

        self.period = DATASET_INFO[env_name][scenario_name]["period"]
        self.sequence_length = DATASET_INFO[env_name][scenario_name]["sequence_length"]
        self.max_episode_length = environment.max_episode_length

    def _decode_fn(self, record_bytes: Any) -> Dict[str, Any]:
        example = tf.io.parse_single_example(
            record_bytes,
            tree.map_structure(lambda x: tf.io.FixedLenFeature([], dtype=tf.string), self._schema),
        )

        for key, dtype in self._schema.items():
            example[key] = tf.io.parse_tensor(example[key], dtype)

        sample: Dict[str, dict] = {
            "observations": {},
            "actions": {},
            "rewards": {},
            "terminals": {},
            "truncations": {},
            "infos": {"legals": {}},
        }
        for agent in self._agents:
            sample["observations"][agent] = example[f"{agent}_observations"]
            sample["actions"][agent] = example[f"{agent}_actions"]
            sample["rewards"][agent] = example[f"{agent}_rewards"]
            sample["terminals"][agent] = 1 - example[f"{agent}_discounts"]
            sample["truncations"][agent] = tf.zeros_like(example[f"{agent}_discounts"])
            sample["infos"]["legals"][agent] = example[f"{agent}_legal_actions"]

        sample["infos"]["mask"] = example["zero_padding_mask"]
        sample["infos"]["state"] = example["env_state"]
        sample["infos"]["episode_return"] = example["episode_return"]

        return sample

    def __getattr__(self, name: Any) -> Any:
        """Expose any other attributes of the underlying environment.

        Args:
        ----
            name (str): attribute.

        Returns:
        -------
            Any: return attribute from env or underlying env.

        """
        if hasattr(self.__class__, name):
            return self.__getattribute__(name)
        else:
            return getattr(self._tf_dataset, name)


def download_and_unzip_dataset(
    env_name: str,
    scenario_name: str,
    dataset_base_dir: str = "./datasets",
) -> None:
    dataset_download_url = DATASET_INFO[env_name][scenario_name]["url"]

    # TODO add check to see if dataset exists already.

    os.makedirs(f"{dataset_base_dir}/tmp/", exist_ok=True)
    os.makedirs(f"{dataset_base_dir}/{env_name}/", exist_ok=True)

    zip_file_path = f"{dataset_base_dir}/tmp/tmp_dataset.zip"

    extraction_path = f"{dataset_base_dir}/{env_name}"

    response = requests.get(dataset_download_url, stream=True)  # type: ignore
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


def download_and_unzip_vault(
    env_name: str,
    scenario_name: str,
    dataset_base_dir: str = "./vaults",
) -> None:
    dataset_download_url = VAULT_INFO[env_name][scenario_name]["url"]

    if check_directory_exists_and_not_empty(f"{dataset_base_dir}/{env_name}/{scenario_name}.vlt"):
        print(f"Vault '{dataset_base_dir}/{env_name}/{scenario_name}' already exists.")
        return

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
    terminals = experience_one_agent[terminal_key]

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
