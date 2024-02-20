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

import copy

import flashbax as fbx
import jax
import numpy as np
from flashbax.vault import Vault
import tensorflow as tf
from tqdm import tqdm
import h5py
import tree

from og_marl.environments.utils import get_environment
from og_marl.offline_dataset import OfflineMARLDataset


def vault_from_dataset(dataset, vault_name):
    batch_size = 200
    batched_dataset = dataset.raw_dataset.batch(batch_size)
    period = dataset.period
    max_episode_length = dataset.max_episode_length
    agents = dataset._agents

    episode = {
        "observations": [],
        "actions": [],
        "rewards": [],
        "terminals": [],
        "truncations": [],
        "infos": {"legals": [], "state": []},
    }

    buffer = fbx.make_flat_buffer(
        max_length=200_000,
        min_length=1,
        sample_batch_size=1,
        add_sequences=True,
        add_batch_size=None,
    )
    buffer_state = ...
    initialised_buffer_state = False
    v = ...
    num_steps_written = 0

    episode_length = 0
    for batch in batched_dataset:
        mask = copy.deepcopy(batch["infos"]["mask"])
        B = mask.shape[0]  # batch_size
        for idx in tqdm(range(B)):
            zero_padding_mask = mask[idx, :period]
            episode_length += np.sum(zero_padding_mask, dtype=int)

            # for agent in agents:
            episode["observations"].append(tf.stack(
                list(jax.tree_map(lambda x: x[idx, :period], batch["observations"]).values()), axis=1
            ))
            episode["actions"].append(tf.stack(
                list(jax.tree_map(lambda x: x[idx, :period], batch["actions"]).values()), axis=1
            ))
            episode["rewards"].append(tf.stack(
                list(jax.tree_map(lambda x: x[idx, :period], batch["rewards"]).values()), axis=1
            ))
            episode["terminals"].append(tf.stack(
                list(jax.tree_map(lambda x: x[idx, :period], batch["terminals"]).values()), axis=1
            ))
            episode["truncations"].append(tf.stack(
                list(jax.tree_map(lambda x: x[idx, :period], batch["truncations"]).values()), axis=1
            ))
            episode["infos"]["legals"].append(tf.stack(
                list(jax.tree_map(lambda x: x[idx, :period], batch["infos"]["legals"]).values()), axis=1
            ))
            episode["infos"]["state"].append(batch["infos"]["state"][idx, :period])

            if (
                episode["terminals"][-1][-1, 0] == 1  # last chunk, last timestep in chunk, agent 0
                or episode_length >= max_episode_length
            ):
                episode_to_save = jax.tree_map(
                    lambda x, ep_len=episode_length: np.concatenate(x, axis=0)[:ep_len],
                    episode,
                    is_leaf=lambda x: isinstance(x, list),
                )
                if not initialised_buffer_state:
                    buffer_state = buffer.init(jax.tree_map(lambda x: x[0, ...], episode_to_save))
                    v = Vault(
                        vault_name=vault_name,
                        experience_structure=buffer_state.experience,
                    )
                    initialised_buffer_state = True

                buffer_state = jax.jit(buffer.add, donate_argnums=0)(buffer_state, episode_to_save)

                # Clear episode
                episode = {
                    "observations": [],
                    "actions": [],
                    "rewards": [],
                    "terminals": [],
                    "truncations": [],
                    "infos": {"legals": [], "state": []},
                }
                episode_length = 0
        num_steps_written += v.write(buffer_state)
        print(f"Wrote {num_steps_written} steps")
    return num_steps_written


##### Main

from og_marl.offline_dataset import DATASET_INFO, download_and_unzip_dataset
from og_marl.environments.utils import get_environment

for env_name in DATASET_INFO.keys():
    if env_name not in ["smac_v2"]:
        continue
    for scenario_name in DATASET_INFO[env_name].keys():
        if scenario_name in []:
            continue
        # if scenario_name not in []:
        #     download_and_unzip_dataset(env_name, scenario_name)

        env = get_environment(env_name, scenario_name)

        for dataset_name in ["Replay"]:
            dataset = OfflineMARLDataset(env, env_name, scenario_name, dataset_name)
            vault_from_dataset(dataset, f"{env_name}_{scenario_name}_{dataset_name}.vlt")
