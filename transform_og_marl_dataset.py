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
from tqdm import tqdm

from og_marl.environments.utils import get_environment
from og_marl.offline_dataset import OfflineMARLDataset


def vault_from_dataset(dataset):
    batch_size = 2048
    batched_dataset = dataset.raw_dataset.batch(batch_size)
    period = dataset.period
    max_episode_length = dataset.max_episode_length
    agents = dataset._agents

    episode = {
        "observations": {agent: [] for agent in agents},
        "actions": {agent: [] for agent in agents},
        "rewards": {agent: [] for agent in agents},
        "terminals": {agent: [] for agent in agents},
        "truncations": {agent: [] for agent in agents},
        "infos": {"legals": {agent: [] for agent in agents}, "state": []},
    }

    buffer = fbx.make_flat_buffer(
        max_length=3_000_000,
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

            for agent in agents:
                episode["observations"][agent].append(batch["observations"][agent][idx, :period])
                episode["actions"][agent].append(batch["actions"][agent][idx, :period])
                episode["rewards"][agent].append(batch["rewards"][agent][idx, :period])
                episode["terminals"][agent].append(batch["terminals"][agent][idx, :period])
                episode["truncations"][agent].append(batch["truncations"][agent][idx, :period])
                episode["infos"]["legals"][agent].append(
                    batch["infos"]["legals"][agent][idx, :period]
                )
            episode["infos"]["state"].append(batch["infos"]["state"][idx, :period])

            if (
                int(list(episode["terminals"].values())[0][-1][-1])
                == 1  # agent 0, last chunk, last timestep in chunk
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
                        vault_name="test.vlt",
                        experience_structure=buffer_state.experience,
                    )
                    initialised_buffer_state = True

                buffer_state = jax.jit(buffer.add, donate_argnums=0)(buffer_state, episode_to_save)

                # Clear episode
                episode = {
                    "observations": {agent: [] for agent in agents},
                    "actions": {agent: [] for agent in agents},
                    "rewards": {agent: [] for agent in agents},
                    "terminals": {agent: [] for agent in agents},
                    "truncations": {agent: [] for agent in agents},
                    "infos": {"legals": {agent: [] for agent in agents}, "state": []},
                }
                episode_length = 0

        num_steps_written += v.write(buffer_state)
        print(f"Wrote {num_steps_written} steps")

    return num_steps_written


##### Main
env_name = "mamujoco"
scenario = "2halfcheetah"
dataset = "Poor"

env = get_environment(env_name, scenario)

dataset = OfflineMARLDataset(env, env_name=env_name, scenario_name=scenario, dataset_type=dataset)

vault_from_dataset(dataset)
