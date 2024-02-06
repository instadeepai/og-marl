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

import numpy as np
import jax
import jax.numpy as jnp
import flashbax as fbx
import copy
from flashbax.buffers.trajectory_buffer import TrajectoryBufferState

from og_marl.offline_dataset import OfflineMARLDataset    

def vault_from_dataset(self, dataset):
    batch_size = 2048
    batched_dataset = dataset.raw_dataset.batch(batch_size)
    period = dataset.period
    max_episode_length = dataset.max_episode_length
    agents = dataset._agents

    experience = {
        "observations": {agent: [] for agent in agents},
        "actions": {agent: [] for agent in agents},
        "rewards": {agent: [] for agent in agents},
        "terminals": {agent: [] for agent in agents},
        "truncations": {agent: [] for agent in agents},
        "infos": {
            "legals": {agent: [] for agent in agents},
            "state": []
        }
    }

    episode = {
        "observations": {agent: [] for agent in agents},
        "actions": {agent: [] for agent in agents},
        "rewards": {agent: [] for agent in agents},
        "terminals": {agent: [] for agent in agents},
        "truncations": {agent: [] for agent in agents},
        "infos": {
            "legals": {agent: [] for agent in agents},
            "state": []
        }
    }

    episode_length = 0
    for batch in batched_dataset:
        mask = copy.deepcopy(batch["infos"]["mask"])
        B = mask.shape[0] # batch_size
        for idx in range(B):
            zero_padding_mask = mask[idx,:period]
            episode_length += np.sum(zero_padding_mask, dtype=int)

            for agent in agents:
                episode["observations"][agent].append(batch["observations"][agent][idx, :period])
                episode["actions"][agent].append(batch["actions"][agent][idx, :period])
                episode["rewards"][agent].append(batch["rewards"][agent][idx, :period])
                episode["terminals"][agent].append(batch["terminals"][agent][idx, :period])
                episode["truncations"][agent].append(batch["truncations"][agent][idx, :period])
                episode["infos"]["legals"][agent].append(batch["infos"]["legals"][agent][idx, :period])
            episode["infos"]["state"].append(batch["infos"]["state"][idx, :period])

            if (
                int(list(episode["terminals"].values())[0][-1][-1]) == 1 # agent 0, last chunck, last timestep in chunk
                or episode_length >= max_episode_length
            ):
                for agent in agents:
                    experience["observations"][agent].append(np.concatenate(episode["observations"][agent], axis=0)[:episode_length])
                    experience["actions"][agent].append(np.concatenate(episode["actions"][agent], axis=0)[:episode_length])
                    experience["rewards"][agent].append(np.concatenate(episode["rewards"][agent], axis=0)[:episode_length])
                    experience["terminals"][agent].append(np.concatenate(episode["terminals"][agent], axis=0)[:episode_length])
                    experience["truncations"][agent].append(np.concatenate(episode["truncations"][agent], axis=0)[:episode_length])
                    experience["infos"]["legals"][agent].append(np.concatenate(episode["infos"]["legals"][agent], axis=0)[:episode_length])
                experience["infos"]["state"].append(np.concatenate(episode["infos"]["state"], axis=0)[:episode_length])

                # Clear episode
                episode = {
                    "observations": {agent: [] for agent in agents},
                    "actions": {agent: [] for agent in agents},
                    "rewards": {agent: [] for agent in agents},
                    "terminals": {agent: [] for agent in agents},
                    "truncations": {agent: [] for agent in agents},
                    "infos": {
                        "legals": {agent: [] for agent in agents},
                        "state": []
                    }
                }
                episode_length = 0

    # Concatenate Episodes Together
    for agent in agents:
        experience["observations"][agent] = np.concatenate(experience["observations"][agent], axis=0)
        experience["actions"][agent] = np.concatenate(experience["actions"][agent], axis=0)
        experience["rewards"][agent] = np.concatenate(experience["rewards"][agent], axis=0)
        experience["terminals"][agent] = np.concatenate(experience["terminals"][agent], axis=0)
        experience["truncations"][agent] = np.concatenate(experience["truncations"][agent], axis=0)
        experience["infos"]["legals"][agent] = np.concatenate(experience["infos"]["legals"][agent], axis=0)
    experience["infos"]["state"] = np.concatenate(experience["infos"]["state"], axis=0)

    experience = jax.tree_map(lambda x: jnp.expand_dims(x, axis=0), experience)

    buffer_state = TrajectoryBufferState(experience=experience, is_full=jnp.array(False, dtype=bool), current_index=jnp.array(0))

    return buffer_state


##### Main
env = "smac_v1"
scenario = "3m"
dataset = "Good"

dataset = OfflineMARLDataset(env, env_name=env, scenario_name=scenario, dataset_type=dataset)

vault_from_dataset(dataset)