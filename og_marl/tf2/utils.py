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
from typing import Any, Dict

import tensorflow as tf
from tensorflow import Module, Tensor


def set_growing_gpu_memory() -> None:
    """Solve gpu mem issues."""
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    physical_devices = tf.config.list_physical_devices("GPU")
    if physical_devices:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)


def gather(values: Tensor, indices: Tensor, axis: int = -1, keepdims: bool = False) -> Tensor:
    one_hot_indices = tf.one_hot(indices, depth=values.shape[axis])
    if len(values.shape) > 4:  # we have extra dim for distributional q-learning
        one_hot_indices = tf.expand_dims(one_hot_indices, axis=-1)
    gathered_values = tf.reduce_sum(values * one_hot_indices, axis=axis, keepdims=keepdims)
    return gathered_values


def switch_two_leading_dims(x: Tensor) -> Tensor:
    trailing_perm = []
    for i in range(2, len(x.shape)):
        trailing_perm.append(i)
    x: Tensor = tf.transpose(x, perm=[1, 0, *trailing_perm])
    return x


def merge_batch_and_agent_dim_of_time_major_sequence(x: Tensor) -> Tensor:
    T, B, N = x.shape[:3]  # assume time major
    trailing_dims = x.shape[3:]
    x = tf.reshape(x, shape=(T, B * N, *trailing_dims))  # type: ignore
    return x


def merge_time_batch_and_agent_dim(x: Tensor) -> Tensor:
    T, B, N = x.shape[:3]  # assume time major
    trailing_dims = x.shape[3:]
    x = tf.reshape(x, shape=(T * B * N, *trailing_dims))  # type: ignore
    return x


def expand_time_batch_and_agent_dim_of_time_major_sequence(
    x: Tensor, T: int, B: int, N: int
) -> Tensor:
    TNB = x.shape[:1]  # assume time major
    assert TNB == T * B * N  # type: ignore
    trailing_dims = x.shape[1:]
    x = tf.reshape(x, shape=(T, B, N, *trailing_dims))  # type: ignore
    return x


def expand_batch_and_agent_dim_of_time_major_sequence(x: Tensor, B: int, N: int) -> Tensor:
    T, NB = x.shape[:2]  # assume time major
    assert NB == B * N
    trailing_dims = x.shape[2:]
    x = tf.reshape(x, shape=(T, B, N, *trailing_dims))  # type: ignore
    return x


def concat_agent_id_to_obs(obs: Tensor, agent_id: int, N: int) -> Tensor:
    is_vector_obs = len(obs.shape) == 1

    if is_vector_obs:
        agent_id = tf.one_hot(agent_id, depth=N)
    else:
        h, w = obs.shape[:2]
        agent_id = tf.zeros((h, w, 1), "float32") + (agent_id / N) + 1 / (2 * N)

    if not is_vector_obs and len(obs.shape) == 2:  # if no channel dim
        obs = tf.expand_dims(obs, axis=-1)

    obs: Tensor = tf.concat([agent_id, obs], axis=-1)

    return obs


def unroll_rnn(rnn_network: Module, inputs: Tensor, resets: Tensor) -> Tensor:
    T, B = inputs.shape[:2]

    outputs = []
    hidden_state = rnn_network.initial_state(B)  # type: ignore
    for i in range(T):  # type: ignore
        output, hidden_state = rnn_network(inputs[i], hidden_state)  # type: ignore
        outputs.append(output)

        hidden_state = (
            tf.where(
                tf.cast(tf.expand_dims(resets[i], axis=-1), "bool"),
                rnn_network.initial_state(B)[0],  # type: ignore
                hidden_state[0],
            ),
        )  # hidden state wrapped in tuple

    return tf.stack(outputs, axis=0)  # type: ignore


def batch_concat_agent_id_to_obs(obs: Tensor) -> Tensor:
    B, T, N = obs.shape[:3]  # batch size, timedim, num_agents
    is_vector_obs = len(obs.shape) == 4

    agent_ids = []
    for i in range(N):  # type: ignore
        if is_vector_obs:
            agent_id = tf.one_hot(i, depth=N)
        else:
            h, w = obs.shape[3:5]
            agent_id = tf.zeros((h, w, 1), "float32") + (i / N) + 1 / (2 * N)  # type: ignore
        agent_ids.append(agent_id)
    agent_ids = tf.stack(agent_ids, axis=0)

    # Repeat along time dim
    agent_ids = tf.stack([agent_ids] * T, axis=0)  # type: ignore

    # Repeat along batch dim
    agent_ids = tf.stack([agent_ids] * B, axis=0)  # type: ignore

    if not is_vector_obs and len(obs.shape) == 5:  # if no channel dim
        obs = tf.expand_dims(obs, axis=-1)

    obs = tf.concat([agent_ids, obs], axis=-1)

    return obs


def batched_agents(agents, batch_dict):  # type: ignore
    batched_agents_dict: Dict[str, Any] = {
        "observations": [],
        "actions": [],
        "rewards": [],
        "terminals": [],
        "truncations": [],
        "infos": {},
    }

    for agent in agents:
        for key in batched_agents_dict:
            if key == "infos":
                continue
            batched_agents_dict[key].append(batch_dict[key][agent])
    for key, value in batched_agents_dict.items():
        if key == "infos":
            continue
        batched_agents_dict[key] = tf.stack(value, axis=2)

    batched_agents_dict["terminals"] = tf.cast(batched_agents_dict["terminals"], "float32")
    batched_agents_dict["truncations"] = tf.cast(batched_agents_dict["truncations"], "float32")

    if "legals" in batch_dict["infos"]:
        batched_agents_dict["infos"]["legals"] = []
        for agent in agents:
            batched_agents_dict["infos"]["legals"].append(batch_dict["infos"]["legals"][agent])
        batched_agents_dict["infos"]["legals"] = tf.stack(
            batched_agents_dict["infos"]["legals"], axis=2
        )

    if "state" in batch_dict["infos"]:
        batched_agents_dict["infos"]["state"] = tf.convert_to_tensor(
            batch_dict["infos"]["state"], "float32"
        )

    if "mask" in batch_dict["infos"]:
        batched_agents_dict["mask"] = tf.convert_to_tensor(batch_dict["infos"]["mask"], "float32")
    else:
        batched_agents_dict["mask"] = tf.ones_like(
            batched_agents_dict["terminals"][:, :, 0], "float32"
        )

    return batched_agents_dict
