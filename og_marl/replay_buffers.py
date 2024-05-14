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

from typing import Any, Dict
import copy

import flashbax as fbx
import jax
import jax.numpy as jnp
import numpy as np
import tree
from chex import Array
from flashbax.buffers.trajectory_buffer import TrajectoryBufferState
from flashbax.buffers.prioritised_trajectory_buffer import PrioritisedTrajectoryBufferState

from flashbax.vault import Vault

Experience = Dict[str, Array]


class FlashbaxReplayBuffer:
    def __init__(
        self,
        sequence_length: int,
        max_size: int = 50_000,
        batch_size: int = 32,
        sample_period: int = 1,
        seed: int = 42,
    ):
        self._sequence_length = sequence_length
        self._max_size = max_size
        self._batch_size = batch_size
        # Flashbax buffer
        self._replay_buffer = fbx.make_trajectory_buffer(
            add_batch_size=1,
            sample_batch_size=batch_size,
            sample_sequence_length=sequence_length,
            period=sample_period,
            min_length_time_axis=1,
            max_size=max_size,
        )
        self._buffer_sample_fn = jax.jit(self._replay_buffer.sample)
        self._buffer_add_fn = jax.jit(self._replay_buffer.add)
        self._buffer_state: TrajectoryBufferState = None
        self._rng_key = jax.random.PRNGKey(seed)

    def add(
        self,
        observations: Dict[str, np.ndarray],
        actions: Dict[str, np.ndarray],
        rewards: Dict[str, np.ndarray],
        terminals: Dict[str, np.ndarray],
        truncations: Dict[str, np.ndarray],
        infos: Dict[str, Any],
    ) -> None:
        new_infos = copy.deepcopy(infos)
        if "legals" in infos:
            new_infos["legals"] = np.stack(list(infos["legals"].values()), axis=0)
        timestep = {
            "observations": np.stack(list(observations.values()), axis=0),
            "actions": np.stack(list(actions.values()), axis=0),
            "rewards": np.stack(list(rewards.values()), axis=0),
            "terminals": np.stack(list(terminals.values()), axis=0),
            "truncations": np.stack(list(truncations.values()), axis=0),
            "infos": new_infos,
        }
        if self._buffer_state is None:
            self._buffer_state = self._replay_buffer.init(timestep)
        timestep = tree.map_structure(
            lambda x: jnp.expand_dims(jnp.expand_dims(jnp.array(x), 0), 0), timestep
        )  # add batch & time dims
        self._buffer_state = self._buffer_add_fn(self._buffer_state, timestep)

    def sample(self) -> Experience:
        self._rng_key, sample_key = jax.random.split(self._rng_key, 2)
        batch = self._buffer_sample_fn(self._buffer_state, sample_key)
        return batch.experience  # type: ignore

    def populate_from_vault(
        self,
        env_name,
        scenario_name,
        # vault_name="tmp",
        dataset_name="tmp",
        rel_dir: str = "vaults",
    ) -> bool:
        self._buffer_state = Vault(
            vault_name=f"{env_name}/{scenario_name}.vlt",
            # vault_name=vault_name,
            vault_uid=dataset_name,
            rel_dir=rel_dir,
        ).read()
        # Recreate the buffer and associated pure functions
        self._replay_buffer = fbx.make_trajectory_buffer(
            add_batch_size=1,
            sample_batch_size=self._batch_size,
            sample_sequence_length=self._sequence_length,
            period=1,
            min_length_time_axis=1,
            max_size=self._sequence_length,
        )
        self._buffer_sample_fn = jax.jit(self._replay_buffer.sample)
        self._buffer_add_fn = jax.jit(self._replay_buffer.add)
        return True


class PrioritisedFlashbaxReplayBuffer:
    def __init__(
        self,
        sequence_length: int,
        max_size: int = 50_000,
        batch_size: int = 32,
        sample_period: int = 1,
        seed: int = 42,
        priority_exponent=0.6,
    ):
        self._sequence_length = sequence_length
        self._max_size = max_size
        self._batch_size = batch_size
        self._priority_exponent = priority_exponent
        # Flashbax buffer
        self._vault_replay_buffer = fbx.make_trajectory_buffer(
            add_batch_size=1,
            sample_batch_size=batch_size,
            sample_sequence_length=sequence_length,
            period=sample_period,
            min_length_time_axis=1,
            max_size=max_size,
        )

        self._replay_buffer = fbx.make_prioritised_trajectory_buffer(
            add_batch_size=1,
            sample_batch_size=batch_size,
            sample_sequence_length=sequence_length,
            period=sample_period,
            min_length_time_axis=1,
            max_size=max_size,
            priority_exponent=self._priority_exponent,
        )
        self._buffer_sample_fn = jax.jit(self._replay_buffer.sample)
        self._buffer_add_fn = jax.jit(self._replay_buffer.add)
        self._buffer_set_priorities_fn = jax.jit(self._replay_buffer.set_priorities)

        self._buffer_state: PrioritisedTrajectoryBufferState = None
        self._rng_key = jax.random.PRNGKey(seed)

    def add(
        self,
        observations: Dict[str, np.ndarray],
        actions: Dict[str, np.ndarray],
        rewards: Dict[str, np.ndarray],
        terminals: Dict[str, np.ndarray],
        truncations: Dict[str, np.ndarray],
        infos: Dict[str, Any],
    ) -> None:
        new_infos = copy.deepcopy(infos)
        if "legals" in infos:
            new_infos["legals"] = np.stack(list(infos["legals"].values()), axis=0)
        timestep = {
            "observations": np.stack(list(observations.values()), axis=0),
            "actions": np.stack(list(actions.values()), axis=0),
            "rewards": np.stack(list(rewards.values()), axis=0),
            "terminals": np.stack(list(terminals.values()), axis=0),
            "truncations": np.stack(list(truncations.values()), axis=0),
            "infos": new_infos,
        }
        if self._buffer_state is None:
            self._buffer_state = self._replay_buffer.init(timestep)
        timestep = tree.map_structure(
            lambda x: jnp.expand_dims(jnp.expand_dims(jnp.array(x), 0), 0), timestep
        )  # add batch & time dims
        self._buffer_state = self._buffer_add_fn(self._buffer_state, timestep)

    def sample(self) -> Experience:
        self._rng_key, sample_key = jax.random.split(self._rng_key, 2)
        batch = self._buffer_sample_fn(self._buffer_state, sample_key)
        return batch  # type: ignore

    def update_priorities(self, indices, priorities):
        self._buffer_state = self._buffer_set_priorities_fn(
            self._buffer_state, indices, priorities.numpy()
        )

    def populate_from_vault(
        self,
        # env_name,
        # scenario_name,
        vault_name="tmp",
        dataset_name="tmp",
        rel_dir: str = "vaults",
    ) -> bool:
        self._vault_buffer_state = Vault(
            # vault_name=f"{env_name}/{scenario_name}.vlt",
            vault_name=vault_name,
            vault_uid=dataset_name,
            rel_dir=rel_dir,
        ).read(percentiles=(0, 100))
        example_timestep = self._vault_replay_buffer.sample(
            self._vault_buffer_state, jax.random.PRNGKey(0)
        )
        example_timestep = jax.tree_map(lambda x: x[0, 0], example_timestep.experience)

        # Recreate the buffer and associated pure functions
        self._replay_buffer = fbx.make_prioritised_trajectory_buffer(
            add_batch_size=1,
            sample_batch_size=self._batch_size,
            sample_sequence_length=self._sequence_length,
            period=1,  # self._sequence_length, #1,
            min_length_time_axis=1,
            max_size=self._vault_buffer_state.experience["truncations"].shape[1] + 1,
            priority_exponent=self._priority_exponent,
        )
        temp_buffer_state = self._replay_buffer.init(example_timestep)
        # del temp_buffer_state

        # self._buffer_state = PrioritisedTrajectoryBufferState(
        #     experience=self._vault_buffer_state.experience,
        #     current_index=self._vault_buffer_state.current_index,
        #     is_full=self._vault_buffer_state.is_full,
        #     priority_state=temp_buffer_state_priorities,
        # )

        self._buffer_sample_fn = jax.jit(self._replay_buffer.sample)
        self._buffer_add_fn = jax.jit(self._replay_buffer.add)
        self._buffer_set_priorities_fn = jax.jit(self._replay_buffer.set_priorities)

        self._buffer_state = self._buffer_add_fn(
            temp_buffer_state, self._vault_buffer_state.experience
        )

        return True
