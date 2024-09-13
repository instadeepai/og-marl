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

import time
import flashbax as fbx
import jax
import jax.numpy as jnp
import numpy as np
import tree
import chex
from chex import Array
from flashbax.buffers.trajectory_buffer import TrajectoryBufferState
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
        store_to_vault: bool = False,
        vault_name: str = "recorded_data",
    ):
        self._sequence_length = sequence_length
        self._max_size = max_size
        self._batch_size = batch_size

        # Vault
        self._store_to_vault = store_to_vault
        self._vault_name = vault_name
        self._write_to_vault_every = max_size - 10
        self._vault_has_initialised = False
        self._step_count = 0

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
        self._buffer_add_fn = jax.jit(self._replay_buffer.add, donate_argnums=[0])

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
        
        stacked_infos = {}
        for key, value in infos.items():
            if isinstance(value, dict):
                stacked_infos[key] = np.stack(list(value.values()), axis=0)
            else:
                stacked_infos[key] = value

        timestep = {
            "observations": np.stack(list(observations.values()), axis=0),
            "actions": np.stack(list(actions.values()), axis=0),
            "rewards": np.stack(list(rewards.values()), axis=0),
            "terminals": np.stack(list(terminals.values()), axis=0),
            "truncations": np.stack(list(truncations.values()), axis=0),
            "infos": stacked_infos,
        }

        if self._buffer_state is None:
            self._buffer_state = self._replay_buffer.init(timestep)

            # Log stuff to vault/flashbax
            if not self._vault_has_initialised and self._store_to_vault:
                self._vault = Vault(
                    vault_name=self._vault_name,
                    experience_structure=self._buffer_state.experience,
                )
                self._vault_has_initialised = True

        timestep = tree.map_structure(
            lambda x: jnp.expand_dims(jnp.expand_dims(jnp.array(x), 0), 0), timestep
        )  # add batch & time dims

        self._buffer_state = self._buffer_add_fn(self._buffer_state, timestep)

        self._step_count += 1

        if self._step_count % self._write_to_vault_every == 0 and self._vault_has_initialised:
            self._vault.write(self._buffer_state)

    def sample(self) -> Experience:
        self._rng_key, sample_key = jax.random.split(self._rng_key, 2)
        batch = self._buffer_sample_fn(self._buffer_state, sample_key)
        return batch.experience  # type: ignore

    def populate_from_vault(
        self,
        source: str,
        env_name: str,
        scenario_name: str,
        dataset_name: str,
        rel_dir: str = "vaults",
    ) -> bool:
        self._buffer_state = Vault(
            vault_name=f"{source}/{env_name}/{scenario_name}.vlt",
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
