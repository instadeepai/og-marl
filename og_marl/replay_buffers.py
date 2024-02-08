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
from flashbax.vault import Vault
import tree

class FlashbaxReplayBuffer:

    def __init__(self, sequence_length, max_size=50_000, batch_size=32, sample_period=1, seed=42):

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

        self._buffer_state = None
        self._rng_key = jax.random.PRNGKey(seed)

    def add(self, observations, actions, rewards, terminals, truncations, infos):
        timestep = {
            "observations": observations,
            "actions": actions,
            "rewards": rewards,
            "terminals": terminals,
            "truncations": truncations,
            "infos": infos
        }

        if self._buffer_state is None:
            self._buffer_state = self._replay_buffer.init(timestep)
        
        timestep = tree.map_structure(lambda x: np.expand_dims(np.expand_dims(x, axis=0), axis=0), timestep) # add batch dim
        self._buffer_state = self._buffer_add_fn(self._buffer_state, timestep)

    def sample(self):
        self._rng_key, sample_key = jax.random.split(self._rng_key,2)
        batch = self._buffer_sample_fn(self._buffer_state, sample_key)
        return batch.experience
    
    def populate_from_vault(self, vault_name, vault_uid, vault_rel_dir="vaults"):
        self._buffer_state = Vault(vault_name, vault_uid=vault_uid, rel_dir=vault_rel_dir).read()
