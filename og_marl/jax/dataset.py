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

import jax
import jax.numpy as jnp
import orbax.checkpoint
from flashbax.buffers.trajectory_buffer import TrajectoryBufferState


class FlashbaxBufferStore:
    def __init__(
        self,
        dataset_path: str,
    ) -> None:
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        options = orbax.checkpoint.CheckpointManagerOptions(
            max_to_keep=1,
            create=True,
        )
        self._manager = orbax.checkpoint.CheckpointManager(
            os.path.join(os.getcwd(), dataset_path),
            orbax_checkpointer,
            options,
            metadata=None,
        )

    def save(self, t, buffer_state):
        return self._manager.save(step=t, items=buffer_state)

    def restore_state(self):
        raw_restored = self._manager.restore(self._manager.latest_step())
        return TrajectoryBufferState(
            experience=jax.tree_util.tree_map(jnp.asarray, raw_restored["experience"]),
            current_index=jnp.asarray(raw_restored["current_index"]),
            is_full=jnp.asarray(raw_restored["is_full"]),
        )

    def restore_metadata(self):
        metadata = self._manager.metadata()
        return metadata
