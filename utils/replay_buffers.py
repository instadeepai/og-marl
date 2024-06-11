from typing import Dict

import flashbax as fbx
import jax
from chex import Array
from flashbax.buffers.trajectory_buffer import TrajectoryBufferState
from flashbax.vault import Vault

Experience = Dict[str, Array]


class FlashbaxReplayBuffer:
    def __init__(
        self,
        sequence_length: int,
        batch_size: int = 64,
        sample_period: int = 1,
        seed: int = 42,
    ):
        self._sequence_length = sequence_length
        self._batch_size = batch_size
        self._sample_period = sample_period

        # Flashbax buffer, made when populating
        self._replay_buffer = None
        self._buffer_state: TrajectoryBufferState = None

        # Random key for sampling
        self._rng_key = jax.random.PRNGKey(seed)

    def sample(self) -> Experience:
            self._rng_key, sample_key = jax.random.split(self._rng_key, 2)
            batch = self._buffer_sample_fn(self._buffer_state, sample_key)
            return batch.experience  # type: ignore

    def populate_from_vault(
        self, env_name: str, scenario_name: str, dataset_name: str, rel_dir: str = "vaults"
    ) -> bool:
        self._buffer_state = Vault(
            vault_name=f"{env_name}/{scenario_name}.vlt",
            vault_uid=dataset_name,
            rel_dir=rel_dir,
        ).read()

        # Recreate the buffer and associated pure functions
        self._replay_buffer = fbx.make_trajectory_buffer(
            add_batch_size=1,
            sample_batch_size=self._batch_size,
            sample_sequence_length=self._sequence_length,
            period=self._sample_period,
            min_length_time_axis=1,
            max_size=self._sequence_length,
        )
        self._buffer_sample_fn = jax.jit(self._replay_buffer.sample)

        return True
