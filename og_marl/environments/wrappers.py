# type: ignore
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

import flashbax as fbx
import jax
import numpy as np
from flashbax.vault import Vault

from og_marl.environments.base import BaseEnvironment, ResetReturn, StepReturn

BUFFER_TIME_AXIS_LEN = 100_000


class ExperienceRecorder:
    def __init__(
        self, environment: BaseEnvironment, vault_name: str, write_to_vault_every: int = 10_000
    ):
        self._environment = environment
        self._buffer = fbx.make_flat_buffer(
            max_length=2 * 10_000,
            min_length=1,
            # Unused:
            sample_batch_size=1,
        )
        self._add_to_buffer = jax.jit(self._buffer.add, donate_argnums=0)

        self.vault_name = vault_name
        self._has_initialised = False

        self._write_to_vault_every = write_to_vault_every
        self._step_count = 0

    def _pack_timestep(
        self,
        observations: Dict[str, np.ndarray],
        actions: Dict[str, np.ndarray],
        rewards: Dict[str, np.ndarray],
        terminals: Dict[str, np.ndarray],
        truncations: Dict[str, np.ndarray],
        infos: Dict[str, Any],
    ) -> Dict[str, Any]:
        packed_timestep = {
            "observations": observations,
            "actions": actions,
            "rewards": rewards,
            "terminals": terminals,
            "truncations": truncations,
            "infos": infos,
        }
        packed_timestep: Dict[str, Any] = jax.tree_map(lambda x: np.array(x), packed_timestep)
        return packed_timestep

    def reset(self) -> ResetReturn:
        observations, infos = self._environment.reset()

        self._observations = observations
        self._infos = infos

        return observations, infos

    def step(self, actions: Dict[str, np.ndarray]) -> StepReturn:
        observations, rewards, terminals, truncations, infos = self._environment.step(actions)

        packed_timestep = self._pack_timestep(
            observations=self._observations,
            actions=actions,
            rewards=rewards,
            terminals=terminals,
            truncations=truncations,
            infos=self._infos,
        )

        # Log stuff to vault/flashbax
        if not self._has_initialised:
            self._buffer_state = self._buffer.init(packed_timestep)
            self._vault = Vault(
                vault_name=self.vault_name,
                init_fbx_state=self._buffer_state,
            )
            self._has_initialised = True

        self._buffer_state = self._add_to_buffer(
            self._buffer_state,
            packed_timestep,
            # jax.tree_map(
            #     lambda x: np.expand_dims(np.expand_dims(np.array(x), axis=0), axis=0),
            #     packed_timestep,
            # ), # NOTE add time dimension and batch dimension. should we use flat buffer?
        )

        # Store new observations and infos
        self._observations = observations
        self._info = infos

        self._step_count += 1
        if self._step_count % self._write_to_vault_every == 0:
            self._vault.write(self._buffer_state)

        return observations, rewards, terminals, truncations, infos

    def __getattr__(self, name: str) -> Any:
        """Expose any other attributes of the underlying environment."""
        if hasattr(self.__class__, name):
            return self.__getattribute__(name)
        else:
            return getattr(self._environment, name)


class Dtype:
    def __init__(self, environment: BaseEnvironment, dtype: str):
        self._environment = environment
        self._dtype = dtype

    def reset(self) -> ResetReturn:
        observations = self._environment.reset()

        if isinstance(observations, tuple):
            observations, infos = observations
        else:
            infos = {}

        for agent, observation in observations.items():
            observations[agent] = observation.astype(self._dtype)

        return observations, infos

    def step(self, actions: Dict[str, np.ndarray]) -> StepReturn:
        next_observations, rewards, terminals, truncations, infos = self._environment.step(actions)

        for agent, observation in next_observations.items():
            next_observations[agent] = observation.astype(self._dtype)

        return next_observations, rewards, terminals, truncations, infos

    def __getattr__(self, name: str) -> Any:
        """Expose any other attributes of the underlying environment."""
        if hasattr(self.__class__, name):
            return self.__getattribute__(name)
        else:
            return getattr(self._environment, name)


class PadObsandActs:
    def __init__(self, environment: BaseEnvironment):
        self._environment = environment

        self._obs_dim = 0
        self._num_actions = 0

        for agent in self._environment.possible_agents:
            act_dim = self._environment.action_spaces[agent].shape[0]
            obs_dim = self._environment.observation_spaces[agent].shape[0]

            if act_dim > self._num_actions:
                self._num_actions = act_dim

            if obs_dim > self._obs_dim:
                self._obs_dim = obs_dim

    def reset(self) -> ResetReturn:
        observations = self._environment.reset()

        if isinstance(observations, tuple):
            observations, infos = observations
        else:
            infos = {}

        for agent, observation in observations.items():
            if observation.shape[0] < self._obs_dim:
                missing_dim = self._obs_dim - observation.shape[0]
                observations[agent] = np.concatenate(
                    (observation, np.zeros((missing_dim,), observation.dtype))
                )

        return observations, infos

    def step(self, actions: Dict[str, np.ndarray]) -> StepReturn:
        actions = {
            agent: action[: self._environment.action_spaces[agent].shape[0]]
            for agent, action in actions.items()
        }
        next_observations, rewards, terminals, truncations, infos = self._environment.step(actions)

        for agent, observation in next_observations.items():
            if observation.shape[0] < self._obs_dim:
                missing_dim = self._obs_dim - observation.shape[0]
                next_observations[agent] = np.concatenate(
                    (observation, np.zeros((missing_dim,), observation.dtype))
                )

        return next_observations, rewards, terminals, truncations, infos

    def __getattr__(self, name: str) -> Any:
        """Expose any other attributes of the underlying environment."""
        if hasattr(self.__class__, name):
            return self.__getattribute__(name)
        else:
            return getattr(self._environment, name)
