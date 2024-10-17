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
# limitations under the License.
"""Wrapper for Pistonball."""
from typing import Dict

import cv2
import dm_env
import numpy as np
import supersuit
from dm_env import specs
from pettingzoo.butterfly import pistonball_v6

from og_marl.environments.base import OLT  # type: ignore
from og_marl.environments.pettingzoo_base import PettingZooBase


class Pistonball(PettingZooBase):

    """Environment wrapper for PettingZoo MARL environments."""

    def __init__(self, n_pistons: int = 15):
        self._environment = pistonball_v6.parallel_env(
            n_pistons=n_pistons, continuous=True, render_mode="rgb_array"
        )
        self.environment_label = "pettingzoo/pistonball"
        # Wrap environment with supersuit pre-process wrappers
        self._environment = supersuit.color_reduction_v0(self._environment, mode="R")
        self._environment = supersuit.resize_v0(self._environment, x_size=22, y_size=80)
        self._environment = supersuit.dtype_v0(self._environment, dtype="float32")
        self._environment = supersuit.normalize_obs_v0(self._environment, env_min=0, env_max=1)

        self._agents = self._environment.possible_agents

        self.num_actions = 1
        self.max_trajectory_length = self._environment.unwrapped.max_cycles

        self._reset_next_step = True
        self._done = False

    def _create_state_representation(self, observations: Dict[str, np.ndarray]) -> np.ndarray:
        if self._step_type == dm_env.StepType.FIRST:
            self._state_history = np.zeros((56, 88, 4), "float32")

        state = self._environment.state()
        state = state.astype("float32")
        state = state[:, :, 0]
        state = cv2.resize(state, dsize=(88, 56))
        state = state / np.amax(state)  # normalize

        # framestacking
        state = np.expand_dims(state, axis=-1)
        self._state_history = np.concatenate((state, self._state_history[:, :, :3]), axis=-1)

        return self._state_history  # type: ignore

    def _convert_observations(
        self, observations: Dict[str, np.ndarray], done: bool
    ) -> Dict[str, np.ndarray]:
        olt_observations = {}
        for _, agent in enumerate(self._agents):
            agent_obs = np.expand_dims(observations[agent][50:, :], axis=-1)
            legal_actions = np.ones(self.num_actions, "float32")  # three actions, all legal

            olt_observations[agent] = OLT(
                observation=agent_obs,
                legal_actions=legal_actions,
                terminal=np.asarray(done, dtype="float32"),
            )

        return olt_observations  # type: ignore

    def extra_spec(self) -> Dict[str, specs.BoundedArray]:
        """Function returns extra spec (format) of the env.

        Returns:
        -------
            Dict[str, specs.BoundedArray]: extra spec.

        """
        state_spec = {"s_t": np.zeros((56, 88, 4), "float32")}  # four stacked frames

        return state_spec

    def action_spec(self) -> Dict[str, specs.BoundedArray]:
        action_spec = {}
        for agent in self._agents:
            spec = specs.BoundedArray(
                shape=(1,), dtype="float32", minimum=-1.0, maximum=1.0, name="act"
            )

            action_spec[agent] = spec

        return action_spec

    def observation_spec(self) -> Dict[str, OLT]:
        """Observation spec.

        Returns:
        -------
            types.Observation: spec for environment.

        """
        observation_specs = {}
        for agent in self._agents:
            obs = np.zeros((30, 22, 1), "float32")

            observation_specs[agent] = OLT(
                observation=obs,
                legal_actions=np.ones(self.num_actions, "float32"),  # three legal actions
                terminal=np.asarray(True, "float32"),
            )

        return observation_specs
