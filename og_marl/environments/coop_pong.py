# python3
# Copyright 2021 InstaDeep Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Wrapper for Cooperative Pettingzoo environments."""
from typing import Any, List, Dict

import numpy as np
from pettingzoo.butterfly import cooperative_pong_v5
import supersuit

from og_marl.environments.base import BaseEnvironment
from og_marl.environments.base import Observations, ResetReturn, StepReturn


class CooperativePong(BaseEnvironment):
    """Environment wrapper PettingZoo Cooperative Pong."""

    def __init__(
        self,
    ) -> None:
        """Constructor."""
        self._environment = cooperative_pong_v5.parallel_env(render_mode="rgb_array")
        # Wrap environment with supersuit pre-process wrappers
        self._environment = supersuit.color_reduction_v0(self._environment, mode="R")
        self._environment = supersuit.resize_v1(self._environment, x_size=145, y_size=84)
        self._environment = supersuit.dtype_v0(self._environment, dtype="float32")
        self._environment = supersuit.normalize_obs_v0(self._environment)

        self._agents = self._environment.possible_agents

        self._num_actions = 3
        self._done = False
        self.max_episode_length = 900
        self._legals = {agent: np.ones((self._num_actions,), "float32") for agent in self._agents}

    def reset(self) -> ResetReturn:
        """Resets the env."""
        # Reset the environment
        observations, _ = self._environment.reset()  # type: ignore

        # Global state
        env_state = self._create_state_representation(observations, first=True)

        # Convert observations
        observations = self._convert_observations(observations)

        # Infos
        info = {"state": env_state, "legals": self._legals}

        return observations, info

    def step(self, actions: Dict[str, np.ndarray]) -> StepReturn:
        """Steps in env."""
        # Step the environment
        observations, rewards, terminals, truncations, _ = self._environment.step(actions)

        # Global state
        env_state = self._create_state_representation(observations)

        # Convert observations
        observations = self._convert_observations(observations)

        # Extra infos
        info = {"state": env_state, "legals": self._legals}

        return observations, rewards, terminals, truncations, info

    def _create_state_representation(self, observations: Observations, first: bool = False) -> Any:
        if first:
            self._state_history = np.zeros((84, 145, 4), "float32")

        state = np.expand_dims(observations["paddle_0"][:, :], axis=-1)

        # framestacking
        self._state_history = np.concatenate((state, self._state_history[:, :, :3]), axis=-1)

        return self._state_history

    def _convert_observations(self, observations: List) -> Observations:
        """Make observations partial."""
        processed_observations = {}
        for agent in self._agents:
            if agent == "paddle_0":
                agent_obs = observations[agent][:, :110]  # hide the other agent
            else:
                agent_obs = observations[agent][:, 35:]  # hide the other agent

            agent_obs = np.expand_dims(agent_obs, axis=-1)
            processed_observations[agent] = agent_obs

        return processed_observations

    def __getattr__(self, name: str) -> Any:
        """Expose any other attributes of the underlying environment.

        Args:
            name (str): attribute.

        Returns:
            Any: return attribute from env or underlying env.
        """
        if hasattr(self.__class__, name):
            return self.__getattribute__(name)
        else:
            return getattr(self._environment, name)
