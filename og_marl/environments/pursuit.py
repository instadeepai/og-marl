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
from typing import Dict

import numpy as np
from gymnasium.spaces import Box, Discrete
from pettingzoo.sisl import pursuit_v4

from og_marl.environments.base import BaseEnvironment, Observations, ResetReturn, StepReturn


class Pursuit(BaseEnvironment):

    """Environment wrapper for Pursuit."""

    def __init__(self) -> None:
        """Constructor for Pursuit"""
        self._environment = pursuit_v4.parallel_env()
        self.possible_agents = self._environment.possible_agents
        self._num_actions = 5
        self._obs_dim = (7, 7, 3)

        self.action_spaces = {agent: Discrete(self._num_actions) for agent in self.possible_agents}
        self.observation_spaces = {
            agent: Box(-np.inf, np.inf, (*self._obs_dim,)) for agent in self.possible_agents
        }

        self._legals = {
            agent: np.ones((self._num_actions,), "int32") for agent in self.possible_agents
        }

        self.info_spec = {"state": np.zeros(8 * 2 + 30 * 2, "float32"), "legals": self._legals}

        self.max_episode_length = 500

    def reset(self) -> ResetReturn:
        """Resets the env."""
        # Reset the environment
        observations, _ = self._environment.reset()  # type: ignore

        # Global state
        env_state = self._create_state_representation(observations)

        # Infos
        info = {"state": env_state, "legals": self._legals}

        return observations, info

    def step(self, actions: Dict[str, np.ndarray]) -> StepReturn:
        """Steps in env."""
        # Step the environment
        observations, rewards, terminals, truncations, _ = self._environment.step(actions)

        # Global state
        env_state = self._create_state_representation(observations)

        # Extra infos
        info = {"state": env_state, "legals": self._legals}

        return observations, rewards, terminals, truncations, info

    def _create_state_representation(self, observations: Observations) -> np.ndarray:
        pursuer_pos = [
            agent.current_position() for agent in self._environment.aec_env.env.env.env.pursuers
        ]
        evader_pos = [
            agent.current_position() for agent in self._environment.aec_env.env.env.env.evaders
        ]
        while len(evader_pos) < 30:
            evader_pos.append(np.array([-1, -1], dtype=np.int32))
        state = np.concatenate(tuple(pursuer_pos + evader_pos), axis=-1).astype("float32")
        state = state / 16  # normalize

        return state  # type: ignore
