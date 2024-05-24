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

import dm_env
import numpy as np
from pettingzoo.mpe import simple_spread_v3

from og_marl.environments.pettingzoo_base import PettingZooBase


class MPE(PettingZooBase):

    """Environment wrapper for PettingZoo MARL environments."""

    def __init__(self):
        self._environment = simple_spread_v3.parallel_env(N=3, local_ratio=0.0, max_cycles=25, continuous_actions=True)
        self.environment_label = "pettingzoo/pistonball"

        self._agents = self._environment.possible_agents

        self._num_actions = 5
        self.max_trajectory_length = self._environment.unwrapped.max_cycles

        self._reset_next_step = True
        self._done = False

    def _create_state_representation(self, observations: Dict[str, np.ndarray]) -> np.ndarray:

        state = np.concatenate(list(observations.values()), axis=-1)

        return state

    def reset(self):
        """Resets the env."""
        # Reset the environment
        observations, _ = self._environment.reset()  # type: ignore

        # Global state
        env_state = self._create_state_representation(observations)

        # Infos
        info = {"state": env_state}

        return observations, info
    
    def step(self, actions: Dict[str, np.ndarray]):
        """Steps in env."""
        # Step the environment
        observations, rewards, terminals, truncations, _ = self._environment.step(actions)

        # Global state
        env_state = self._create_state_representation(observations)

        # Extra infos
        info = {"state": env_state}

        return observations, rewards, terminals, truncations, info