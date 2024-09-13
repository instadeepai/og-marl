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

"""Wrapper for CityLearn."""
from typing import Dict, List, Optional

import numpy as np
from citylearn.citylearn import CityLearnEnv
from citylearn.reward_function import SolarPenaltyAndComfortReward

from og_marl.environment_wrappers.base import BaseEnvironment, ResetReturn, StepReturn


class CityLearn(BaseEnvironment):

    """Environment wrapper CityLearn."""

    def __init__(
        self,
        scenario_name: str,
        seed: Optional[int] = None,
    ):
        self._environment = CityLearnEnv(
            'citylearn_challenge_2023_phase_2_local_evaluation', 
            reward_function=SolarPenaltyAndComfortReward,
            central_agent=False,
            random_seed=seed
            )

        self.agents = [f"agent_{n}" for n in range(len(self._environment.observation_space))]
        self.num_agents = len(self.agents)
        self.num_actions = self._environment.action_space[0].shape[0]

    def reset(self) -> ResetReturn:
        """Resets the env."""
        # Reset the environment
        observations = self._environment.reset()

        # Get observation from env
        observations = {agent: np.array(observations[i], "float32") for i, agent in enumerate(self.agents)}

        info = {"state": np.ones((10,), "float32")}

        return observations, info

    def step(self, actions: Dict[str, np.ndarray]) -> StepReturn:
        """Step in env."""
        # Convert dict of actions to list for SMAC
        citylearn_actions = []
        for agent in self.agents:
            citylearn_actions.append((actions[agent] / 2.0) + 0.5) # rescale

        # Step the environment
        observations, rewards, done, _= self._environment.step(citylearn_actions)

        # Get the next observations
        observations = {agent: np.array(observations[i], "float32") for i, agent in enumerate(self.agents)}

        # Convert team reward to agent-wise rewards
        rewards = {agent: np.array(rewards[i], "float32") for i, agent in enumerate(self.agents)}

        terminals = {agent: np.array(done) for agent in self.agents}
        truncations = {agent: np.array(False) for agent in self.agents}

        info = {"state": np.ones((10,), "float32")}

        return observations, rewards, terminals, truncations, info
