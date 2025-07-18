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

"""Wrapper for SMAC."""
from typing import Dict, List, Optional

import numpy as np
from smac.env import StarCraft2Env

from og_marl.wrapped_environments.base import BaseEnvironment, ResetReturn, StepReturn


class SMACv1(BaseEnvironment):

    """Environment wrapper SMACv1."""

    def __init__(
        self,
        map_name: str,
        seed: Optional[int] = None,
    ):
        self._environment = StarCraft2Env(map_name=map_name, seed=seed)

        self.agents = [f"agent_{n}" for n in range(self._environment.n_agents)]
        self.num_agents = len(self.agents)
        self.num_actions = self._environment.n_actions
        self.observation_shape = (self._environment.get_obs_size(),)

    def reset(self) -> ResetReturn:
        """Resets the env."""
        # Reset the environment
        self._environment.reset()

        # Get observation from env
        observations = self._environment.get_obs()
        observations = {agent: observations[i] for i, agent in enumerate(self.agents)}

        legal_actions = self._get_legal_actions()
        legals = {agent: legal_actions[i] for i, agent in enumerate(self.agents)}

        env_state = self._environment.get_state().astype("float32")

        info = {"legals": legals, "state": env_state}

        return observations, info

    def step(self, actions: Dict[str, np.ndarray]) -> StepReturn:
        """Step in env."""
        # Convert dict of actions to list for SMAC
        smac_actions = []
        for agent in self.agents:
            smac_actions.append(actions[agent])

        # Step the SMAC environment
        reward, done, _ = self._environment.step(smac_actions)

        # Get the next observations
        observations = self._environment.get_obs()
        observations = {agent: observations[i] for i, agent in enumerate(self.agents)}

        legal_actions = self._get_legal_actions()
        legals = {agent: legal_actions[i] for i, agent in enumerate(self.agents)}

        env_state = self._environment.get_state().astype("float32")

        # Convert team reward to agent-wise rewards
        rewards = {agent: np.array(reward, "float32") for agent in self.agents}

        terminals = {agent: np.array(done) for agent in self.agents}
        truncations = {agent: np.array(False) for agent in self.agents}

        info = {"legals": legals, "state": env_state}

        return observations, rewards, terminals, truncations, info

    def _get_legal_actions(self) -> List[np.ndarray]:
        """Get legal actions from the environment."""
        legal_actions = []
        for i, _ in enumerate(self.agents):
            legal_actions.append(
                np.array(self._environment.get_avail_agent_actions(i), dtype="float32")
            )
        return legal_actions
