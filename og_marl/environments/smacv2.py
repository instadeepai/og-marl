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

"""Wrapper for SMACv2."""
from typing import Any, Dict, List

import numpy as np
from gymnasium.spaces import Box, Discrete
from smacv2.env.starcraft2.wrapper import StarCraftCapabilityEnvWrapper

from og_marl.environments.base import BaseEnvironment, ResetReturn, StepReturn

DISTRIBUTION_CONFIGS = {
    "terran_5_vs_5": {
        "n_units": 5,
        "n_enemies": 5,
        "team_gen": {
            "dist_type": "weighted_teams",
            "unit_types": ["marine", "marauder", "medivac"],
            "exception_unit_types": ["baneling"],
            "weights": [0.45, 0.45, 0.1],
            "observe": True,
        },
        "start_positions": {
            "dist_type": "surrounded_and_reflect",
            "p": 0.5,
            "n_enemies": 5,
            "map_x": 32,
            "map_y": 32,
        },
    },
    "zerg_5_vs_5": {
        "n_units": 5,
        "n_enemies": 5,
        "team_gen": {
            "dist_type": "weighted_teams",
            "unit_types": ["zergling", "baneling", "hydralisk"],
            "exception_unit_types": ["baneling"],
            "weights": [0.45, 0.1, 0.45],
            "observe": True,
        },
        "start_positions": {
            "dist_type": "surrounded_and_reflect",
            "p": 0.5,
            "n_enemies": 5,
            "map_x": 32,
            "map_y": 32,
        },
    },
    "terran_10_vs_10": {
        "n_units": 10,
        "n_enemies": 10,
        "team_gen": {
            "dist_type": "weighted_teams",
            "unit_types": ["marine", "marauder", "medivac"],
            "exception_unit_types": ["baneling"],
            "weights": [0.45, 0.45, 0.1],
            "observe": True,
        },
        "start_positions": {
            "dist_type": "surrounded_and_reflect",
            "p": 0.5,
            "n_enemies": 5,
            "map_x": 32,
            "map_y": 32,
        },
    },
}

MAP_NAMES = {
    "terran_5_vs_5": "10gen_terran",
    "zerg_5_vs_5": "10gen_zerg",
    "terran_10_vs_10": "10gen_terran",
}


class SMACv2(BaseEnvironment):

    """Environment wrapper SMAC."""

    def __init__(self, scenario: str):
        self._environment = StarCraftCapabilityEnvWrapper(
            capability_config=DISTRIBUTION_CONFIGS[scenario],
            map_name=MAP_NAMES[scenario],
            debug=False,
            conic_fov=False,
            obs_own_pos=True,
            use_unit_ranges=True,
            min_attack_range=2,
        )

        self.possible_agents = [f"agent_{n}" for n in range(self._environment.n_agents)]

        self._num_agents = len(self.possible_agents)
        self._num_actions = self._environment.n_actions
        self._obs_dim = self._environment.get_obs_size()

        self.action_spaces = {agent: Discrete(self._num_actions) for agent in self.possible_agents}
        self.observation_spaces = {
            agent: Box(-np.inf, np.inf, (self._obs_dim,)) for agent in self.possible_agents
        }

        self.info_spec = {
            "state": np.zeros((self._environment.get_state_size(),), "float32"),
            "legals": {
                agent: np.zeros((self._num_actions,), "int64") for agent in self.possible_agents
            },
        }

        self.max_episode_length = self._environment.episode_limit

    def reset(self) -> ResetReturn:
        """Resets the env."""
        # Reset the environment
        self._environment.reset()
        self._done = False

        # Get observation from env
        observations = self._environment.get_obs()
        observations = {agent: observations[i] for i, agent in enumerate(self.possible_agents)}

        legal_actions = self._get_legal_actions()
        legals = {agent: legal_actions[i] for i, agent in enumerate(self.possible_agents)}

        env_state = self._environment.get_state().astype("float32")

        info = {"legals": legals, "state": env_state}

        return observations, info

    def step(self, actions: Dict[str, np.ndarray]) -> StepReturn:
        """Step in env."""
        # Convert dict of actions to list for SMAC
        smac_actions = []
        for agent in self.possible_agents:
            smac_actions.append(actions[agent])

        # Step the SMAC environment
        reward, done, _ = self._environment.step(smac_actions)

        # Get the next observations
        observations = self._environment.get_obs()
        observations = {agent: observations[i] for i, agent in enumerate(self.possible_agents)}

        legal_actions = self._get_legal_actions()
        legals = {agent: legal_actions[i] for i, agent in enumerate(self.possible_agents)}

        env_state = self._environment.get_state().astype("float32")

        # Convert team reward to agent-wise rewards
        rewards = {agent: np.array(reward, "float32") for agent in self.possible_agents}

        terminals = {agent: np.array(done) for agent in self.possible_agents}
        truncations = {agent: np.array(False) for agent in self.possible_agents}

        info = {"legals": legals, "state": env_state}

        return observations, rewards, terminals, truncations, info

    def _get_legal_actions(self) -> List[np.ndarray]:
        """Get legal actions from the environment."""
        legal_actions = []
        for i, _ in enumerate(self.possible_agents):
            legal_actions.append(
                np.array(self._environment.get_avail_agent_actions(i), dtype="float32")
            )
        return legal_actions

    def get_stats(self) -> Any:
        """Return extra stats to be logged."""
        return self._environment.get_stats()
