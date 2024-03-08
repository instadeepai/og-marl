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
"""Base wrapper for Jumanji environments."""
from typing import Any, Dict

import jax
import numpy as np
from jaxmarl import make
from jaxmarl.environments.smax import map_name_to_scenario
from og_marl.environments.base import BaseEnvironment, ResetReturn, StepReturn


class SMAX(BaseEnvironment):

    """Environment wrapper for Jumanji environments."""

    def __init__(self, scenario_name: str = "3m", seed: int = 0) -> None:
        """Constructor."""
        scenario = map_name_to_scenario(scenario_name)

        self._environment = make(
            "HeuristicEnemySMAX",
            enemy_shoots=True,
            scenario=scenario,
            use_self_play_reward=False,
            walls_cause_death=True,
            see_enemy_actions=False,
        )

        self._num_agents = self._environment.num_agents
        self.possible_agents = self._environment.agents
        self._num_actions = int(self._environment.action_spaces[self.possible_agents[0]].n)

        self._state = ...  # Jaxmarl environment state

        self.info_spec: Dict[str, Any] = {}  # TODO add global state spec

        self._key = jax.random.PRNGKey(seed)

        self._env_step = jax.jit(self._environment.step)

    def reset(self) -> ResetReturn:
        """Resets the env."""
        # Reset the environment
        self._key, sub_key = jax.random.split(self._key)
        obs, self._state = self._environment.reset(sub_key)

        observations = {
            agent: np.asarray(obs[agent], dtype=np.float32) for agent in self.possible_agents
        }
        legals = {
            agent: np.array(legal, "int64")
            for agent, legal in self._environment.get_avail_actions(self._state).items()
        }
        state = np.asarray(obs["world_state"], "float32")

        # Infos
        info = {"legals": legals, "state": state}

        return observations, info

    def step(self, actions: Dict[str, np.ndarray]) -> StepReturn:
        """Steps in env."""
        self._key, sub_key = jax.random.split(self._key)

        # Step the environment
        obs, self._state, reward, done, infos = self._environment.step(
            sub_key, self._state, actions
        )

        observations = {
            agent: np.asarray(obs[agent], dtype=np.float32) for agent in self.possible_agents
        }
        legals = {
            agent: np.array(legal, "int64")
            for agent, legal in self._environment.get_avail_actions(self._state).items()
        }
        state = np.asarray(obs["world_state"], "float32")

        # Infos
        info = {"legals": legals, "state": state}

        rewards = {agent: reward[agent] for agent in self.possible_agents}
        terminals = {agent: done["__all__"] for agent in self.possible_agents}
        truncations = {agent: False for agent in self.possible_agents}

        return observations, rewards, terminals, truncations, info
