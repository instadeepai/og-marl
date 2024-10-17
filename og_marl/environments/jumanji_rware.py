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

import copy
import jax
import jax.numpy as jnp
import jumanji
import numpy as np
from jumanji.environments.routing.robot_warehouse.generator import RandomGenerator

from og_marl.environments.base import BaseEnvironment, ResetReturn, StepReturn

task_configs = {
    "tiny-4ag": {
        "column_height": 8,
        "shelf_rows": 1,
        "shelf_columns": 3,
        "num_agents": 4,
        "sensor_range": 1,
        "request_queue_size": 4,
    },
    "tiny-2ag": {
        "column_height": 8,
        "shelf_rows": 1,
        "shelf_columns": 3,
        "num_agents": 2,
        "sensor_range": 1,
        "request_queue_size": 2,
    },
    "small-4ag-easy": {
        "column_height": 8,
        "shelf_rows": 2,
        "shelf_columns": 3,
        "num_agents": 4,
        "sensor_range": 1,
        "request_queue_size": 8,
    },
}


class JumanjiRware(BaseEnvironment):

    """Environment wrapper for Jumanji environments."""

    def __init__(self, scenario_name: str = "tiny-4ag", seed: int = 0) -> None:
        """Constructor."""
        self._environment = jumanji.make(
            "RobotWarehouse-v0",
            generator=RandomGenerator(**task_configs[scenario_name]),
        )
        self._num_agents = self._environment.num_agents
        self._num_actions = int(self._environment.action_spec().num_values[0])
        self.possible_agents = [f"agent_{i}" for i in range(self._num_agents)]
        self._state = ...  # Jumanji environment state

        self.info_spec: Dict[str, Any] = {}  # TODO add global state spec

        self._key = jax.random.PRNGKey(seed)

        self._env_step = jax.jit(self._environment.step, donate_argnums=0)
        self._env_reset = jax.jit(self._environment.reset)

    def reset(self) -> ResetReturn:
        """Resets the env."""
        # Reset the environment
        self._key, sub_key = jax.random.split(self._key)
        self._state, timestep = self._env_reset(sub_key)

        observations = {
            agent: np.asarray(timestep.observation.agents_view[i], dtype=np.float32)
            for i, agent in enumerate(self.possible_agents)
        }
        legals = {
            agent: np.asarray(timestep.observation.action_mask[i], dtype=np.int32)
            for i, agent in enumerate(self.possible_agents)
        }

        # Infos
        info = {"legals": legals}

        return observations, info

    def step(self, actions: Dict[str, np.ndarray]) -> StepReturn:
        """Steps in env."""
        actions = jnp.array(list(actions.values()))  # .squeeze(-1)
        # Step the environment
        self._state, timestep = self._env_step(self._state, actions)

        observations = {
            agent: np.asarray(timestep.observation.agents_view[i], dtype=np.float32)
            for i, agent in enumerate(self.possible_agents)
        }
        legals = {
            agent: np.asarray(timestep.observation.action_mask[i], dtype=np.int32)
            for i, agent in enumerate(self.possible_agents)
        }
        rewards = {agent: np.asarray(timestep.reward) for agent in self.possible_agents}
        terminals = {agent: np.asarray(timestep.last()) for agent in self.possible_agents}
        truncations = {agent: np.asarray(False) for agent in self.possible_agents}

        # # Global state # TODO
        # env_state = self._create_state_representation(observations)

        # Extra infos
        info = {"legals": legals}

        return observations, rewards, terminals, truncations, info

    def render(self) -> Any:
        frame = self._state

        return copy.deepcopy(frame)
