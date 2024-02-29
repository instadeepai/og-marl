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

import numpy as np
from gymnasium.spaces import Box
from multiagent_mujoco.mujoco_multi import MujocoMulti

from og_marl.environments.base import BaseEnvironment, ResetReturn, StepReturn


def get_mamujoco_args(scenario: str) -> Dict[str, Any]:
    env_args = {
        "agent_obsk": 1,
        "episode_limit": 1000,
        "global_categories": "qvel,qpos",
    }
    if scenario.lower() == "4ant":
        env_args["scenario"] = "Ant-v2"
        env_args["agent_conf"] = "4x2"
    elif scenario.lower() == "2ant":
        env_args["scenario"] = "Ant-v2"
        env_args["agent_conf"] = "2x4"
    elif scenario.lower() == "2halfcheetah":
        env_args["scenario"] = "HalfCheetah-v2"
        env_args["agent_conf"] = "2x3"
    else:
        raise ValueError("Not a valid mamujoco scenario")
    return env_args


class MAMuJoCo(BaseEnvironment):

    """Environment wrapper Multi-Agent MuJoCo."""

    def __init__(self, scenario: str):
        env_args = get_mamujoco_args(scenario)
        self._environment = MujocoMulti(env_args=env_args)

        self.possible_agents = [f"agent_{n}" for n in range(self._environment.n_agents)]
        self._num_actions = self._environment.n_actions

        self.observation_spaces = {
            agent: Box(float(), 1, (self._environment.obs_size,), "float32")
            for i, agent in enumerate(self.possible_agents)
        }

        self.action_spaces = {
            agent: self._environment.action_space[i] for i, agent in enumerate(self.possible_agents)
        }

        self.info_spec = {
            "state": self._environment.get_state(),
            "legals": {
                agent: np.zeros(self.action_spaces[agent].shape, "int64")
                for agent in self.possible_agents
            },
        }

        self.max_episode_length = 1000

    def reset(self) -> ResetReturn:
        self._environment.reset()

        observations = self._environment.get_obs()

        observations = {
            agent: observations[i].astype("float32") for i, agent in enumerate(self.possible_agents)
        }

        info = {"state": self._environment.get_state()}

        return observations, info

    def step(self, actions: Dict[str, np.ndarray]) -> StepReturn:
        mujoco_actions = []
        for agent in self.possible_agents:
            mujoco_actions.append(actions[agent])

        reward, done, info = self._environment.step(mujoco_actions)

        terminals = {agent: done for agent in self.possible_agents}
        trunctations = {agent: False for agent in self.possible_agents}

        rewards = {agent: reward for agent in self.possible_agents}

        observations = self._environment.get_obs()

        observations = {
            agent: observations[i].astype("float32") for i, agent in enumerate(self.possible_agents)
        }

        info["state"] = self._environment.get_state()

        return observations, rewards, terminals, trunctations, info  # type: ignore

    def __getattr__(self, name: str) -> Any:
        """Expose any other attributes of the underlying environment."""
        if hasattr(self.__class__, name):
            return self.__getattribute__(name)
        else:
            return getattr(self._environment, name)
