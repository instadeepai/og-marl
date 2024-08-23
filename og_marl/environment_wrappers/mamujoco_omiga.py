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
from typing import Any, Dict, Optional

import numpy as np
from og_marl.custom_environments.multiagent_mujoco.mujoco_multi import MujocoMulti

from og_marl.environment_wrappers.base import BaseEnvironment, ResetReturn, StepReturn


def get_mamujoco_args(scenario: str) -> Dict[str, Any]:
    env_args = {
        "agent_obsk": None,
        "episode_limit": 1000,
    }
    if scenario.lower() == "3hopper":
        env_args["scenario"] = "Hopper-v2"
        env_args["agent_conf"] = "3x1"
    elif scenario.lower() == "2ant":
        env_args["scenario"] = "Ant-v2"
        env_args["agent_conf"] = "2x4"
    elif scenario.lower() == "6halfcheetah":
        env_args["scenario"] = "HalfCheetah-v2"
        env_args["agent_conf"] = "6x1"
    else:
        raise ValueError("Not a valid omiga mamujoco scenario")
    return env_args


class MAMuJoCoOMIGA(BaseEnvironment):

    """Environment wrapper Multi-Agent MuJoCo."""

    def __init__(self, scenario: str, seed: Optional[int] = 42):
        env_args = get_mamujoco_args(scenario)
        self._environment = MujocoMulti(env_args=env_args, seed=seed)

        self.agents = [f"agent_{n}" for n in range(self._environment.n_agents)]
        self.num_actions = self._environment.n_actions

    def reset(self) -> ResetReturn:
        self._environment.reset()

        observations = self._environment.get_obs()

        observations = {
            agent: observations[i].astype("float32") for i, agent in enumerate(self.agents)
        }

        observations = self.add_agent_id_and_normalise(observations)

        info = {"state": self._environment.get_state()}

        return observations, info

    def step(self, actions: Dict[str, np.ndarray]) -> StepReturn:
        mujoco_actions = []
        for agent in self.agents:
            mujoco_actions.append(actions[agent])

        reward, done, info = self._environment.step(mujoco_actions)

        terminals = {agent: np.array(done) for agent in self.agents}
        truncations = {agent: np.array(False) for agent in self.agents}

        rewards = {agent: reward for agent in self.agents}

        observations = self._environment.get_obs()

        observations = {
            agent: observations[i].astype("float32") for i, agent in enumerate(self.agents)
        }
        observations = self.add_agent_id_and_normalise(observations)

        info["state"] = self._environment.get_state()

        return observations, rewards, terminals, truncations, info  # type: ignore
    
    def add_agent_id_and_normalise(self, observations):
        for i, agent in enumerate(self.agents):
            one_hot = np.zeros((len(self.agents),), "float32")
            one_hot[i] = 1
            agent_obs = observations[agent].astype("float32")
            agent_obs = np.concatenate([agent_obs, one_hot], axis=-1)
            agent_obs = agent_obs - np.mean(agent_obs) / np.std(agent_obs)
            observations[agent] = agent_obs
        return observations


    def __getattr__(self, name: str) -> Any:
        """Expose any other attributes of the underlying environment."""
        if hasattr(self.__class__, name):
            return self.__getattribute__(name)
        else:
            return getattr(self._environment, name)
