from typing import Any, Dict

import numpy as np
from multiagent_mujoco.mujoco_multi import MujocoMulti

from environment_wrappers.base import BaseEnvironment, ResetReturn, StepReturn


class MAMuJoCo(BaseEnvironment):

    """Environment wrapper Multi-Agent MuJoCo."""

    def __init__(self, scenario: str):
        env_args = self._get_mamujoco_args(scenario)
        self._environment = MujocoMulti(env_args=env_args)

        self.possible_agents = [f"agent_{n}" for n in range(self._environment.n_agents)]
        self._num_actions = self._environment.n_actions
        self.max_episode_length = 1000

    def _get_mamujoco_args(self, scenario: str) -> Dict[str, Any]:
        env_args = {
            "agent_obsk": 1,
            "episode_limit": 1000,
            "global_categories": "qvel,qpos",
        }
        if scenario.lower() == "2halfcheetah":
            env_args["scenario"] = "HalfCheetah-v2"
            env_args["agent_conf"] = "2x3"
        elif scenario.lower() == "2ant":
            env_args["scenario"] = "Ant-v2"
            env_args["agent_conf"] = "2x4"
        elif scenario.lower() == "4ant":
            env_args["scenario"] = "Ant-v2"
            env_args["agent_conf"] = "4x2"
        else:
            raise ValueError("Not a valid mamujoco scenario")
        return env_args

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

        info = {}
        info["state"] = self._environment.get_state()

        return observations, rewards, terminals, trunctations, info  # type: ignore

    def __getattr__(self, name: str) -> Any:
        """Expose any other attributes of the underlying environment."""
        if hasattr(self.__class__, name):
            return self.__getattribute__(name)
        else:
            return getattr(self._environment, name)
