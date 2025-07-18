from typing import Any, Dict

import numpy as np

from og_marl.custom_environments.multiagent_mujoco.mujoco_multi import MujocoMulti

from og_marl.wrapped_environments.base import BaseEnvironment, ResetReturn, StepReturn


class MAMuJoCo(BaseEnvironment):

    """Environment wrapper Multi-Agent MuJoCo."""

    def __init__(self, scenario: str, seed=None):
        env_args = self._get_mamujoco_args(scenario)

        self._environment = MujocoMulti(env_args=env_args)

        self.agents = [f"agent_{n}" for n in range(self._environment.n_agents)]
        self.num_actions = self._environment.n_actions
        self.observation_shape = (self._environment.get_obs_size(),)
        self.state_shape = (self._environment.get_state_size(),)

    def _get_mamujoco_args(self, scenario: str) -> Dict[str, Any]:
        env_args = {
            "agent_obsk": 0,
            "episode_limit": 1000,
        }
        if scenario.lower() == "2halfcheetah":
            env_args["scenario"] = "HalfCheetah-v2"
            env_args["agent_conf"] = "2x3"
        else:
            raise ValueError("Not a valid omar mamujoco scenario.")
        return env_args
    
    def reset(self) -> ResetReturn:
        self._environment.reset()

        observations = self._environment.get_obs()

        observations = {
            agent: observations[i].astype("float32") for i, agent in enumerate(self.agents)
        }

        info = {"state": self._environment.get_state()}

        return observations, info

    def step(self, actions: Dict[str, np.ndarray]) -> StepReturn:
        mujoco_actions = []
        for agent in self.agents:
            mujoco_actions.append(actions[agent])

        reward, done, info = self._environment.step(mujoco_actions)

        terminals = {agent: done for agent in self.agents}
        trunctations = {agent: False for agent in self.agents}

        rewards = {agent: reward for agent in self.agents}

        observations = self._environment.get_obs()

        observations = {
            agent: observations[i].astype("float32") for i, agent in enumerate(self.agents)
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