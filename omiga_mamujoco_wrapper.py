from typing import Any, Dict, Tuple

import numpy as np

from omiga_envs.multiagent_mujoco.mujoco_multi import MujocoMulti

Observations = Dict[str, np.ndarray]
NextObservations = Observations
Rewards = Dict[str, np.ndarray]
Terminals = Dict[str, np.ndarray]
Truncations = Dict[str, np.ndarray]
Info = Dict[str, Any]

ResetReturn = Tuple[Observations, Info]
StepReturn = Tuple[NextObservations, Rewards, Terminals, Truncations, Info]

class OmigaMAMuJoCo:

    def __init__(self, scenario: str):
        env_args = self._get_mamujoco_args(scenario)
        self._environment = MujocoMulti(env_args=env_args)

        self.possible_agents = [f"agent_{n}" for n in range(self._environment.n_agents)]
        self._num_actions = self._environment.n_actions
        self.max_episode_length = 1000

    def _get_mamujoco_args(self, scenario: str) -> Dict[str, Any]:
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