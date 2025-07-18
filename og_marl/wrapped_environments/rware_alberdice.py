from typing import Any, Dict

import numpy as np
from og_marl.custom_environments.warehouse.warehouse_env import WareHouseEnv

from .base import BaseEnvironment, ResetReturn, StepReturn


class RWAREAlberDICE(BaseEnvironment):
    """Note: currently only supports simple spread."""

    def __init__(self, scenario, seed=None):

        n_agents = int(scenario.split("-")[1].split("ag")[0])
        scenario = "rware" + "-" + scenario.split("-")[0]
        env = WareHouseEnv(scenario, n_agents)

        self.environment = env

        self.agents = [f"agent_{n}" for n in range(self.environment.n_agents)]
        self.num_actions = 5
        self.num_agents = len(self.agents)

        self.observation_shape = None # TODO: pytorch systems need this.

    def reset(self) -> ResetReturn:
        obs, state, info = self.environment.reset()
        self.ep_len = 0

        observations = {agent: obs[i].astype("float32") for i, agent in enumerate(self.agents)}
        legals = {agent: np.array(info["avail_actions"][i], "float32") for i, agent in enumerate(self.agents)}

        return observations, {"legals": legals}

    def step(self, actions: Dict[str, np.ndarray]) -> StepReturn:
        rware_actions = []
        for agent in self.agents:
            rware_actions.append(actions[agent])

        next_observation, reward, state, done, info = self.environment.step(rware_actions)

        terminals = {agent: done for i, agent in enumerate(self.agents)}

        rewards = {agent: reward[0] for i, agent in enumerate(self.agents)}

        observations = {
            agent: next_observation[i].astype("float32") for i, agent in enumerate(self.agents)
        }
        legals = {agent: np.array(info["avail_actions"][i], "float32") for i, agent in enumerate(self.agents)}


        info = {"legals": legals}

        self.ep_len += 1

        if self.ep_len >= 500:
            trunctations = {agent: True for i, agent in enumerate(self.agents)}
        else:
            trunctations = {agent: False for i, agent in enumerate(self.agents)}

        return observations, rewards, terminals, trunctations, info  # type: ignore

    def __getattr__(self, name: str) -> Any:
        """Expose any other attributes of the underlying environment."""
        if hasattr(self.__class__, name):
            return self.__getattribute__(name)
        else:
            return getattr(self.environment, name)
