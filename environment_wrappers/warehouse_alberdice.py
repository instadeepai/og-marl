from typing import Dict, List

import numpy as np

from .base import BaseEnvironment, ResetReturn, StepReturn

from environments.alberdice_warehouse.warehouse_env import WareHouseEnv

class Warehouse(BaseEnvironment):

    """Environment wrapper for AlberDice Warehouse Env."""

    def __init__(
        self,
        scenario: str,
    ):
        if scenario == "tiny-2ag":
            scenario = "rware-tiny"
            n_agents = 2
        elif scenario == "tiny-6ag":
            scenario = "rware-tiny"
            n_agents = 6
        elif scenario == "small-2ag":
            scenario = "rware-small"
            n_agents = 2
        elif scenario == "small-6ag":
            scenario = "rware-small"
            n_agents = 6
        else:
            raise ValueError("Scenario not valid.")
        
        self._environment = WareHouseEnv(scenario=scenario, n_agents=n_agents)
        self.possible_agents = [f"agent_{n}" for n in range(self._environment.n_agents)]

        self._num_agents = self._environment.n_agents
        self._num_actions = self._environment.num_actions[0]
        self.t = 0


    def reset(self) -> ResetReturn:
        """Resets the env."""
        self.t = 0

        # Reset the environment
        obs, env_state, infos = self._environment.reset()
        self._done = False

        observations = {agent: obs[i] for i, agent in enumerate(self.possible_agents)}
        legals = {agent: infos["avail_actions"][i] for i, agent in enumerate(self.possible_agents)}

        info = {"legals": legals, "state": env_state.astype("float32")}

        return observations, info

    def step(self, actions: Dict[str, np.ndarray]) -> StepReturn:
        """Step in env."""

        # Convert dict of actions to list
        list_actions = []
        for agent in self.possible_agents:
            list_actions.append(actions[agent])


        obs, rewards, env_state, terminated, infos = self._environment.step(list_actions)

        observations = {agent: obs[i] for i, agent in enumerate(self.possible_agents)}
        legals = {agent: infos["avail_actions"][i] for i, agent in enumerate(self.possible_agents)}

        info = {"legals": legals, "state": env_state.astype("float32")}

        rewards = {agent: np.array(rewards, "float32") for i, agent in enumerate(self.possible_agents)}

        self.t += 1
        if self.t == 500:
            terminated = True

        terminals = {agent: np.array(terminated) for agent in self.possible_agents}
        truncations = {agent: np.array(False) for agent in self.possible_agents}

        info = {"legals": legals, "state": env_state}

        return observations, rewards, terminals, truncations, info
