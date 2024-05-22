from typing import Any, Dict

import numpy as np
from gymnasium.spaces import Box
from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios

from og_marl.environments.base import BaseEnvironment, ResetReturn, StepReturn


class MPE(BaseEnvironment):

    """Environment wrapper Multi-Agent MuJoCo."""

    def __init__(self):
        # load scenario from script
        scenario = scenarios.load("simple_spread" + ".py").Scenario()
        # create world
        world = scenario.make_world()

        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)

        self._environment = env

        self.possible_agents = [f"agent_{n}" for n in range(3)]
        self._num_actions = 2

        self.max_episode_length = 25

        self.t = 0

    def reset(self) -> ResetReturn:
        obs = self._environment.reset()

        observations = {agent: obs[i].astype("float32") for i, agent in enumerate(self.possible_agents)}

        self.t = 0

        return observations, {}

    def step(self, actions: Dict[str, np.ndarray]) -> StepReturn:
        mpe_actions = []
        for agent in self.possible_agents:
            mpe_actions.append(actions[agent])

        next_observation, reward, done, info = self._environment.step(mpe_actions)

        terminals = {agent: done[i] for i, agent in enumerate(self.possible_agents)}
        trunctations = {agent: False for i, agent in enumerate(self.possible_agents)}

        rewards = {agent: reward[i] for i, agent in enumerate(self.possible_agents)}

        observations = {
            agent: next_observation[i].astype("float32") for i, agent in enumerate(self.possible_agents)
        }

        if self.t == 25:
            terminals = {agent: True for i, agent in enumerate(self.possible_agents)}

        self.t += 1

        return observations, rewards, terminals, trunctations, info  # type: ignore

    def __getattr__(self, name: str) -> Any:
        """Expose any other attributes of the underlying environment."""
        if hasattr(self.__class__, name):
            return self.__getattribute__(name)
        else:
            return getattr(self._environment, name)
