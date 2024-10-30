from typing import Any, Dict

import numpy as np
from og_marl.custom_environments.multiagent_particle_envs.multiagent.environment import (
    MultiAgentEnv,
)
from og_marl.custom_environments.multiagent_particle_envs.multiagent.simple_spread import Scenario

from .base import BaseEnvironment, ResetReturn, StepReturn


class MPEOMAR(BaseEnvironment):
    """Note: currently only supports simple spread."""

    def __init__(self, scenario, seed=None):
        # load scenario from script
        scenario = Scenario()  # TODO: make variable
        # create world
        world = scenario.make_world()

        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)

        self.environment = env

        self.agents = [f"agent_{n}" for n in range(3)]
        self.num_actions = 2
        self.num_agents = len(self.agents)

        self.max_episode_length = 25

        self.t = 0

    def reset(self) -> ResetReturn:
        obs = self.environment.reset()

        observations = {agent: obs[i].astype("float32") for i, agent in enumerate(self.agents)}

        self.t = 0

        return observations, {}

    def step(self, actions: Dict[str, np.ndarray]) -> StepReturn:
        mpe_actions = []
        for agent in self.agents:
            mpe_actions.append(actions[agent])

        next_observation, reward, done, info = self.environment.step(mpe_actions)

        terminals = {agent: done[i] for i, agent in enumerate(self.agents)}
        trunctations = {agent: False for i, agent in enumerate(self.agents)}

        rewards = {agent: reward[i] for i, agent in enumerate(self.agents)}

        observations = {
            agent: next_observation[i].astype("float32") for i, agent in enumerate(self.agents)
        }

        if self.t == 25:
            terminals = {agent: True for i, agent in enumerate(self.agents)}

        self.t += 1

        return observations, rewards, terminals, trunctations, info  # type: ignore

    def __getattr__(self, name: str) -> Any:
        """Expose any other attributes of the underlying environment."""
        if hasattr(self.__class__, name):
            return self.__getattribute__(name)
        else:
            return getattr(self.environment, name)
