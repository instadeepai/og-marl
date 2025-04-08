from typing import Any, Dict

import numpy as np
from og_marl.custom_environments.multiagent_particle_envs.multiagent.environment import (
    MultiAgentEnv,
)
from og_marl.custom_environments.multiagent_particle_envs.multiagent.scenarios.simple_spread import Scenario
import og_marl.custom_environments.multiagent_particle_envs.multiagent.scenarios as scenarios
# import torch

from .base import BaseEnvironment, ResetReturn, StepReturn


class MPEOMAR(BaseEnvironment):
    """Note: currently only supports simple spread."""

    def __init__(self, scenario, seed=None):
        self.scenario_id = scenario

        # load scenario from script
        loaded_scenario = scenarios.load(scenario + ".py").Scenario()

        # create world
        world = loaded_scenario.make_world()

        env = MultiAgentEnv(world, loaded_scenario.reset_world, loaded_scenario.reward, loaded_scenario.observation)

        self.environment = env

        self.agents = [f"agent_{n}" for n in range(env.n)]
        self.num_actions = 2
        self.num_agents = len(self.agents)

        self.max_episode_length = 25

        self.t = 0

    def load_pretrained_preys(self, filename):
        save_dict = torch.load(filename, map_location=torch.device('cpu'))

        if self.env_id in ['simple_tag', 'simple_world']:
            prey_params = save_dict['agent_params'][self.num_predators:]

        for i, params in zip(range(self.num_preys), prey_params):
            self.preys[i].load_params_without_optims(params)

        for p in self.preys:
            p.policy.eval()
            p.target_policy.eval()

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
