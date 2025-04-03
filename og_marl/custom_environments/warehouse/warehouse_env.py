import random

import numpy as np
import rware
import gym

from gym import ObservationWrapper, spaces
from gym.spaces import flatdim, Box
from gym.wrappers import TimeLimit

from .multiagentenv import MultiAgentEnv
from .reward_calculator import RewardCalculator


class FlattenObservation(ObservationWrapper):
    r"""Observation wrapper that flattens the observation of individual agents."""

    def __init__(self, env):
        super(FlattenObservation, self).__init__(env)

        ma_spaces = []

        for sa_obs in env.observation_space:
            flatdim = spaces.flatdim(sa_obs)
            ma_spaces += [
                spaces.Box(
                    low=-float("inf"),
                    high=float("inf"),
                    shape=(flatdim,),
                    dtype=np.float32,
                )
            ]

        self.observation_space = spaces.Tuple(tuple(ma_spaces))

    def observation(self, observation):
        return tuple(
            [
                spaces.flatten(obs_space, obs)
                for obs_space, obs in zip(self.env.observation_space, observation)
            ]
        )


class WareHouseEnv(MultiAgentEnv):
    def __init__(
            self,
            scenario="rware-small",
            n_agents=2,
            seed=None,
            difficulty="easy",
            episode_limit=500,
            render=False,
            add_agent_id=False
    ):
        self.scenario = scenario
        self.n_agents = n_agents
        if difficulty == "none":
            self.difficulty = ""
        else:
            self.difficulty = "-" + difficulty
        self.episode_limit = episode_limit
        self.render = render
        self.add_agent_id = add_agent_id
        self._env = TimeLimit(gym.make(f'rware:{self.scenario}-{self.n_agents}ag{self.difficulty}-v1'),
                              max_episode_steps=self.episode_limit)
        self._env = FlattenObservation(self._env)
        self.num_actions = [self._env.action_space[i].n for i in range(self.n_agents)]
        self.action_space = self._env.action_space

        agent_id_offset = int(self.add_agent_id) * self.n_agents

        # create shape
        self._env.reset()

        self.observation_space = [
            Box(low=float("-inf"), high=float("inf"),
                shape=(self._env.observation_space.spaces[0].shape[0] + agent_id_offset,),
                dtype=np.float32) for n in range(self.n_agents)]

        self.share_observation_space = [
            Box(low=float("-inf"), high=float("inf"),
                shape=(self.n_agents * 3 + len(self._env.request_queue) * 2,),
                dtype=np.float32) for _ in range(self.n_agents)]

        self._obs = None

        self.longest_action_space = max(self._env.action_space, key=lambda x: x.n)
        self.longest_observation_space = max(
            self._env.observation_space, key=lambda x: x.shape
        )

        # add for randomizing
        self.agent_permutation = None
        self.agent_recovery = None

    def reset(self):
        """ Returns initial observations and states"""
        obs_dicts = self._env.reset()
        self._obs = obs_dicts

        obs = []
        avail = []
        for agent_id, obs_i in enumerate(obs_dicts):
            avail_i = self.get_avail_agent_actions(agent_id)
            if self.add_agent_id:
                agent_id_onehot = np.zeros(self.n_agents)
                agent_id_onehot[agent_id] = 1
                obs_i = np.concatenate((obs_i, agent_id_onehot))
            obs.append(obs_i)
            avail.append(avail_i)

        shared_obs = self.get_shared_obs()

        return obs, shared_obs, {"avail_actions": avail}

    def step(self, actions, t=0):
        """ Returns reward, terminated, info """
        actions_int = [int(a) for a in actions]
        # add for randomizing

        o, rewards, done, infos = self._env.step(actions_int)
        obs = []
        avail = []
        for agent_id, obs_i in enumerate(o):
            avail_i = self.get_avail_agent_actions(agent_id)
            if self.add_agent_id:
                agent_id_onehot = np.zeros(self.n_agents)
                agent_id_onehot[agent_id] = 1
                obs_i = np.concatenate((obs_i, agent_id_onehot))
            obs.append(obs_i)
            avail.append(avail_i)

        shared_obs = self.get_shared_obs()

        rewards = [[RewardCalculator.calculate(self._env, _reward, _prev_obs, _obs)]
                   for _reward, _prev_obs, _obs in zip(rewards, self._obs, o)]
        rewards = np.array(rewards)[0]

        self._obs = o

        if t >= self.episode_limit - 1 or t >= self.episode_limit - 1:
            terminated = True
        else:
            terminated = False

        return obs, rewards, shared_obs, terminated, {"avail_actions": avail}

    def seed(self, seed=None):
        if seed is None:
            random.seed(1)
        else:
            random.seed(seed)

    def close(self):
        self._env.close()

    def get_obs(self):
        """Returns all agent observations in a list."""
        pass

    def get_obs_agent(self, agent_id):
        """Returns observation for agent_id."""
        pass

    def get_obs_size(self):
        """Returns the size of the observation."""
        return self.observation_space[0].shape[0]

    def get_shared_obs(self):
        agent_obs = np.array([[agent.x, agent.y, agent.dir.value] for agent in self._env.agents]).flatten()
        requested_shelf_obs = np.array([[shelf.x, shelf.y] for shelf in self._env.request_queue]).flatten()
        shared_obs = np.concatenate([agent_obs, requested_shelf_obs])
        state = [shared_obs for _ in range(self.n_agents)]
        return shared_obs

    def get_state(self):
        """Returns the global state."""
        pass

    def get_state_size(self):
        """Returns the size of the global state."""
        return self.share_observation_space[0].shape[0]

    def get_avail_actions(self, info=None):
        """Returns the available actions of all agents in a list. Only used internally"""
        pass

    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for agent_id."""
        return [1] * self.num_actions[0]

    def get_total_actions(self):
        """Returns the total number of actions an agent could ever take."""
        return self.num_actions[0]

    def save_replay(self):
        """Save a replay."""
        pass

    def render(self):
        """Save a replay."""
        pass

    def get_env_info(self):
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.n_agents,
                    "episode_limit": self.episode_limit,
                    "unit_dim":  self.get_obs_size()}
        return env_info
