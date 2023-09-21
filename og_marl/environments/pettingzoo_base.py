"""Base wraper for Cooperative Pettingzoo environments."""
import numpy as np
from gymnasium.spaces import Discrete, Box
from og_marl.environments.base import BaseEnvironment

class PettingZooBase(BaseEnvironment):
    """Environment wrapper for PettingZoo environments."""

    def __init__(self):
        """Constructor."""
        self._environment = None
        self.possible_agents = None

        self._num_actions = None

        self.action_spaces = {agent: None for agent in self.possible_agents}
        self.observation_spaces = {agent: None for agent in self.possible_agents}

        self.info_spec = {}


    def reset(self):
        """Resets the env."""

        # Reset the environment
        observations = self._environment.reset()

        # Global state
        env_state = self._create_state_representation(observations)

        # Infos
        info = {"state": env_state}

        # Convert observations to OLT format
        observations = self._convert_observations(observations)

        return observations, info


    def step(self, actions):
        """Steps in env."""

        # Step the environment
        observations, rewards, terminals, truncations, _ = self._environment.step(
            actions
        )

        # Global state
        env_state = self._create_state_representation(observations)
        
        # Extra infos
        info = {"state": env_state}

        return observations, rewards, terminals, truncations, info


    def _add_zero_obs_for_missing_agent(self, observations):
        for agent in self._agents:
            if agent not in observations:
                observations[agent] = np.zeros(self.observation_spaces[agent].shape, self.observation_spaces[agent].dtype)
        return observations


    def _convert_observations(
        self, observations
    ):
        """Convert observations"""
        raise NotImplementedError

    def _create_state_representation(self, observations):
        """Create global state representation from agent observations."""
        raise NotImplementedError