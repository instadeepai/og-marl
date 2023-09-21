"""Wraper for SMAC."""
import numpy as np
from og_marl.environments.base import BaseEnvironment
from smac.env import StarCraft2Env
from gymnasium.spaces import Discrete, Box

class SMACv1(BaseEnvironment):
    """Environment wrapper SMACv1."""

    def __init__(
        self,
        map_name: str, 
    ):
        self._environment = StarCraft2Env(map_name=map_name)
        self.possible_agents = [f"agent_{n}" for n in range(self._environment.n_agents)]

        self._num_agents = len(self.possible_agents)
        self._num_actions = self._environment.n_actions
        self._obs_dim = self._environment.get_obs_size()

        self.action_spaces = {agent: Discrete(self._num_actions) for agent in self.possible_agents}
        self.observation_spaces = {agent: Box(-np.inf, np.inf, (self._obs_dim,)) for agent in self.possible_agents}

        self.info_spec = {
            "state": np.zeros((self._environment.get_state_size(),), "float32"),
            "legals": {agent: np.zeros((self._num_actions,), "int64") for agent in self.possible_agents}
        }

    def reset(self):
        """Resets the env."""

        # Reset the environment
        self._environment.reset()
        self._done = False

        # Get observation from env
        observations = self._environment.get_obs()
        observations = {agent: observations[i] for i, agent in enumerate(self.possible_agents)}

        legal_actions = self._get_legal_actions()
        legals = {agent: legal_actions[i] for i, agent in enumerate(self.possible_agents)}
        
        env_state = self._environment.get_state().astype("float32")

        info = {
            "legals": legals,
            "state": env_state
        }

        return observations, info

    def step(self, actions):
        """Step in env."""

        # Convert dict of actions to list for SMAC
        smac_actions = []
        for agent in self.possible_agents:
            smac_actions.append(actions[agent])

        # Step the SMAC environment
        reward, done, _ = self._environment.step(smac_actions)

        # Get the next observations
        observations = self._environment.get_obs()
        observations = {agent: observations[i] for i, agent in enumerate(self.possible_agents)}

        legal_actions = self._get_legal_actions()
        legals = {agent: legal_actions[i] for i, agent in enumerate(self.possible_agents)}

        env_state = self._environment.get_state().astype("float32")

        # Convert team reward to agent-wise rewards
        rewards = {agent: np.array(reward, "float32") for agent in self.possible_agents}

        terminals = {agent: np.array(done) for agent in self.possible_agents}
        truncations = {agent: False for agent in self.possible_agents}

        info = {
            "legals": legals,
            "state": env_state
        }

        return observations, rewards, terminals, truncations, info

    def _get_legal_actions(self):
        """Get legal actions from the environment."""
        legal_actions = []
        for i, _ in enumerate(self.possible_agents):
            legal_actions.append(
                np.array(self._environment.get_avail_agent_actions(i), dtype="float32")
            )
        return legal_actions

    def get_stats(self):
        """Return extra stats to be logged."""
        return self._environment.get_stats()
    
