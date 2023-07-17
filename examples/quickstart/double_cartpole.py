import gym
import dm_env
from acme import specs
from mava import types
import numpy as np
from og_marl.environments.base import BaseEnvironment
from mava.utils.wrapper_utils import parameterized_restart

"""IMPORTANT: This is a part of the OG-MARL quickstart tutorial. If you have not already,
go to `examples/quickstart/generate_dataset.py` to get started with the 
tutorial."""

class DoubleCartPole(BaseEnvironment):

    def __init__(self):

        self.num_agents = 2
        self.num_actions = 2
        self.agents = ["agent_1", "agent_2"]
        self.possible_agents = self.agents
        self._agents = self.agents

        self._env1 = gym.make("CartPole-v1")
        self._env2 = gym.make("CartPole-v1")

    def reset(self):
        # Reset the environment
        obs1 = self._env1.reset()
        obs2 = self._env2.reset()

        self._done = False
        self._step_type = dm_env.StepType.FIRST

        # We will just make a dummy global state filled with ones
        extras = {"s_t": np.array([1,1,1,1], "float32")}

        observations = self._convert_observations(obs1, obs2)

        # Set env discount to 1 for all agents
        self._discounts = {agent: np.array(1, "float32") for agent in self.agents}

        # Set reward to zero for all agents
        rewards = {agent: np.array(0, "float32") for agent in self.agents}

        return parameterized_restart(rewards, self._discounts, observations), extras

    def step(self, actions):
        """Step the env."""

        # Step the environments
        next_obs1, reward1, done1, _ = self._env1.step(
            actions["agent_1"]
        )
        next_obs2, reward2, done2, _ = self._env2.step(
            actions["agent_2"]
        )

        rewards = {"agent_1": reward1, "agent_2": reward2}

        # Set done flag
        dones = [done1, done2]
        self._done = done1 or done2 # game over if either falls down.

        # We will just make a dummy global state filled with ones
        extras = {"s_t": np.array([1,1,1,1], "float32")}

        # Convert next observations to OLT format
        next_obs = [next_obs1, next_obs2]
        next_observations = self._convert_observations(next_obs, dones)

        if self._done:
            self._step_type = dm_env.StepType.LAST

            # Discount on last timestep set to zero
            self._discounts = {agent: np.array(0, "float32") for agent in self.agents}
        else:
            self._step_type = dm_env.StepType.MID

        # Create timestep object
        timestep = dm_env.TimeStep(
            observation=next_observations,
            reward=rewards,
            discount=self._discounts,
            step_type=self._step_type,
        )

        return timestep, extras


    def _convert_observations(self, obs, dones=[]):
        olt_observations = {}
        for i, agent in enumerate(self.agents):
            
            if len(dones)==0:
                agent_obs = np.array(obs[i], "float32")
                olt_observations[agent] = types.OLT(
                    observation=agent_obs,
                    legal_actions=np.ones(2, "float32"),  # two legal actions
                    terminal=np.asarray(dones[i], dtype="float32"),
                )
            else:
                if dones[i]:
                    # If agent is done, give it zeros as obs
                    agent_obs = np.zeros((4,), "float32")
                else:
                    agent_obs = np.array(obs[i], "float32")
                olt_observations[agent] = types.OLT(
                    observation=agent_obs,
                    legal_actions=np.ones(2, "float32"),  # two legal actions
                    terminal=np.asarray(dones[i], dtype="float32"),
                )

        return olt_observations

    def extra_spec(self):
        """Function returns extra spec."""
        state_spec = {"s_t": np.zeros((4,), "float32")}

        return state_spec

    def observation_spec(self):
        """Observation spec."""
        observation_specs = {}
        for agent in self.agents:

            obs = np.zeros((4,), "float32")

            observation_specs[agent] = types.OLT(
                observation=obs,
                legal_actions=np.ones(2, "float32"),
                terminal=np.asarray(True, "float32"),
            )

        return observation_specs

    def action_spec(
        self,
    ):
        """Action spec."""
        action_specs = {}
        for agent in self.agents:
            action_specs[agent] = specs.DiscreteArray(
                num_values=2, dtype="int64"  # two actions
            )
        return action_specs