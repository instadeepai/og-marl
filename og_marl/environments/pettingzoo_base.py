"""Base wraper for Cooperative Pettingzoo environments."""
from typing import Dict, List
import dm_env
import numpy as np
from og_marl.environments.base import parameterized_restart, BaseEnvironment, OLT, convert_space_to_spec


class PettingZooBase(BaseEnvironment):
    """Environment wrapper for MARL environments."""

    def __init__(self):
        """Constructor for parallel wrapper."""
        self._environment = None
        self._agents = None

        self.num_actions = None
        self.action_dim = None
        self.max_trajectory_length = None

        self._reset_next_step = True
        self._done = False

    def reset(self) -> dm_env.TimeStep:
        """Resets the env.

        Returns:
            dm_env.TimeStep: dm timestep.
        """
        # Reset the environment
        observations = self._environment.reset()
        self._done = False
        self._reset_next_step = False
        self._step_type = dm_env.StepType.FIRST

        # Global state
        state = self._create_state_representation(observations)
        if state is not None:
            extras = {"s_t": state}
        else:
            extras = {}

        # Convert observations to OLT format
        observations = self._convert_observations(observations, self._done)

        # Set env discount to 1 for all agents and all non-terminal timesteps
        self._discounts = {agent: np.array(1, "float32") for agent in self._agents}

        # Set reward to zero for all agents
        rewards = {agent: np.array(0, "float32") for agent in self._agents}

        return parameterized_restart(rewards, self._discounts, observations), extras

    def step(self, actions: Dict[str, np.ndarray]) -> dm_env.TimeStep:
        """Steps in env.

        Args:
            actions (Dict[str, np.ndarray]): actions per agent.

        Returns:
            dm_env.TimeStep: dm timestep
        """
        # Possibly reset the environment
        if self._reset_next_step:
            return self.reset()

        actions = self._preprocess_actions(actions)

        # Step the environment
        next_observations, pz_rewards, dones, truncated, _ = self._environment.step(
            actions
        )

        # Add zero-observations to missing agents
        next_observations = self._add_zero_obs_for_missing_agent(next_observations)

        rewards = {}
        for agent in self._agents:
            if agent in pz_rewards:
                rewards[agent] = np.array(pz_rewards[agent], "float32")
            else:
                rewards[agent] = np.array(0, "float32")

        # Set done flag
        self._done = all(dones.values()) or all(truncated.values())

        # Global state
        state = self._create_state_representation(next_observations)
        if state is not None:
            extras = {"s_t": state}
        else:
            extras = {}

        # for i in range(4):
        #     plt.imshow(state[:,:,i])
        #     plt.savefig(f"state_{i}.png")

        # Convert next observations to OLT format
        next_observations = self._convert_observations(next_observations, self._done)

        # for i, observation in enumerate(next_observations.values()):
        #     plt.imshow(observation.observation)
        #     plt.savefig(f"obs_{i}.png")

        if self._done:
            self._step_type = dm_env.StepType.LAST
            self._reset_next_step = True

            # Discount on last timestep set to zero
            self._discounts = {agent: np.array(0, "float32") for agent in self._agents}
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

    def _add_zero_obs_for_missing_agent(self, observations):
        for agent in self._agents:
            if agent not in observations:
                observations[agent] = np.zeros_like(self.observation_spec()[agent].observation)
        return observations

    def _preprocess_actions(self, actions):
        return actions

    def _convert_observations(
        self, observations: List, done: bool
    ):
        """Convert observation so it's dm_env compatible.

        Args:
            observes (Dict[str, np.ndarray]): observations per agent.
            dones (Dict[str, bool]): dones per agent.

        Returns:
            types.Observation: dm compatible observations.
        """
        raise NotImplementedError

    def _create_state_representation(self, observations):

        raise NotImplementedError

    def action_spec(
        self,
    ) -> Dict:
        """Action spec.

        Returns:
            spec for actions.
        """
        action_specs = {}
        for agent in self._agents:
            action_specs[agent] = convert_space_to_spec(
                self._environment.action_space(agent)
            )
        return action_specs