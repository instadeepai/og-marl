"""City Learn Environment Wrapper."""
from typing import Any, Dict, List
import dm_env
from dm_env import specs
import numpy as np
from og_marl.environments.base import BaseEnvironment, parameterized_restart, OLT
from citylearn.citylearn import CityLearnEnv
from citylearn.wrappers import NormalizedObservationWrapper

class CityLearn(BaseEnvironment):
    """Environment wrapper for City Learn environment."""

    def __init__(self):
        """Constructor for CityLearn."""
        dataset_name = 'citylearn_challenge_2022_phase_all'
        self._environment = CityLearnEnv(dataset_name)
        self._environment = NormalizedObservationWrapper(self._environment)
        self._agents = [f"agent_{id}" for id in range(len(self._environment.action_space))]

        self.num_agents = len(self._agents)
        self.num_actions = self._environment.action_space[0].shape[0]
        self.action_dim = self.num_actions
        self.max_trajectory_length = None
        self.environment_label = "city_learn/2022_all_phases"

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
        state = np.concatenate(observations, axis=0)
        extras = {"s_t": state.astype("float32")}

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
        next_observations, rewards_list, done, _ = self._environment.step(
            actions
        )

        rewards = {}
        for i, agent in enumerate(self._agents):
            rewards[agent] = np.array(rewards_list[i], "float32")

        # Set done flag
        self._done = done

        state = np.concatenate(next_observations, axis=0).astype("float32")

        next_observations = self._convert_observations(next_observations, done)

        # Global state
        extras = {"s_t": state}

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

    def render(self, mode):
        return self._environment.render()

    def _preprocess_actions(self, actions):
        actions_list = []
        for agent in self._agents:
            actions_list.append(actions[agent])
        return actions_list

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
        olt_observations = {}
        for i, agent in enumerate(self._agents):
            obs = np.array(observations[i], "float32")
            olt_observations[agent] = OLT(
                observation=obs,
                legal_actions=np.zeros((1,), "float32"),
                terminal=np.asarray([done], dtype=np.float32),
            )
        return olt_observations

    def extra_spec(self) -> Dict:
        joint_obs_shape = (self._environment.observation_space[0].shape[0]*len(self._environment.observation_space),)
        return {"s_t": np.zeros(joint_obs_shape, "float32")}

    def observation_spec(self) -> Dict:
        """Observation spec.

        Returns:
            types.Observation: spec for environment.
        """
        olt_observations = {}
        for agent in self._agents:
            obs = np.zeros(self._environment.observation_space[0].shape, "float32")
            olt_observations[agent] = OLT(
                observation=obs,
                legal_actions=np.zeros((1,), "float32"),
                terminal=np.asarray([False], dtype=np.float32),
            )
        return olt_observations

    def action_spec(
        self,
    ) -> Dict:
        """Action spec.

        Returns:
            spec for actions.
        """
        action_specs = {}
        for agent in self._agents:
            action_shape = self._environment.action_space[0].shape
            action_specs[agent] = specs.BoundedArray(action_shape, "float32", -1, 1)
        return action_specs