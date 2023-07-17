"""MAPDN Environment Wrapper."""
from typing import Any, Dict, List, Union
import dm_env
import numpy as np
from dm_env import specs
from og_marl.environments.base import parameterized_restart, BaseEnvironment, OLT
from var_voltage_control.voltage_control_env import VoltageControl


class VoltageControlEnv(BaseEnvironment):
    """Environment wrapper for MAPDN environment."""

    def __init__(self):
        """Constructor for VoltageControl."""
        self._environment = VoltageControl()
        self.environment_label = "voltage_control/case33_3min_final"
        self._agents = [f"agent_{id}" for id in range(self._environment.get_num_of_agents())]

        self.num_actions = self.environment.get_total_actions()
        self.action_dim = self.num_actions
        self.max_trajectory_length = None

        self._reset_next_step = True
        self._done = False

    def reset(self) -> dm_env.TimeStep:
        """Resets the env.

        Returns:
            dm_env.TimeStep: dm timestep.
        """
        # Reset the environment
        observations, state = self._environment.reset()
        self._done = False
        self._reset_next_step = False
        self._step_type = dm_env.StepType.FIRST

        # Global state
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
        reward, done, _ = self._environment.step(
            actions
        )

        rewards = {}
        for agent in self._agents:
                rewards[agent] = np.array(reward, "float32")

        # Set done flag
        self._done = done

        next_observations = self._environment.get_obs()

        next_observations = self._convert_observations(next_observations, done)

        state = self._environment.get_state().astype("float32")

        # Global state
        if state is not None:
            extras = {"s_t": state}
        else:
            extras = {}

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

    def _preprocess_actions(self, actions):
        concat_action = []
        for agent in self._agents:
            concat_action.append(actions[agent])
        concat_action = np.concatenate(concat_action)
        return concat_action

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

    def extra_spec(self) -> Dict[str, specs.BoundedArray]:
        return {"s_t": np.zeros((144,), "float32")}

    def observation_spec(self) -> Dict:
        """Observation spec.

        Returns:
            types.Observation: spec for environment.
        """
        olt_observations = {}
        for agent in self._agents:
            obs = np.zeros((50,), "float32")
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
            action_specs[agent] = specs.BoundedArray((1,), "float32", -1, 1)
        return action_specs