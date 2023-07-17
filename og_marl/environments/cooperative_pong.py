"""Wraper for Cooperative Pong."""
from typing import Dict, List, Union

import dm_env
from dm_env import specs
import numpy as np
from pettingzoo.butterfly import cooperative_pong_v5
import supersuit

from og_marl.environments.base import parameterized_restart, BaseEnvironment, OLT
from og_marl.environments.pettingzoo_base import PettingZooBase
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    

class CooperativePong(PettingZooBase):
    """Environment wrapper for PettingZoo MARL environments."""

    def __init__(
        self,
    ):
        """Constructor for parallel PZ wrapper.

        Args:
            environment (ParallelEnv): parallel PZ env.
            env_preprocess_wrappers (Optional[List], optional): Wrappers
                that preprocess envs.
                Format (env_preprocessor, dict_with_preprocessor_params).
            return_state_info: return extra state info
        """
        self._environment = cooperative_pong_v5.parallel_env(render_mode="rgb_array")
        # Wrap environment with supersuit pre-process wrappers
        self._environment = supersuit.color_reduction_v0(self._environment, mode="R")
        self._environment = supersuit.resize_v0(
            self._environment, x_size=145, y_size=84
        )
        self._environment = supersuit.dtype_v0(self._environment, dtype="float32")
        self._environment = supersuit.normalize_obs_v0(self._environment)

        self._agents = self._environment.possible_agents
        self._reset_next_step = True
        self._done = False
        self.environment_label = "pettingzoo/coop_pong"

    def _create_state_representation(self, observations):
        if self._step_type == dm_env.StepType.FIRST:
            self._state_history = np.zeros((84, 145, 4), "float32")

        state = np.expand_dims(observations["paddle_0"][:, :], axis=-1)

        # framestacking
        self._state_history = np.concatenate(
            (state, self._state_history[:, :, :3]), axis=-1
        )

        return self._state_history

    def _add_zero_obs_for_missing_agent(self, observations):
        for agent in self._agents:
            if agent not in observations:
                observations[agent] = np.zeros((84, 145), "float32")
        return observations

    def _convert_observations(
        self, observations: List, done: bool
    ):
        """Convert SMAC observation so it's dm_env compatible.

        Args:
            observes (Dict[str, np.ndarray]): observations per agent.
            dones (Dict[str, bool]): dones per agent.

        Returns:
            types.Observation: dm compatible observations.
        """
        olt_observations = {}
        for i, agent in enumerate(self._agents):

            if agent == "paddle_0":
                agent_obs = observations[agent][:, :110]  # hide the other agent
            else:
                agent_obs = observations[agent][:, 35:]  # hide the other agent

            agent_obs = np.expand_dims(agent_obs, axis=-1)
            olt_observations[agent] = OLT(
                observation=agent_obs,
                legal_actions=np.ones(3, "float32"),  # three actions in pong, all legal
                terminal=np.asarray(done, dtype="float32"),
            )

        return olt_observations

    def extra_spec(self) -> Dict[str, specs.BoundedArray]:
        """Function returns extra spec (format) of the env.

        Returns:
            Dict[str, specs.BoundedArray]: extra spec.
        """
        state_spec = {"s_t": np.zeros((84, 145, 4), "float32")}  # four stacked frames

        return state_spec

    def observation_spec(self) -> Dict:
        """Observation spec.

        Returns:
            types.Observation: spec for environment.
        """
        observation_specs = {}
        for agent in self._agents:

            obs = np.zeros((84, 110, 1), "float32")

            observation_specs[agent] = OLT(
                observation=obs,
                legal_actions=np.ones(3, "float32"),
                terminal=np.asarray(True, "float32"),
            )

        return observation_specs

    def action_spec(
        self,
    ) -> Dict[str, Union[specs.DiscreteArray, specs.BoundedArray]]:
        """Action spec.

        Returns:
            spec for actions.
        """
        action_specs = {}
        for agent in self._agents:
            action_specs[agent] = specs.DiscreteArray(
                num_values=3, dtype="int64"  # three actions
            )
        return action_specs