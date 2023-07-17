from typing import Dict, Union
import numpy as np
from dm_env import specs
from og_marl.environments.base import OLT
from pettingzoo.sisl import pursuit_v4
from og_marl.environments.pettingzoo_base import PettingZooBase


class Pursuit(PettingZooBase):
    """Environment wrapper for Pursuit."""

    def __init__(self):
        """Constructor for parallel PZ wrapper.

        Args:
            environment (ParallelEnv): parallel PZ env.
            env_preprocess_wrappers (Optional[List], optional): Wrappers
                that preprocess envs.
                Format (env_preprocessor, dict_with_preprocessor_params).
            return_state_info: return extra state info
        """
        self._environment = pursuit_v4.parallel_env()
        self.environment_label = "pettingzoo/pursuit"
        self._agents = self._environment.possible_agents
        self._reset_next_step = True
        self._done = False
        self.num_actions = 5
        self.max_trajectory_length = 500

    def _convert_observations(self, observations, done):
        """Convert SMAC observation so it's dm_env compatible.

        Args:
            observes (Dict[str, np.ndarray]): observations per agent.
            dones (Dict[str, bool]): dones per agent.

        Returns:
            types.Observation: dm compatible observations.
        """
        olt_observations = {}
        for i, agent in enumerate(self._agents):

            obs = observations[agent].astype("float32")

            olt_observations[agent] = OLT(
                observation=obs,
                legal_actions=np.ones(self.num_actions, "float32"),
                terminal=np.asarray(done, dtype="float32"),
            )

        return olt_observations

    def _create_state_representation(self, observations):

        pursuer_pos = [
            agent.current_position()
            for agent in self._environment.aec_env.env.env.env.pursuers
        ]
        evader_pos = [
            agent.current_position()
            for agent in self._environment.aec_env.env.env.env.evaders
        ]
        while len(evader_pos) < 30:
            evader_pos.append(np.array([-1, -1], dtype=np.int32))
        state = np.concatenate(tuple(pursuer_pos + evader_pos), axis=-1).astype(
            "float32"
        )
        state = state / 16  # normalize

        return state

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
                num_values=5, dtype="int64"
            )
        return action_specs

    def extra_spec(self) -> Dict[str, specs.BoundedArray]:
        """Function returns extra spec (format) of the env.

        Returns:
            Dict[str, specs.BoundedArray]: extra spec.
        """
        state_spec = {"s_t": np.zeros(8 * 2 + 30 * 2, "float32")}
        return state_spec

    def observation_spec(self) -> Dict[str, OLT]:
        """Observation spec.

        Returns:
            types.Observation: spec for environment.
        """
        observation_specs = {}
        for agent in self._agents:

            obs = np.ones((7, 7, 3), "float32")

            observation_specs[agent] = OLT(
                observation=obs,
                legal_actions=np.ones(5, "float32"),
                terminal=np.asarray(True, "float32"),
            )

        return observation_specs
