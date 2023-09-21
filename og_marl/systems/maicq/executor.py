# Copyright 2023 InstaDeep Ltd. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict
import tree
import dm_env
import tensorflow as tf
import sonnet as snt
from acme import types
from acme.tf import utils as tf2_utils

from og_marl.systems import ExecutorBase
from og_marl.utils.executor_utils import concat_agent_id_to_obs, epsilon_greedy_action_selection


class MAICQExecutor(ExecutorBase):
    def __init__(
        self,
        agents,
        variable_client,
        policy_network: snt.Module,
        add_agent_id_to_obs=False,
        checkpoint_subpath="",
        must_checkpoint=False
    ):

        super().__init__(
            agents=agents,
            variable_client=variable_client,
            add_agent_id_to_obs=add_agent_id_to_obs,
            checkpoint_subpath=checkpoint_subpath,
            must_checkpoint=must_checkpoint
        )

        # Policy network
        self._policy_network = policy_network

        # Recurrent core states for policy-network, per agent
        self._core_states = {agent: None for agent in self._agents}

    def observe_first(self, timestep: dm_env.TimeStep, extras: Dict = {}):
        # Re-initialize the recurrent core states with policy network.
        for agent in self._agents:
            self._core_states[agent] = self._policy_network.initial_state(1)

        return

    def observe(
        self,
        actions: Dict[str, types.NestedArray],
        next_timestep: dm_env.TimeStep,
        next_extras: Dict[str, types.NestedArray] = {},
    ) -> None:
        """Dont need to store observation for offline systems."""
        return

    def select_actions(self, observations):

        actions, next_core_states = self._select_actions(
            observations, self._core_states
        )

        # Update core states
        for agent in self._core_states.keys():
            self._core_states[agent] = next_core_states[agent]

        # Convert actions to numpy
        actions = tree.map_structure(tf2_utils.to_numpy_squeeze, actions)

        return actions

    @tf.function
    def _select_actions(self, observations, core_states):
        actions = {}
        new_core_states = {}
        for agent in observations.keys():
            actions[agent], new_core_states[agent] = self._select_action(
                agent,
                observations[agent].observation,
                observations[agent].legal_actions,
                core_states[agent],
            )

        return actions, new_core_states

    def _select_action(self, agent, observation, legal_actions, core_state):
        if self._add_agent_id_to_obs:
            agent_id = self._agents.index(agent)
            observation = concat_agent_id_to_obs(observation, agent_id, len(self._agents))

        # Add a dummy batch dimension
        observation = tf.expand_dims(observation, axis=0)
        legal_actions = tf.expand_dims(legal_actions, axis=0)

        # Pass observation embedding through Q-network
        logits, next_core_state = self._policy_network(observation, core_state)

        # Pass action values through action selector
        action, _ = epsilon_greedy_action_selection(
            logits=logits, legal_actions=legal_actions, epsilon=0.0
        )

        return action, next_core_state

    def get_stats(self):
        """Return extra stats to log."""
        return {}
