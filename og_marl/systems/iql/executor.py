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

import tree
import numpy as np
import tensorflow as tf
from acme.tf import utils as tf2_utils

from og_marl.systems import ExecutorBase
from og_marl.utils.executor_utils import (
    epsilon_greedy_action_selection, 
    concat_agent_id_to_obs
)

class IQLExecutor(ExecutorBase):
    def __init__(
        self,
        agents,
        variable_client,
        q_network,
        adder=None,
        eps_start=1.0,
        eps_dec=1e-5,
        eps_min=0.05,
        add_agent_id_to_obs=False,
        checkpoint_subpath="",
        must_checkpoint=False,
    ):

        super().__init__(
            agents=agents,
            variable_client=variable_client,
            add_agent_id_to_obs=add_agent_id_to_obs,
            checkpoint_subpath=checkpoint_subpath,
            must_checkpoint=must_checkpoint,
        )

        # Store optional adder
        self._adder = adder

        # Store Q-network
        self._q_network = q_network

        self._variables_to_checkpoint.update({"q_network": self._q_network.variables})

        # Epsilon-greedy exploration
        self._eps = eps_start
        self._eps_dec = eps_dec
        self._eps_min = eps_min

        # Recurrent core states for Q-network, per agent
        self._core_states = {agent: None for agent in self._agents}

        if self._must_checkpoint:
            self.restore_checkpoint()

    def observe_first(self, timestep, extras={}):
        # Re-initialize the recurrent core states for Q-network
        for agent in self._agents:
            self._core_states[agent] = self._q_network.initial_state(1)

        if self._adder is not None:

            extras.update({"zero_padding_mask": np.array(1)})

            # Adder first timestep to adder
            self._adder.add_first(timestep, extras)

    def observe(self, actions, next_timestep, next_extras={}):

        if self._adder is not None:

            # Add core states to extras
            next_extras.update({"zero_padding_mask": np.array(1)})

            # Add timestep to adder
            self._adder.add(actions, next_timestep, next_extras)

    def select_actions(self, observations):
        # Get agent actions
        epsilon = self._decay_epsilon()
        epsilon = tf.convert_to_tensor(epsilon, dtype="float32")
        actions, next_core_states = self._select_actions(
            observations, self._core_states, epsilon
        )

        # Update core states
        for agent in self._core_states.keys():
            self._core_states[agent] = next_core_states[agent]

        # Convert actions to numpy
        actions = tree.map_structure(tf2_utils.to_numpy_squeeze, actions)

        return actions

    @tf.function
    def _select_actions(self, observations, core_states, eps):
        actions = {}
        new_core_states = {}
        for agent in observations.keys():
            actions[agent], new_core_states[agent] = self._select_action(
                agent,
                observations[agent].observation,
                observations[agent].legal_actions,
                core_states[agent],
                eps,
            )

        return actions, new_core_states

    def _select_action(self, agent, observation, legal_actions, core_state, eps):
        # Add agent ID to embed
        if self._add_agent_id_to_obs:
            agent_id = self._agents.index(agent)
            observation = concat_agent_id_to_obs(
                observation, agent_id, len(self._agents)
            )

        # Add a dummy batch dimension
        observation = tf.expand_dims(observation, axis=0)
        legal_actions = tf.expand_dims(legal_actions, axis=0)

        # Pass observation through Q-network
        action_values, next_core_state = self._q_network(observation, core_state)

        # Pass action values through action selector
        action, _ = epsilon_greedy_action_selection(
            action_values=action_values, legal_actions=legal_actions, epsilon=eps
        )

        return action, next_core_state

    def _decay_epsilon(self):
        if self._eps_dec != 0:
            self._eps = self._eps - self._eps_dec
        self._eps = max(self._eps, self._eps_min)
        return self._eps

    def get_stats(self):
        """Return extra stats to log."""
        return {"Epsilon": self._eps}
