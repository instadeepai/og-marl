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
import tensorflow as tf
import tensorflow_probability as tfp
from acme.tf import utils as tf2_utils

from og_marl.systems import ExecutorBase
from og_marl.utils.executor_utils import concat_agent_id_to_obs


class DiscreteBCExecutor(ExecutorBase):
    def __init__(
        self,
        agents,
        variable_client,
        behaviour_cloning_network,
        add_agent_id_to_obs=False,
        must_checkpoint=False,
        checkpoint_subpath=".",
    ):

        super().__init__(
            agents=agents,
            variable_client=variable_client,
            add_agent_id_to_obs=add_agent_id_to_obs,
            must_checkpoint=must_checkpoint,
            checkpoint_subpath=checkpoint_subpath,
        )

        # Store BC Network
        self._behaviour_cloning_network = behaviour_cloning_network

        # Recurrent core states for Q-network, per agent
        self._bc_core_states = {agent: None for agent in self._agents}

        # Checkpointing
        self._variables_to_checkpoint.update(
            {"bc_network": self._behaviour_cloning_network.variables}
        )
        if self._must_checkpoint:
            self.restore_checkpoint()

    def observe_first(self, timestep, extras={}):
        # Re-initialize the recurrent core states for Q-network
        for agent in self._agents:
            self._bc_core_states[agent] = self._behaviour_cloning_network.initial_state(
                1
            )

        return

    def observe(self, actions, next_timestep, next_extras={}):
        return

    def select_actions(self, observations):
        # Get agent actions
        actions, next_bc_core_states = self._select_actions(
            observations, self._bc_core_states
        )

        # Update core states
        for agent in self._bc_core_states.keys():
            self._bc_core_states[agent] = next_bc_core_states[agent]

        # Convert actions to numpy
        actions = tree.map_structure(tf2_utils.to_numpy_squeeze, actions)

        return actions

    @tf.function
    def _select_actions(self, observations, bc_core_states):
        actions = {}
        new_bc_core_states = {}
        for agent in observations.keys():
            actions[agent], new_bc_core_states[agent] = self._select_action(
                agent,
                observations[agent].observation,
                observations[agent].legal_actions,
                bc_core_states[agent],
            )

        return actions, new_bc_core_states

    def _select_action(self, agent, observation, legal_actions, bc_core_state):
        # Add agent ID to obs
        if self._add_agent_id_to_obs:
            agent_id = self._agents.index(agent)
            observation = concat_agent_id_to_obs(
                observation, agent_id, len(self._agents)
            )

        # Add a dummy batch dimension
        observation = tf.expand_dims(observation, axis=0)
        legal_actions = tf.expand_dims(legal_actions, axis=0)

        probs, next_bc_core_state = self._behaviour_cloning_network(
            observation, bc_core_state
        )

        probs = probs * tf.cast(legal_actions, "float32")  # mask illegal actions
        probs_sum = tf.reduce_sum(probs, axis=-1, keepdims=True)
        probs = probs / probs_sum  # renormalize

        action = tfp.distributions.Categorical(probs=probs).sample()

        return action, next_bc_core_state


class ContinuousBCExecutor(DiscreteBCExecutor):
    def __init__(
        self,
        agents,
        variable_client,
        behaviour_cloning_network,
        add_agent_id_to_obs=False,
        must_checkpoint=False,
        checkpoint_subpath=".",
    ):

        super().__init__(
            agents=agents,
            variable_client=variable_client,
            behaviour_cloning_network=behaviour_cloning_network,
            add_agent_id_to_obs=add_agent_id_to_obs,
            must_checkpoint=must_checkpoint,
            checkpoint_subpath=checkpoint_subpath,
        )

    def _select_action(self, agent, observation, legal_actions, bc_core_state):
        # Add agent ID to obs
        if self._add_agent_id_to_obs:
            agent_id = self._agents.index(agent)
            observation = concat_agent_id_to_obs(
                observation, agent_id, len(self._agents)
            )

        # Add a dummy batch dimension
        observation = tf.expand_dims(observation, axis=0)

        action, next_bc_core_state = self._behaviour_cloning_network(
            observation, bc_core_state
        )

        return action, next_bc_core_state
