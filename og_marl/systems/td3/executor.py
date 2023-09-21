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
import tensorflow_probability as tfp

from og_marl.systems import ExecutorBase
from og_marl.utils.executor_utils import concat_agent_id_to_obs


class DeterministicPolicyExecutor(ExecutorBase):
    def __init__(
        self,
        agents,
        variable_client,
        policy_network,
        adder=None,
        add_agent_id_to_obs=False,
        gaussian_noise_network=None,
        exploration_timesteps=0,
        checkpoint_subpath="",
        must_checkpoint=False,
    ):
        # Call super init
        super().__init__(
            agents=agents,
            variable_client=variable_client,
            add_agent_id_to_obs=add_agent_id_to_obs,
            checkpoint_subpath=checkpoint_subpath,
            must_checkpoint=must_checkpoint,
        )

        # Store optional adder
        self._adder = adder

        # Store networks
        self._policy_network = policy_network
        self._gaussian_noise_network = gaussian_noise_network

        # Recurrent core states for policy network, per agent
        self._core_states = {agent: None for agent in agents}

        # Exploration
        self._exploration_timesteps = exploration_timesteps

        # Counter
        self._timestep = 0

        # Checkpointing
        self._variables_to_checkpoint.update(
            {"policy_network": self._policy_network.variables}
        )
        if self._must_checkpoint:
            self.restore_checkpoint()

    def observe_first(self, timestep, extras={}):
        # Re-initialize the recurrent core states for Q-network
        for agent in self._agents:
            self._core_states[agent] = self._policy_network.initial_state(1)

        if self._adder is not None:

            # Adder first timestep to adder
            extras.update({"zero_padding_mask": np.array(1)})

            self._adder.add_first(timestep, extras)

    def observe(self, actions, next_timestep, next_extras={}):

        if self._adder is not None:
            next_extras.update({"zero_padding_mask": np.array(1)})

            self._adder.add(actions, next_timestep, next_extras)

    def select_actions(self, observations):
        # Get agent actions
        actions, next_core_states = self._select_actions(
            observations, self._core_states
        )

        # Update core states
        for agent in self._core_states.keys():
            self._core_states[agent] = next_core_states[agent]

        # TODO: either do this or _select_action, not both
        if (
            self._timestep < self._exploration_timesteps
        ):
            num_actions = list(actions.values())[0].shape[-1] # find another way to get num actions
            for agent in self._agents:
                actions[agent] = tf.expand_dims(
                    tfp.distributions.Uniform(
                        low=[-1.0] * num_actions,
                        high=[1.0] * num_actions,
                        validate_args=False,
                        allow_nan_stats=True,
                        name="Uniform",
                    ).sample(),
                    axis=0,
                )

        self._timestep += 1

        # Convert actions to numpy
        actions = tree.map_structure(tf2_utils.to_numpy_squeeze, actions)

        return actions

    @tf.function
    def _select_actions(self, observations, core_states):
        actions = {}
        next_core_states = {}
        for agent in observations.keys():
            action, next_core_states[agent] = self._select_action(
                agent,
                observations[agent].observation,
                observations[agent].legal_actions,
                core_states[agent],
            )

            if self._gaussian_noise_network is not None:
                action = self._gaussian_noise_network(action)

            actions[agent] = action

        return actions, next_core_states

    def _select_action(self, agent, observation, legal_actions, core_state):
        # Add agent ID to embed
        if self._add_agent_id_to_obs:
            agent_id = self._agents.index(agent)
            observation = concat_agent_id_to_obs(
                observation, agent_id, len(self._agents)
            )

        # Add a dummy batch dimension
        observation = tf.expand_dims(observation, axis=0)
        legal_actions = tf.expand_dims(legal_actions, axis=0)

        # Pass observation embedding through policy network
        action, next_core_state = self._policy_network(observation, core_state)

        return action, next_core_state

    def get_stats(self):
        """Return extra stats to log."""
        return {}