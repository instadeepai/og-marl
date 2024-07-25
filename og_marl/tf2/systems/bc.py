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

"""Implementation of Behaviour Cloning"""
from typing import Any, Dict, Optional, Tuple

import numpy as np
import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp
import tree
from chex import Numeric

from og_marl.environments.base import BaseEnvironment
from og_marl.loggers import BaseLogger
from og_marl.replay_buffers import Experience
from og_marl.tf2.systems.base import BaseMARLSystem
from og_marl.tf2.utils import (
    batch_concat_agent_id_to_obs,
    concat_agent_id_to_obs,
    expand_batch_and_agent_dim_of_time_major_sequence,
    merge_batch_and_agent_dim_of_time_major_sequence,
    switch_two_leading_dims,
    unroll_rnn,
)
from og_marl.tf2.networks import IdentityNetwork


class DicreteActionBehaviourCloning(BaseMARLSystem):
    """Behaviour cloning."""

    def __init__(
        self,
        environment: BaseEnvironment,
        logger: BaseLogger,
        linear_layer_dim: int = 100,
        recurrent_layer_dim: int = 100,
        discount: float = 0.99,
        learning_rate: float = 1e-3,
        add_agent_id_to_obs: bool = True,
        observation_embedding_network: Optional[snt.Module] = None,
    ):
        super().__init__(
            environment, logger, discount=discount, add_agent_id_to_obs=add_agent_id_to_obs
        )

        # Policy network
        self._policy_network = snt.DeepRNN(
            [
                snt.Linear(linear_layer_dim),
                tf.nn.relu,
                snt.GRU(recurrent_layer_dim),
                tf.nn.relu,
                snt.Linear(self._environment._num_actions),
            ]
        )  # shared network for all agents
        if observation_embedding_network is None:
            observation_embedding_network = IdentityNetwork()
        self._policy_embedding_network = observation_embedding_network

        self._optimizer = snt.optimizers.RMSProp(learning_rate=learning_rate)

        # Reset the recurrent neural network
        self._rnn_states = {
            agent: self._policy_network.initial_state(1)
            for agent in self._environment.possible_agents
        }

    def reset(self) -> None:
        """Called at the start of a new episode."""
        # Reset the recurrent neural network
        self._rnn_states = {
            agent: self._policy_network.initial_state(1)
            for agent in self._environment.possible_agents
        }

        return

    def select_actions(
        self,
        observations: Dict[str, np.ndarray],
        legal_actions: Optional[Dict[str, np.ndarray]] = None,
        explore: bool = True,
    ) -> Dict[str, np.ndarray]:
        observations, legal_actions = tree.map_structure(
            tf.convert_to_tensor, (observations, legal_actions)
        )

        actions, next_rnn_states = self._tf_select_actions(
            observations, self._rnn_states, legal_actions
        )
        self._rnn_states = next_rnn_states
        return tree.map_structure(  # type: ignore
            lambda x: x[0].numpy(), actions
        )  # convert to numpy and squeeze batch dim

    @tf.function(jit_compile=True)
    def _tf_select_actions(
        self,
        observations: Dict[str, tf.Tensor],
        rnn_states: Dict[str, tf.Tensor],
        legal_actions: Optional[Dict[str, tf.Tensor]] = None,
    ) -> Tuple[Dict[str, tf.Tensor], Dict[str, tf.Tensor]]:
        actions = {}
        next_rnn_states = {}
        for i, agent in enumerate(self._environment.possible_agents):
            agent_observation = observations[agent]
            if self._add_agent_id_to_obs:
                agent_observation = concat_agent_id_to_obs(
                    agent_observation, i, len(self._environment.possible_agents)
                )
            agent_observation = tf.cast(
                tf.expand_dims(agent_observation, axis=0), "float32"
            )  # add batch dimension
            embedding = self._policy_embedding_network(agent_observation)
            logits, next_rnn_states[agent] = self._policy_network(embedding, rnn_states[agent])

            probs = tf.nn.softmax(logits)

            if legal_actions is not None:
                agent_legals = tf.cast(tf.expand_dims(legal_actions[agent], axis=0), "float32")
                probs = (probs * agent_legals) / tf.reduce_sum(
                    probs * agent_legals
                )  # mask and renorm

            action = tfp.distributions.Categorical(probs=probs).sample(1)

            # Store agent action
            actions[agent] = action[0]

        return actions, next_rnn_states

    def train_step(self, experience: Experience) -> Dict[str, Numeric]:
        logs = self._tf_train_step(experience)
        return logs  # type: ignore

    @tf.function(jit_compile=True)
    def _tf_train_step(self, experience: Dict[str, Any]) -> Dict[str, Numeric]:
        # Unpack the relevant quantities
        observations = tf.cast(experience["observations"], "float32")
        actions = experience["actions"]
        truncations = tf.cast(experience["truncations"], "float32")  # (B,T,N)
        terminals = tf.cast(experience["terminals"], "float32")  # (B,T,N)

        # When to reset the RNN hidden state
        resets = tf.maximum(terminals, truncations)  # equivalent to logical 'or'

        # Get batch size, max sequence length, num agents and num actions
        B, T, N, A = experience["infos"]["legals"].shape

        # Maybe add agent ids to observation
        if self._add_agent_id_to_obs:
            observations = batch_concat_agent_id_to_obs(observations)

        # Make time-major
        observations = switch_two_leading_dims(observations)
        resets = switch_two_leading_dims(resets)
        actions = switch_two_leading_dims(actions)

        # Merge batch_dim and agent_dim
        observations = merge_batch_and_agent_dim_of_time_major_sequence(observations)
        resets = merge_batch_and_agent_dim_of_time_major_sequence(resets)

        with tf.GradientTape() as tape:
            embeddings = self._policy_embedding_network(observations)
            probs_out = unroll_rnn(
                self._policy_network,
                embeddings,
                resets,
            )
            probs_out = expand_batch_and_agent_dim_of_time_major_sequence(probs_out, B, N)

            # Behaviour cloning loss
            one_hot_actions = tf.one_hot(actions, depth=probs_out.shape[-1], axis=-1)
            bc_loss = tf.keras.metrics.categorical_crossentropy(
                one_hot_actions, probs_out, from_logits=True
            )
            bc_loss = tf.reduce_mean(bc_loss)

        # Apply gradients to policy
        variables = (
            *self._policy_network.trainable_variables,
            *self._policy_embedding_network.trainable_variables,
        )  # Get trainable variables

        gradients = tape.gradient(bc_loss, variables)  # Compute gradients.
        self._optimizer.apply(gradients, variables)

        logs = {"Policy Loss": bc_loss}

        return logs


# class ContinuousMaBcTrainer(DiscreteMaBcTrainer):
#     def __init__(
#         self,
#         agents,
#         dataset,
#         logger,
#         behaviour_cloning_network,
#         optimizer,
#         max_gradient_norm=20.0,
#         add_agent_id_to_obs=False,
#     ):
#         super().__init__(
#             agents=agents,
#             dataset=dataset,
#             optimizer=optimizer,
#             behaviour_cloning_network=behaviour_cloning_network,
#             logger=logger,
#             max_gradient_norm=max_gradient_norm,
#             add_agent_id_to_obs=add_agent_id_to_obs,
#         )

#     @tf.function
#     def _train(self, sample, trainer_step):
#         batch = sample_batch_agents(self._agents, sample)

#         # Get the relevant quantities
#         observations = batch["observations"]
#         actions = batch["actions"]
#         legal_actions = batch["legals"]
#         mask = tf.cast(batch["mask"], "float32")  # shape=(B,T)

#         # Get dims
#         B, T, N = legal_actions.shape[:3]

#         # Maybe add agent ids to observation
#         if self._add_agent_id_to_obs:
#             observations = batch_concat_agent_id_to_obs(observations)

#         # Make time-major
#         observations = switch_two_leading_dims(observations)
#         actions = switch_two_leading_dims(actions)
#         mask = switch_two_leading_dims(mask)

#         # Do forward passes through the networks and calculate the losses
#         with tf.GradientTape() as tape:
#             # Unroll network
#             a_out, _ = snt.static_unroll(
#                 self._behaviour_cloning_network,
#                 merge_batch_and_agent_dim_of_time_major_sequence(observations),
#                 self._behaviour_cloning_network.initial_state(B*N)
#             )
#             a_out = expand_batch_and_agent_dim_of_time_major_sequence(a_out, B, N)

#             # BC loss
#             bc_loss = (a_out - actions) ** 2

#             # Masking zero-padded elements
#             mask = tf.concat([mask] * N, axis=-1)
#             bc_loss = tf.reduce_sum(tf.expand_dims(mask, axis=-1) * bc_loss) / tf.reduce_sum(mask)

#         # Get trainable variables
#         variables = (*self._behaviour_cloning_network.trainable_variables,)

#         # Compute gradients.
#         gradients = tape.gradient(bc_loss, variables)

#         # Maybe clip gradients.
#         gradients = tf.clip_by_global_norm(gradients, self._max_gradient_norm)[0]

#         self._optimizer.apply(gradients, variables)

#         del tape

#         logs = {
#             "Trainer Steps": trainer_step,
#             "BC Loss": bc_loss,
#         }

#         return logs
