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

"""Implementation of MAICQ"""
from typing import Any, Dict, Tuple, Optional

import numpy as np
import sonnet as snt
import tensorflow as tf
import tree
from chex import Numeric
from tensorflow import Tensor

from og_marl.environments.base import BaseEnvironment
from og_marl.loggers import BaseLogger
from og_marl.tf2.systems.qmix import QMIXSystem
from og_marl.tf2.utils import (
    batch_concat_agent_id_to_obs,
    concat_agent_id_to_obs,
    expand_batch_and_agent_dim_of_time_major_sequence,
    gather,
    merge_batch_and_agent_dim_of_time_major_sequence,
    switch_two_leading_dims,
    unroll_rnn,
)
from og_marl.tf2.networks import IdentityNetwork


class MAICQSystem(QMIXSystem):

    """MAICQ System"""

    def __init__(
        self,
        environment: BaseEnvironment,
        logger: BaseLogger,
        icq_advantages_beta: float = 0.1,  # from MAICQ code
        icq_target_q_taken_beta: int = 1000,  # from MAICQ code
        linear_layer_dim: int = 64,
        recurrent_layer_dim: int = 64,
        mixer_embed_dim: int = 32,
        mixer_hyper_dim: int = 64,
        discount: float = 0.99,
        target_update_period: int = 200,
        learning_rate: float = 3e-4,
        add_agent_id_to_obs: bool = False,
        observation_embedding_network: Optional[snt.Module] = None,
        state_embedding_network: Optional[snt.Module] = None,
    ):
        super().__init__(
            environment,
            logger,
            linear_layer_dim=linear_layer_dim,
            recurrent_layer_dim=recurrent_layer_dim,
            add_agent_id_to_obs=add_agent_id_to_obs,
            discount=discount,
            target_update_period=target_update_period,
            learning_rate=learning_rate,
            mixer_embed_dim=mixer_embed_dim,
            mixer_hyper_dim=mixer_hyper_dim,
            observation_embedding_network=observation_embedding_network,
            state_embedding_network=state_embedding_network,
        )

        # ICQ hyper-params
        self._icq_advantages_beta = icq_advantages_beta
        self._icq_target_q_taken_beta = icq_target_q_taken_beta

        # Embedding Networks
        if observation_embedding_network is None:
            observation_embedding_network = IdentityNetwork()
        self._policy_embedding_network = observation_embedding_network

        # Policy Network
        self._policy_network = snt.DeepRNN(
            [
                snt.Linear(linear_layer_dim),
                tf.nn.relu,
                snt.GRU(recurrent_layer_dim),
                tf.nn.relu,
                snt.Linear(self._environment._num_actions),
                tf.nn.softmax,
            ]
        )

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
        legal_actions: Dict[str, np.ndarray],
        explore: bool = False,
    ) -> Dict[str, np.ndarray]:
        observations = tree.map_structure(tf.convert_to_tensor, observations)
        actions, next_rnn_states = self._tf_select_actions(
            observations, legal_actions, self._rnn_states
        )
        self._rnn_states = next_rnn_states
        return tree.map_structure(  # type: ignore
            lambda x: x.numpy(), actions
        )  # convert to numpy and squeeze batch dim

    @tf.function()
    def _tf_select_actions(
        self,
        observations: Dict[str, Tensor],
        legal_actions: Dict[str, Tensor],
        rnn_states: Dict[str, Tensor],
    ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        actions = {}
        next_rnn_states = {}
        for i, agent in enumerate(self._environment.possible_agents):
            agent_observation = observations[agent]
            agent_observation = concat_agent_id_to_obs(
                agent_observation, i, len(self._environment.possible_agents)
            )
            agent_observation = tf.expand_dims(agent_observation, axis=0)  # add batch dimension
            embedding = self._policy_embedding_network(agent_observation)
            probs, next_rnn_states[agent] = self._policy_network(embedding, rnn_states[agent])

            agent_legal_actions = legal_actions[agent]
            masked_probs = tf.where(
                tf.equal(agent_legal_actions, 1),
                probs[0],
                -99999999,
            )

            # Max Q-value over legal actions
            actions[agent] = tf.argmax(masked_probs)

        return actions, next_rnn_states

    @tf.function(jit_compile=True)
    def _tf_train_step(
        self,
        train_step_ctr: int,
        experience: Dict[str, Any],
    ) -> Dict[str, Numeric]:
        # Unpack the batch
        observations = experience["observations"]  # (B,T,N,O)
        actions = experience["actions"]  # (B,T,N)
        env_states = experience["infos"]["state"]  # (B,T,S)
        rewards = experience["rewards"]  # (B,T,N)
        truncations = experience["truncations"]  # (B,T,N)
        terminals = experience["terminals"]  # (B,T,N)
        legal_actions = experience["infos"]["legals"]  # (B,T,N,A)

        # When to reset the RNN hidden state
        resets = tf.maximum(terminals, truncations)  # equivalent to logical 'or'

        # Get dims
        B, T, N, A = legal_actions.shape

        # Maybe add agent ids to observation
        if self._add_agent_id_to_obs:
            observations = batch_concat_agent_id_to_obs(observations)

        # Make time-major
        observations = switch_two_leading_dims(observations)
        resets = switch_two_leading_dims(resets)

        # Merge batch_dim and agent_dim
        observations = merge_batch_and_agent_dim_of_time_major_sequence(observations)
        resets = merge_batch_and_agent_dim_of_time_major_sequence(resets)

        # Unroll target network
        target_embeddings = self._target_q_embedding_network(observations)
        target_qs_out = unroll_rnn(self._target_q_network, target_embeddings, resets)

        # Expand batch and agent_dim
        target_qs_out = expand_batch_and_agent_dim_of_time_major_sequence(target_qs_out, B, N)

        # Make batch-major again
        target_q_vals = switch_two_leading_dims(target_qs_out)

        with tf.GradientTape(persistent=True) as tape:
            # Unroll online network
            q_embeddings = self._q_embedding_network(observations)
            qs_out = unroll_rnn(self._q_network, q_embeddings, resets)

            # Expand batch and agent_dim
            qs_out = expand_batch_and_agent_dim_of_time_major_sequence(qs_out, B, N)

            # Make batch-major again
            q_vals = switch_two_leading_dims(qs_out)

            # Unroll the policy
            policy_embeddings = self._policy_embedding_network(observations)
            probs_out = unroll_rnn(self._policy_network, policy_embeddings, resets)

            # Expand batch and agent_dim
            probs_out = expand_batch_and_agent_dim_of_time_major_sequence(probs_out, B, N)

            # Make batch-major again
            probs_out = switch_two_leading_dims(probs_out)

            # Mask illegal actions
            probs_out = probs_out * tf.cast(legal_actions, "float32")
            probs_sum = (
                tf.reduce_sum(probs_out, axis=-1, keepdims=True) + 1e-10
            )  # avoid div by zero
            probs_out = probs_out / probs_sum

            action_values = gather(q_vals, actions)
            baseline = tf.reduce_sum(probs_out * q_vals, axis=-1)
            advantages = action_values - baseline
            advantages = tf.nn.softmax(advantages / self._icq_advantages_beta, axis=0)
            advantages = tf.stop_gradient(advantages)

            pi_taken = gather(probs_out, actions, keepdims=False)
            log_pi_taken = tf.math.log(pi_taken)

            env_state_embeddings = self._state_embedding_network(env_states)
            target_env_state_embeddings = self._target_state_embedding_network(env_states)
            coe = self._mixer.k(env_state_embeddings)

            coma_loss = -tf.reduce_mean(coe * (len(advantages) * advantages * log_pi_taken))

            # Critic learning
            q_taken = gather(q_vals, actions, axis=-1)
            target_q_taken = gather(target_q_vals, actions, axis=-1)

            # Mixing critics
            q_taken = self._mixer(q_taken, env_state_embeddings)
            target_q_taken = self._target_mixer(target_q_taken, target_env_state_embeddings)

            advantage_Q = tf.nn.softmax(target_q_taken / self._icq_target_q_taken_beta, axis=0)
            target_q_taken = len(advantage_Q) * advantage_Q * target_q_taken

            # Compute targets
            targets = (
                rewards[:, :-1] + (1 - terminals[:, :-1]) * self._discount * target_q_taken[:, 1:]
            )
            targets = tf.stop_gradient(targets)

            # TD error
            td_error = targets - q_taken[:, :-1]
            q_loss = 0.5 * tf.square(td_error)

            # Masking
            q_loss = tf.reduce_mean(q_loss)

            # Add losses together
            loss = q_loss + coma_loss

        # Apply gradients to policy
        variables = (
            *self._policy_network.trainable_variables,
            *self._q_network.trainable_variables,
            *self._mixer.trainable_variables,
            *self._q_embedding_network.trainable_variables,
            *self._policy_embedding_network.trainable_variables,
        )  # Get trainable variables

        gradients = tape.gradient(loss, variables)  # Compute gradients.

        self._optimizer.apply(gradients, variables)  # One optimizer for whole system

        # Online variables
        online_variables = (
            *self._q_network.variables,
            *self._mixer.variables,
            *self._q_embedding_network.variables,
            *self._state_embedding_network.variables,
        )

        # Get target variables
        target_variables = (
            *self._target_q_network.variables,
            *self._target_mixer.variables,
            *self._target_q_embedding_network.variables,
            *self._target_state_embedding_network.variables,
        )

        # Maybe update target network
        self._update_target_network(train_step_ctr, online_variables, target_variables)

        return {
            "Critic Loss": q_loss,
            "Policy Loss": coma_loss,
            "Loss": loss,
        }
