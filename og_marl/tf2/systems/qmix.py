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

"""Implementation of QMIX"""
from typing import Any, Dict, Tuple, Optional

import copy
import sonnet as snt
import tensorflow as tf
from chex import Numeric
from tensorflow import Tensor

from og_marl.environments.base import BaseEnvironment
from og_marl.loggers import BaseLogger
from og_marl.tf2.systems.idrqn import IDRQNSystem
from og_marl.tf2.utils import (
    batch_concat_agent_id_to_obs,
    expand_batch_and_agent_dim_of_time_major_sequence,
    gather,
    merge_batch_and_agent_dim_of_time_major_sequence,
    switch_two_leading_dims,
    unroll_rnn,
)
from og_marl.tf2.networks import IdentityNetwork, QMixer


class QMIXSystem(IDRQNSystem):

    """QMIX System"""

    def __init__(
        self,
        environment: BaseEnvironment,
        logger: BaseLogger,
        linear_layer_dim: int = 64,
        recurrent_layer_dim: int = 64,
        mixer_embed_dim: int = 32,
        mixer_hyper_dim: int = 64,
        discount: float = 0.99,
        target_update_period: int = 200,
        learning_rate: float = 1e-3,
        eps_min: float = 0.05,
        eps_decay_timesteps: int = 50_000,
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
            eps_min=eps_min,
            target_update_period=target_update_period,
            learning_rate=learning_rate,
            eps_decay_timesteps=eps_decay_timesteps,
            observation_embedding_network=observation_embedding_network,
        )

        if state_embedding_network is None:
            state_embedding_network = IdentityNetwork()
        self._state_embedding_network = state_embedding_network
        self._target_state_embedding_network = copy.deepcopy(state_embedding_network)

        self._mixer = QMixer(
            len(self._environment.possible_agents), mixer_embed_dim, mixer_hyper_dim
        )
        self._target_mixer = QMixer(
            len(self._environment.possible_agents), mixer_embed_dim, mixer_hyper_dim
        )

    @tf.function(jit_compile=True)  # NOTE: comment this out if using debugger
    def _tf_train_step(self, train_step_ctr: int, experience: Dict[str, Any]) -> Dict[str, Numeric]:
        # Unpack the batch
        observations = experience["observations"]  # (B,T,N,O)
        actions = experience["actions"]  # (B,T,N)
        env_states = experience["infos"]["state"]  # (B,T,S)
        rewards = experience["rewards"]  # (B,T,N)
        truncations = tf.cast(experience["truncations"], "float32")  # (B,T,N)
        terminals = tf.cast(experience["terminals"], "float32")  # (B,T,N)
        legal_actions = experience["infos"]["legals"]  # (B,T,N,A)

        done = terminals

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
        target_qs_out = switch_two_leading_dims(target_qs_out)

        with tf.GradientTape() as tape:
            # Unroll online network
            q_network_embeddings = self._q_embedding_network(observations)
            qs_out = unroll_rnn(self._q_network, q_network_embeddings, resets)

            # Expand batch and agent_dim
            qs_out = expand_batch_and_agent_dim_of_time_major_sequence(qs_out, B, N)

            # Make batch-major again
            qs_out = switch_two_leading_dims(qs_out)

            # Pick the Q-Values for the actions taken by each agent
            chosen_action_qs = gather(qs_out, tf.cast(actions, "int32"), axis=3, keepdims=False)

            # Max over target Q-Values/ Double q learning
            qs_out_selector = tf.where(
                tf.cast(legal_actions, "bool"), qs_out, -9999999
            )  # legal action masking
            cur_max_actions = tf.argmax(qs_out_selector, axis=3)
            target_max_qs = gather(target_qs_out, cur_max_actions, axis=-1, keepdims=False)

            # Q-MIXING
            env_state_embeddings, target_env_state_embeddings = (
                self._state_embedding_network(env_states),
                self._target_state_embedding_network(env_states),
            )
            chosen_action_qs, target_max_qs, rewards = self._mixing(
                chosen_action_qs,
                target_max_qs,
                env_state_embeddings,
                target_env_state_embeddings,
                rewards,
            )

            # Reduce Agent Dim
            done = tf.reduce_mean(done, axis=2, keepdims=True)  # NOTE Assumes all the same

            # Compute targets
            targets = rewards[:, :-1] + (1 - done[:, :-1]) * self._discount * target_max_qs[:, 1:]
            targets = tf.stop_gradient(targets)

            # Chop off last time step
            chosen_action_qs = chosen_action_qs[:, :-1]  # shape=(B,T-1)

            # TD-Error Loss
            loss = 0.5 * tf.square(targets - chosen_action_qs)
            loss = tf.reduce_mean(loss)

        # Get trainable variables
        variables = (
            *self._q_network.trainable_variables,
            *self._q_embedding_network.trainable_variables,
            *self._mixer.trainable_variables,
            *self._state_embedding_network.trainable_variables,
        )

        # Compute gradients.
        gradients = tape.gradient(loss, variables)

        # Apply gradients.
        self._optimizer.apply(gradients, variables)

        # Online variables
        online_variables = (
            *self._q_network.variables,
            *self._q_embedding_network.variables,
            *self._mixer.variables,
            *self._state_embedding_network.variables,
        )

        # Get target variables
        target_variables = (
            *self._target_q_network.variables,
            *self._q_embedding_network.variables,
            *self._target_mixer.variables,
            *self._target_state_embedding_network.variables,
        )

        # Maybe update target network
        self._update_target_network(train_step_ctr, online_variables, target_variables)

        return {
            "Loss": loss,
            "Mean Q-values": tf.reduce_mean(qs_out),
            "Mean Chosen Q-values": tf.reduce_mean(chosen_action_qs),
        }

    def _mixing(
        self,
        chosen_action_qs: Tensor,
        target_max_qs: Tensor,
        state_embeddings: Tensor,
        target_state_embeddings: Tensor,
        rewards: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """QMIX"""
        # VDN
        # chosen_action_qs = tf.reduce_sum(chosen_action_qs, axis=2, keepdims=True)
        # target_max_qs = tf.reduce_sum(target_max_qs, axis=2, keepdims=True)
        # VDN

        chosen_action_qs = self._mixer(chosen_action_qs, state_embeddings)
        target_max_qs = self._target_mixer(target_max_qs, target_state_embeddings)
        rewards = tf.reduce_mean(rewards, axis=2, keepdims=True)
        return chosen_action_qs, target_max_qs, rewards
