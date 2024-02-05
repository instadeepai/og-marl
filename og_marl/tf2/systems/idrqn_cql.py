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
"""Implementation of IDRQN+CQL"""
import sonnet as snt
import tensorflow as tf

from og_marl.tf2.systems.qmix import QMIXSystem
from og_marl.tf2.utils import (
    batch_concat_agent_id_to_obs, batched_agents,
    expand_batch_and_agent_dim_of_time_major_sequence, gather,
    merge_batch_and_agent_dim_of_time_major_sequence, switch_two_leading_dims)


class IDRQNCQLSystem(QMIXSystem):
    """IDRQN+CQL System"""

    def __init__(
        self,
        environment,
        logger,
        num_ood_actions=5,
        cql_weight=1.0,
        linear_layer_dim=64,
        recurrent_layer_dim=64,
        mixer_embed_dim=32,
        mixer_hyper_dim=64,
        discount=0.99,
        target_update_period=200,
        learning_rate=3e-4,
        add_agent_id_to_obs=False,
    ):

        super().__init__(
            environment,
            logger,
            linear_layer_dim=linear_layer_dim,
            recurrent_layer_dim=recurrent_layer_dim,
            mixer_embed_dim=mixer_embed_dim,
            mixer_hyper_dim=mixer_hyper_dim,
            add_agent_id_to_obs=add_agent_id_to_obs,
            discount=discount,
            target_update_period=target_update_period,
            learning_rate=learning_rate
        )

        # CQL
        self._num_ood_actions = num_ood_actions
        self._cql_weight = cql_weight

    @tf.function(jit_compile=True)
    def _tf_train_step(self, train_step, batch):
        batch = batched_agents(self._environment.possible_agents, batch)

        # Unpack the batch
        observations = batch["observations"] # (B,T,N,O)
        actions = tf.cast(batch["actions"], "int32") # (B,T,N)
        env_states = batch["state"] # (B,T,S)
        rewards = batch["rewards"] # (B,T,N)
        truncations = batch["truncations"] # (B,T,N)
        terminals = batch["terminals"] # (B,T,N)
        zero_padding_mask = batch["mask"] # (B,T)
        legal_actions = batch["legals"]  # (B,T,N,A)

        done = terminals

        # Get dims
        B, T, N, A = legal_actions.shape

        # Maybe add agent ids to observation
        if self._add_agent_id_to_obs:
            observations = batch_concat_agent_id_to_obs(observations)

        # Make time-major
        observations = switch_two_leading_dims(observations)

        # Merge batch_dim and agent_dim
        observations = merge_batch_and_agent_dim_of_time_major_sequence(observations)

        # Unroll target network
        target_qs_out, _ = snt.static_unroll(
            self._target_q_network, 
            observations,
            self._target_q_network.initial_state(B*N)
        )

        # Expand batch and agent_dim
        target_qs_out = expand_batch_and_agent_dim_of_time_major_sequence(target_qs_out, B, N)

        # Make batch-major again
        target_qs_out = switch_two_leading_dims(target_qs_out)

        with tf.GradientTape() as tape:
            # Unroll online network
            qs_out, _ = snt.static_unroll(
                self._q_network, 
                observations, 
                self._q_network.initial_state(B*N)
            )

            # Expand batch and agent_dim
            qs_out = expand_batch_and_agent_dim_of_time_major_sequence(qs_out, B, N)

            # Make batch-major again
            qs_out = switch_two_leading_dims(qs_out)

            # Pick the Q-Values for the actions taken by each agent
            chosen_action_qs = gather(qs_out, actions, axis=3, keepdims=False)

            # Max over target Q-Values/ Double q learning
            qs_out_selector = tf.where(
                tf.cast(legal_actions, "bool"), qs_out, -9999999
            )  # legal action masking
            cur_max_actions = tf.argmax(qs_out_selector, axis=3)
            target_max_qs = gather(target_qs_out, cur_max_actions, axis=-1)

            # Compute targets
            targets = rewards[:, :-1] + (1-done[:, :-1]) * self._discount * target_max_qs[:, 1:]
            targets = tf.stop_gradient(targets)

            # TD-Error Loss
            loss = 0.5 * tf.square(targets - chosen_action_qs[:, :-1])

            #############
            #### CQL ####
            #############

            random_ood_actions = tf.random.uniform(
                                shape=(self._num_ood_actions, B, T, N),
                                minval=0,
                                maxval=A,
                                dtype=tf.dtypes.int64
            ) # [Ra, B, T, N]

            all_ood_qs = []
            for i in range(self._num_ood_actions):
                # Gather
                one_hot_indices = tf.one_hot(random_ood_actions[i], depth=qs_out.shape[-1])
                ood_qs = tf.reduce_sum(
                    qs_out * one_hot_indices, axis=-1, keepdims=False
                ) # [B, T, N]

                # Mixing
                all_ood_qs.append(ood_qs) # [B, T, Ra]

            all_ood_qs.append(chosen_action_qs) # [B, T, Ra + 1]
            all_ood_qs = tf.concat(all_ood_qs, axis=-1)

            cql_loss = self._apply_mask(tf.reduce_logsumexp(all_ood_qs, axis=-1, keepdims=True)[:, :-1], zero_padding_mask) - self._apply_mask(chosen_action_qs[:, :-1], zero_padding_mask)

            #############
            #### end ####
            #############

            # Mask out zero-padded timesteps
            loss = self._apply_mask(loss, zero_padding_mask) + cql_loss

        # Get trainable variables
        variables = (
            *self._q_network.trainable_variables,
        )

        # Compute gradients.
        gradients = tape.gradient(loss, variables)

        # Apply gradients.
        self._optimizer.apply(gradients, variables)

        # Online variables
        online_variables = (
            *self._q_network.variables,
        )

        # Get target variables
        target_variables = (
            *self._target_q_network.variables,
        )

        # Maybe update target network
        self._update_target_network(train_step, online_variables, target_variables)

        return {
            "Loss": loss,
            "Mean Q-values": tf.reduce_mean(qs_out),
            "Mean Chosen Q-values": tf.reduce_mean(chosen_action_qs),
        }