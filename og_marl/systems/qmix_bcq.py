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

"""Implementation of QMIX+BCQ"""
import tensorflow as tf
import sonnet as snt

from og_marl.systems.qmix import QMIXSystem
from og_marl.utils import (
    gather,
    batch_concat_agent_id_to_obs,
    switch_two_leading_dims,
    merge_batch_and_agent_dim_of_time_major_sequence,
    expand_batch_and_agent_dim_of_time_major_sequence,
    dict_to_tensor,
)

class QMIXBCQSystem(QMIXSystem):
    """QMIX+BCQ System"""

    def __init__(
        self,
        environment,
        logger,
        bc_threshold=0.4,
        linear_layer_dim=100,
        recurrent_layer_dim=100,
        mixer_embed_dim=64,
        mixer_hyper_dim=32,
        batch_size=64,
        discount=0.99,
        target_update_rate=0.005,
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
            batch_size=batch_size,
            discount=discount,
            target_update_rate=target_update_rate,
            learning_rate=learning_rate
        )

        self._threshold = bc_threshold
        self._behaviour_cloning_network = snt.DeepRNN(
            [
                snt.Linear(self._linear_layer_dim),
                tf.nn.relu,
                snt.GRU(self._recurrent_layer_dim),
                tf.nn.relu,
                snt.Linear(self._environment.num_actions),
                tf.nn.softmax,
            ]
        )

    @tf.function(jit_compile=True)
    def _tf_train_step(self, batch):
        batch = dict_to_tensor(self._environment._agents, batch)

        # Unpack the batch
        observations = batch.observations # (B,T,N,O)
        actions = batch.actions # (B,T,N,A)
        legal_actions = batch.legal_actions # (B,T,N,A)
        env_states = batch.env_state # (B,T,S)
        rewards = batch.rewards # (B,T,N)
        done = batch.done # (B,T)
        zero_padding_mask = batch.zero_padding_mask # (B,T)

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

            ###################
            ####### BCQ #######
            ###################

            # Unroll behaviour cloning network
            probs_out, _ = snt.static_unroll(
                self._behaviour_cloning_network, 
                observations, 
                self._behaviour_cloning_network.initial_state(B*N)
            )

            # Expand batch and agent_dim
            probs_out = expand_batch_and_agent_dim_of_time_major_sequence(probs_out, B, N)

            # Make batch-major again
            probs_out = switch_two_leading_dims(probs_out)

            # Behaviour Cloning Loss
            one_hot_actions = tf.one_hot(actions, depth=probs_out.shape[-1], axis=-1)
            bc_mask = tf.concat([zero_padding_mask] * N, axis=-1)
            probs_out = tf.where(
                tf.cast(tf.expand_dims(bc_mask, axis=-1), "bool"),
                probs_out,
                1 / A * tf.ones(A, "float32"),
            )  # avoid nans, get masked out later
            bc_loss = tf.keras.metrics.categorical_crossentropy(
                one_hot_actions, probs_out
            )
            bc_loss = tf.reduce_sum(bc_loss * bc_mask) / tf.reduce_sum(bc_mask)

            # Legal action masking plus bc probs
            masked_probs_out = probs_out * tf.cast(legal_actions, "float32")
            masked_probs_out_sum = tf.reduce_sum(masked_probs_out, axis=-1, keepdims=True)
            masked_probs_out = masked_probs_out / masked_probs_out_sum

            # Behaviour cloning action mask
            bc_action_mask = (
                masked_probs_out / tf.reduce_max(masked_probs_out, axis=-1, keepdims=True)
            ) >= self._threshold


            q_selector = tf.where(bc_action_mask, qs_out, -999999)
            max_actions = tf.argmax(q_selector, axis=-1)
            target_max_qs = gather(target_qs_out, max_actions, axis=-1)

            ###################
            ####### END #######
            ###################

            # Maybe do mixing (e.g. QMIX) but not in independent system
            chosen_action_qs, target_max_qs, rewards = self._mixing(
                chosen_action_qs, target_max_qs, env_states, rewards
            )

            # Compute targets
            targets = rewards[:, :-1] + tf.expand_dims((1-done[:, :-1]), axis=-1) * self._discount * target_max_qs[:, 1:]
            targets = tf.stop_gradient(targets)

            # Chop off last time step
            chosen_action_qs = chosen_action_qs[:, :-1]  # shape=(B,T-1)

            # TD-Error Loss
            loss = 0.5 * tf.square(targets - chosen_action_qs)

            # Mask out zero-padded timesteps
            loss = self._apply_mask(loss, zero_padding_mask) + bc_loss

        # Get trainable variables
        variables = (
            *self._q_network.trainable_variables,
            *self._mixer.trainable_variables,
            *self._behaviour_cloning_network.trainable_variables
        )

        # Compute gradients.
        gradients = tape.gradient(loss, variables)

        # Apply gradients.
        self._optimizer.apply(gradients, variables)

        # Online variables
        online_variables = (
            *self._q_network.variables,
            *self._mixer.variables,
        )

        # Get target variables
        target_variables = (
            *self._target_q_network.variables,
            *self._target_mixer.variables,
        )

        # Maybe update target network
        self._update_target_network(online_variables, target_variables)

        return {
            "Loss": loss,
            "Mean Q-values": tf.reduce_mean(qs_out),
            "Mean Chosen Q-values": tf.reduce_mean(chosen_action_qs),
        }