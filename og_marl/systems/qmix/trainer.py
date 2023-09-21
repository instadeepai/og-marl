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

import copy
import tensorflow as tf
import sonnet as snt

from og_marl.systems.iql import IQLTrainer 
from og_marl.utils.trainer_utils import (
    gather,
    batch_concat_agent_id_to_obs,
    switch_two_leading_dims,
    merge_batch_and_agent_dim_of_time_major_sequence,
    expand_batch_and_agent_dim_of_time_major_sequence,
    sample_batch_agents
)

class QmixTrainer(IQLTrainer):
    def __init__(
        self,
        agents,
        dataset,
        logger,
        optimizer,
        q_network,
        mixer,
        discount=0.99,
        lambda_=0.6,
        max_gradient_norm=20.0,
        add_agent_id_to_obs=False,
        target_update_rate=0.001,
        max_trainer_steps=1e6,
    ):
        super().__init__(
            agents=agents,
            dataset=dataset,
            logger=logger,
            optimizer=optimizer,
            q_network=q_network,
            discount=discount,
            lambda_=lambda_,
            max_gradient_norm=max_gradient_norm,
            add_agent_id_to_obs=add_agent_id_to_obs,
            target_update_rate=target_update_rate,
            max_trainer_steps=max_trainer_steps,
        )

        self._mixer = mixer
        self._target_mixer = copy.deepcopy(mixer)

    def _batch_agents(self, agents, sample):
        return sample_batch_agents(agents, sample, independent=False)

    def _mixing(self, chosen_action_qs, target_max_qs, states):
        """QMIX"""
        chosen_action_qs = self._mixer(chosen_action_qs, states)
        target_max_qs = self._target_mixer(target_max_qs, states)
        return chosen_action_qs, target_max_qs

    def _get_trainable_variables(self):
        variables = (
            *self._q_network.trainable_variables,
            *self._mixer.trainable_variables,
        )
        return variables

    def _get_variables_to_update(self):
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

        return online_variables, target_variables


class QmixBcqTrainer(QmixTrainer):

    def __init__(
        self,
        agents,
        dataset,
        logger,
        optimizer,
        behaviour_cloning_network,
        q_network,
        mixer,
        threshold=0.3,
        discount=0.99,
        lambda_=0.6,
        max_gradient_norm=20.0,
        add_agent_id_to_obs=False,
        target_update_rate=0.001,
        max_trainer_steps=1e6,
    ):
        """Initialise trainer.

        Args:
            TODO
        """

        super().__init__(
            agents=agents,
            dataset=dataset,
            logger=logger,
            optimizer=optimizer,
            q_network=q_network,
            mixer=mixer,
            discount=discount,
            lambda_=lambda_,
            max_gradient_norm=max_gradient_norm,
            add_agent_id_to_obs=add_agent_id_to_obs,
            target_update_rate=target_update_rate,
            max_trainer_steps=max_trainer_steps,
        )

        # BCQ
        self._behaviour_cloning_network = behaviour_cloning_network
        self._threshold = threshold

        # Expose the network variables.
        self._system_variables.update({
            "behaviour_cloning_network": self._behaviour_cloning_network.variables,
        })

    @tf.function
    def _train(self, sample, trainer_step):
        batch = self._batch_agents(self._agents, sample)

        # Get the relevant quantities
        observations = batch["observations"]
        actions = batch["actions"]
        legal_actions = batch["legals"]
        states = batch["states"]
        rewards = batch["rewards"]
        env_discounts = tf.cast(batch["discounts"], "float32")
        mask = tf.cast(batch["mask"], "float32")  # shape=(B,T)

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
            bc_mask = tf.concat([mask] * N, axis=-1)
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

            # Mixing
            chosen_action_qs, target_max_qs = self._mixing(
                chosen_action_qs, target_max_qs, states
            )

            # Compute targets
            targets = self._compute_targets(
                rewards, env_discounts, target_max_qs
            )  # shape=(B,T-1)

            # Chop off last time step
            chosen_action_qs = chosen_action_qs[:, :-1]  # shape=(B,T-1)

            # TD-Error Loss
            q_loss = 0.5 * tf.square(targets - chosen_action_qs)

            # Mask out zero-padded timesteps
            q_loss = self._apply_mask(q_loss, mask)

            # Combine losses
            loss = bc_loss + q_loss

        # Get trainable variables for Q-learning
        variables = self._get_trainable_variables()

        # Compute gradients.
        gradients = tape.gradient(loss, variables)

        # Maybe clip gradients.
        gradients = tf.clip_by_global_norm(gradients, self._max_gradient_norm)[0]

        # Apply gradients.
        self._optimizer.apply(gradients, variables)

        # Get online and target variables
        online_variables, target_variables = self._get_variables_to_update()

        # Maybe update target network
        self._update_target_network(online_variables, target_variables, trainer_step)

        return {
            "Loss": loss,
            "Q-Loss": q_loss,
            "BC Loss": bc_loss,
            "Mean Mixed Q-values": tf.reduce_mean(chosen_action_qs),
            "Trainer Steps": trainer_step,
        }

    def _get_trainable_variables(self):
        variables = (
            *self._q_network.trainable_variables,
            *self._mixer.trainable_variables,
            *self._behaviour_cloning_network.trainable_variables
        )
        return variables

class QmixCqlTrainer(QmixTrainer):

    def __init__(
        self,
        agents,
        dataset,
        logger,
        optimizer,
        q_network,
        mixer,
        num_ood_actions=20, # CQL
        cql_weight=2.0, # CQL
        discount=0.99,
        lambda_=0.6,
        max_gradient_norm=20.0,
        add_agent_id_to_obs=False,
        target_update_rate=0.001,
        max_trainer_steps=1e6,
    ):
        """Initialise trainer.

        Args:
            TODO
        """

        super().__init__(
            agents=agents,
            dataset=dataset,
            logger=logger,
            optimizer=optimizer,
            q_network=q_network,
            mixer=mixer,
            discount=discount,
            lambda_=lambda_,
            max_gradient_norm=max_gradient_norm,
            add_agent_id_to_obs=add_agent_id_to_obs,
            target_update_rate=target_update_rate,
            max_trainer_steps=max_trainer_steps,
        )

        # CQL
        self._num_ood_actions = num_ood_actions
        self._cql_weight = cql_weight

    @tf.function
    def _train(self, sample, trainer_step):
        batch = self._batch_agents(self._agents, sample)

        # Get the relevant quantities
        observations = batch["observations"]
        actions = batch["actions"]
        legal_actions = batch["legals"]
        states = batch["states"]
        rewards = batch["rewards"]
        env_discounts = tf.cast(batch["discounts"], "float32")
        mask = tf.cast(batch["mask"], "float32")  # shape=(B,T)

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
            target_max_qs = self._get_target_max_qs(
                qs_out, target_qs_out, legal_actions
            )

            # Mixing
            chosen_action_qs, target_max_qs = self._mixing(
                chosen_action_qs, target_max_qs, states
            )

            # Compute targets
            targets = self._compute_targets(
                rewards, env_discounts, target_max_qs
            )  # shape=(B,T-1)

            # TD-Error Loss
            td_loss = 0.5 * tf.square(targets - chosen_action_qs[:, :-1])

            # Mask out zero-padded timesteps
            td_loss = self._apply_mask(td_loss, mask)

            # CQL
            random_ood_actions = tf.random.uniform(
                                shape=(self._num_ood_actions, B, T, N),
                                minval=0,
                                maxval=A,
                                dtype=tf.dtypes.int64
            ) # [Ra, B, T, N]

            all_mixed_ood_qs = []
            for i in range(self._num_ood_actions):
                # Gather
                one_hot_indices = tf.one_hot(random_ood_actions[i], depth=qs_out.shape[-1])
                ood_qs = tf.reduce_sum(
                    qs_out * one_hot_indices, axis=-1, keepdims=False
                ) # [B, T, N]

                # Mixing
                mixed_ood_qs = self._mixer(ood_qs, states) # [B, T, 1]
                all_mixed_ood_qs.append(mixed_ood_qs) # [B, T, Ra]

            all_mixed_ood_qs.append(chosen_action_qs) # [B, T, Ra + 1]
            all_mixed_ood_qs = tf.concat(all_mixed_ood_qs, axis=-1)

            cql_loss = self._apply_mask(tf.reduce_logsumexp(all_mixed_ood_qs, axis=-1, keepdims=True)[:, :-1], mask) - self._apply_mask(chosen_action_qs[:, :-1], mask)

            # Add CQL loss to loss
            loss = td_loss + self._cql_weight * cql_loss

        # Get trainable variables for Q-learning
        variables = self._get_trainable_variables()

        # Compute gradients.
        gradients = tape.gradient(loss, variables)

        # Maybe clip gradients.
        gradients = tf.clip_by_global_norm(gradients, self._max_gradient_norm)[0]

        # Apply gradients.
        self._optimizer.apply(gradients, variables)

        # Get online and target variables
        online_variables, target_variables = self._get_variables_to_update()

        # Maybe update target network
        self._update_target_network(online_variables, target_variables, trainer_step)

        return {
            "Loss": loss,
            "TD Loss": td_loss,
            "CQL Loss": cql_loss,
            "Mean Mixed Q-values": tf.reduce_mean(chosen_action_qs),
            "Mean Mixed OOD Q-values": tf.reduce_mean(all_mixed_ood_qs),
            "Trainer Steps": trainer_step,
        }