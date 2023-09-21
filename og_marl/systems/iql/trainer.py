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

"""Implementation of independent Q-learning (DRQN style)"""
import copy
import tensorflow as tf
import trfl
import sonnet as snt

from og_marl.systems import TrainerBase
from og_marl.utils.trainer_utils import (
    sample_batch_agents,
    gather,
    batch_concat_agent_id_to_obs,
    switch_two_leading_dims,
    merge_batch_and_agent_dim_of_time_major_sequence,
    expand_batch_and_agent_dim_of_time_major_sequence,
)


class IQLTrainer(TrainerBase):
    def __init__(
        self,
        agents,
        dataset,
        logger,
        optimizer,
        q_network,
        discount=0.99,
        lambda_=0.6,
        target_update_period=200,
        max_gradient_norm=20.0,
        add_agent_id_to_obs=False,
        target_update_rate=0.01,
        max_trainer_steps=1e5,
    ):
        super().__init__(
            agents=agents,
            dataset=dataset,
            logger=logger,
            discount=discount,
            max_gradient_norm=max_gradient_norm,
            add_agent_id_to_obs=add_agent_id_to_obs,
            max_trainer_steps=max_trainer_steps,
        )

        # Q-network
        self._q_network = q_network
        self._system_variables.update({"q_network": self._q_network.variables})

        # Optimizer
        self._optimizer = optimizer

        # Q-lambda
        self._lambda = lambda_

        # Target networks
        self._target_q_network = copy.deepcopy(q_network)
        self._target_update_period = target_update_period
        self._target_update_rate = target_update_rate

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

            # Maybe do some extra reshaping
            qs_out = self._reshape_qs(qs_out)
            target_qs_out = self._reshape_qs(target_qs_out)

            # Pick the Q-Values for the actions taken by each agent
            chosen_action_qs = gather(qs_out, actions, axis=3, keepdims=False)

            # Max over target Q-Values/ Double q learning
            target_max_qs = self._get_target_max_qs(
                qs_out, target_qs_out, legal_actions
            )

            # Maybe do mixing (Noop in IQL)
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
            loss = 0.5 * tf.square(targets - chosen_action_qs)

            # Mask out zero-padded timesteps
            loss = self._apply_mask(loss, mask)

        # Get trainable variables
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
            "Mean Q-values": tf.reduce_mean(qs_out),
            "Mean Chosen Q-values": tf.reduce_mean(chosen_action_qs),
            "Trainer Steps": trainer_step,
        }

    def _batch_agents(self, agents, sample):
        """Batch all agent transitions into larger tensor of shape [T,B,N,...],
        where N is the number of agents."""
        return sample_batch_agents(agents, sample, independent=True)

    def _reshape_qs(self, qs):
        return qs  # NOOP

    def _get_target_max_qs(self, qs_out, target_qs_out, legal_actions):
        qs_out_selector = tf.where(
            legal_actions, qs_out, -9999999
        )  # legal action masking
        cur_max_actions = tf.argmax(qs_out_selector, axis=3)
        target_max_qs = gather(target_qs_out, cur_max_actions, axis=-1)
        return target_max_qs

    def _mixing(self, chosen_action_qs, target_max_qs, states):
        """Noop in IQL."""
        return chosen_action_qs, target_max_qs

    def _compute_targets(self, rewards, env_discounts, target_max_qs):
        if self._lambda is not None:
            # Get time and batch dim
            B, T = rewards.shape[:2]

            # Duplicate rewards and discount for all agents
            rewards = tf.broadcast_to(rewards, target_max_qs.shape)
            env_discounts = tf.broadcast_to(env_discounts, target_max_qs.shape)

            # Make time major for trfl
            rewards = tf.transpose(rewards, perm=[1, 0, 2])
            env_discounts = tf.transpose(env_discounts, perm=[1, 0, 2])
            target_max_qs = tf.transpose(target_max_qs, perm=[1, 0, 2])

            # Flatten agent dim into batch-dim
            rewards = tf.reshape(rewards, shape=(T, -1))
            env_discounts = tf.reshape(env_discounts, shape=(T, -1))
            target_max_qs = tf.reshape(target_max_qs, shape=(T, -1))

            # Q(lambda)
            targets = trfl.multistep_forward_view(
                rewards[:-1],
                self._discount * env_discounts[:-1],
                target_max_qs[1:],
                lambda_=self._lambda,
                back_prop=False,
            )
            # Unpack agent dim again
            targets = tf.reshape(targets, shape=(T - 1, B, -1))

            # Make batch major again
            targets = tf.transpose(targets, perm=[1, 0, 2])
        else:
            targets = (
                rewards[:, :-1]
                + self._discount * env_discounts[:, :-1] * target_max_qs[:, 1:]
            )
        return tf.stop_gradient(targets)

    def _apply_mask(self, loss, mask):
        mask = tf.broadcast_to(mask[:, :-1], loss.shape)
        loss = tf.reduce_sum(loss * mask) / tf.reduce_sum(mask)
        return loss

    def _get_trainable_variables(self):
        variables = (
            *self._q_network.trainable_variables,
        )
        return variables

    def _get_variables_to_update(self):
        # Online variables
        online_variables = (
            *self._q_network.variables,
        )

        # Get target variables
        target_variables = (
            *self._target_q_network.variables,
        )

        return online_variables, target_variables

    def _update_target_network(self, online_variables, target_variables, trainer_step):
        """Update the target networks."""
        tau = self._target_update_rate
        for src, dest in zip(online_variables, target_variables):
            dest.assign(dest * (1.0 - tau) + src * tau)

    def after_train_step(self):
        info = {}
        return info