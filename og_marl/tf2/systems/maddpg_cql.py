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

"""Implementation of MADDPG+CQL"""
from typing import Any, Dict

import numpy as np
import tensorflow as tf
from chex import Numeric

from og_marl.environments.base import BaseEnvironment
from og_marl.loggers import BaseLogger
from og_marl.tf2.systems.maddpg import MADDPGSystem
from og_marl.tf2.utils import (
    batch_concat_agent_id_to_obs,
    expand_batch_and_agent_dim_of_time_major_sequence,
    merge_batch_and_agent_dim_of_time_major_sequence,
    switch_two_leading_dims,
    unroll_rnn,
)


class MADDPGCQLSystem(MADDPGSystem):

    """MA Deep Recurrent Q-Networs with CQL System"""

    def __init__(
        self,
        environment: BaseEnvironment,
        logger: BaseLogger,
        linear_layer_dim: int = 64,
        recurrent_layer_dim: int = 64,
        discount: float = 0.99,
        target_update_rate: float = 0.005,
        critic_learning_rate: float = 1e-3,
        policy_learning_rate: float = 1e-3,
        add_agent_id_to_obs: bool = False,
        random_exploration_timesteps: int = 0,
        num_ood_actions: int = 10,  # CQL
        cql_weight: float = 5.0,  # CQL
        cql_sigma: float = 0.2,  # CQL
    ):
        super().__init__(
            environment=environment,
            logger=logger,
            linear_layer_dim=linear_layer_dim,
            recurrent_layer_dim=recurrent_layer_dim,
            discount=discount,
            target_update_rate=target_update_rate,
            critic_learning_rate=critic_learning_rate,
            policy_learning_rate=policy_learning_rate,
            add_agent_id_to_obs=add_agent_id_to_obs,
            random_exploration_timesteps=random_exploration_timesteps,
        )

        self._num_ood_actions = num_ood_actions
        self._cql_weight = cql_weight
        self._cql_sigma = cql_sigma

    @tf.function(jit_compile=True)  # NOTE: comment this out if using debugger
    def _tf_train_step(self, experience: Dict[str, Any]) -> Dict[str, Numeric]:
        # Unpack the batch
        observations = experience["observations"]  # (B,T,N,O)
        actions = experience["actions"]  # (B,T,N,A)
        env_states = experience["infos"]["state"]  # (B,T,S)
        rewards = experience["rewards"]  # (B,T,N)
        truncations = tf.cast(experience["truncations"], "float32")  # (B,T,N)
        terminals = tf.cast(experience["terminals"], "float32")  # (B,T,N)

        # When to reset the RNN hidden state
        resets = tf.maximum(terminals, truncations)  # equivalent to logical 'or'

        # Get dims
        B, T, N = actions.shape[:3]

        # Maybe add agent ids to observation
        if self._add_agent_id_to_obs:
            observations = batch_concat_agent_id_to_obs(observations)

        # Make time-major
        observations = switch_two_leading_dims(observations)
        replay_actions = switch_two_leading_dims(actions)
        rewards = switch_two_leading_dims(rewards)
        terminals = switch_two_leading_dims(terminals)
        env_states = switch_two_leading_dims(env_states)
        resets = switch_two_leading_dims(resets)

        # Unroll target policy
        target_actions = unroll_rnn(
            self._target_policy_network,
            merge_batch_and_agent_dim_of_time_major_sequence(observations),
            merge_batch_and_agent_dim_of_time_major_sequence(resets),
        )
        target_actions = expand_batch_and_agent_dim_of_time_major_sequence(target_actions, B, N)

        # Target critics
        target_qs_1 = self._target_critic_network_1(env_states, target_actions, target_actions)
        target_qs_2 = self._target_critic_network_2(env_states, target_actions, target_actions)

        # Take minimum between two target critics
        target_qs = tf.minimum(target_qs_1, target_qs_2)

        # Compute Bellman targets
        targets = rewards[:-1] + self._discount * (1 - terminals[:-1]) * tf.squeeze(
            target_qs[1:], axis=-1
        )

        # Do forward passes through the networks and calculate the losses
        with tf.GradientTape(persistent=True) as tape:
            # Online critics
            qs_1 = tf.squeeze(
                self._critic_network_1(env_states, replay_actions, replay_actions),
                axis=-1,
            )
            qs_2 = tf.squeeze(
                self._critic_network_2(env_states, replay_actions, replay_actions),
                axis=-1,
            )

            # Squared TD-error
            critic_loss_1 = tf.reduce_mean(0.5 * (targets - qs_1[:-1]) ** 2)
            critic_loss_2 = tf.reduce_mean(0.5 * (targets - qs_2[:-1]) ** 2)

            ###########
            ### CQL ###
            ###########

            online_actions = unroll_rnn(
                self._policy_network,
                merge_batch_and_agent_dim_of_time_major_sequence(observations),
                merge_batch_and_agent_dim_of_time_major_sequence(resets),
            )
            online_actions = expand_batch_and_agent_dim_of_time_major_sequence(online_actions, B, N)

            # Repeat all tensors num_ood_actions times andadd  next to batch dim
            repeat_observations = tf.stack(
                [observations] * self._num_ood_actions, axis=2
            )  # next to batch dim
            repeat_env_states = tf.stack(
                [env_states] * self._num_ood_actions, axis=2
            )  # next to batch dim
            repeat_online_actions = tf.stack(
                [online_actions] * self._num_ood_actions, axis=2
            )  # next to batch dim

            # Flatten into batch dim
            repeat_observations = tf.reshape(
                repeat_observations, (T, -1, *repeat_observations.shape[3:])
            )
            repeat_env_states = tf.reshape(repeat_env_states, (T, -1, *repeat_env_states.shape[3:]))
            repeat_online_actions = tf.reshape(
                repeat_online_actions, (T, -1, *repeat_online_actions.shape[3:])
            )

            # CQL Loss
            random_ood_actions = tf.random.uniform(
                shape=repeat_online_actions.shape,
                minval=-1.0,
                maxval=1.0,
                dtype=repeat_online_actions.dtype,
            )
            random_ood_action_log_pi = tf.math.log(0.5 ** (random_ood_actions.shape[-1]))

            ood_qs_1 = (
                self._critic_network_1(repeat_env_states, random_ood_actions, random_ood_actions)[
                    :-1
                ]
                - random_ood_action_log_pi
            )
            ood_qs_2 = (
                self._critic_network_2(repeat_env_states, random_ood_actions, random_ood_actions)[
                    :-1
                ]
                - random_ood_action_log_pi
            )

            # # Actions near true actions
            mu = 0.0
            std = self._cql_sigma
            action_noise = tf.random.normal(
                repeat_online_actions.shape,
                mean=mu,
                stddev=std,
                dtype=repeat_online_actions.dtype,
            )
            current_ood_actions = tf.clip_by_value(repeat_online_actions + action_noise, -1.0, 1.0)

            ood_actions_prob = (1 / (self._cql_sigma * tf.math.sqrt(2 * np.pi))) * tf.exp(
                -((action_noise - mu) ** 2) / (2 * self._cql_sigma**2)
            )
            ood_actions_log_prob = tf.math.log(
                tf.reduce_prod(ood_actions_prob, axis=-1, keepdims=True)
            )

            current_ood_qs_1 = (
                self._critic_network_1(
                    repeat_env_states[:-1],
                    current_ood_actions[:-1],
                    current_ood_actions[:-1],
                )
                - ood_actions_log_prob[:-1]
            )
            current_ood_qs_2 = (
                self._critic_network_2(
                    repeat_env_states[:-1],
                    current_ood_actions[:-1],
                    current_ood_actions[:-1],
                )
                - ood_actions_log_prob[:-1]
            )

            next_current_ood_qs_1 = (
                self._critic_network_1(
                    repeat_env_states[:-1],
                    current_ood_actions[1:],
                    current_ood_actions[1:],
                )
                - ood_actions_log_prob[1:]
            )
            next_current_ood_qs_2 = (
                self._critic_network_2(
                    repeat_env_states[:-1],
                    current_ood_actions[1:],
                    current_ood_actions[1:],
                )
                - ood_actions_log_prob[1:]
            )

            # Reshape
            ood_qs_1 = tf.reshape(ood_qs_1, (T - 1, B, self._num_ood_actions, N))
            ood_qs_2 = tf.reshape(ood_qs_2, (T - 1, B, self._num_ood_actions, N))
            current_ood_qs_1 = tf.reshape(current_ood_qs_1, (T - 1, B, self._num_ood_actions, N))
            current_ood_qs_2 = tf.reshape(current_ood_qs_2, (T - 1, B, self._num_ood_actions, N))
            next_current_ood_qs_1 = tf.reshape(
                next_current_ood_qs_1, (T - 1, B, self._num_ood_actions, N)
            )
            next_current_ood_qs_2 = tf.reshape(
                next_current_ood_qs_2, (T - 1, B, self._num_ood_actions, N)
            )

            all_ood_qs_1 = tf.concat((ood_qs_1, current_ood_qs_1, next_current_ood_qs_1), axis=2)
            all_ood_qs_2 = tf.concat((ood_qs_2, current_ood_qs_2, next_current_ood_qs_2), axis=2)

            cql_loss_1 = tf.reduce_mean(
                tf.reduce_logsumexp(all_ood_qs_1, axis=2, keepdims=False)
            ) - tf.reduce_mean(qs_1[:-1])
            cql_loss_2 = tf.reduce_mean(
                tf.reduce_logsumexp(all_ood_qs_2, axis=2, keepdims=False)
            ) - tf.reduce_mean(qs_2[:-1])

            critic_loss_1 += self._cql_weight * cql_loss_1
            critic_loss_2 += self._cql_weight * cql_loss_2

            ### END CQL ###

            critic_loss = (critic_loss_1 + critic_loss_2) / 2

            # Policy Loss
            # Unroll online policy
            onlin_actions = unroll_rnn(
                self._policy_network,
                merge_batch_and_agent_dim_of_time_major_sequence(observations),
                merge_batch_and_agent_dim_of_time_major_sequence(resets),
            )
            online_actions = expand_batch_and_agent_dim_of_time_major_sequence(onlin_actions, B, N)

            qs_1 = self._critic_network_1(env_states, online_actions, replay_actions)
            qs_2 = self._critic_network_2(env_states, online_actions, replay_actions)
            qs = tf.minimum(qs_1, qs_2)

            policy_loss = -tf.reduce_mean(qs) + 1e-3 * tf.reduce_mean(online_actions**2)

        # Train critics
        variables = (
            *self._critic_network_1.trainable_variables,
            *self._critic_network_2.trainable_variables,
        )
        gradients = tape.gradient(critic_loss, variables)
        self._critic_optimizer.apply(gradients, variables)

        # Train policy
        variables = (*self._policy_network.trainable_variables,)
        gradients = tape.gradient(policy_loss, variables)
        self._policy_optimizer.apply(gradients, variables)

        # Update target networks
        online_variables = (
            *self._critic_network_1.variables,
            *self._critic_network_2.variables,
            *self._policy_network.variables,
        )
        target_variables = (
            *self._target_critic_network_1.variables,
            *self._target_critic_network_2.variables,
            *self._target_policy_network.variables,
        )
        self._update_target_network(
            online_variables,
            target_variables,
        )

        del tape

        logs = {
            "Mean Q-values": tf.reduce_mean((qs_1 + qs_2) / 2),
            "Mean Critic Loss": (critic_loss),
            "Policy Loss": policy_loss,
        }

        return logs
