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

"""Implementation of TD3"""

from typing import Dict

import numpy as np
import tensorflow as tf
from chex import Numeric
from tensorflow import Tensor

from og_marl.environments.base import BaseEnvironment
from og_marl.loggers import BaseLogger
from og_marl.tf2.systems.facmac import FACMACSystem
from og_marl.tf2.utils import (
    batch_concat_agent_id_to_obs,
    batched_agents,
    expand_batch_and_agent_dim_of_time_major_sequence,
    merge_batch_and_agent_dim_of_time_major_sequence,
    switch_two_leading_dims,
    unroll_rnn,
)


class FACMACCQLSystem(FACMACSystem):
    def __init__(
        self,
        environment: BaseEnvironment,
        logger: BaseLogger,
        linear_layer_dim: int = 64,
        recurrent_layer_dim: int = 64,
        discount: float = 0.99,
        target_update_rate: float = 0.005,
        critic_learning_rate: float = 1e-4,
        policy_learning_rate: float = 1e-4,
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
    def _tf_train_step(self, batch: Dict[str, Tensor]) -> Dict[str, Numeric]:
        batch = batched_agents(self._environment.possible_agents, batch)

        # Unpack the batch
        observations = batch["observations"]  # (B,T,N,O)
        actions = batch["actions"]  # (B,T,N,A)
        env_states = batch["state"]  # (B,T,S)
        rewards = batch["rewards"]  # (B,T,N)
        truncations = tf.cast(batch["truncations"], "float32")  # (B,T,N)
        terminals = tf.cast(batch["terminals"], "float32")  # (B,T,N)

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
        target_qs = self._target_critic_network(observations, target_actions, resets)
        target_qs = self._target_mixer(target_qs, env_states)

        # Compute Bellman targets
        targets = (
            tf.reduce_mean(rewards[:-1], axis=2, keepdims=True)
            + self._discount
            * tf.reduce_mean((1 - terminals[:-1]), axis=2, keepdims=True)
            * target_qs[1:]
        )

        # Do forward passes through the networks and calculate the losses
        with tf.GradientTape(persistent=True) as tape:
            # Online critics
            qs = self._critic_network(observations, replay_actions, resets)
            qs = self._mixer(qs, env_states)

            # Squared TD-error
            critic_loss = tf.reduce_mean(0.5 * (targets - qs[:-1]) ** 2)

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
            repeat_resets = tf.stack([resets] * self._num_ood_actions, axis=2)  # next to batch dim

            # Flatten into batch dim
            repeat_observations = tf.reshape(
                repeat_observations, (T, -1, *repeat_observations.shape[3:])
            )
            repeat_env_states = tf.reshape(repeat_env_states, (T, -1, *repeat_env_states.shape[3:]))
            repeat_online_actions = tf.reshape(
                repeat_online_actions, (T, -1, *repeat_online_actions.shape[3:])
            )
            repeat_resets = tf.reshape(repeat_resets, (T, -1, *repeat_resets.shape[3:]))

            # CQL Loss
            all_ood_qs = []
            random_ood_actions = tf.random.uniform(
                shape=repeat_online_actions.shape,
                minval=-1.0,
                maxval=1.0,
                dtype=repeat_online_actions.dtype,
            )
            random_ood_action_log_pi = tf.math.log(0.5 ** (random_ood_actions.shape[-1]))

            ood_qs = (
                self._critic_network(
                    repeat_observations,
                    random_ood_actions,
                    repeat_resets,
                )[:-1]
                - random_ood_action_log_pi
            )
            ood_qs = self._mixer(ood_qs, repeat_env_states[:-1])
            all_ood_qs.append(ood_qs)

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

            ood_qs = (
                self._critic_network(
                    repeat_observations[:-1],
                    current_ood_actions[:-1],
                    repeat_resets[:-1],
                )
                - ood_actions_log_prob[:-1]
            )
            ood_qs = self._mixer(ood_qs, repeat_env_states[:-1])
            all_ood_qs.append(ood_qs)

            next_ood_qs = (
                self._critic_network(
                    repeat_observations[:-1],
                    current_ood_actions[1:],  # next action
                    repeat_resets[:-1],
                )
                - ood_actions_log_prob[1:]
            )
            next_ood_qs = self._mixer(next_ood_qs, repeat_env_states[:-1])
            all_ood_qs.append(next_ood_qs)

            # Reshape
            all_ood_qs = [tf.reshape(x, (T - 1, B, self._num_ood_actions)) for x in all_ood_qs]
            all_ood_qs = tf.concat(all_ood_qs, axis=2)

            cql_loss = tf.reduce_mean(
                tf.reduce_logsumexp(all_ood_qs, axis=2, keepdims=True)
            ) - tf.reduce_mean(qs[:-1])

            critic_loss += self._cql_weight * cql_loss

            ### END CQL ###

            # Policy Loss
            qs = self._critic_network(observations, online_actions, resets)
            qs = self._mixer(qs, env_states)

            policy_loss = -tf.reduce_mean(qs) + 1e-3 * tf.reduce_mean(tf.square(online_actions))

        # Train critics
        variables = (*self._critic_network.trainable_variables, *self._mixer.trainable_variables)
        gradients = tape.gradient(critic_loss, variables)
        self._critic_optimizer.apply(gradients, variables)

        # Train policy
        variables = (*self._policy_network.trainable_variables,)
        gradients = tape.gradient(policy_loss, variables)
        self._policy_optimizer.apply(gradients, variables)

        # Update target networks
        online_variables = (
            *self._critic_network.variables,
            *self._policy_network.variables,
            *self._mixer.variables,
        )
        target_variables = (
            *self._target_critic_network.variables,
            *self._target_policy_network.variables,
            *self._target_mixer.variables,
        )
        self._update_target_network(
            online_variables,
            target_variables,
        )

        del tape

        logs = {
            "Mean Q-values": tf.reduce_mean(qs),
            "Mean Critic Loss": critic_loss,
            "Policy Loss": policy_loss,
            # "CQL Alpha Loss": cql_alpha_loss,
            "CQL Loss": cql_loss,
        }

        return logs
