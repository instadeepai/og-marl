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
import time
from typing import Any, Dict, Optional

import numpy as np
import tree
import tensorflow as tf
import tensorflow_probability as tfp
import sonnet as snt
from chex import Numeric

from og_marl.environments.base import BaseEnvironment
from og_marl.loggers import BaseLogger
from og_marl.replay_buffers import FlashbaxReplayBuffer
from og_marl.tf2.systems.maddpg import MADDPGSystem
from og_marl.tf2.utils import (
    batch_concat_agent_id_to_obs,
    expand_batch_and_agent_dim_of_time_major_sequence,
    merge_batch_and_agent_dim_of_time_major_sequence,
    switch_two_leading_dims,
    unroll_rnn,
)


tfd = tfp.distributions
snt_init = snt.initializers


class MADDPGCQLBCSystem(MADDPGSystem):

    """MA Deep Recurrent Q-Network with CQL System"""

    def __init__(
        self,
        environment: BaseEnvironment,
        logger: BaseLogger,
        linear_layer_dim: int = 64,
        recurrent_layer_dim: int = 64,
        discount: float = 0.99,
        target_update_rate: float = 0.005,
        critic_learning_rate: float = 1e-3,
        policy_learning_rate: float = 3e-4,
        add_agent_id_to_obs: bool = False,
        random_exploration_timesteps: int = 0,
        num_ood_actions: int = 10,  # CQL
        cql_weight: float = 5.0,  # CQL
        cql_sigma: float = 0.3,  # CQL
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

        # Policy network
        self._bc_network = snt.Sequential(
            [
                snt.Linear(linear_layer_dim),
                tf.nn.relu,
                snt.Linear(recurrent_layer_dim),
                tf.nn.relu,
                snt.Linear(self._environment._num_actions * len(self._environment.possible_agents)),
                tf.nn.tanh,
            ]
        )  # shared network for all agents

        self._bc_optimiser = snt.optimizers.RMSProp(learning_rate=1e-3)

    @tf.function(jit_compile=True)
    def _train_bc(self, experience):
        observations = experience["observations"]  # (B,T,N,O)
        actions = experience["actions"]  # (B,T,N,A)

        ### Compute new replay priorities
        T, B, N, A = actions.shape[:4]
        O = observations.shape[3]

        joint_action = tf.reshape(actions, (T, B, N * A))
        joint_observation = tf.reshape(observations, (T, B, N * O))

        with tf.GradientTape(persistent=True) as tape:
            bc_joint_action = self._bc_network(joint_observation)
            loss = tf.reduce_mean((joint_action - bc_joint_action)**2)

        # Train BC
        variables = (
            *self._bc_network.trainable_variables,
        )
        gradients = tape.gradient(loss, variables)
        self._bc_optimiser.apply(gradients, variables)

        logs = {
            "BC Loss": loss,
        }

        del tape

        return logs
    
    def select_actions(
        self,
        observations: Dict[str, np.ndarray],
        legal_actions: Optional[Dict[str, np.ndarray]] = None,
        explore: bool = True,
    ) -> Dict[str, np.ndarray]:
        actions, next_rnn_states = self._tf_select_actions(observations, self._rnn_states, explore)
        self._rnn_states = next_rnn_states

        # TODO make sure this is commented out
        # actions = self._select_bc_action(observations)

        return tree.map_structure(  # type: ignore
            lambda x: x[0].numpy(), actions
        )  # convert to numpy and squeeze batch dim
    
    @tf.function(jit_compile=True)
    def _select_bc_action(self, observations):
        actions = {}

        joint_observation = tf.concat(list(observations.values()), axis=-1)
        joint_observation = tf.expand_dims(joint_observation, axis=0)
        joint_action = self._bc_network(joint_observation)
        agent_actions = tf.reshape(joint_action, (1,2,3))

        for i, agent in enumerate(self._environment.possible_agents):
            # Store agent action
            actions[agent] = agent_actions[:,i]

        return actions


    def train_step(self, experience, trainer_step_ctr) -> Dict[str, Numeric]:
        trainer_step_ctr = tf.convert_to_tensor(trainer_step_ctr)

        logs = self._tf_train_step(experience, trainer_step_ctr)

        return logs  # type: ignore

    @tf.function(jit_compile=True)  # NOTE: comment this out if using debugger
    def _tf_train_step(self, experience: Dict[str, Any], train_step) -> Dict[str, Numeric]:
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
        B, T, N, A = actions.shape[:4]
        O = observations.shape[3]

        joint_observation = tf.reshape(observations, (T, B, N * O))

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

            # Policy Loss
            # Unroll online policy
            online_actions = unroll_rnn(
                self._policy_network,
                merge_batch_and_agent_dim_of_time_major_sequence(observations),
                merge_batch_and_agent_dim_of_time_major_sequence(resets),
            )
            online_actions = expand_batch_and_agent_dim_of_time_major_sequence(online_actions, B, N)

            policy_qs_1 = self._critic_network_1(env_states, online_actions, online_actions)
            policy_qs_2 = self._critic_network_2(env_states, online_actions, online_actions)

            policy_qs = tf.minimum(policy_qs_1, policy_qs_2)

            bc_lambda = (2.5/tf.reduce_mean(tf.abs(tf.stop_gradient(policy_qs))))
            policy_loss =  tf.reduce_mean((replay_actions - online_actions)**2) - bc_lambda * tf.reduce_mean(policy_qs) # + 1e-3 * tf.reduce_mean(online_actions**2)

            # Critic Loss
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
            ) - tf.reduce_mean(qs_1[:-1])- tf.reduce_mean(policy_qs_1)
            cql_loss_2 = tf.reduce_mean(
                tf.reduce_logsumexp(all_ood_qs_2, axis=2, keepdims=False)
            ) - tf.reduce_mean(qs_2[:-1]) - tf.reduce_mean(policy_qs_2)

            critic_loss_1 += self._cql_weight * cql_loss_1
            critic_loss_2 += self._cql_weight * cql_loss_2

            ### END CQL ###

            critic_loss = (critic_loss_1 + critic_loss_2) / 2

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
    
    def train_offline(
        self,
        replay_buffer: FlashbaxReplayBuffer,
        max_trainer_steps: int = int(1e5),
        evaluate_every: int = 1000,
        num_eval_episodes: int = 4,
        json_writer: Optional[int] = None,
    ) -> None:
        """Method to train the system offline.

        WARNING: make sure evaluate_every % log_every == 0 and log_every < evaluate_every,
        else you won't log evaluation.
        """
        trainer_step_ctr = 0
        while trainer_step_ctr < max_trainer_steps:
            if evaluate_every is not None and trainer_step_ctr % evaluate_every == 0:
                print("EVALUATION")
                eval_logs = self.evaluate(num_eval_episodes)
                self._logger.write(
                    eval_logs | {"Trainer Steps (eval)": trainer_step_ctr}, force=True
                )
                if json_writer is not None:
                    json_writer.write(
                        trainer_step_ctr,
                        "evaluator/episode_return",
                        eval_logs["evaluator/episode_return"],
                        trainer_step_ctr // evaluate_every,
                    )

            start_time = time.time()
            batch = replay_buffer.sample()
            # batch = replay_buffer.good_sample()
            end_time = time.time()
            time_to_sample = end_time - start_time

            start_time = time.time()
            train_logs = self.train_step(batch.experience, trainer_step_ctr)
            end_time = time.time()
            time_train_step = end_time - start_time

            train_steps_per_second = 1 / (time_train_step + time_to_sample)

            logs = {
                **train_logs,
                "Trainer Steps": trainer_step_ctr,
                "Time to Sample": time_to_sample,
                "Time for Train Step": time_train_step,
                "Train Steps Per Second": train_steps_per_second,
            }

            self._logger.write(logs)

            trainer_step_ctr += 1

        print("FINAL EVALUATION")
        eval_logs = self.evaluate(num_eval_episodes)
        self._logger.write(eval_logs | {"Trainer Steps (eval)": trainer_step_ctr}, force=True)
        if json_writer is not None:
            eval_logs = {f"absolute/{key.split('/')[1]}": value for key, value in eval_logs.items()}
            json_writer.write(
                trainer_step_ctr,
                "absolute/episode_return",
                eval_logs["absolute/episode_return"],
                trainer_step_ctr // evaluate_every,
            )
            json_writer.close()
