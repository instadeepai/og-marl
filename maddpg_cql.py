"""Implementation of MADDPG+CQL"""
from typing import Dict
import copy
import time

from replay_buffers import PrioritisedFlashbaxReplayBuffer

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import sonnet as snt
import tree
from chex import Numeric

from utils import (
    batch_concat_agent_id_to_obs,
    concat_agent_id_to_obs,
    expand_batch_and_agent_dim_of_time_major_sequence,
    merge_batch_and_agent_dim_of_time_major_sequence,
    switch_two_leading_dims,
    unroll_rnn,
)


tfd = tfp.distributions
snt_init = snt.initializers


class StateAndJointActionCritic(snt.Module):
    def __init__(self, num_agents: int, num_actions: int):
        self.N = num_agents
        self.A = num_actions

        self._critic_network = snt.Sequential(
            [
                snt.Linear(128),
                tf.keras.layers.ReLU(),
                snt.Linear(128),
                tf.keras.layers.ReLU(),
                snt.Linear(1),
            ]
        )

        super().__init__()

    def __call__(
        self,
        states,
        agent_actions,
        other_actions,
        stop_other_actions_gradient: bool = True,
    ):
        """Forward pass of critic network.

        observations [T,B,N,O]
        states [T,B,S]
        agent_actions [T,B,N,A]: the actions the agent took.
        other_actions [T,B,N,A]: the actions the other agents took.
        """
        if stop_other_actions_gradient:
            other_actions = tf.stop_gradient(other_actions)

        # Make joint action
        joint_actions = make_joint_action(agent_actions, other_actions)

        # Repeat states for each agent
        states = tf.stack([states] * self.N, axis=2)  # [T,B,S] -> [T,B,N,S]

        # Concat states and joint actions
        critic_input = tf.concat([states, joint_actions], axis=-1)

        # Concat agent IDs to critic input
        # critic_input = batch_concat_agent_id_to_obs(critic_input)

        q_values = self._critic_network(critic_input)

        return q_values


def make_joint_action(agent_actions, other_actions):
    """Method to construct the joint action.

    agent_actions [T,B,N,A]: tensor of actions the agent took. Usually
        the actions from the learnt policy network.
    other_actions [[T,B,N,A]]: tensor of actions the agent took. Usually
        the actions from the replay buffer.
    """
    T, B, N, A = agent_actions.shape[:4]  # (B,N,A)
    all_joint_actions = []
    for i in range(N):  # type: ignore
        one_hot = tf.expand_dims(
            tf.cast(
                tf.stack([tf.stack([tf.one_hot(i, N)] * B, axis=0)] * T, axis=0), "bool"
            ),  # type: ignore
            axis=-1,
        )
        joint_action = tf.where(one_hot, agent_actions, other_actions)
        joint_action = tf.reshape(joint_action, (T, B, N * A))  # type: ignore
        all_joint_actions.append(joint_action)
    all_joint_actions = tf.stack(all_joint_actions, axis=2)

    return all_joint_actions


class MADDPGCQLSystem:

    """Rec-MADDPG with CQL System"""

    def __init__(
        self,
        environment,
        logger,
        priority_on_ramp=10_000,
        gaussian_steepness=2,
        min_priority=0.001,
        linear_layer_dim: int = 64,
        recurrent_layer_dim: int = 64,
        discount: float = 0.99,
        target_update_rate: float = 0.005,
        critic_learning_rate: float = 1e-3,
        policy_learning_rate: float = 3e-4,
        add_agent_id_to_obs: bool = False,
        random_exploration_timesteps: int = 0,
        num_ood_actions: int = 10,  # CQL
        cql_weight: float = 3.0,  # CQL
        cql_sigma: float = 0.2,  # CQL
        is_omiga: bool = False,  # is an omiga dataset
    ):
        self._environment = environment
        self._agents = environment.possible_agents
        self._logger = logger
        self._discount = discount
        self._add_agent_id_to_obs = add_agent_id_to_obs
        self._env_step_ctr = 0.0

        self._linear_layer_dim = linear_layer_dim
        self._recurrent_layer_dim = recurrent_layer_dim

        # Policy network
        self._policy_network = snt.DeepRNN(
            [
                snt.Linear(linear_layer_dim),
                tf.nn.relu,
                snt.GRU(recurrent_layer_dim),
                tf.nn.relu,
                snt.Linear(self._environment._num_actions),
                tf.nn.tanh,
            ]
        )  # shared network for all agents

        # Target policy network
        self._target_policy_network = copy.deepcopy(self._policy_network)

        # Critic network
        self._critic_network_1 = StateAndJointActionCritic(
            len(self._environment.possible_agents), self._environment._num_actions
        )  # shared network for all agents
        self._critic_network_2 = copy.deepcopy(self._critic_network_1)

        # Target critic network
        self._target_critic_network_1 = copy.deepcopy(self._critic_network_1)
        self._target_critic_network_2 = copy.deepcopy(self._critic_network_1)
        self._target_update_rate = target_update_rate

        # Optimizers
        self._critic_optimizer = snt.optimizers.RMSProp(
            learning_rate=critic_learning_rate
        )
        self._policy_optimizer = snt.optimizers.RMSProp(
            learning_rate=policy_learning_rate
        )

        # Exploration
        self._random_exploration_timesteps = tf.Variable(
            tf.constant(random_exploration_timesteps)
        )

        # Reset the recurrent neural network
        self._rnn_states = {
            agent: self._policy_network.initial_state(1)
            for agent in self._environment.possible_agents
        }

        # PER
        self.priority_on_ramp = priority_on_ramp
        self.gaussian_steepness = gaussian_steepness
        self.min_priority = min_priority

        # CQL
        self._num_ood_actions = num_ood_actions
        self._cql_weight = cql_weight
        self._cql_sigma = cql_sigma

        # Is an OMIGA dataset
        self._is_omiga = is_omiga

    def train_offline(
        self,
        replay_buffer,
        max_trainer_steps: int = int(1e5),
        evaluate_every: int = 1000,
        num_eval_episodes: int = 4,
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

            start_time = time.time()
            data_batch = replay_buffer.sample()
            end_time = time.time()
            time_to_sample = end_time - start_time

            start_time = time.time()
            train_logs, priority = self.train_step(
                data_batch.experience, trainer_step_ctr
            )
            end_time = time.time()
            time_train_step = end_time - start_time

            if isinstance(replay_buffer, PrioritisedFlashbaxReplayBuffer):
                start_time = time.time()
                replay_buffer.update_priorities(data_batch.indices, priority)
                end_time = time.time()
                time_priority = end_time - start_time
                train_logs["Priority Update Time"] = time_priority
            else:
                time_priority = 0

            train_steps_per_second = 1 / (
                time_train_step + time_to_sample + time_priority
            )

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
        eval_logs = self.evaluate(num_eval_episodes * 10)
        self._logger.write(
            eval_logs | {"Trainer Steps (eval)": trainer_step_ctr}, force=True
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
    ) -> Dict[str, np.ndarray]:
        actions, next_rnn_states = self._tf_select_actions(
            observations, self._rnn_states
        )
        self._rnn_states = next_rnn_states
        return tree.map_structure(  # type: ignore
            lambda x: x[0].numpy(), actions
        )  # convert to numpy and squeeze batch dim

    @tf.function(jit_compile=True)
    def _tf_select_actions(
        self,
        observations,
        rnn_states,
    ):
        actions = {}
        next_rnn_states = {}
        for i, agent in enumerate(self._environment.possible_agents):
            agent_observation = observations[agent]
            if self._add_agent_id_to_obs:
                agent_observation = concat_agent_id_to_obs(
                    agent_observation,
                    i,
                    len(self._environment.possible_agents),
                    at_back=self._is_omiga,
                )
                if self._is_omiga:
                    agent_observation = (
                        agent_observation - tf.reduce_mean(agent_observation)
                    ) / tf.math.reduce_std(agent_observation)
            agent_observation = tf.expand_dims(
                agent_observation, axis=0
            )  # add batch dimension
            action, next_rnn_states[agent] = self._policy_network(
                agent_observation, rnn_states[agent]
            )

            # Store agent action
            actions[agent] = action

        return actions, next_rnn_states

    def evaluate(self, num_eval_episodes: int = 4) -> Dict[str, Numeric]:
        """Method to evaluate the system online (i.e. in the environment)."""
        episode_returns = []
        for _ in range(num_eval_episodes):
            self.reset()
            observations_ = self._environment.reset()

            if isinstance(observations_, tuple):
                observations, infos = observations_
            else:
                observations = observations_
                infos = {}

            done = False
            episode_return = 0.0
            while not done:
                actions = self.select_actions(observations)

                (
                    observations,
                    rewards,
                    terminals,
                    truncations,
                    infos,
                ) = self._environment.step(actions)
                episode_return += np.mean(list(rewards.values()), dtype="float")
                done = all(terminals.values()) or all(truncations.values())
            episode_returns.append(episode_return)
        logs = {"evaluator/episode_return": np.mean(episode_returns)}
        return logs

    def train_step(self, experience, trainer_step_ctr) -> Dict[str, Numeric]:
        trainer_step_ctr = tf.convert_to_tensor(trainer_step_ctr, "float32")

        logs, new_priorities = self._tf_train_step(experience, trainer_step_ctr)

        return logs, new_priorities

    @tf.function(jit_compile=True)
    def _tf_train_step(self, experience, train_steps):
        # Unpack the batch
        observations = experience["observations"]  # (B,T,N,O)
        actions = tf.clip_by_value(experience["actions"], -1, 1)  # (B,T,N,A)
        env_states = experience["infos"]["state"]  # (B,T,S)
        rewards = experience["rewards"]  # (B,T,N)
        truncations = tf.cast(experience["truncations"], "float32")  # (B,T,N)
        terminals = tf.cast(experience["terminals"], "float32")  # (B,T,N)

        # When to reset the RNN hidden state
        resets = tf.maximum(terminals, truncations)  # equivalent to logical 'or'

        # Get dims
        B, T, N = actions.shape[:3]

        # Maybe add agent ids to observation
        if self._add_agent_id_to_obs and not self._is_omiga:
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
        target_actions = expand_batch_and_agent_dim_of_time_major_sequence(
            target_actions, B, N
        )

        # Target critics
        target_qs_1 = self._target_critic_network_1(
            env_states, target_actions, target_actions
        )
        target_qs_2 = self._target_critic_network_2(
            env_states, target_actions, target_actions
        )

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
            online_actions = expand_batch_and_agent_dim_of_time_major_sequence(
                online_actions, B, N
            )

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
            repeat_env_states = tf.reshape(
                repeat_env_states, (T, -1, *repeat_env_states.shape[3:])
            )
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
            random_ood_action_log_pi = tf.math.log(
                0.5 ** (random_ood_actions.shape[-1])
            )

            ood_qs_1 = (
                self._critic_network_1(
                    repeat_env_states, random_ood_actions, random_ood_actions
                )[:-1]
                - random_ood_action_log_pi
            )
            ood_qs_2 = (
                self._critic_network_2(
                    repeat_env_states, random_ood_actions, random_ood_actions
                )[:-1]
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
            current_ood_actions = tf.clip_by_value(
                repeat_online_actions + action_noise, -1.0, 1.0
            )

            ood_actions_prob = (
                1 / (self._cql_sigma * tf.math.sqrt(2 * np.pi))
            ) * tf.exp(-((action_noise - mu) ** 2) / (2 * self._cql_sigma**2))
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
            current_ood_qs_1 = tf.reshape(
                current_ood_qs_1, (T - 1, B, self._num_ood_actions, N)
            )
            current_ood_qs_2 = tf.reshape(
                current_ood_qs_2, (T - 1, B, self._num_ood_actions, N)
            )
            next_current_ood_qs_1 = tf.reshape(
                next_current_ood_qs_1, (T - 1, B, self._num_ood_actions, N)
            )
            next_current_ood_qs_2 = tf.reshape(
                next_current_ood_qs_2, (T - 1, B, self._num_ood_actions, N)
            )

            all_ood_qs_1 = tf.concat(
                (ood_qs_1, current_ood_qs_1, next_current_ood_qs_1), axis=2
            )
            all_ood_qs_2 = tf.concat(
                (ood_qs_2, current_ood_qs_2, next_current_ood_qs_2), axis=2
            )

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
            online_actions = unroll_rnn(
                self._policy_network,
                merge_batch_and_agent_dim_of_time_major_sequence(observations),
                merge_batch_and_agent_dim_of_time_major_sequence(resets),
            )
            online_actions = expand_batch_and_agent_dim_of_time_major_sequence(
                online_actions, B, N
            )

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

        for src, dest in zip(online_variables, target_variables):
            dest.assign(
                dest * (1.0 - self._target_update_rate) + src * self._target_update_rate
            )

        del tape

        # Compute new replay priorities
        A = replay_actions.shape[-1]
        joint_replay_action = tf.reshape(replay_actions, (T, B, N * A))

        # Target joint action
        joint_target_actions = tf.reshape(target_actions, (T, B, N * A))

        ## Compute distance
        distance = tf.reduce_mean(
            tf.abs(joint_target_actions - joint_replay_action), axis=-1
        )

        # Aggregate across time
        sequence_distance = tf.reduce_mean(distance, axis=0)

        sequence_distance = tf.maximum(self.min_priority, sequence_distance)

        ## Priority
        priority_on_ramp = tf.minimum(1.0, train_steps * (1 / self.priority_on_ramp))
        priority = tf.exp(
            -((self.gaussian_steepness * priority_on_ramp * sequence_distance) ** 2)
        )
        # priority = tf.clip_by_value(priority, self.min_priority, 1.0)

        # priority = 1 / (sequence_distance * self.gaussian_steepness)

        logs = {
            "Mean Q-values": tf.reduce_mean((qs_1 + qs_2) / 2),
            "Mean Critic Loss": (critic_loss),
            "Policy Loss": policy_loss,
            "Max Priority": tf.reduce_max(priority),
            "Mean Priority": tf.reduce_mean(priority),
            "Min Priority": tf.reduce_min(priority),
            "mean action distance": tf.reduce_mean(distance),
            "max action distance": tf.reduce_max(distance),
            "min action distance": tf.reduce_min(distance),
            "mean sequence distance": tf.reduce_mean(sequence_distance),
            "max sequence distance": tf.reduce_max(sequence_distance),
            "min sequence distance": tf.reduce_min(sequence_distance),
        }

        return logs, priority
