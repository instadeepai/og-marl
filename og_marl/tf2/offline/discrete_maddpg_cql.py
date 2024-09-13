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

"""Implementation discrete action MADDPG+CQL"""
from typing import Any, Dict, Optional, Tuple
import copy

import hydra
import numpy as np
import sonnet as snt
from omegaconf import DictConfig
import tensorflow_probability as tfp
import tensorflow as tf
from tensorflow import Tensor
import tree
from chex import Numeric

from og_marl.environments import get_environment, BaseEnvironment
from og_marl.tf2.networks import StateAndJointActionCritic
from og_marl.tf2.offline.base import BaseOfflineSystem
from og_marl.loggers import BaseLogger, WandbLogger
from og_marl.vault_utils.download_vault import download_and_unzip_vault
from og_marl.replay_buffers import Experience, FlashbaxReplayBuffer
from og_marl.tf2.utils import (
    batch_concat_agent_id_to_obs,
    concat_agent_id_to_obs,
    expand_batch_and_agent_dim_of_time_major_sequence,
    merge_batch_and_agent_dim_of_time_major_sequence,
    set_growing_gpu_memory,
    switch_two_leading_dims,
    unroll_rnn,
)

set_growing_gpu_memory()

tfd = tfp.distributions

def replace_gradient(value, surrogate):
    """Returns `value` but backpropagates gradients through `surrogate`."""
    return tf.cast(surrogate + tf.stop_gradient(value - surrogate), "float32")


class DiscreteGradientEstimator(snt.Module):

    def __init__(self, temperature, gap=1.0):
        self.temperature = temperature
        self.gap = gap
        super().__init__()

    def _calculate_movements(self, logits, one_hot_action):
        max_logit = tf.reduce_max(logits, axis=-1, keepdims=True)
        selected_logit = tf.expand_dims(tf.gather_nd(logits, indices=tf.expand_dims(tf.argmax(one_hot_action, axis=-1), axis=-1), batch_dims=3), axis=-1)
        m1 = (max_logit - selected_logit) * one_hot_action
        m2 = tf.maximum(logits + self.gap - max_logit, 0.0) * (1 - one_hot_action)
        return m1, m2

    def __call__(self, logits, need_gradients=True):
        one_hot_action = tf.cast(tfd.OneHotCategorical(logits=logits).sample(), "float32")
        if need_gradients:
            m1, m2 = self._calculate_movements(logits, one_hot_action)
            m1, m2 = tf.stop_gradient(m1), tf.stop_gradient(m2)
            surrogate = tf.nn.softmax((logits + m1 - m2) / self.temperature, axis=-1)
            return replace_gradient(one_hot_action, surrogate)
        else:
            return tf.cast(one_hot_action, "float32")


class DiscreteMADDPGCQLSystem(BaseOfflineSystem):
    """Multi-Agent Deep Deterministic Policy Gradients with CQL."""

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
        add_agent_id_to_obs: bool = True,
        num_ood_actions: int = 10,  # CQL
        cql_weight: float = 3.0,  # CQL
        cql_sigma: float = 0.2,  # CQL
    ):
        super().__init__(environment=environment, logger=logger)

        self.add_agent_id_to_obs = add_agent_id_to_obs
        self.discount = discount

        # Policy network
        self.policy_network = snt.DeepRNN(
            [
                snt.Linear(linear_layer_dim),
                tf.nn.relu,
                snt.GRU(recurrent_layer_dim),
                tf.nn.relu,
                snt.Linear(self.environment.num_actions),
            ]
        )  # shared network for all agents

        # Target policy network
        self.target_policy_network = copy.deepcopy(self.policy_network)

        self.gradient_estimator = DiscreteGradientEstimator(temperature=0.7)

        # Critic network
        self.critic_network_1 = StateAndJointActionCritic(
            len(self.environment.agents), self.environment.num_actions
        )  # shared network for all agents
        self.critic_network_2 = copy.deepcopy(self.critic_network_1)

        # Target critic network
        self.target_critic_network_1 = copy.deepcopy(self.critic_network_1)
        self.target_critic_network_2 = copy.deepcopy(self.critic_network_1)
        self.target_update_rate = target_update_rate

        # Optimizers
        self.critic_optimizer = snt.optimizers.Adam(learning_rate=critic_learning_rate)
        self.policy_optimizer = snt.optimizers.Adam(learning_rate=policy_learning_rate)

        # Reset the recurrent neural network
        self.rnn_states = {
            agent: self.policy_network.initial_state(1) for agent in self.environment.agents
        }

        # CQL
        self.num_ood_actions = num_ood_actions
        self.cql_weight = cql_weight
        self.cql_sigma = cql_sigma

    def reset(self) -> None:
        """Called at the start of a new episode."""
        # Reset the recurrent neural network
        self.rnn_states = {
            agent: self.policy_network.initial_state(1) for agent in self.environment.agents
        }
        return

    def select_actions(
        self,
        observations: Dict[str, np.ndarray],
        legal_actions: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, np.ndarray]:
        actions, next_rnn_states = self._tf_select_actions(observations, self.rnn_states, legal_actions)
        self.rnn_states = next_rnn_states
        return tree.map_structure(  # type: ignore
            lambda x: x[0].numpy(), actions
        )  # convert to numpy and squeeze batch dim

    @tf.function(jit_compile=True)
    def _tf_select_actions(
        self,
        observations: Dict[str, Tensor],
        rnn_states: Dict[str, Tensor],
        legals: Dict[str, Tensor],
    ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        actions = {}
        next_rnn_states = {}
        for i, agent in enumerate(self.environment.agents):
            agent_observation = observations[agent]
            if self.add_agent_id_to_obs:
                agent_observation = concat_agent_id_to_obs(
                    agent_observation, i, len(self.environment.agents)
                )
            agent_observation = tf.expand_dims(agent_observation, axis=0)  # add batch dimension
            logits, next_rnn_states[agent] = self.policy_network(
                agent_observation, rnn_states[agent]
            )

            logits = tf.where(legals[agent]==1.0, logits, -np.inf)
            action = self.gradient_estimator(logits, need_gradients=False)

            # Store agent action
            actions[agent] = tf.argmax(action, axis=-1)

        return actions, next_rnn_states

    def train_step(self, experience: Experience) -> Dict[str, Numeric]:
        logs = self._tf_train_step(experience)
        return logs  # type: ignore

    @tf.function(jit_compile=True)  # NOTE: comment this out if using debugger
    def _tf_train_step(self, experience: Dict[str, Any]) -> Dict[str, Numeric]:
        # Unpack the batch
        observations = experience["observations"]  # (B,T,N,O)
        actions = experience["actions"]  # (B,T,N) 
        env_states = experience["infos"]["state"]  # (B,T,S)
        legals = experience["infos"]["legals"]  # (B,T,S,A)
        rewards = experience["rewards"]  # (B,T,N)
        truncations = tf.cast(experience["truncations"], "float32")  # (B,T,N)
        terminals = tf.cast(experience["terminals"], "float32")  # (B,T,N)

        # When to reset the RNN hidden state
        resets = tf.maximum(terminals, truncations)  # equivalent to logical 'or'

        # Get dims
        B, T, N, A = legals.shape[:4]

        # Maybe add agent ids to observation
        if self.add_agent_id_to_obs:
            observations = batch_concat_agent_id_to_obs(observations)

        one_hot_actions = tf.one_hot(actions, self.environment.num_actions, axis=-1)

        # Make time-major
        observations = switch_two_leading_dims(observations)
        one_hot_actions = switch_two_leading_dims(one_hot_actions)
        rewards = switch_two_leading_dims(rewards)
        terminals = switch_two_leading_dims(terminals)
        env_states = switch_two_leading_dims(env_states)
        resets = switch_two_leading_dims(resets)
        legals = switch_two_leading_dims(legals)

        # Unroll target policy
        target_logits = unroll_rnn(
            self.target_policy_network,
            merge_batch_and_agent_dim_of_time_major_sequence(observations),
            merge_batch_and_agent_dim_of_time_major_sequence(resets),
        )
        target_logits = expand_batch_and_agent_dim_of_time_major_sequence(target_logits, B, N)
        target_logits = tf.where(legals==1.0, target_logits, -np.inf)
        target_actions = self.gradient_estimator(target_logits, need_gradients=False)

        # Target critics
        target_qs_1 = self.target_critic_network_1(env_states, target_actions, target_actions)
        target_qs_2 = self.target_critic_network_2(env_states, target_actions, target_actions)

        # Take minimum between two target critics
        target_qs = tf.minimum(target_qs_1, target_qs_2)

        # Compute Bellman targets
        targets = rewards[:-1] + self.discount * (1 - terminals[:-1]) * tf.squeeze(
            target_qs[1:], axis=-1
        )

        # Do forward passes through the networks and calculate the losses
        with tf.GradientTape(persistent=True) as tape:
            # Online critics
            qs_1 = tf.squeeze(
                self.critic_network_1(env_states, one_hot_actions, one_hot_actions),
                axis=-1,
            )
            qs_2 = tf.squeeze(
                self.critic_network_2(env_states, one_hot_actions, one_hot_actions),
                axis=-1,
            )

            # Squared TD-error
            td_error_1 = tf.reduce_mean(0.5 * (targets - qs_1[:-1]) ** 2)
            td_error_2 = tf.reduce_mean(0.5 * (targets - qs_2[:-1]) ** 2)

            ###########
            ### CQL ###
            ###########

            online_logits = unroll_rnn(
                self.policy_network,
                merge_batch_and_agent_dim_of_time_major_sequence(observations),
                merge_batch_and_agent_dim_of_time_major_sequence(resets),
            )
            online_logits = expand_batch_and_agent_dim_of_time_major_sequence(online_logits, B, N)
            online_logits = tf.where(legals==1.0, online_logits, -np.inf)
            online_one_hot_actions = self.gradient_estimator(online_logits, need_gradients=True)

            # Sample legal random actions
            repeated_legals = tf.stack([legals] * self.num_ood_actions, axis=0)
            repeated_legals = tf.reshape(repeated_legals, (-1, A))
            random_ood_actions = tf.random.categorical(
                repeated_legals / tf.reduce_sum(repeated_legals, axis=-1, keepdims=True),
                1,
                dtype="int32",
            )
            random_ood_actions = tf.reshape(random_ood_actions, (self.num_ood_actions, T, B, N))
            random_ood_actions_one_hot = tf.cast(tf.one_hot(random_ood_actions, A, axis=-1), "float32")

            all_ood_qs_1 = []
            all_ood_qs_2 = []
            for i in range(self.num_ood_actions):
                ood_qs_1 = self.critic_network_1(env_states, random_ood_actions_one_hot[i], random_ood_actions_one_hot[i])
                ood_qs_2 = self.critic_network_2(env_states, random_ood_actions_one_hot[i], random_ood_actions_one_hot[i])

                all_ood_qs_1.append(ood_qs_1)
                all_ood_qs_2.append(ood_qs_2)

            all_ood_qs_1.append(tf.expand_dims(qs_1, axis=-1))
            all_ood_qs_2.append(tf.expand_dims(qs_2, axis=-1))

            all_ood_qs_1 = tf.concat(all_ood_qs_1, axis=-1)
            all_ood_qs_2 = tf.concat(all_ood_qs_2, axis=-1)

            cql_loss_1 = tf.reduce_mean(
                tf.reduce_logsumexp(all_ood_qs_1, axis=-1, keepdims=True)
            ) - tf.reduce_mean(qs_1)
            cql_loss_2 = tf.reduce_mean(
                tf.reduce_logsumexp(all_ood_qs_2, axis=-1, keepdims=True)
            ) - tf.reduce_mean(qs_2)

            critic_loss_1 = td_error_1 + self.cql_weight * cql_loss_1
            critic_loss_2 = td_error_2 + self.cql_weight * cql_loss_2
            critic_loss = (critic_loss_1 + critic_loss_2) / 2

            ### END CQL ###

            # Policy Loss
            policy_qs_1 = self.critic_network_1(env_states, online_one_hot_actions, one_hot_actions)
            policy_qs_2 = self.critic_network_2(env_states, online_one_hot_actions, one_hot_actions)
            policy_qs = tf.minimum(policy_qs_1, policy_qs_2)

            policy_loss = -tf.reduce_mean(policy_qs)

        # Train critics
        variables = (
            *self.critic_network_1.trainable_variables,
            *self.critic_network_2.trainable_variables,
        )
        gradients = tape.gradient(critic_loss, variables)
        self.critic_optimizer.apply(gradients, variables)

        # Train policy
        variables = (*self.policy_network.trainable_variables,)
        gradients = tape.gradient(policy_loss, variables)
        self.policy_optimizer.apply(gradients, variables)

        # Update target networks
        online_variables = (
            *self.critic_network_1.variables,
            *self.critic_network_2.variables,
            *self.policy_network.variables,
        )
        target_variables = (
            *self.target_critic_network_1.variables,
            *self.target_critic_network_2.variables,
            *self.target_policy_network.variables,
        )

        # Soft target update
        tau = self.target_update_rate
        for src, dest in zip(online_variables, target_variables):
            dest.assign(dest * (1.0 - tau) + src * tau)

        del tape

        logs = {
            "mean_dataset_q_values": tf.reduce_mean((qs_1 + qs_2) / 2),
            "critic_loss": critic_loss,
            "td_loss": (td_error_1 + td_error_2) / 2,
            "cql_loss": (cql_loss_1 + cql_loss_2) / 2.0,
            "policy_loss": policy_loss,
            "mean_chosen_q_values": tf.reduce_mean((policy_qs_1 + policy_qs_2) / 2),
        }

        return logs


@hydra.main(version_base=None, config_path="configs", config_name="discrete_maddpg_cql")
def run_experiment(cfg: DictConfig) -> None:
    print(cfg)

    env = get_environment(
        cfg["task"]["source"], cfg["task"]["env"], cfg["task"]["scenario"], seed=cfg["seed"]
    )

    buffer = FlashbaxReplayBuffer(
        sequence_length=cfg["replay"]["sequence_length"],
        sample_period=cfg["replay"]["sample_period"],
        seed=cfg["seed"],
    )

    download_and_unzip_vault(cfg["task"]["source"], cfg["task"]["env"], cfg["task"]["scenario"])

    buffer.populate_from_vault(
        cfg["task"]["source"], cfg["task"]["env"], cfg["task"]["scenario"], cfg["task"]["dataset"]
    )

    wandb_config = {
        "system": cfg["system_name"],
        "seed": cfg["seed"],
        "training_steps": cfg["training_steps"],
        **cfg["task"],
        **cfg["replay"],
        **cfg["system"],
    }
    logger = WandbLogger(project=cfg["wandb_project"], config=wandb_config)

    system = DiscreteMADDPGCQLSystem(env, logger, **cfg["system"])

    tf.random.set_seed(cfg["seed"])

    system.train(buffer, training_steps=int(cfg["training_steps"]))


if __name__ == "__main__":
    run_experiment()
