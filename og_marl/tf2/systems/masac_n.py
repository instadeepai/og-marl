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

"""Implementation of ISACN"""
from typing import Any, Dict, Optional, Tuple

import random
import copy
import hydra
import numpy as np
from omegaconf import DictConfig
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import Tensor
import sonnet as snt
import tree
from chex import Numeric

from og_marl.environments import get_environment, BaseEnvironment
from og_marl.loggers import BaseLogger, WandbLogger
from og_marl.vault_utils.download_vault import download_and_unzip_vault
from og_marl.replay_buffers import Experience, FlashbaxReplayBuffer
from og_marl.tf2.networks import StateAndActionCritic
from og_marl.tf2.systems.base import BaseOfflineSystem
from og_marl.tf2.utils import (
    batch_concat_agent_id_to_obs,
    concat_agent_id_to_obs,
    expand_batch_and_agent_dim_of_time_major_sequence,
    merge_batch_and_agent_dim_of_time_major_sequence,
    set_growing_gpu_memory,
    switch_two_leading_dims,
    unroll_rnn,
)

tfd = tfp.distributions

set_growing_gpu_memory()

    
class Actor(snt.Module):
    def __init__(
        self,
        action_dim: int,
    ):
        super().__init__()
        self.action_dim = action_dim

        self.base_network = snt.Sequential(
            [
                snt.Linear(256),
                tf.nn.relu,
                snt.Linear(256),
                tf.nn.relu,
                snt.Linear(256),
                tf.nn.relu,
            ]
        )

        self.mu = snt.Linear(self.action_dim)
        self.log_sigma = snt.Linear(self.action_dim)

    def __call__(
        self,
        observations: tf.Tensor,
        deterministic: bool = False,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        
        hidden = self.base_network(observations)

        mu = self.mu(hidden)
        log_sigma = self.log_sigma(hidden)

        log_sigma = tf.clip_by_value(log_sigma, -5, 2)
        policy_dist = tfd.Normal(mu, tf.exp(log_sigma))

        action = policy_dist.sample()
        tanh_action = tf.tanh(action)

        log_prob = tf.reduce_sum(policy_dist.log_prob(action), axis=-1)
        log_prob = log_prob - tf.reduce_sum(tf.math.log(1 - tf.pow(tanh_action, 2) + 1e-6), axis=-1)

        return tanh_action, log_prob
    
class CriticNetwork(snt.Module):
    def __init__(
        self,
        action_dim: int,
        orthogonal_init: bool = False,
        n_hidden_layers: int = 3,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.orthogonal_init = orthogonal_init

        layers = []
        for _ in range(n_hidden_layers):
            layers.append(snt.Linear(256, w_init=snt.initializers.RandomUniform(-3e-3, 3e-3), b_init=snt.initializers.RandomUniform(-3e-3, 3e-3)))
            layers.append(tf.nn.relu)
        layers.append(snt.Linear(1, w_init=snt.initializers.RandomUniform(-3e-3, 3e-3), b_init=snt.initializers.RandomUniform(-3e-3, 3e-3)))

        self.network = snt.Sequential(layers)

    def make_joint_action(self, agent_actions: Tensor, other_actions: Tensor) -> Tensor:
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
                tf.cast(tf.stack([tf.stack([tf.one_hot(i, N)] * B, axis=0)] * T, axis=0), "bool"),  # type: ignore
                axis=-1,
            )
            joint_action = tf.where(one_hot, agent_actions, other_actions)
            joint_action = tf.reshape(joint_action, (T, B, N * A))  # type: ignore
            all_joint_actions.append(joint_action)
        all_joint_actions: Tensor = tf.stack(all_joint_actions, axis=2)

        return all_joint_actions

    def __call__(self, observations: tf.Tensor, actions: tf.Tensor, other_actions: tf.Tensor) -> tf.Tensor:
        joint_actions = self.make_joint_action(actions, tf.stop_gradient(other_actions))
        input_tensor = tf.concat([observations, joint_actions], axis=-1)
        q_values = tf.squeeze(self.network(input_tensor),axis=-1)
        return q_values


class ISACNSystem(BaseOfflineSystem):
    """Independent SAC + CQL.

    NOTE: the critic conditions on states and individual agent actions.

    """

    def __init__(
        self,
        environment: BaseEnvironment,
        logger: BaseLogger,
        discount: float = 0.99,
        num_critics: int = 100,
        critic_learning_rate: float = 3e-4,
        policy_learning_rate: float = 3e-4,
        alpha_learning_rate: float = 3e-4,
        target_update_rate: float = 0.005,
        add_agent_id_to_obs: bool = True,
        use_automatic_entropy_tuning: bool = True
    ):
        super().__init__(
            environment=environment,
            logger=logger,
        )

        self.discount = discount
        self.target_entropy = -self.environment.num_actions
        self.num_critics = num_critics
        self.add_agent_id_to_obs = add_agent_id_to_obs
        self.target_update_rate = target_update_rate
        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning

        # Policy network
        self.actor = Actor(environment.num_actions)

        # Critic network, shared network for all agents
        self.critics = [CriticNetwork(self.environment.num_actions) for _ in range(self.num_critics)]
        self.target_critics = copy.deepcopy(self.critics)

        # Optimizers
        self.critic_optimizer = snt.optimizers.Adam(learning_rate=critic_learning_rate)
        self.policy_optimizer = snt.optimizers.Adam(learning_rate=policy_learning_rate)

        # Entropy
        if self.use_automatic_entropy_tuning:
            self.log_alpha = tf.Variable(tf.zeros((len(self.agents),), "float32"))
            self.alpha_optimizer = snt.optimizers.Adam(
                learning_rate=alpha_learning_rate,
            )
        else:
            self.log_alpha = None

    def reset(self) -> None:
        """Called at the start of a new episode."""
        return

    def select_actions(
        self,
        observations: Dict[str, np.ndarray],
        legal_actions: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, np.ndarray]:
        actions = self._tf_select_actions(observations)
        return tree.map_structure(  # type: ignore
            lambda x: x[0].numpy(), actions
        )  # convert to numpy and squeeze batch dim

    @tf.function(jit_compile=True)
    def _tf_select_actions(
        self,
        observations: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        actions = {}
        for i, agent in enumerate(self.environment.agents):
            agent_observation = observations[agent]
            if self.add_agent_id_to_obs:
                agent_observation = concat_agent_id_to_obs(
                    agent_observation, i, len(self.environment.agents)
                )
            agent_observation = tf.expand_dims(agent_observation, axis=0)  # add batch dimension
            action, _ = self.actor(
                agent_observation, deterministic=True
            )

            # Store agent action
            actions[agent] = action

        return actions

    def train_step(self, experience: Experience) -> Dict[str, Numeric]:
        logs = self._tf_train_step(experience)
        return logs  # type: ignore

    @tf.function(jit_compile=True)  # NOTE: comment this out if using debugger
    def _tf_train_step(self, experience: Dict[str, Any]) -> Dict[str, Numeric]:
        # Unpack the batch
        observations_ = tf.cast(experience["observations"], "float32") # (B,T,N,O)
        replay_actions = tf.cast(
            tf.clip_by_value(experience["actions"], -1.0, 1.0), "float32"
        ) # (B,T,N,A)
        env_states_ = tf.cast(experience["infos"]["state"], "float32") # (B,T,S)
        rewards = tf.cast(experience["rewards"], "float32") # (B,T,N)
        terminals = tf.cast(experience["terminals"], "float32") # (B,T,N)

        # Get dims
        B, T, N = replay_actions.shape[:3]

        # Repeat states for each agent
        env_states_ = tf.stack([env_states_]*N, axis=2)

        # Maybe add agent ids to observation
        if self.add_agent_id_to_obs:
            observations_ = batch_concat_agent_id_to_obs(observations_)
            env_states_ = batch_concat_agent_id_to_obs(env_states_)

        observations = observations_[:,:-1]
        next_observations = observations_[:,1:]
        env_states = env_states_[:,:-1]
        next_env_states = env_states_[:,1:]
        rewards = rewards[:,:-1]
        replay_actions = replay_actions[:,:-1]
        terminals = terminals[:,:-1]

        # Target actions
        target_actions, target_action_log_prob = self.actor(next_observations)

        # Target critics
        target_qs = []
        for target_critic in self.target_critics:
            target_q = target_critic(next_env_states, target_actions, target_actions)
            target_qs.append(target_q)

        # Take minimum between all target critics
        target_qs = tf.reduce_min(target_qs, axis=0)

        # Entropy
        alpha = tf.reshape(tf.exp(self.log_alpha), (1,1,*self.log_alpha.shape))
        target_qs = target_qs - alpha * target_action_log_prob

        # Compute Bellman targets
        targets = rewards + self.discount * (1 - terminals) * target_qs

        # Do forward passes through the networks and calculate the losses
        with tf.GradientTape(persistent=True) as tape:

            ###################
            ### Critic Loss ###
            ###################

            critic_losses = []
            in_dist_qs = []
            for critic in self.critics:
                # Critic
                qs = critic(env_states, replay_actions, replay_actions)
                in_dist_qs.append(qs)

                # Squared TD-error
                critic_losses.append(tf.reduce_mean(0.5 * (targets - qs) ** 2, axis=0)) # mean across batch
            critic_loss = tf.reduce_sum(critic_losses) # sum across critics and agents

            ###################
            ### Policy Loss ###
            ###################
            online_actions, online_log_probs = self.actor(observations)

            policy_qs = []
            for critic in self.critics:
                policy_qs.append(critic(env_states, online_actions, replay_actions))
            policy_qs = tf.reduce_min(policy_qs, axis=0)

            policy_qs = tf.reduce_mean(policy_qs, axis=0) # mean across batch
            policy_qs = tf.reduce_sum(policy_qs) # sum across agent

            alpha = tf.reshape(tf.exp(self.log_alpha), (1,1,*self.log_alpha.shape))
            entropy = alpha * online_log_probs
            entropy = tf.reduce_mean(entropy, axis=0)
            entropy = tf.reduce_sum(entropy)

            policy_loss = entropy - policy_qs
            # policy_loss = tf.reduce_mean((online_actions-replay_actions)**2)

            ###################
            ### Alpha Loss ###
            ###################

            alpha_loss = (-self.log_alpha * (tf.stop_gradient(online_log_probs) + self.target_entropy))
            alpha_loss = tf.reduce_mean(alpha_loss, axis=0)
            alpha_loss = tf.reduce_sum(alpha_loss)

        # Train critics
        variables = []
        for critic in self.critics:
            variables += [*critic.trainable_variables,]
        gradients = tape.gradient(critic_loss, variables)
        self.critic_optimizer.apply(gradients, variables)

        # Train actor
        variables = (*self.actor.trainable_variables,)
        gradients = tape.gradient(policy_loss, variables)
        self.policy_optimizer.apply(gradients, variables)

        # Train alpha
        variables = (self.log_alpha,)
        gradients = tape.gradient(alpha_loss, variables)
        self.alpha_optimizer.apply(gradients, variables)

        # Update target networks
        online_variables = []
        target_variables = []
        for i in range(self.num_critics):
            online_variables += [*self.critics[i].variables,]
            target_variables += [*self.target_critics[i].variables,]

        # Soft target update
        tau = self.target_update_rate
        for src, dest in zip(online_variables, target_variables):
            dest.assign(dest * (1.0 - tau) + src * tau)

        # Log standard deviation of Q-values for OOD actions
        rand_actions = tf.random.uniform(replay_actions.shape, -1.0, 1.0, "float32")
        q_rands = []
        for critic in self.critics:
            q_rands.append(critic(env_states, rand_actions, rand_actions))
        q_random_std = tf.reduce_mean(tf.math.reduce_std(q_rands,axis=0))

        in_dist_qs_std = tf.reduce_mean(tf.math.reduce_std(in_dist_qs,axis=0))

        del tape

        logs = {
            # "mean_dataset_q_values": tf.reduce_mean((qs_1 + qs_2) / 2),
            "critic_loss": critic_loss,
            "alpha_loss": alpha_loss,
            "alpha": tf.reduce_mean(tf.exp(self.log_alpha)),
            "ood_q_values_std": q_random_std,
            "in_dist_q_values_std": in_dist_qs_std,
            "mean_policy_q_values": tf.reduce_mean(policy_qs),
            # "cql_loss": (cql_loss_1 + cql_loss_2) / 2.0,
            "policy_loss": policy_loss,
            # "mean_chosen_q_values": tf.reduce_mean((policy_qs_1 + policy_qs_2) / 2),
        }

        return logs


@hydra.main(version_base=None, config_path="configs", config_name="masac_n")
def run_experiment(cfg: DictConfig) -> None:
    print(cfg)

    seed = random.randint(1, 10000)

    env = get_environment(cfg["task"]["source"], cfg["task"]["env"], cfg["task"]["scenario"], seed=seed)

    buffer = FlashbaxReplayBuffer(
        sequence_length=cfg["replay"]["sequence_length"],
        sample_period=cfg["replay"]["sample_period"],
        seed=seed,
    )

    download_and_unzip_vault(cfg["task"]["source"], cfg["task"]["env"], cfg["task"]["scenario"])

    buffer.populate_from_vault(cfg["task"]["source"], cfg["task"]["env"], cfg["task"]["scenario"], cfg["task"]["dataset"])

    wandb_config = {
        "system": cfg["system_name"],
        "seed": seed,
        "training_steps": cfg["training_steps"],
        **cfg["task"],
        **cfg["replay"],
        **cfg["system"],
    }
    logger = WandbLogger(project=cfg["wandb_project"], config=wandb_config)

    system = ISACNSystem(env, logger, **cfg["system"])

    tf.random.set_seed(seed)

    system.train(buffer, num_eval_episodes=10, training_steps=int(cfg["training_steps"]))


if __name__ == "__main__":
    run_experiment()
