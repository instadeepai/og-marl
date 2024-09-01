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

"""Implementation of MASAC+CQL"""
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
        B, N, A = agent_actions.shape[:3]  # (B,N,A)
        all_joint_actions = []
        for i in range(N):  # type: ignore
            one_hot = tf.expand_dims(
                tf.cast(tf.stack([tf.one_hot(i, N)] * B, axis=0), "bool"),  # type: ignore
                axis=-1,
            )
            joint_action = tf.where(one_hot, agent_actions, other_actions)
            joint_action = tf.reshape(joint_action, (B, N * A))  # type: ignore
            all_joint_actions.append(joint_action)
        all_joint_actions: Tensor = tf.stack(all_joint_actions, axis=1)

        return all_joint_actions

    def __call__(self, observations: tf.Tensor, actions: tf.Tensor, other_actions: tf.Tensor) -> tf.Tensor:
        joint_actions = self.make_joint_action(actions, tf.stop_gradient(other_actions))
        input_tensor = tf.concat([observations, joint_actions], axis=-1)
        q_values = tf.squeeze(self.network(input_tensor),axis=-1)
        return q_values


class MASACCQLSystem(BaseOfflineSystem):
    """MA-SAC + CQL.

    NOTE: the critic conditions on states and individual agent actions.

    """

    def __init__(
        self,
        environment: BaseEnvironment,
        logger: BaseLogger,
        discount: float = 0.99,
        num_critics: int = 2,
        critic_learning_rate: float = 3e-4,
        policy_learning_rate: float = 3e-4,
        alpha_learning_rate: float = 3e-4,
        alpha_prime_learning_rate: float = 3e-3,
        target_update_rate: float = 0.005,
        add_agent_id_to_obs: bool = True,
        use_automatic_entropy_tuning: bool = True,
        cql_n_actions=10,
        cql_temp=1.0,
        cql_target_action_gap=-1.0,
        cql_alpha=5.0
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

        # CQL
        self.cql_n_actions = cql_n_actions
        self.log_alpha_prime = tf.Variable(tf.zeros((len(self.agents),), "float32"))
        self.alpha_prime_optimizer = snt.optimizers.Adam(
            learning_rate=alpha_prime_learning_rate,
        )
        self.cql_temp = cql_temp
        self.cql_clip_diff_min = -np.inf
        self.cql_clip_diff_max = np.inf
        self.cql_target_action_gap = cql_target_action_gap
        self.cql_alpha = cql_alpha

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
        B, T, N, A = replay_actions.shape[:4]

        # Repeat states for each agent
        env_states_ = tf.stack([env_states_]*N, axis=2)

        # Maybe add agent ids to observation
        if self.add_agent_id_to_obs:
            observations_ = batch_concat_agent_id_to_obs(observations_)
            env_states_ = batch_concat_agent_id_to_obs(env_states_)

        observations = observations_[:,0]
        next_observations = observations_[:,1]
        env_states = env_states_[:,0]
        next_env_states = env_states_[:,1]
        rewards = rewards[:,0]
        replay_actions = replay_actions[:,0]
        terminals = terminals[:,0]

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

            td_loss = []
            in_dist_qs = []
            for critic in self.critics:
                # Critic
                qs = critic(env_states, replay_actions, replay_actions)
                in_dist_qs.append(qs)

                # Squared TD-error
                td_loss.append(tf.reduce_mean(0.5 * (targets - qs) ** 2, axis=0)) # mean across batch
            td_loss = tf.reduce_sum(td_loss) # sum across critics and agents

            # CQL
            cql_random_actions = tf.random.uniform((B,self.cql_n_actions,N,A), -1, 1)

            repeat_observations = tf.stack([observations]*self.cql_n_actions, axis=1)
            repeat_next_observations = tf.stack([next_observations]*self.cql_n_actions, axis=1)
            repeat_env_states = tf.stack([env_states]*self.cql_n_actions, axis=1)

            cql_current_actions, cql_current_log_pis = self.actor(
                repeat_observations
            )
            cql_next_actions, cql_next_log_pis = self.actor(
                repeat_next_observations
            )

            # Merge cql dim into batch dim
            repeat_env_states = tf.reshape(repeat_env_states, shape=(-1, *repeat_env_states.shape[2:]))
            cql_random_actions = tf.reshape(cql_random_actions, shape=(-1, *cql_random_actions.shape[2:]))
            cql_current_actions = tf.reshape(cql_current_actions, shape=(-1, *cql_current_actions.shape[2:]))
            cql_next_actions = tf.reshape(cql_next_actions, shape=(-1, *cql_next_actions.shape[2:]))

            cql_min_qf_loss = []
            for i, critic in enumerate(self.critics):
                cql_q_rand = critic(repeat_env_states, cql_random_actions, cql_random_actions)
                cql_q_current_actions = critic(repeat_env_states, cql_current_actions, cql_current_actions)
                cql_q_next_actions = critic(repeat_env_states, cql_next_actions, cql_next_actions)

                # Split cql and batch dims
                cql_q_rand = tf.reshape(cql_q_rand, shape=(B, self.cql_n_actions, *cql_q_rand.shape[1:]))
                cql_q_current_actions = tf.reshape(cql_q_current_actions, shape=(B, self.cql_n_actions, *cql_q_current_actions.shape[1:]))
                cql_q_next_actions = tf.reshape(cql_q_next_actions, shape=(B, self.cql_n_actions, *cql_q_next_actions.shape[1:]))

                cql_cat_q = tf.concat(
                    [
                        cql_q_rand - tf.math.log(0.5**A),
                        cql_q_current_actions - cql_current_log_pis,
                        cql_q_next_actions - cql_next_log_pis,
                    ],
                    axis=1,
                )

                cql_qf_ood = tf.reduce_logsumexp(cql_cat_q / self.cql_temp, axis=1) * self.cql_temp

                cql_qf_diff = tf.reduce_mean(
                    tf.clip_by_value(
                        cql_qf_ood - in_dist_qs[i],
                        self.cql_clip_diff_min,
                        self.cql_clip_diff_max,
                    ),
                    axis=0
                ) # mean across batch

                alpha_prime = 1.0# tf.clip_by_value(tf.exp(self.log_alpha_prime), 0.0, 1000000.0)
                
                cql_min_qf_loss.append(
                    alpha_prime
                    * self.cql_alpha
                    * (cql_qf_diff) # - self.cql_target_action_gap)
                )

            cql_min_qf_loss = tf.reduce_sum(cql_min_qf_loss) # sum across critics and agents
            # alpha_prime_loss = -0.5*tf.reduce_sum(cql_min_qf_loss) # sum across critics and agents

            critic_loss = cql_min_qf_loss + td_loss

            ###################
            ### Policy Loss ###
            ###################
            online_actions, online_log_probs = self.actor(observations)

            policy_qs = []
            for critic in self.critics:
                policy_qs.append(critic(env_states, online_actions, online_actions))
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

        # Alpha prime loss
        # variables = (self.log_alpha_prime,)
        # gradients = tape.gradient(alpha_prime_loss, variables)
        # self.alpha_prime_optimizer.apply(gradients, variables)

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

        del tape

        logs = {
            # "mean_dataset_q_values": tf.reduce_mean((qs_1 + qs_2) / 2),
            "critic_loss": critic_loss,
            "td_loss": td_loss,
            "cql_loss": cql_min_qf_loss,
            # "alpha_prime_loss": alpha_prime_loss,
            "alpha_loss": alpha_loss,
            "alpha_prime": tf.reduce_mean(tf.exp(self.log_alpha_prime)),
            "alpha": tf.reduce_mean(tf.exp(self.log_alpha)),
            "mean_policy_q_values": tf.reduce_mean(policy_qs),
            # "cql_loss": (cql_loss_1 + cql_loss_2) / 2.0,
            "policy_loss": policy_loss,
            # "mean_chosen_q_values": tf.reduce_mean((policy_qs_1 + policy_qs_2) / 2),
        }

        return logs


@hydra.main(version_base=None, config_path="configs", config_name="masac_cql")
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

    system = MASACCQLSystem(env, logger, **cfg["system"])

    tf.random.set_seed(seed)

    system.train(buffer, num_eval_episodes=10, training_steps=int(cfg["training_steps"]))


if __name__ == "__main__":
    run_experiment()
