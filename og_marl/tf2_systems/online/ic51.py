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

"""Implementation of Independent Categorical 51 (IC51)."""
from typing import Any, Dict, Optional, Tuple

import copy
import hydra
import numpy as np
import jax
from omegaconf import DictConfig
import tensorflow as tf
from tensorflow import Tensor
import sonnet as snt
import tree
from chex import Numeric

from og_marl.tf2_systems.networks import RecurrentCategoricalQNetwork
from og_marl.environments import get_environment, BaseEnvironment
from og_marl.loggers import BaseLogger, WandbLogger
from og_marl.replay_buffers import Experience, FlashbaxReplayBuffer
from og_marl.tf2_systems.online.base import BaseOnlineSystem
from og_marl.tf2_systems.utils import (
    batch_concat_agent_id_to_obs,
    concat_agent_id_to_obs,
    expand_batch_and_agent_dim_of_time_major_sequence,
    merge_batch_and_agent_dim_of_time_major_sequence,
    set_growing_gpu_memory,
    switch_two_leading_dims,
    unroll_rnn,
    gather
)

set_growing_gpu_memory()


class IC51System(BaseOnlineSystem):
    """Independent Categorical 51"""

    def __init__(
        self,
        environment: BaseEnvironment,
        evaluation_environment: BaseEnvironment,
        logger: BaseLogger,
        linear_layer_dim: int = 64,
        recurrent_layer_dim: int = 64,
        discount: float = 0.99,
        target_update_period: int = 200,
        learning_rate: float = 3e-4,
        eps_decay_steps: int = 10_000,
        eps_min: float = 0.05,
        add_agent_id_to_obs: bool = True,
        env_steps_before_train: int = 5000,
        train_period: int = 4,
        n_atoms: int = 51,
        v_min: float = -10.0,
        v_max: float = 20.0,
    ):
        super().__init__(
            environment=environment,
            evaluation_environment=evaluation_environment,
            logger=logger,
            env_steps_before_train=env_steps_before_train,
            train_period=train_period,
        )

        self.add_agent_id_to_obs = add_agent_id_to_obs
        self.discount = discount

        # Categorical parameters
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.delta_z = (v_max - v_min) / (n_atoms - 1)
        self.atoms = tf.linspace(v_min, v_max, n_atoms)

        # Distributional Q-network
        self.q_network = RecurrentCategoricalQNetwork(
            self.environment.num_actions, n_atoms, 
            linear_layer_dim, recurrent_layer_dim
        ) # shared network for all agents

        # Target Distributional Q-network
        self.target_q_network = copy.deepcopy(self.q_network)
        self.target_update_period = target_update_period
        self.train_step_ctr = tf.Variable(0.0, trainable=False)

        # Optimizer
        self.optimizer = snt.optimizers.Adam(learning_rate=learning_rate)

        # Recurrent neural network hidden states for evaluation
        self.rnn_states = {
            agent: self.q_network.initial_state(1) for agent in self.environment.agents
        }

        self.env_steps = tf.Variable(0.0, trainable=False)
        self.eps_decay_steps = eps_decay_steps
        self.eps_min = eps_min
        self.eps_denominator = (1/(1-eps_min))*eps_decay_steps

    def reset(self) -> None:
        """Called at the start of a new episode."""
        # Reset the recurrent neural network
        self.rnn_states = {
            agent: self.q_network.initial_state(1) for agent in self.environment.agents
        }
        return

    def select_actions(
        self,
        observations: Dict[str, np.ndarray],
        legal_actions: Optional[Dict[str, np.ndarray]] = None,
        explore: bool = True,
    ) -> Dict[str, np.ndarray]:
        actions, next_rnn_states = self._tf_select_actions(observations, self.rnn_states, legal_actions, explore)
        self.rnn_states = next_rnn_states
        return tree.map_structure(  # type: ignore
            lambda x: x[0].numpy().astype("int32"), actions
        )  # convert to numpy and squeeze batch dim

    @tf.function(jit_compile=True)
    def _tf_select_actions(
        self, observations: Dict[str, Tensor], rnn_states: Dict[str, Tensor], legals: Dict[str, Tensor], explore: bool
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
            
            # Get probability mass function over atoms for each action
            action_pmfs, next_rnn_states[agent] = self.q_network(  # shapes (1,A,n_atoms), (1,hidden_dim)
                agent_observation, rnn_states[agent]
            )

            # Get Q-values by taking the expectation over the atoms
            q_values = tf.reduce_sum(action_pmfs * self.atoms, axis=-1)

            agent_legals = tf.expand_dims(legals[agent], axis=0)
            q_values = tf.where(agent_legals==1, q_values, -np.inf)
            action = tf.argmax(q_values, axis=-1)

            self.env_steps.assign_add(1.0)
            eps = tf.maximum(1 - (1/self.eps_denominator) * self.env_steps, self.eps_min)
            if explore and tf.random.uniform(()) < eps:
                agent_log_probs = tf.math.log(agent_legals / tf.reduce_sum(agent_legals))
                action = tf.random.categorical(logits=agent_log_probs, num_samples=1)[0]

            # Store agent action
            actions[agent] = action

        return actions, next_rnn_states

    def train_step(self, experience: Experience) -> Dict[str, Numeric]:
        logs = self._tf_train_step(experience)
        return logs  # type: ignore

    @tf.function(jit_compile=True)  # NOTE: comment this out if using debugger
    def _tf_train_step(self, experience: Dict[str, Any]) -> Dict[str, Numeric]:
        # Unpack the batch
        observations = experience["observations"]  # (B,T,N,O)
        actions = experience["actions"]  # (B,T,N)
        rewards = experience["rewards"]  # (B,T,N)
        truncations = tf.cast(experience["truncations"], "float32")  # (B,T,N)
        terminals = tf.cast(experience["terminals"], "float32")  # (B,T,N)
        legal_actions = experience["infos"]["legals"]  # (B,T,N,A)

        # When to reset the RNN hidden state
        resets = tf.maximum(terminals, truncations)  # equivalent to logical 'or'

        # Get dims
        B, T, N, A = legal_actions.shape

        # Maybe add agent ids to observation
        if self.add_agent_id_to_obs:
            observations = batch_concat_agent_id_to_obs(observations)

        # Make time-major
        observations = switch_two_leading_dims(observations)
        resets = switch_two_leading_dims(resets)

        # Merge batch_dim and agent_dim
        observations = merge_batch_and_agent_dim_of_time_major_sequence(observations)
        resets = merge_batch_and_agent_dim_of_time_major_sequence(resets)

        # Unroll target network
        target_distributions = unroll_rnn(self.target_q_network, observations, resets)

        # Expand batch and agent_dim
        target_distributions = expand_batch_and_agent_dim_of_time_major_sequence(target_distributions, B, N)

        # Make batch-major again
        target_distributions = switch_two_leading_dims(target_distributions)

        with tf.GradientTape() as tape:
            # Unroll online network
            distributions = unroll_rnn(self.q_network, observations, resets)

            # Expand batch and agent_dim
            distributions = expand_batch_and_agent_dim_of_time_major_sequence(distributions, B, N)

            # Make batch-major again
            distributions = switch_two_leading_dims(distributions)

            # Compute Q-Values
            qs_out = tf.reduce_sum(distributions * self.atoms, axis=-1)

            # legal action masking
            qs_out_selector = tf.where(legal_actions==1, qs_out, -np.inf)

            # Get best actions
            max_actions = tf.argmax(qs_out_selector, axis=-1)

            # Gather target distributions for the selected actions at the next time step
            target_distributions_next = tf.gather(
                target_distributions[:, 1:], 
                max_actions[:, 1:], 
                batch_dims=3
            )  # (B,T-1,N,n_atoms)
            
            # Expand rewards and terminals for broadcasting
            rewards_expanded = tf.expand_dims(rewards[:, :-1], axis=-1)  # (B, T-1, N, 1)
            terminals_expanded = tf.expand_dims(terminals[:, :-1], axis=-1)  # (B, T-1, N, 1)
            
            # Compute Tz = r + gamma * z * (1 - done) for all atoms
            # (B, T-1, N, n_atoms)
            Tz = rewards_expanded + self.discount * (1.0 - terminals_expanded) * tf.reshape(self.atoms, [1, 1, 1, -1])
            
            # Clip target values to support range [v_min, v_max]
            Tz = tf.clip_by_value(Tz, self.v_min, self.v_max)
            
            # Compute indices on the support for the projected distribution
            b = (Tz - self.v_min) / self.delta_z  # (B, T-1, N, n_atoms)
            
            # Compute lower and upper indices for projection
            l = tf.math.floor(b)  # (B, T-1, N, n_atoms)
            u = tf.math.ceil(b)   # (B, T-1, N, n_atoms)
            
            # Handle case where b is exactly an integer to avoid l == u
            u = tf.where(tf.equal(u, l), u + 1.0, u)
            l_idx = tf.cast(tf.clip_by_value(l, 0, self.n_atoms - 1), tf.int32)
            u_idx = tf.cast(tf.clip_by_value(u, 0, self.n_atoms - 1), tf.int32)
            
            # Calculate probability mass to assign to each supported atom
            # Weight for lower bound: (u - b) * p
            # Weight for upper bound: (b - l) * p
            u_minus_b = u - b
            b_minus_l = b - l
            
            # Initialize projected distribution with zeros
            # Shape: (B, T-1, N, n_atoms)
            target_dist = tf.zeros_like(target_distributions_next)
            
            # Create batch indices for the batch dimension
            batch_indices = tf.range(B, dtype=tf.int32)
            batch_indices = tf.reshape(batch_indices, [B, 1, 1, 1])
            batch_indices = tf.broadcast_to(batch_indices, [B, T-1, N, self.n_atoms])
            
            # Create time indices for the time dimension
            time_indices = tf.range(T-1, dtype=tf.int32)
            time_indices = tf.reshape(time_indices, [1, T-1, 1, 1])
            time_indices = tf.broadcast_to(time_indices, [B, T-1, N, self.n_atoms])
            
            # Create agent indices for the agent dimension
            agent_indices = tf.range(N, dtype=tf.int32)
            agent_indices = tf.reshape(agent_indices, [1, 1, N, 1])
            agent_indices = tf.broadcast_to(agent_indices, [B, T-1, N, self.n_atoms])
            
            # For lower bound indices (l_idx)
            l_scatter_indices = tf.stack([batch_indices, time_indices, agent_indices, l_idx], axis=-1)
            l_updates = u_minus_b * target_distributions_next
            
            # Add lower bound contributions to target distribution
            target_dist = tf.tensor_scatter_nd_add(target_dist, l_scatter_indices, l_updates)
            
            # For upper bound indices (u_idx)
            u_scatter_indices = tf.stack([batch_indices, time_indices, agent_indices, u_idx], axis=-1)
            u_updates = b_minus_l * target_distributions_next
            
            # Add upper bound contributions to target distribution
            target_dist = tf.tensor_scatter_nd_add(target_dist, u_scatter_indices, u_updates)
            
            # Get online network distributions for the actions actually taken
            # Shape: (B, T, N, n_atoms)
            online_dists = tf.gather(distributions, actions, batch_dims=3)
            
            # Focus on distributions for steps other than the last one
            online_dists = online_dists[:, :-1]
            
            # Compute cross-entropy loss (avoiding log(0))
            online_dists = tf.clip_by_value(online_dists, 1e-5, 1.0 - 1e-5)
            loss = -tf.reduce_sum(target_dist * tf.math.log(online_dists), axis=-1)
            loss = tf.reduce_mean(loss)

        # Get trainable variables
        variables = self.q_network.trainable_variables

        # Compute gradients
        gradients = tape.gradient(loss, variables)

        # Apply gradients
        self.optimizer.apply(gradients, variables)

        # Online variables
        online_variables = self.q_network.variables

        # Get target variables
        target_variables = self.target_q_network.variables

        # Maybe update target network
        self.train_step_ctr.assign_add(1.0)
        if self.train_step_ctr % self.target_update_period == 0:
            for src, dest in zip(online_variables, target_variables):
                dest.assign(src)

        # Calculate mean Q values for logging
        mean_q_values = tf.reduce_mean(tf.reduce_sum(distributions * self.atoms, axis=-1))
        
        # Calculate mean Q values for chosen actions
        chosen_q_values = tf.reduce_sum(online_dists * self.atoms, axis=-1)
        mean_chosen_q_values = tf.reduce_mean(chosen_q_values)

        return {
            "distributional_loss": loss,
            "mean_q_values": mean_q_values,
            "mean_chosen_q_values": mean_chosen_q_values,
        }

@hydra.main(version_base=None, config_path="configs", config_name="ic51")
def run_experiment(cfg: DictConfig) -> None:
    print(cfg)

    jax.config.update('jax_platform_name', 'cpu')

    env = get_environment(
        cfg["task"]["source"], cfg["task"]["env"], cfg["task"]["scenario"], seed=cfg["seed"]
    )
    eval_env = get_environment(
        cfg["task"]["source"], cfg["task"]["env"], cfg["task"]["scenario"], seed=cfg["seed"] + 1
    )

    buffer = FlashbaxReplayBuffer(
        sequence_length=cfg["replay"]["sequence_length"],
        sample_period=cfg["replay"]["sample_period"],
        seed=cfg["seed"],
        store_to_vault=cfg["replay"]["store_to_vault"],
        vault_name=f"recorded_data/{cfg['task']['env']}/{cfg['task']['scenario']}.vlt"
    )

    wandb_config = {
        "system": cfg["system_name"],
        "seed": cfg["seed"],
        "environment_steps": cfg["environment_steps"],
        **cfg["task"],
        **cfg["replay"],
        **cfg["system"],
    }
    logger = WandbLogger(project=cfg["wandb_project"], config=wandb_config)

    system = IC51System(env, eval_env, logger, **cfg["system"])

    tf.random.set_seed(cfg["seed"])

    system.train(buffer, evaluation_every=5000, num_eval_episodes=32, environment_steps=int(cfg["environment_steps"]))


if __name__ == "__main__":
    run_experiment()
