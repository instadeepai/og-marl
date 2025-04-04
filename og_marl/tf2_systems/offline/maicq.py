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

"""Implementation of MAICQ"""
from typing import Any, Dict, Tuple

import copy
import hydra
import numpy as np
import jax
from omegaconf import DictConfig
import sonnet as snt
import tensorflow as tf
import tree
from chex import Numeric
from tensorflow import Tensor

from og_marl.environments import get_environment, BaseEnvironment
from og_marl.loggers import BaseLogger, WandbLogger
from og_marl.vault_utils.download_vault import download_and_unzip_vault
from og_marl.replay_buffers import Experience, FlashbaxReplayBuffer
from og_marl.tf2_systems.networks import QMixer
from og_marl.tf2_systems.offline.base import BaseOfflineSystem
from og_marl.tf2_systems.utils import (
    batch_concat_agent_id_to_obs,
    concat_agent_id_to_obs,
    expand_batch_and_agent_dim_of_time_major_sequence,
    gather,
    merge_batch_and_agent_dim_of_time_major_sequence,
    set_growing_gpu_memory,
    switch_two_leading_dims,
    unroll_rnn,
)

set_growing_gpu_memory()


class MAICQSystem(BaseOfflineSystem):

    """MAICQ System"""

    def __init__(
        self,
        environment: BaseEnvironment,
        logger: BaseLogger,
        linear_layer_dim: int = 64,
        recurrent_layer_dim: int = 64,
        mixer_embed_dim: int = 32,
        mixer_hyper_dim: int = 64,
        discount: float = 0.99,
        target_update_period: int = 200,
        learning_rate: float = 3e-4,
        add_agent_id_to_obs: bool = True,
        icq_advantages_beta: float = 0.1,  # from MAICQ code
        icq_target_q_taken_beta: int = 1000,  # from MAICQ code
    ):
        super().__init__(
            environment,
            logger,
        )

        self.discount = discount
        self.add_agent_id_to_obs = add_agent_id_to_obs

        # Q-network
        self.q_network = snt.DeepRNN(
            [
                snt.Linear(linear_layer_dim),
                tf.nn.relu,
                snt.GRU(recurrent_layer_dim),
                tf.nn.relu,
                snt.Linear(self.environment.num_actions),
            ]
        )  # shared network for all agents

        # Target Q-network
        self.target_q_network = copy.deepcopy(self.q_network)
        self.target_update_period = target_update_period

        # Optimizer
        self.optimizer = snt.optimizers.Adam(learning_rate=learning_rate)

        # Recurrent neural network hidden states for evaluation
        self.rnn_states = {
            agent: self.q_network.initial_state(1) for agent in self.environment.agents
        }

        self.mixer = QMixer(len(self.environment.agents), mixer_embed_dim, mixer_hyper_dim)
        self.target_mixer = QMixer(len(self.environment.agents), mixer_embed_dim, mixer_hyper_dim)

        # ICQ hyper-params
        self.icq_advantages_beta = icq_advantages_beta
        self.icq_target_q_taken_beta = icq_target_q_taken_beta

        # Policy Network
        self.policy_network = snt.DeepRNN(
            [
                snt.Linear(linear_layer_dim),
                tf.nn.relu,
                snt.GRU(recurrent_layer_dim),
                tf.nn.relu,
                snt.Linear(self.environment.num_actions),
                tf.nn.softmax,
            ]
        )

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
        legal_actions: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        observations = tree.map_structure(tf.convert_to_tensor, observations)
        actions, next_rnn_states = self._tf_select_actions(
            observations, legal_actions, self.rnn_states
        )
        self.rnn_states = next_rnn_states
        return tree.map_structure(  # type: ignore
            lambda x: x.numpy(), actions
        )  # convert to numpy and squeeze batch dim

    @tf.function(jit_compile=True)
    def _tf_select_actions(
        self,
        observations: Dict[str, Tensor],
        legal_actions: Dict[str, Tensor],
        rnn_states: Dict[str, Tensor],
    ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        actions = {}
        next_rnn_states = {}
        for i, agent in enumerate(self.environment.agents):
            agent_observation = observations[agent]
            agent_observation = concat_agent_id_to_obs(
                agent_observation, i, len(self.environment.agents)
            )
            agent_observation = tf.expand_dims(agent_observation, axis=0)  # add batch dimension
            probs, next_rnn_states[agent] = self.policy_network(
                agent_observation, rnn_states[agent]
            )

            agent_legal_actions = legal_actions[agent]
            masked_probs = tf.where(
                tf.equal(agent_legal_actions, 1),
                probs[0],
                -99999999,
            )

            # Max Q-value over legal actions
            actions[agent] = tf.argmax(masked_probs)

        return actions, next_rnn_states

    def train_step(self, experience: Experience) -> Dict[str, Numeric]:
        train_step = tf.convert_to_tensor(self.training_step_ctr)
        logs = self._tf_train_step(train_step, experience)
        return logs  # type: ignore

    @tf.function(jit_compile=True)
    def _tf_train_step(
        self,
        train_step: int,
        experience: Dict[str, Any],
    ) -> Dict[str, Numeric]:
        # Unpack the batch
        observations = experience["observations"]  # (B,T,N,O)
        actions = tf.squeeze(tf.cast(experience["actions"], "int32"), -1)  # (B,T,N)
        env_states = experience["infos"]["state"]  # (B,T,S)
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
        target_qs_out = unroll_rnn(self.target_q_network, observations, resets)

        # Expand batch and agent_dim
        target_qs_out = expand_batch_and_agent_dim_of_time_major_sequence(target_qs_out, B, N)

        # Make batch-major again
        target_q_vals = switch_two_leading_dims(target_qs_out)

        with tf.GradientTape(persistent=True) as tape:
            # Unroll online network
            qs_out = unroll_rnn(self.q_network, observations, resets)

            # Expand batch and agent_dim
            qs_out = expand_batch_and_agent_dim_of_time_major_sequence(qs_out, B, N)

            # Make batch-major again
            q_vals = switch_two_leading_dims(qs_out)

            # Unroll the policy
            probs_out = unroll_rnn(self.policy_network, observations, resets)

            # Expand batch and agent_dim
            probs_out = expand_batch_and_agent_dim_of_time_major_sequence(probs_out, B, N)

            # Make batch-major again
            probs_out = switch_two_leading_dims(probs_out)

            # Mask illegal actions
            probs_out = probs_out * tf.cast(legal_actions, "float32")
            probs_sum = (
                tf.reduce_sum(probs_out, axis=-1, keepdims=True) + 1e-10
            )  # avoid div by zero
            probs_out = probs_out / probs_sum

            action_values = gather(q_vals, actions)
            baseline = tf.reduce_sum(probs_out * q_vals, axis=-1)
            advantages = action_values - baseline
            advantages = tf.nn.softmax(advantages / self.icq_advantages_beta, axis=0)
            advantages = tf.stop_gradient(advantages)

            pi_taken = gather(probs_out, actions, keepdims=False)
            log_pi_taken = tf.math.log(pi_taken)

            # coe = self.mixer.k(env_states)
            coe = 1.

            coma_loss = -tf.reduce_mean(coe * (len(advantages) * advantages * log_pi_taken))

            # Critic learning
            q_taken = gather(q_vals, actions, axis=-1)
            target_q_taken = gather(target_q_vals, actions, axis=-1)

            # Mixing critics
            # q_taken = self.mixer(q_taken, env_states)
            # target_q_taken = self.target_mixer(target_q_taken, env_states)

            advantage_Q = tf.nn.softmax(target_q_taken / self.icq_target_q_taken_beta, axis=0)
            target_q_taken = len(advantage_Q) * advantage_Q * target_q_taken

            # Compute targets
            targets = (
                rewards[:, :-1] + (1 - terminals[:, :-1]) * self.discount * target_q_taken[:, 1:]
            )
            targets = tf.stop_gradient(targets)

            # TD error
            td_error = targets - q_taken[:, :-1]
            q_loss = 0.5 * tf.square(td_error)

            # Masking
            q_loss = tf.reduce_mean(q_loss)

            # Add losses together
            loss = q_loss + coma_loss

        # Apply gradients to policy
        variables = (
            *self.policy_network.trainable_variables,
            *self.q_network.trainable_variables,
            # *self.mixer.trainable_variables,
        )  # Get trainable variables

        gradients = tape.gradient(loss, variables)  # Compute gradients.

        self.optimizer.apply(gradients, variables)  # One optimizer for whole system

        # Online variables
        online_variables = (
            *self.q_network.variables,
            # *self.mixer.variables,
        )

        # Get target variables
        target_variables = (
            *self.target_q_network.variables,
            # *self.target_mixer.variables,
        )

        # Maybe update target network
        if train_step % self.target_update_period == 0:
            for src, dest in zip(online_variables, target_variables):
                dest.assign(src)

        return {
            "critic_oss": q_loss,
            "policy_loss": coma_loss,
        }


@hydra.main(version_base=None, config_path="configs", config_name="maicq")
def run_experiment(cfg: DictConfig) -> None:
    print(cfg)

    jax.config.update('jax_platform_name', 'cpu')

    env = get_environment(cfg["task"]["source"], cfg["task"]["env"], cfg["task"]["scenario"], seed=cfg["seed"])

    buffer = FlashbaxReplayBuffer(
        sequence_length=cfg["replay"]["sequence_length"],
        sample_period=cfg["replay"]["sample_period"],
        seed=cfg["seed"],
    )

    download_and_unzip_vault(cfg["task"]["source"], cfg["task"]["env"], cfg["task"]["scenario"])

    buffer.populate_from_vault(cfg["task"]["source"], cfg["task"]["env"], cfg["task"]["scenario"], str(cfg["task"]["dataset"]))

    wandb_config = {
        "system": cfg["system_name"],
        "seed": cfg["seed"],
        "training_steps": cfg["training_steps"],
        **cfg["task"],
        **cfg["replay"],
        **cfg["system"],
    }
    logger = WandbLogger(project=cfg["wandb_project"], config=wandb_config)

    system = MAICQSystem(env, logger, **cfg["system"])

    tf.random.set_seed(cfg["seed"])

    system.train(buffer, training_steps=int(cfg["training_steps"]))


if __name__ == "__main__":
    run_experiment()
