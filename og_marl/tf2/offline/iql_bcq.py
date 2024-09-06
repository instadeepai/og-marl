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

"""Implementation of IQL+BCQ"""
from typing import Any, Dict, Tuple

import copy
import hydra
from omegaconf import DictConfig
import sonnet as snt
import tensorflow as tf
import numpy as np
from tensorflow import Tensor
from chex import Numeric
import tree

from og_marl.environments import get_environment, BaseEnvironment
from og_marl.loggers import BaseLogger, WandbLogger
from og_marl.vault_utils.download_vault import download_and_unzip_vault
from og_marl.replay_buffers import Experience, FlashbaxReplayBuffer
from og_marl.tf2.offline.base import BaseOfflineSystem
from og_marl.tf2.utils import (
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


class IQLBCQSystem(BaseOfflineSystem):

    """IQL+BCQ System"""

    def __init__(
        self,
        environment: BaseEnvironment,
        logger: BaseLogger,
        linear_layer_dim: int = 64,
        recurrent_layer_dim: int = 64,
        discount: float = 0.99,
        target_update_period: int = 200,
        learning_rate: float = 3e-4,
        add_agent_id_to_obs: bool = True,
        bc_threshold: float = 0.4,
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

        # Optimizers
        self.optimizer = snt.optimizers.Adam(learning_rate=learning_rate)
        self.bc_optimizer = snt.optimizers.Adam(learning_rate=learning_rate)

        # Recurrent neural network hidden states for evaluation
        self.rnn_states = {
            agent: self.q_network.initial_state(1) for agent in self.environment.agents
        }

        self.bc_threshold = bc_threshold
        self.behaviour_cloning_network = snt.DeepRNN(
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
        """Called at the start of a new episode during evaluation."""
        self.rnn_states = {
            agent: self.q_network.initial_state(1) for agent in self.environment.agents
        }  # reset the rnn hidden states

    def select_actions(
        self,
        observations: Dict[str, np.ndarray],
        legal_actions: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        observations, legal_actions = tree.map_structure(
            tf.convert_to_tensor, (observations, legal_actions)
        )
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
        for i, agent in enumerate(self.agents):
            agent_observation = observations[agent]
            if self.add_agent_id_to_obs:
                agent_observation = concat_agent_id_to_obs(agent_observation, i, len(self.agents))
            agent_observation = tf.expand_dims(agent_observation, axis=0)  # add batch dimension
            q_values, next_rnn_states[agent] = self.q_network(agent_observation, rnn_states[agent])

            agent_legal_actions = legal_actions[agent]
            masked_q_values = tf.where(
                tf.cast(agent_legal_actions, "bool"),
                q_values[0],
                -99999999,
            )
            greedy_action = tf.argmax(masked_q_values)

            actions[agent] = greedy_action

        return actions, next_rnn_states

    def train_step(self, experience: Experience) -> Dict[str, Numeric]:
        train_step = tf.convert_to_tensor(self.training_step_ctr)
        logs = self._tf_train_step(train_step, experience)
        return logs  # type: ignore

    @tf.function(jit_compile=True)
    def _tf_train_step(self, train_step: int, experience: Dict[str, Any]) -> Dict[str, Numeric]:
        # Unpack the batch
        observations = tf.cast(experience["observations"], "float32")  # (B,T,N,O)
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
        target_qs_out = unroll_rnn(self.target_q_network, observations, resets)

        # Expand batch and agent_dim
        target_qs_out = expand_batch_and_agent_dim_of_time_major_sequence(target_qs_out, B, N)

        # Make batch-major again
        target_qs_out = switch_two_leading_dims(target_qs_out)

        with tf.GradientTape(persistent=True) as tape:
            # Unroll online network
            qs_out = unroll_rnn(self.q_network, observations, resets)

            # Expand batch and agent_dim
            qs_out = expand_batch_and_agent_dim_of_time_major_sequence(qs_out, B, N)

            # Make batch-major again
            qs_out = switch_two_leading_dims(qs_out)

            # Pick the Q-Values for the actions taken by each agent
            chosen_action_qs = gather(qs_out, actions, axis=3, keepdims=False)

            ###################
            ####### BCQ #######
            ###################

            # Unroll behaviour cloning network
            probs_out = unroll_rnn(self.behaviour_cloning_network, observations, resets)

            # Expand batch and agent_dim
            probs_out = expand_batch_and_agent_dim_of_time_major_sequence(probs_out, B, N)

            # Make batch-major again
            probs_out = switch_two_leading_dims(probs_out)

            # Behaviour Cloning Loss
            one_hot_actions = tf.one_hot(actions, depth=probs_out.shape[-1], axis=-1)
            bc_loss = tf.keras.metrics.categorical_crossentropy(one_hot_actions, probs_out)
            bc_loss = tf.reduce_mean(bc_loss)

            # Legal action masking plus bc probs
            masked_probs_out = probs_out * tf.cast(legal_actions, "float32")
            masked_probs_out_sum = tf.reduce_sum(masked_probs_out, axis=-1, keepdims=True)
            masked_probs_out = masked_probs_out / masked_probs_out_sum

            # Behaviour cloning action mask
            bc_action_mask = (
                masked_probs_out / tf.reduce_max(masked_probs_out, axis=-1, keepdims=True)
            ) >= self.bc_threshold

            q_selector = tf.where(bc_action_mask, qs_out, -999999)
            max_actions = tf.argmax(q_selector, axis=-1)
            target_max_qs = gather(target_qs_out, max_actions, axis=-1)

            ###################
            ####### END #######
            ###################

            # Compute targets
            targets = (
                rewards[:, :-1] + (1 - terminals[:, :-1]) * self.discount * target_max_qs[:, 1:]
            )
            targets = tf.stop_gradient(targets)

            # Chop off last time step
            chosen_action_qs = chosen_action_qs[:, :-1]  # shape=(B,T-1)

            # TD-Error Loss
            td_loss = 0.5 * tf.reduce_mean(tf.square(targets - chosen_action_qs))

        # Get trainable variables
        variables = (*self.q_network.trainable_variables,)

        # Compute gradients.
        gradients = tape.gradient(td_loss, variables)

        # Apply gradients.
        self.optimizer.apply(gradients, variables)

        # BC network update
        variables = (*self.behaviour_cloning_network.trainable_variables,)
        # Compute gradients.
        gradients = tape.gradient(bc_loss, variables)
        # Apply gradients.
        self.bc_optimizer.apply(gradients, variables)

        # Online variables
        online_variables = (*self.q_network.variables,)
        # Get target variables
        target_variables = (*self.target_q_network.variables,)
        # Maybe update target network
        if train_step % self.target_update_period == 0:
            for src, dest in zip(online_variables, target_variables):
                dest.assign(src)

        del tape

        return {
            "td_loss": td_loss,
            "bc_loss": bc_loss,
            "mean_q_values": tf.reduce_mean(qs_out),
            "mean_chosen_q_values": tf.reduce_mean(chosen_action_qs),
        }


@hydra.main(version_base=None, config_path="configs", config_name="iql_bcq")
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

    system = IQLBCQSystem(env, logger, **cfg["system"])

    tf.random.set_seed(cfg["seed"])

    system.train(buffer, training_steps=int(cfg["training_steps"]))


if __name__ == "__main__":
    run_experiment()
