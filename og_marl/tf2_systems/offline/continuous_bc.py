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

"""Implementation of Behaviour Cloning"""
from typing import Any, Dict, Optional, Tuple

import hydra
import numpy as np
from omegaconf import DictConfig
import sonnet as snt
import tensorflow as tf
import tree
from chex import Numeric

from og_marl.environments import get_environment, BaseEnvironment
from og_marl.loggers import BaseLogger, WandbLogger
from og_marl.vault_utils.download_vault import download_and_unzip_vault
from og_marl.replay_buffers import Experience, FlashbaxReplayBuffer
from og_marl.tf2_systems.offline.base import BaseOfflineSystem
from og_marl.tf2_systems.utils import (
    batch_concat_agent_id_to_obs,
    concat_agent_id_to_obs,
    expand_batch_and_agent_dim_of_time_major_sequence,
    merge_batch_and_agent_dim_of_time_major_sequence,
    set_growing_gpu_memory,
    switch_two_leading_dims,
    unroll_rnn,
)

set_growing_gpu_memory()


class ContinuousActionBehaviourCloning(BaseOfflineSystem):
    """Behaviour cloning for continuous action spaces."""

    def __init__(
        self,
        environment: BaseEnvironment,
        logger: BaseLogger,
        linear_layer_dim: int = 64,
        recurrent_layer_dim: int = 64,
        discount: float = 0.99,
        learning_rate: float = 1e-3,
        add_agent_id_to_obs: bool = True,
    ):
        super().__init__(environment, logger)

        self.discount = discount
        self.add_agent_id_to_obs = add_agent_id_to_obs

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

        self.optimizer = snt.optimizers.Adam(learning_rate=learning_rate)

        # Reset the recurrent neural network
        self.rnn_states = {
            agent: self.policy_network.initial_state(1) for agent in self.environment.agents
        }

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
        observations = tree.map_structure(tf.convert_to_tensor, observations)

        actions, next_rnn_states = self._tf_select_actions(observations, self.rnn_states)
        self.rnn_states = next_rnn_states
        return tree.map_structure(  # type: ignore
            lambda x: x[0].numpy(), actions
        )  # convert to numpy and squeeze batch dim

    @tf.function(jit_compile=True)
    def _tf_select_actions(
        self,
        observations: Dict[str, tf.Tensor],
        rnn_states: Dict[str, tf.Tensor],
    ) -> Tuple[Dict[str, tf.Tensor], Dict[str, tf.Tensor]]:
        actions = {}
        next_rnn_states = {}
        for i, agent in enumerate(self.environment.agents):
            agent_observation = observations[agent]
            if self.add_agent_id_to_obs:
                agent_observation = concat_agent_id_to_obs(
                    agent_observation, i, len(self.environment.agents)
                )
            agent_observation = tf.cast(
                tf.expand_dims(agent_observation, axis=0), "float32"
            )  # add batch dimension
            action, next_rnn_states[agent] = self.policy_network(
                agent_observation, rnn_states[agent]
            )

            # Store agent action
            actions[agent] = action

        return actions, next_rnn_states

    def train_step(self, experience: Experience) -> Dict[str, Numeric]:
        logs = self._tf_train_step(experience)
        return logs  # type: ignore

    @tf.function(jit_compile=True)
    def _tf_train_step(self, experience: Dict[str, Any]) -> Dict[str, Numeric]:
        # Unpack the relevant quantities
        observations = tf.cast(experience["observations"], "float32")  # (B,T,N,O)
        actions = tf.cast(experience["actions"], "float32")  # (B,T,N,A)
        truncations = tf.cast(experience["truncations"], "float32")  # (B,T,N)
        terminals = tf.cast(experience["terminals"], "float32")  # (B,T,N)

        # When to reset the RNN hidden state
        resets = tf.maximum(terminals, truncations)  # equivalent to logical 'or'

        # Get batch size, max sequence length, num agents and num actions
        B, T, N, A = experience["actions"].shape

        # Maybe add agent ids to observation
        if self.add_agent_id_to_obs:
            observations = batch_concat_agent_id_to_obs(observations)

        # Make time-major
        observations = switch_two_leading_dims(observations)
        resets = switch_two_leading_dims(resets)
        actions = switch_two_leading_dims(actions)

        # Merge batch_dim and agent_dim
        observations = merge_batch_and_agent_dim_of_time_major_sequence(observations)
        resets = merge_batch_and_agent_dim_of_time_major_sequence(resets)

        with tf.GradientTape() as tape:
            policy_actions = unroll_rnn(
                self.policy_network,
                observations,
                resets,
            )
            policy_actions = expand_batch_and_agent_dim_of_time_major_sequence(policy_actions, B, N)

            # Behaviour cloning loss
            mse = (policy_actions - actions) ** 2
            bc_loss = tf.reduce_mean(mse)

        # Apply gradients to policy
        variables = (*self.policy_network.trainable_variables,)  # Get trainable variables

        gradients = tape.gradient(bc_loss, variables)  # Compute gradients.
        self.optimizer.apply(gradients, variables)

        logs = {"policy_loss": bc_loss}

        return logs


@hydra.main(version_base=None, config_path="configs", config_name="continuous_bc")
def run_experiment(cfg: DictConfig) -> None:
    print(cfg)

    env = get_environment(cfg["task"]["source"], cfg["task"]["env"], cfg["task"]["scenario"], seed=cfg["seed"])

    buffer = FlashbaxReplayBuffer(
        sequence_length=cfg["replay"]["sequence_length"],
        sample_period=cfg["replay"]["sample_period"],
        seed=cfg["seed"],
    )

    download_and_unzip_vault(cfg["task"]["source"], cfg["task"]["env"], cfg["task"]["scenario"])

    buffer.populate_from_vault(cfg["task"]["source"], cfg["task"]["env"], cfg["task"]["scenario"], cfg["task"]["dataset"])

    wandb_config = {
        "system": cfg["system_name"],
        "seed": cfg["seed"],
        "training_steps": cfg["training_steps"],
        **cfg["task"],
        **cfg["replay"],
        **cfg["system"],
    }
    logger = WandbLogger(project=cfg["wandb_project"], config=wandb_config)

    system = ContinuousActionBehaviourCloning(env, logger, **cfg["system"])

    tf.random.set_seed(cfg["seed"])

    system.train(buffer, training_steps=int(cfg["training_steps"]))


if __name__ == "__main__":
    run_experiment()
