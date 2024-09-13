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

"""Implementation of Independent DDPG"""
from typing import Any, Dict, Optional, Tuple

import copy
import hydra
import numpy as np
from omegaconf import DictConfig
import tensorflow as tf
from tensorflow import Tensor
import sonnet as snt
import tree
from chex import Numeric

from og_marl.environments import get_environment, BaseEnvironment
from og_marl.loggers import BaseLogger, WandbLogger
from og_marl.replay_buffers import Experience, FlashbaxReplayBuffer
from og_marl.tf2.networks import QMixer, StateAndActionCritic
from og_marl.tf2.online.base import BaseOnlineSystem
from og_marl.tf2.utils import (
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


class QMIXSystem(BaseOnlineSystem):
    """QMIX"""

    def __init__(
        self,
        environment: BaseEnvironment,
        evaluation_environment: BaseEnvironment,
        logger: BaseLogger,
        linear_layer_dim: int = 64,
        recurrent_layer_dim: int = 64,
        mixer_embed_dim: int = 32,
        mixer_hyper_dim: int = 64,
        discount: float = 0.99,
        target_update_period: int = 200,
        learning_rate: float = 3e-4,
        eps_decay_steps: int = 10_000,
        eps_min: float = 0.05,
        add_agent_id_to_obs: bool = True,
        env_steps_before_train: int = 5000,
        train_period: int = 4,
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
        self.train_step_ctr = tf.Variable(0.0, trainable=False)

        # Optimizer
        self.optimizer = snt.optimizers.Adam(learning_rate=learning_rate)

        # Recurrent neural network hidden states for evaluation
        self.rnn_states = {
            agent: self.q_network.initial_state(1) for agent in self.environment.agents
        }

        self.mixer = QMixer(len(self.environment.agents), mixer_embed_dim, mixer_hyper_dim)
        self.target_mixer = QMixer(len(self.environment.agents), mixer_embed_dim, mixer_hyper_dim)

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
            q_values, next_rnn_states[agent] = self.q_network(
                agent_observation, rnn_states[agent]
            )
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
        target_qs_out = switch_two_leading_dims(target_qs_out)

        with tf.GradientTape() as tape:
            # Unroll online network
            qs_out = unroll_rnn(self.q_network, observations, resets)

            # Expand batch and agent_dim
            qs_out = expand_batch_and_agent_dim_of_time_major_sequence(qs_out, B, N)

            # Make batch-major again
            qs_out = switch_two_leading_dims(qs_out)

            # Pick the Q-Values for the actions taken by each agent
            chosen_action_qs = gather(qs_out, actions, axis=3, keepdims=False)

            # Max over target Q-Values/ Double q learning
            qs_out_selector = tf.where(
                tf.cast(legal_actions, "bool"), qs_out, -9999999
            )  # legal action masking
            cur_max_actions = tf.argmax(qs_out_selector, axis=3)
            target_max_qs = gather(target_qs_out, cur_max_actions, axis=-1)

            # Q-MIXING
            chosen_action_qs, target_max_qs, rewards = self.mixing(
                chosen_action_qs,
                target_max_qs,
                env_states,
                rewards,
            )

            # Compute targets
            targets = (
                rewards[:, :-1] + (1 - terminals[:, :-1]) * self.discount * target_max_qs[:, 1:]
            )
            targets = tf.stop_gradient(targets)

            # TD-Error Loss
            td_loss = 0.5 * tf.reduce_mean(tf.square(targets - chosen_action_qs[:, :-1]))


        # Get trainable variables
        variables = (
            *self.q_network.trainable_variables,
            *self.mixer.trainable_variables,
        )

        # Compute gradients.
        gradients = tape.gradient(td_loss, variables)

        # Apply gradients.
        self.optimizer.apply(gradients, variables)

        # Online variables
        online_variables = (
            *self.q_network.variables,
            *self.mixer.variables,
        )

        # Get target variables
        target_variables = (
            *self.target_q_network.variables,
            *self.target_mixer.variables,
        )

        # Maybe update target network
        self.train_step_ctr.assign_add(1.0)
        if self.train_step_ctr % self.target_update_period == 0:
            for src, dest in zip(online_variables, target_variables):
                dest.assign(src)

        return {
            "td_loss": td_loss,
            "mean_q_values": tf.reduce_mean(qs_out),
            "mean_chosen_q_values": tf.reduce_mean(chosen_action_qs),
        }

    def mixing(
        self,
        chosen_action_qs: Tensor,
        target_max_qs: Tensor,
        states: Tensor,
        rewards: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """QMIX"""
        # VDN
        # chosen_action_qs = tf.reduce_sum(chosen_action_qs, axis=2, keepdims=True)
        # target_max_qs = tf.reduce_sum(target_max_qs, axis=2, keepdims=True)
        # VDN

        chosen_action_qs = self.mixer(chosen_action_qs, states)
        target_max_qs = self.target_mixer(target_max_qs, states)
        rewards = tf.reduce_mean(rewards, axis=2, keepdims=True)
        return chosen_action_qs, target_max_qs, rewards


@hydra.main(version_base=None, config_path="configs", config_name="qmix")
def run_experiment(cfg: DictConfig) -> None:
    print(cfg)

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
        store_to_vault=True,
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

    system = QMIXSystem(env, eval_env, logger, **cfg["system"])

    tf.random.set_seed(cfg["seed"])

    system.train(buffer, evaluation_every=5000, num_eval_episodes=10, environment_steps=int(cfg["environment_steps"]))


if __name__ == "__main__":
    run_experiment()
