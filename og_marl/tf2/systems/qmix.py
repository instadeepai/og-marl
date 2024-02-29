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

"""Implementation of QMIX"""
from typing import Any, Dict, Tuple

import sonnet as snt
import tensorflow as tf
from chex import Numeric
from tensorflow import Tensor

from og_marl.environments.base import BaseEnvironment
from og_marl.loggers import BaseLogger
from og_marl.tf2.systems.idrqn import IDRQNSystem
from og_marl.tf2.utils import (
    batch_concat_agent_id_to_obs,
    expand_batch_and_agent_dim_of_time_major_sequence,
    gather,
    merge_batch_and_agent_dim_of_time_major_sequence,
    switch_two_leading_dims,
    unroll_rnn,
)


class QMIXSystem(IDRQNSystem):

    """QMIX System"""

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
        eps_decay_timesteps: int = 50_000,
        add_agent_id_to_obs: bool = False,
    ):
        super().__init__(
            environment,
            logger,
            linear_layer_dim=linear_layer_dim,
            recurrent_layer_dim=recurrent_layer_dim,
            add_agent_id_to_obs=add_agent_id_to_obs,
            discount=discount,
            target_update_period=target_update_period,
            learning_rate=learning_rate,
            eps_decay_timesteps=eps_decay_timesteps,
        )

        self._mixer = QMixer(
            len(self._environment.possible_agents), mixer_embed_dim, mixer_hyper_dim
        )
        self._target_mixer = QMixer(
            len(self._environment.possible_agents), mixer_embed_dim, mixer_hyper_dim
        )

    @tf.function(jit_compile=True)  # NOTE: comment this out if using debugger
    def _tf_train_step(self, train_step_ctr: int, experience: Dict[str, Any]) -> Dict[str, Numeric]:
        # Unpack the batch
        observations = experience["observations"]  # (B,T,N,O)
        actions = experience["actions"]  # (B,T,N)
        env_states = experience["infos"]["state"]  # (B,T,S)
        rewards = experience["rewards"]  # (B,T,N)
        truncations = experience["truncations"]  # (B,T,N)
        terminals = experience["terminals"]  # (B,T,N)
        legal_actions = experience["infos"]["legals"]  # (B,T,N,A)

        done = terminals

        # When to reset the RNN hidden state
        resets = tf.maximum(terminals, truncations)  # equivalent to logical 'or'

        # Get dims
        B, T, N, A = legal_actions.shape

        # Maybe add agent ids to observation
        if self._add_agent_id_to_obs:
            observations = batch_concat_agent_id_to_obs(observations)

        # Make time-major
        observations = switch_two_leading_dims(observations)
        resets = switch_two_leading_dims(resets)

        # Merge batch_dim and agent_dim
        observations = merge_batch_and_agent_dim_of_time_major_sequence(observations)
        resets = merge_batch_and_agent_dim_of_time_major_sequence(resets)

        # Unroll target network
        target_qs_out = unroll_rnn(self._target_q_network, observations, resets)

        # Expand batch and agent_dim
        target_qs_out = expand_batch_and_agent_dim_of_time_major_sequence(target_qs_out, B, N)

        # Make batch-major again
        target_qs_out = switch_two_leading_dims(target_qs_out)

        with tf.GradientTape() as tape:
            # Unroll online network
            qs_out = unroll_rnn(self._q_network, observations, resets)

            # Expand batch and agent_dim
            qs_out = expand_batch_and_agent_dim_of_time_major_sequence(qs_out, B, N)

            # Make batch-major again
            qs_out = switch_two_leading_dims(qs_out)

            # Pick the Q-Values for the actions taken by each agent
            chosen_action_qs = gather(qs_out, tf.cast(actions, "int32"), axis=3, keepdims=False)

            # Max over target Q-Values/ Double q learning
            qs_out_selector = tf.where(
                tf.cast(legal_actions, "bool"), qs_out, -9999999
            )  # legal action masking
            cur_max_actions = tf.argmax(qs_out_selector, axis=3)
            target_max_qs = gather(target_qs_out, cur_max_actions, axis=-1, keepdims=False)

            # Q-MIXING
            chosen_action_qs, target_max_qs, rewards = self._mixing(
                chosen_action_qs, target_max_qs, env_states, rewards
            )

            # Reduce Agent Dim
            done = tf.reduce_mean(done, axis=2, keepdims=True)  # NOTE Assumes all the same

            # Compute targets
            targets = rewards[:, :-1] + (1 - done[:, :-1]) * self._discount * target_max_qs[:, 1:]
            targets = tf.stop_gradient(targets)

            # Chop off last time step
            chosen_action_qs = chosen_action_qs[:, :-1]  # shape=(B,T-1)

            # TD-Error Loss
            loss = 0.5 * tf.square(targets - chosen_action_qs)
            loss = tf.reduce_mean(loss)

        # Get trainable variables
        variables = (*self._q_network.trainable_variables, *self._mixer.trainable_variables)

        # Compute gradients.
        gradients = tape.gradient(loss, variables)

        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)

        # Apply gradients.
        self._optimizer.apply(gradients, variables)

        # Online variables
        online_variables = (
            *self._q_network.variables,
            *self._mixer.variables,
        )

        # Get target variables
        target_variables = (
            *self._target_q_network.variables,
            *self._target_mixer.variables,
        )

        # Maybe update target network
        self._update_target_network(train_step_ctr, online_variables, target_variables)

        return {
            "Loss": loss,
            "Mean Q-values": tf.reduce_mean(qs_out),
            "Mean Chosen Q-values": tf.reduce_mean(chosen_action_qs),
        }

    def _mixing(
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

        chosen_action_qs = self._mixer(chosen_action_qs, states)
        target_max_qs = self._target_mixer(target_max_qs, states)
        rewards = tf.reduce_mean(rewards, axis=2, keepdims=True)
        return chosen_action_qs, target_max_qs, rewards


class QMixer(snt.Module):

    """QMIX mixing network."""

    def __init__(
        self,
        num_agents: int,
        embed_dim: int = 32,
        hypernet_embed: int = 64,
        preprocess_network: snt.Module = None,
        non_monotonic: bool = False,
    ):
        """Initialise QMIX mixing network

        Args:
        ----
            num_agents: Number of agents in the environment
            state_dim: Dimensions of the global environment state
            embed_dim: The dimension of the output of the first layer
                of the mixer.
            hypernet_embed: Number of units in the hyper network

        """
        super().__init__()
        self.num_agents = num_agents
        self.embed_dim = embed_dim
        self.hypernet_embed = hypernet_embed
        self._non_monotonic = non_monotonic

        self.hyper_w_1 = snt.Sequential(
            [
                snt.Linear(self.hypernet_embed),
                tf.nn.relu,
                snt.Linear(self.embed_dim * self.num_agents),
            ]
        )

        self.hyper_w_final = snt.Sequential(
            [snt.Linear(self.hypernet_embed), tf.nn.relu, snt.Linear(self.embed_dim)]
        )

        # State dependent bias for hidden layer
        self.hyper_b_1 = snt.Linear(self.embed_dim)

        # V(s) instead of a bias for the last layers
        self.V = snt.Sequential([snt.Linear(self.embed_dim), tf.nn.relu, snt.Linear(1)])

    def __call__(self, agent_qs: Tensor, states: Tensor) -> Tensor:
        """Forward method."""
        B = agent_qs.shape[0]  # batch size
        state_dim = states.shape[2:]

        agent_qs = tf.reshape(agent_qs, (-1, 1, self.num_agents))

        # states = tf.ones_like(states)
        states = tf.reshape(states, (-1, *state_dim))

        # First layer
        w1 = self.hyper_w_1(states)
        if not self._non_monotonic:
            w1 = tf.abs(w1)
        b1 = self.hyper_b_1(states)
        w1 = tf.reshape(w1, (-1, self.num_agents, self.embed_dim))
        b1 = tf.reshape(b1, (-1, 1, self.embed_dim))
        hidden = tf.nn.elu(tf.matmul(agent_qs, w1) + b1)

        # Second layer
        w_final = self.hyper_w_final(states)
        if not self._non_monotonic:
            w_final = tf.abs(w_final)
        w_final = tf.reshape(w_final, (-1, self.embed_dim, 1))

        # State-dependent bias
        v = tf.reshape(self.V(states), (-1, 1, 1))

        # Compute final output
        y = tf.matmul(hidden, w_final) + v

        # Reshape and return
        q_tot = tf.reshape(y, (B, -1, 1))

        return q_tot

    def k(self, states: Tensor) -> Tensor:
        """Method used by MAICQ."""
        B, T = states.shape[:2]

        w1 = tf.math.abs(self.hyper_w_1(states))
        w_final = tf.math.abs(self.hyper_w_final(states))
        w1 = tf.reshape(w1, shape=(-1, self.num_agents, self.embed_dim))
        w_final = tf.reshape(w_final, shape=(-1, self.embed_dim, 1))
        k = tf.matmul(w1, w_final)
        k = tf.reshape(k, shape=(B, -1, self.num_agents))
        k = k / (tf.reduce_sum(k, axis=2, keepdims=True) + 1e-10)
        return k
