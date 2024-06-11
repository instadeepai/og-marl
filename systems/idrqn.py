import copy
from typing import Any, Dict, Sequence, Tuple, Optional

import numpy as np
import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp
import tree
from chex import Numeric
from tensorflow import Tensor, Variable

from environment_wrappers.base import BaseEnvironment
from utils.loggers import BaseLogger
from utils.networks import IdentityNetwork
from utils.replay_buffers import Experience
from systems.base import BaseMARLSystem
from utils.utils import (
    batch_concat_agent_id_to_obs,
    concat_agent_id_to_obs,
    expand_batch_and_agent_dim_of_time_major_sequence,
    gather,
    merge_batch_and_agent_dim_of_time_major_sequence,
    switch_two_leading_dims,
    unroll_rnn,
)

class IDRQNSystem(BaseMARLSystem):

    def __init__(
        self,
        environment: BaseEnvironment,
        logger: BaseLogger,
        linear_layer_dim: int = 64,
        recurrent_layer_dim: int = 64,
        discount: float = 0.99,
        target_update_period: int = 200,
        learning_rate: float = 3e-4,
        add_agent_id_to_obs_in_trainer = True,
        add_agent_id_to_obs_in_action_selection: bool = True,
    ):
        super().__init__(
            environment, logger, add_agent_id_to_obs_in_trainer=add_agent_id_to_obs_in_trainer, 
            add_agent_id_to_obs_in_action_selection=add_agent_id_to_obs_in_action_selection, 
            discount=discount
        )

        # Q-network
        self._q_network = snt.DeepRNN(
            [
                snt.Linear(linear_layer_dim),
                tf.nn.relu,
                snt.GRU(recurrent_layer_dim),
                tf.nn.relu,
                snt.Linear(self._environment._num_actions),
            ]
        )  # shared network for all agents

        # Embedding network (not used!)
        observation_embedding_network = IdentityNetwork()
        self._q_embedding_network = observation_embedding_network
        self._target_q_embedding_network = copy.deepcopy(observation_embedding_network)

        # Target Q-network
        self._target_q_network = copy.deepcopy(self._q_network)
        self._target_update_period = target_update_period
        self._train_step_ctr = 0

        # Optimizer
        self._optimizer = snt.optimizers.Adam(learning_rate=learning_rate)

        # Reset the recurrent neural network
        self._rnn_states = {
            agent: self._q_network.initial_state(1) for agent in self._environment.possible_agents
        }

    def reset(self) -> None:
        """Called at the start of a new episode."""
        # Reset the recurrent neural network
        self._rnn_states = {
            agent: self._q_network.initial_state(1) for agent in self._environment.possible_agents
        }

        return

    def select_actions(
        self,
        observations: Dict[str, np.ndarray],
        legal_actions: Dict[str, np.ndarray],
        explore: bool = False,
    ) -> Dict[str, np.ndarray]:

        env_step_ctr, observations, legal_actions = tree.map_structure(
            tf.convert_to_tensor, (self._env_step_ctr, observations, legal_actions)
        )
        actions, next_rnn_states = self._tf_select_actions(
            env_step_ctr, observations, legal_actions, self._rnn_states, explore
        )
        self._rnn_states = next_rnn_states
        return tree.map_structure(  # type: ignore
            lambda x: x.numpy(), actions
        )  # convert to numpy and squeeze batch dim

    @tf.function(jit_compile=True)
    def _tf_select_actions(
        self,
        env_step_ctr: int,
        observations: Dict[str, Tensor],
        legal_actions: Dict[str, Tensor],
        rnn_states: Dict[str, Tensor],
        explore: bool,
    ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        actions = {}
        next_rnn_states = {}
        for i, agent in enumerate(self._environment.possible_agents):
            agent_observation = observations[agent]
            if self._add_agent_id_to_obs_in_action_selection:
                agent_observation = concat_agent_id_to_obs(
                    agent_observation, i, len(self._environment.possible_agents)
                )
            agent_observation = tf.expand_dims(agent_observation, axis=0)  # add batch dimension
            embedding = self._q_embedding_network(agent_observation)
            q_values, next_rnn_states[agent] = self._q_network(embedding, rnn_states[agent])

            agent_legal_actions = legal_actions[agent]
            masked_q_values = tf.where(
                tf.equal(agent_legal_actions, 1),
                q_values[0],
                -99999999,
            )
            greedy_action = tf.argmax(masked_q_values)
            action = greedy_action

            # Max Q-value over legal actions
            actions[agent] = action

        return actions, next_rnn_states

    def train_step(self, experience: Experience) -> Dict[str, Numeric]:
        self._train_step_ctr += 1
        logs = self._tf_train_step(tf.convert_to_tensor(self._train_step_ctr), experience)
        return logs  # type: ignore

    @tf.function(jit_compile=True)  # NOTE: comment this out if using debugger
    def _tf_train_step(self, train_step_ctr: int, experience: Dict[str, Any]) -> Dict[str, Numeric]:
        # Unpack the batch
        observations = experience["observations"]  # (B,T,N,O)
        actions = experience["actions"]  # (B,T,N)
        rewards = experience["rewards"]  # (B,T,N)
        truncations = experience["truncations"]  # (B,T,N)
        terminals = experience["terminals"]  # (B,T,N)
        legal_actions = experience["infos"]["legals"]  # (B,T,N,A)

        # When to reset the RNN hidden state
        resets = tf.maximum(terminals, truncations)  # equivalent to logical 'or'

        # Get dims
        B, T, N, A = legal_actions.shape

        # Maybe add agent ids to observation
        if self._add_agent_id_to_obs_in_trainer:
            observations = batch_concat_agent_id_to_obs(observations)

        # Make time-major
        observations = switch_two_leading_dims(observations)
        resets = switch_two_leading_dims(resets)

        # Merge batch_dim and agent_dim
        observations = merge_batch_and_agent_dim_of_time_major_sequence(observations)
        resets = merge_batch_and_agent_dim_of_time_major_sequence(resets)

        # Unroll target network
        target_embeddings = self._target_q_embedding_network(observations)
        target_qs_out = unroll_rnn(self._target_q_network, target_embeddings, resets)

        # Expand batch and agent_dim
        target_qs_out = expand_batch_and_agent_dim_of_time_major_sequence(target_qs_out, B, N)

        # Make batch-major again
        target_qs_out = switch_two_leading_dims(target_qs_out)

        with tf.GradientTape() as tape:
            # Unroll online network
            embeddings = self._q_embedding_network(observations)
            qs_out = unroll_rnn(self._q_network, embeddings, resets)

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

            # Compute targets
            targets = (
                rewards[:, :-1] + (1 - terminals[:, :-1]) * self._discount * target_max_qs[:, 1:]
            )
            targets = tf.stop_gradient(targets)

            # Chop off last time step
            chosen_action_qs = chosen_action_qs[:, :-1]  # shape=(B,T-1)

            # TD-Error Loss
            loss = 0.5 * tf.square(targets - chosen_action_qs)

            # Mask out zero-padded timesteps
            loss = tf.reduce_mean(loss)

        # Get trainable variables
        variables = (
            *self._q_network.trainable_variables,
            *self._q_embedding_network.trainable_variables,
        )

        # Compute gradients.
        gradients = tape.gradient(loss, variables)

        # Apply gradients.
        self._optimizer.apply(gradients, variables)

        # Online variables
        online_variables = (*self._q_network.variables, *self._q_embedding_network.variables)

        # Get target variables
        target_variables = (
            *self._target_q_network.variables,
            *self._target_q_embedding_network.variables,
        )

        # Maybe update target network
        self._update_target_network(train_step_ctr, online_variables, target_variables)

        return {
            "Loss": loss,
            "Mean Q-values": tf.reduce_mean(qs_out),
            "Mean Chosen Q-values": tf.reduce_mean(chosen_action_qs),
        }

    def get_stats(self) -> Dict[str, Numeric]:
        return {"Epsilon": max(1.0 - self._env_step_ctr * self._eps_dec, self._eps_min)}

    def _apply_mask(self, loss: Tensor, mask: Tensor) -> Numeric:
        mask = tf.expand_dims(mask, axis=-1)
        mask = tf.broadcast_to(mask[:, :-1], loss.shape)
        loss = tf.reduce_sum(loss * mask) / tf.reduce_sum(mask)
        return loss

    def _update_target_network(
        self,
        train_step: int,
        online_variables: Sequence[Variable],
        target_variables: Sequence[Variable],
    ) -> None:
        """Update the target networks."""
        if train_step % self._target_update_period == 0:
            for src, dest in zip(online_variables, target_variables):
                dest.assign(src)
