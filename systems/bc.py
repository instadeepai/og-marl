from typing import Any, Dict, Optional, Tuple

import numpy as np
import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp
import tree
from chex import Numeric

from environment_wrappers.base import BaseEnvironment
from utils.loggers import BaseLogger
from utils.replay_buffers import Experience
from systems.base import BaseMARLSystem
from utils.utils import (
    batch_concat_agent_id_to_obs,
    concat_agent_id_to_obs,
    expand_batch_and_agent_dim_of_time_major_sequence,
    merge_batch_and_agent_dim_of_time_major_sequence,
    switch_two_leading_dims,
    unroll_rnn,
)
from utils.networks import IdentityNetwork


class DicreteActionBehaviourCloning(BaseMARLSystem):

    def __init__(
        self,
        environment: BaseEnvironment,
        logger: BaseLogger,
        linear_layer_dim: int = 64,
        recurrent_layer_dim: int = 64,
        discount: float = 0.99,
        learning_rate: float = 1e-3,
        add_agent_id_to_obs_in_trainer: bool = True,
        add_agent_id_to_obs_in_action_selection = True
    ):
        super().__init__(
            environment, logger, discount=discount, add_agent_id_to_obs_in_action_selection=add_agent_id_to_obs_in_action_selection, add_agent_id_to_obs_in_trainer=add_agent_id_to_obs_in_trainer
        )

        # Policy network
        self._policy_network = snt.DeepRNN(
            [
                snt.Linear(linear_layer_dim),
                tf.nn.relu,
                snt.GRU(recurrent_layer_dim),
                tf.nn.relu,
                snt.Linear(self._environment._num_actions),
            ]
        )  # shared network for all agents
        self._policy_embedding_network = IdentityNetwork()

        self._optimizer = snt.optimizers.RMSProp(learning_rate=learning_rate)

        # Reset the recurrent neural network
        self._rnn_states = {
            agent: self._policy_network.initial_state(1)
            for agent in self._environment.possible_agents
        }

    def reset(self) -> None:
        """Called at the start of a new episode."""
        # Reset the recurrent neural network
        self._rnn_states = {
            agent: self._policy_network.initial_state(1)
            for agent in self._environment.possible_agents
        }

        return

    def select_actions(
        self,
        observations: Dict[str, np.ndarray],
        legal_actions: Optional[Dict[str, np.ndarray]] = None,
        explore: bool = True,
    ) -> Dict[str, np.ndarray]:
        observations, legal_actions = tree.map_structure(
            tf.convert_to_tensor, (observations, legal_actions)
        )

        actions, next_rnn_states = self._tf_select_actions(
            observations, self._rnn_states, legal_actions
        )
        self._rnn_states = next_rnn_states
        return tree.map_structure(  # type: ignore
            lambda x: x[0].numpy(), actions
        )  # convert to numpy and squeeze batch dim

    @tf.function()
    def _tf_select_actions(
        self,
        observations: Dict[str, tf.Tensor],
        rnn_states: Dict[str, tf.Tensor],
        legal_actions: Optional[Dict[str, tf.Tensor]] = None,
    ) -> Tuple[Dict[str, tf.Tensor], Dict[str, tf.Tensor]]:
        actions = {}
        next_rnn_states = {}
        for i, agent in enumerate(self._environment.possible_agents):
            agent_observation = observations[agent]
            if self._add_agent_id_to_obs_in_action_selection:
                agent_observation = concat_agent_id_to_obs(
                    agent_observation, i, len(self._environment.possible_agents)
                )
            agent_observation = tf.expand_dims(agent_observation, axis=0)  # add batch dimension
            embedding = self._policy_embedding_network(agent_observation)
            logits, next_rnn_states[agent] = self._policy_network(embedding, rnn_states[agent])

            probs = tf.nn.softmax(logits)

            if legal_actions is not None:
                agent_legals = tf.cast(tf.expand_dims(legal_actions[agent], axis=0), "float32")
                probs = (probs * agent_legals) / tf.reduce_sum(
                    probs * agent_legals
                )  # mask and renorm

            action = tfp.distributions.Categorical(probs=probs).sample(1)

            # Store agent action
            actions[agent] = action[0]

        return actions, next_rnn_states

    def train_step(self, experience: Experience) -> Dict[str, Numeric]:
        logs = self._tf_train_step(experience)
        return logs  # type: ignore

    @tf.function(jit_compile=True)
    def _tf_train_step(self, experience: Dict[str, Any]) -> Dict[str, Numeric]:
        # Unpack the relevant quantities
        observations = experience["observations"]
        actions = tf.cast(experience["actions"], "int32")[:,:,:,0]
        truncations = tf.cast(experience["truncations"], "float32")  # (B,T,N)
        terminals = tf.cast(experience["terminals"], "float32")  # (B,T,N)

        # When to reset the RNN hidden state
        resets = tf.maximum(terminals, truncations)  # equivalent to logical 'or'

        # Get batch size, max sequence length, num agents and num actions
        B, T, N, A = experience["infos"]["legals"].shape

        # Maybe add agent ids to observation
        if self._add_agent_id_to_obs_in_trainer:
            observations = batch_concat_agent_id_to_obs(observations)

        # Make time-major
        observations = switch_two_leading_dims(observations)
        resets = switch_two_leading_dims(resets)
        actions = switch_two_leading_dims(actions)

        # Merge batch_dim and agent_dim
        observations = merge_batch_and_agent_dim_of_time_major_sequence(observations)
        resets = merge_batch_and_agent_dim_of_time_major_sequence(resets)

        with tf.GradientTape() as tape:
            embeddings = self._policy_embedding_network(observations)
            probs_out = unroll_rnn(
                self._policy_network,
                embeddings,
                resets,
            )
            probs_out = expand_batch_and_agent_dim_of_time_major_sequence(probs_out, B, N)

            # Behaviour cloning loss
            one_hot_actions = tf.one_hot(actions, depth=probs_out.shape[-1], axis=-1)
            bc_loss = tf.keras.metrics.categorical_crossentropy(
                one_hot_actions, probs_out, from_logits=True
            )
            bc_loss = tf.reduce_mean(bc_loss)

        # Apply gradients to policy
        variables = (
            *self._policy_network.trainable_variables,
            *self._policy_embedding_network.trainable_variables,
        )  # Get trainable variables

        gradients = tape.gradient(bc_loss, variables)  # Compute gradients.
        self._optimizer.apply(gradients, variables)

        logs = {"Policy Loss": bc_loss}

        return logs