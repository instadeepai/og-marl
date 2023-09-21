"""Implementation of independent Q-learning (DRQN style)"""
import copy
import tensorflow as tf
import sonnet as snt
import tree

from og_marl.systems.base import BaseMARLSystem
from og_marl.utils import (
    batched_agents,
    gather,
    batch_concat_agent_id_to_obs,
    switch_two_leading_dims,
    merge_batch_and_agent_dim_of_time_major_sequence,
    expand_batch_and_agent_dim_of_time_major_sequence,
    set_growing_gpu_memory,
    concat_agent_id_to_obs
)

set_growing_gpu_memory()


class IDRQNSystem(BaseMARLSystem):
    """Independent Deep Recurrent Q-Networs System"""

    def __init__(
        self,
        environment,
        logger,
        linear_layer_dim=100,
        recurrent_layer_dim=100,
        discount=0.99,
        target_update_period=200,
        learning_rate=3e-4,
        eps_min=0.05,
        eps_decay_timesteps=50_000,
        add_agent_id_to_obs=True,
    ):
        super().__init__(
            environment,
            logger,
            add_agent_id_to_obs=add_agent_id_to_obs,
            discount=discount
        )

        self._linear_layer_dim = linear_layer_dim
        self._recurrent_layer_dim = recurrent_layer_dim

        # Exploration
        self._eps_dec_timesteps = eps_decay_timesteps
        self._eps_min = eps_min
        self._eps_dec = (1.0-self._eps_min) / self._eps_dec_timesteps

        # Q-network
        self._q_network =snt.DeepRNN(
            [
                snt.Linear(linear_layer_dim),
                tf.nn.relu,
                snt.GRU(recurrent_layer_dim),
                tf.nn.relu,
                snt.Linear(self._environment._num_actions)
            ]
        ) # shared network for all agents

        # Target Q-network
        self._target_q_network = copy.deepcopy(self._q_network)
        self._target_update_period = target_update_period
        self._train_step_ctr = 0

        # Optimizer
        self._optimizer=snt.optimizers.Adam(learning_rate=learning_rate)

        # Reset the recurrent neural network
        self._rnn_states = {agent: self._q_network.initial_state(1) for agent in self._environment.possible_agents}

    def reset(self):
        """Called at the start of a new episode."""

        # Reset the recurrent neural network
        self._rnn_states = {agent: self._q_network.initial_state(1) for agent in self._environment.possible_agents}

        return

    def select_actions(self, observations, legal_actions=None, explore=True):
        if explore:
            self._env_step_ctr += 1.0

        env_step_ctr, observations, legal_actions = tree.map_structure(tf.convert_to_tensor, (self._env_step_ctr, observations, legal_actions))
        actions, next_rnn_states = self._tf_select_actions(env_step_ctr, observations, legal_actions, self._rnn_states, explore)
        self._rnn_states = next_rnn_states
        return tree.map_structure(lambda x: x.numpy(), actions) # convert to numpy and squeeze batch dim

    @tf.function(jit_compile=True)
    def _tf_select_actions(self, env_step_ctr, observations, legal_actions, rnn_states, explore):
        actions = {}
        next_rnn_states = {}
        for i, agent in enumerate(self._environment.possible_agents):
            agent_observation = observations[agent]
            if self._add_agent_id_to_obs:
                agent_observation = concat_agent_id_to_obs(agent_observation, i, len(self._environment.possible_agents))
            agent_observation = tf.expand_dims(agent_observation, axis=0) # add batch dimension
            q_values, next_rnn_states[agent] = self._q_network(agent_observation, rnn_states[agent])

            agent_legal_actions = legal_actions[agent]
            masked_q_values = tf.where(
                tf.equal(agent_legal_actions, 1),
                q_values[0],
                -99999999,
            )
            greedy_action = tf.argmax(masked_q_values)

            epsilon = tf.maximum(1.0 - self._eps_dec * env_step_ctr, self._eps_min)

            greedy_logits = tf.math.log(tf.one_hot(greedy_action, masked_q_values.shape[-1]))
            logits = (1.0-epsilon) * greedy_logits + epsilon * tf.math.log(agent_legal_actions)
            logits = tf.expand_dims(logits, axis=0)

            if explore:
                action = tf.random.categorical(logits, 1)
            else:
                action = greedy_action

            # Max Q-value over legal actions
            actions[agent] = action

        return actions, next_rnn_states
    
    def train_step(self, batch):
        self._train_step_ctr += 1
        logs = self._tf_train_step(tf.convert_to_tensor(self._train_step_ctr), batch)
        return logs

    @tf.function(jit_compile=True) # NOTE: comment this out if using debugger
    def _tf_train_step(self, train_step_ctr, batch):
        batch = batched_agents(self._environment.possible_agents, batch)

        # Unpack the batch
        observations = batch["observations"] # (B,T,N,O)
        actions = batch["actions"] # (B,T,N)
        env_states = batch["state"] # (B,T,S)
        rewards = batch["rewards"] # (B,T,N)
        truncations = batch["truncations"] # (B,T,N)
        terminals = batch["terminals"] # (B,T,N)
        zero_padding_mask = batch["mask"] # (B,T)
        legal_actions = batch["legals"]  # (B,T,N,A)

        # done = tf.cast(tf.logical_or(tf.cast(truncations, "bool"), tf.cast(terminals, "bool")), "float32")
        done = terminals

        # Get dims
        B, T, N, A = legal_actions.shape

        # Maybe add agent ids to observation
        if self._add_agent_id_to_obs:
            observations = batch_concat_agent_id_to_obs(observations)

        # Make time-major
        observations = switch_two_leading_dims(observations)

        # Merge batch_dim and agent_dim
        observations = merge_batch_and_agent_dim_of_time_major_sequence(observations)

        # Unroll target network
        target_qs_out, _ = snt.static_unroll(
            self._target_q_network, 
            observations,
            self._target_q_network.initial_state(B*N)
        )

        # Expand batch and agent_dim
        target_qs_out = expand_batch_and_agent_dim_of_time_major_sequence(target_qs_out, B, N)

        # Make batch-major again
        target_qs_out = switch_two_leading_dims(target_qs_out)

        with tf.GradientTape() as tape:
            # Unroll online network
            qs_out, _ = snt.static_unroll(
                self._q_network, 
                observations, 
                self._q_network.initial_state(B*N)
            )

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
            targets = rewards[:, :-1] + (1-done[:, :-1]) * self._discount * target_max_qs[:, 1:]
            targets = tf.stop_gradient(targets)

            # Chop off last time step
            chosen_action_qs = chosen_action_qs[:, :-1]  # shape=(B,T-1)

            # TD-Error Loss
            loss = 0.5 * tf.square(targets - chosen_action_qs)

            # Mask out zero-padded timesteps
            loss = self._apply_mask(loss, zero_padding_mask)

        # Get trainable variables
        variables = (
            *self._q_network.trainable_variables,
        )

        # Compute gradients.
        gradients = tape.gradient(loss, variables)

        # Apply gradients.
        self._optimizer.apply(gradients, variables)

        # Online variables
        online_variables = (
            *self._q_network.variables,
        )

        # Get target variables
        target_variables = (
            *self._target_q_network.variables,
        )

        # Maybe update target network
        self._update_target_network(train_step_ctr, online_variables, target_variables)

        return {
            "Loss": loss,
            "Mean Q-values": tf.reduce_mean(qs_out),
            "Mean Chosen Q-values": tf.reduce_mean(chosen_action_qs),
        }
    
    def get_stats(self):
        return {"Epsilon": max(1.0 - self._env_step_ctr * self._eps_dec, self._eps_min)}

    def _apply_mask(self, loss, mask):
        mask = tf.expand_dims(mask, axis=-1)
        mask = tf.broadcast_to(mask[:, :-1], loss.shape)
        loss = tf.reduce_sum(loss * mask) / tf.reduce_sum(mask)
        return loss

    def _update_target_network(self, train_step, online_variables, target_variables):
        """Update the target networks."""

        if train_step % self._target_update_period == 0:
            for src, dest in zip(online_variables, target_variables):
                dest.assign(src)

        # tau = self._target_update_rate
        # for src, dest in zip(online_variables, target_variables):
        #     dest.assign(dest * (1.0 - tau) + src * tau)