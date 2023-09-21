"""Implementation of MAICQ"""
import tensorflow as tf
import sonnet as snt
import tree

from og_marl.systems.qmix import QMIXSystem
from og_marl.utils import (
    concat_agent_id_to_obs,
    gather,
    batch_concat_agent_id_to_obs,
    switch_two_leading_dims,
    merge_batch_and_agent_dim_of_time_major_sequence,
    expand_batch_and_agent_dim_of_time_major_sequence,
    set_growing_gpu_memory,
    dict_to_tensor,
)

set_growing_gpu_memory()


class MAICQSystem(QMIXSystem):
    """MAICQ System"""

    def __init__(
        self,
        environment,
        logger,
        icq_advantages_beta = 0.1,  # from MAICQ code
        icq_target_q_taken_beta = 1000,  # from MAICQ code
        linear_layer_dim=100,
        recurrent_layer_dim=100,
        mixer_embed_dim=64,
        mixer_hyper_dim=32,
        batch_size=64,
        discount=0.99,
        target_update_rate=0.005,
        learning_rate=3e-4,
        add_agent_id_to_obs=False,
    ):

        super().__init__(
            environment,
            logger,
            linear_layer_dim=linear_layer_dim,
            recurrent_layer_dim=recurrent_layer_dim,
            mixer_embed_dim=mixer_embed_dim,
            mixer_hyper_dim=mixer_hyper_dim,
            add_agent_id_to_obs=add_agent_id_to_obs,
            batch_size=batch_size,
            discount=discount,
            target_update_rate=target_update_rate,
            learning_rate=learning_rate
        )

        # ICQ
        self._icq_advantages_beta = icq_advantages_beta
        self._icq_target_q_taken_beta = icq_target_q_taken_beta

        # Policy Network
        self._policy_network = snt.DeepRNN(
            [
                snt.Linear(self._linear_layer_dim),
                tf.nn.relu,
                snt.GRU(self._recurrent_layer_dim),
                tf.nn.relu,
                snt.Linear(self._environment.num_actions),
                tf.nn.softmax,
            ]
        )
    
    def reset(self):
        """Called at the start of a new episode."""

        # Reset the recurrent neural network
        self._rnn_states = {agent: self._policy_network.initial_state(1) for agent in self._environment._agents}

        return

    def select_actions(self, observations):
        observations = tree.map_structure(tf.convert_to_tensor, observations)
        actions, next_rnn_states = self._tf_select_actions(observations, self._rnn_states)
        self._rnn_states = next_rnn_states
        return tree.map_structure(lambda x: x.numpy(), actions) # convert to numpy and squeeze batch dim

    @tf.function()
    def _tf_select_actions(self, observations, rnn_states):
        actions = {}
        next_rnn_states = {}
        for i, agent in enumerate(self._environment._agents):
            agent_observation = observations[agent].observation
            agent_observation = concat_agent_id_to_obs(agent_observation, i, len(self._environment._agents))
            agent_observation = tf.expand_dims(agent_observation, axis=0) # add batch dimension
            probs, next_rnn_states[agent] = self._policy_network(agent_observation, rnn_states[agent])

            agent_legal_actions = observations[agent].legal_actions
            masked_probs = tf.where(
                tf.equal(agent_legal_actions, 1),
                probs[0],
                -99999999,
            )

            # Max Q-value over legal actions
            actions[agent] = tf.argmax(masked_probs)

        return actions, next_rnn_states

    @tf.function(jit_compile=True)
    def _tf_train_step(self, batch):
        batch = dict_to_tensor(self._environment._agents, batch)

        # Unpack the batch
        observations = batch.observations # (B,T,N,O)
        actions = batch.actions # (B,T,N,A)
        legal_actions = batch.legal_actions # (B,T,N,A)
        env_states = batch.env_state # (B,T,S)
        rewards = batch.rewards # (B,T,N)
        done = batch.done # (B,T)
        zero_padding_mask = batch.zero_padding_mask # (B,T)

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
        target_q_vals = switch_two_leading_dims(target_qs_out)

        with tf.GradientTape(persistent=True) as tape:
            # Unroll online network
            qs_out, _ = snt.static_unroll(
                self._q_network, 
                observations, 
                self._q_network.initial_state(B*N)
            )

            # Expand batch and agent_dim
            qs_out = expand_batch_and_agent_dim_of_time_major_sequence(qs_out, B, N)

            # Make batch-major again
            q_vals = switch_two_leading_dims(qs_out)

            # Unroll the policy
            probs_out, _ = snt.static_unroll(
                self._policy_network,
                observations,
                self._policy_network.initial_state(B*N)
            )

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
            advantages = tf.nn.softmax(advantages / self._icq_advantages_beta, axis=0)
            advantages = tf.stop_gradient(advantages)

            pi_taken = gather(probs_out, actions, keepdims=False)
            pi_taken = tf.where(tf.cast(zero_padding_mask, "bool"), pi_taken, 1.0)
            log_pi_taken = tf.math.log(pi_taken)

            coe = self._mixer.k(env_states)

            coma_mask = tf.concat([zero_padding_mask] * N, axis=2)
            coma_loss = -tf.reduce_sum(
                coe * (len(advantages) * advantages * log_pi_taken) * coma_mask
            ) / tf.reduce_sum(coma_mask)

            # Critic learning
            q_taken = gather(q_vals, actions, axis=-1)
            target_q_taken = gather(target_q_vals, actions, axis=-1)

            # Mixing critics
            q_taken = self._mixer(q_taken, env_states)
            target_q_taken = self._target_mixer(target_q_taken, env_states)

            advantage_Q = tf.nn.softmax(
                target_q_taken / self._icq_target_q_taken_beta, axis=0
            )
            target_q_taken = len(advantage_Q) * advantage_Q * target_q_taken

            # Compute targets
            targets = rewards[:, :-1] + tf.expand_dims((1-done[:, :-1]), axis=-1) * self._discount * target_q_taken[:, 1:]
            targets = tf.stop_gradient(targets)

            # TD error
            td_error = targets - q_taken[:, :-1]
            q_loss = 0.5 * tf.square(td_error)

            # Masking
            q_loss = tf.reduce_sum(q_loss * zero_padding_mask[:, :-1]) / tf.reduce_sum(
                zero_padding_mask[:, :-1]
            )

            # Add losses together
            loss = q_loss + coma_loss

        # Apply gradients to policy
        variables = (
            *self._policy_network.trainable_variables,
            *self._q_network.trainable_variables,
            *self._mixer.trainable_variables
        )  # Get trainable variables

        gradients = tape.gradient(loss, variables)  # Compute gradients.

        self._optimizer.apply(
            gradients, variables
        )  # One optimizer for whole system

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
        self._update_target_network(online_variables, target_variables)

        return {
            "Critic Loss": q_loss,
            "Policy Loss": coma_loss,
            "Loss": loss,
        }