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

from typing import List, Dict, Optional
import copy
import trfl
import tensorflow as tf
import sonnet as snt

from og_marl.systems.qmix import QMixer
from og_marl.systems.trainer_base import TrainerBase
from og_marl.utils.trainer_utils import (
    batch_concat_agent_id_to_obs, 
    expand_batch_and_agent_dim_of_time_major_sequence, 
    merge_batch_and_agent_dim_of_time_major_sequence, 
    switch_two_leading_dims,
    sample_batch_agents,
    gather
)

class MAICQTrainer(TrainerBase):
    def __init__(
        self,
        agents,
        dataset,
        logger,
        policy_network: snt.Module,
        policy_optimizer,
        critic_network: snt.Module,
        critic_optimizer: snt.Optimizer,
        mixer: snt.Module,
        discount: float = 0.99,
        lambda_: float = 0.6,
        target_update_period: int = 600,
        max_gradient_norm: float = 20.0,
        add_agent_id_to_obs = False,
        max_trainer_steps=1e6
    ):
        """Initialise trainer."""

        super().__init__(
            agents,
            dataset,
            logger,
            discount=discount,
            max_gradient_norm=max_gradient_norm,
            add_agent_id_to_obs=add_agent_id_to_obs,
            max_trainer_steps=max_trainer_steps
        )

        # Optimizers
        self._policy_optimizer = policy_optimizer
        self._critic_optimizer = critic_optimizer

        # Store Q-network and Policy Network and Mixer
        self._policy_network = policy_network
        self._critic_network = critic_network
        self._mixer = mixer

        # Target networks
        self._target_critic_network = copy.deepcopy(critic_network)
        self._target_update_period = target_update_period
        self._target_mixer = copy.deepcopy(mixer)

        # Expose the network variables.
        self._system_variables: Dict = {
            "policy_network": self._policy_network.variables,
        }

        self._lambda = lambda_

        # ICQ Hyper-parameters
        self._epsilon = 0.5  # from MAICQ code
        self._advantages_beta = 0.1  # from MAICQ code
        self._target_q_taken_beta = 1000  # from MAICQ code

    @tf.function
    def _train(self, sample, trainer_step):

        # Batch agent inputs together
        batch = sample_batch_agents(self._agents, sample)

        # Get max sequence length, batch size, num agents and num actions
        B, T, N, A = batch["legals"].shape

        # Unpack the relevant quantities
        observations = batch["observations"]
        actions = batch["actions"]
        legal_actions = batch["legals"]
        states = batch["states"]
        rewards = batch["rewards"]
        env_discounts = batch["discounts"]
        zero_padding_mask = batch["mask"]

        if self._add_agent_id_to_obs:
            observations = batch_concat_agent_id_to_obs(observations)

        with tf.GradientTape(persistent=True) as tape:

            critic_states = [states] * N
            critic_states = tf.stack(critic_states, axis=2) # duplicate states for all agents

            q_vals = self._critic_network(observations, critic_states)
            target_q_vals = self._target_critic_network(observations, critic_states)

            # Unroll the policy
            logits_out, _ = snt.static_unroll(
                self._policy_network,
                merge_batch_and_agent_dim_of_time_major_sequence(switch_two_leading_dims(observations)),
                self._policy_network.initial_state(B*N)
            )

            logits_out = switch_two_leading_dims(expand_batch_and_agent_dim_of_time_major_sequence(logits_out, B, N))

            # Compute probabilities
            probs_out = tf.nn.softmax(logits_out, axis=-1)

            # Mask illegal actions
            probs_out = probs_out * tf.cast(legal_actions, "float32")
            probs_sum = (
                tf.reduce_sum(probs_out, axis=-1, keepdims=True) + 1e-10
            )  # avoid div by zero
            probs_out = probs_out / probs_sum

            action_values = gather(q_vals, actions)
            baseline = tf.reduce_sum(probs_out * q_vals, axis=-1)
            advantages = action_values - baseline
            advantages = tf.nn.softmax(advantages / self._advantages_beta, axis=0)
            advantages = tf.stop_gradient(advantages)

            pi_taken = gather(probs_out, actions, keepdims=False)
            pi_taken = tf.where(tf.cast(zero_padding_mask, "bool"), pi_taken, 1.0)
            log_pi_taken = tf.math.log(pi_taken)

            coe = self._mixer.k(states)

            coma_mask = tf.concat([zero_padding_mask] * N, axis=2)
            coma_loss = -tf.reduce_sum(
                coe * (len(advantages) * advantages * log_pi_taken) * coma_mask
            ) / tf.reduce_sum(coma_mask)

            # Critic learning
            q_taken = gather(q_vals, actions, axis=-1)
            target_q_taken = gather(target_q_vals, actions, axis=-1)

            # Mixing critics
            q_taken = self._mixer(q_taken, states)
            target_q_taken = self._target_mixer(target_q_taken, states)

            advantage_Q = tf.nn.softmax(
                target_q_taken / self._target_q_taken_beta, axis=0
            )
            target_q_taken = len(advantage_Q) * advantage_Q * target_q_taken

            # Make time major for trfl
            rewards = switch_two_leading_dims(rewards)
            env_discounts = switch_two_leading_dims(env_discounts)
            target_q_taken = switch_two_leading_dims(target_q_taken)

            # Q(lambda)
            target_q = trfl.multistep_forward_view(
                tf.squeeze(rewards[:-1, :]),
                tf.squeeze(self._discount * env_discounts[:-1, :]),
                tf.squeeze(target_q_taken[1:, :]),
                lambda_=self._lambda,
                back_prop=False,
            )
            # Make batch major again
            target_q = switch_two_leading_dims(target_q)
            target_q = tf.expand_dims(target_q, axis=-1)

            # TD error
            td_error = tf.stop_gradient(target_q) - q_taken[:, :-1]
            q_loss = 0.5 * tf.square(td_error)

            # Masking
            q_loss = tf.reduce_sum(q_loss * zero_padding_mask[:, :-1]) / tf.reduce_sum(
                zero_padding_mask[:, :-1]
            )

        # Apply gradients to policy
        variables = (
            *self._policy_network.trainable_variables,
        )  # Get trainable variables

        gradients = tape.gradient(coma_loss, variables)  # Compute gradients.
        gradients = tf.clip_by_global_norm(gradients, self._max_gradient_norm)[
            0
        ]  # Maybe clip gradients.
        self._policy_optimizer.apply(
            gradients, variables
        )  # One optimizer for whole system

        # Apply gradients to critic and mixer
        variables = (
            *self._critic_network.trainable_variables,
            *self._mixer.trainable_variables,
        )  # Get trainable variables

        gradients = tape.gradient(q_loss, variables)  # Compute gradients.
        gradients = tf.clip_by_global_norm(gradients, self._max_gradient_norm)[
            0
        ]  # Maybe clip gradients.
        self._critic_optimizer.apply(
            gradients, variables
        )  # One optimizer for whole system

        # Maybe update target network
        self._update_target_network(trainer_step)

        del tape

        return {
            "critic_loss": q_loss,
            "policy_loss": coma_loss,
            "mean_q_vals": tf.reduce_mean(q_vals),
            "mean_q_taken_after_mix": tf.reduce_mean(q_taken),
            "trainer_steps": trainer_step,
        }

    def _update_target_network(self, trainer_step):
        # Online variables
        online_variables = (
            *self._critic_network.variables,
            *self._mixer.variables,
        )

        # Get target variables
        target_variables = (
            *self._target_critic_network.variables,
            *self._target_mixer.variables,
        )

        if tf.math.mod(trainer_step, self._target_update_period) == 0:
            # Make online -> target network update ops.
            for src, dest in zip(online_variables, target_variables):
                dest.assign(src)
