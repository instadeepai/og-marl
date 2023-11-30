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

"""Implementation of OMAR"""
import tensorflow as tf
import sonnet as snt
import numpy as np
import tensorflow_probability as tfp 

from og_marl.tf2.systems.iddpg_cql import IDDPGCQLSystem
from og_marl.tf2.utils import (
    batch_concat_agent_id_to_obs,
    batched_agents,
    switch_two_leading_dims,
    merge_batch_and_agent_dim_of_time_major_sequence,
    expand_batch_and_agent_dim_of_time_major_sequence,
)

class OMARSystem(IDDPGCQLSystem):

    def __init__(
        self, 
        environment,
        logger,
        linear_layer_dim=64,
        recurrent_layer_dim=64,
        discount=0.99,
        target_update_rate=0.005,
        critic_learning_rate=3e-4,
        policy_learning_rate=1e-3,
        add_agent_id_to_obs=False,
        num_ood_actions=4, # CQL
        cql_weight=10.0, # CQL  
        cql_sigma=0.2, # CQL
        target_action_gap=10.0, # CQL
        cql_alpha_learning_rate=3e-4, # CQL
        omar_iters=3, # OMAR
        omar_num_samples=5, # OMAR
        omar_num_elites=5, # OMAR
        omar_sigma=2.0, # OMAR
        omar_coe=0.7, # OMAR
    ):
        super().__init__(
            environment=environment,
            logger=logger,
            linear_layer_dim=linear_layer_dim,
            recurrent_layer_dim=recurrent_layer_dim,
            discount=discount,
            target_update_rate=target_update_rate,
            critic_learning_rate=critic_learning_rate,
            policy_learning_rate=policy_learning_rate,
            add_agent_id_to_obs=add_agent_id_to_obs,
            num_ood_actions=num_ood_actions,
            cql_weight=cql_weight,
            cql_sigma=cql_sigma,
            target_action_gap=target_action_gap,
            cql_alpha_learning_rate=cql_alpha_learning_rate
        )

        self._omar_iters = omar_iters
        self._omar_num_samples = omar_num_samples
        self._omar_num_elites = omar_num_elites
        self._omar_coe = omar_coe

        self._init_omar_mu, self._init_omar_sigma = 0.0, omar_sigma

    def _policy_loss(self, observations, actions, env_states, zero_padding_mask):
        T,B,N,A = actions.shape[:4]

        with tf.GradientTape() as tape:
            # Unroll online policy
            onlin_actions, _ = snt.static_unroll(
                self._policy_network,
                merge_batch_and_agent_dim_of_time_major_sequence(observations),
                self._policy_network.initial_state(B*N)
            )
            curr_pol_out = expand_batch_and_agent_dim_of_time_major_sequence(onlin_actions, B, N)

            pred_qvals = self._critic_network_1(observations, env_states, curr_pol_out, actions)

            omar_mu = tf.zeros_like(curr_pol_out) + self._init_omar_mu
            omar_sigma = tf.zeros_like(curr_pol_out) + self._init_omar_sigma

            # Repeat all tensors num_ood_actions times andadd  next to batch dim
            observations = tf.stack([observations]*self._omar_num_samples, axis=2) # next to batch dim
            env_states = tf.stack([env_states]*self._omar_num_samples, axis=2) # next to batch dim

            # Flatten into batch dim
            observations = tf.reshape(observations, (T, -1, *observations.shape[3:]))
            env_states = tf.reshape(env_states, (T, -1, *env_states.shape[3:]))

            last_top_k_qvals, last_elite_acs = None, None
            for iter_idx in range(self._omar_iters):
                dist = tfp.distributions.Normal(omar_mu, omar_sigma)
                cem_sampled_acs = dist.sample((self._omar_num_samples,))
                cem_sampled_acs = tf.transpose(cem_sampled_acs, (1, 2, 0, 3, 4))
                cem_sampled_acs = tf.clip_by_value(cem_sampled_acs, -1.0, 1.0)

                formatted_cem_sampled_acs = tf.reshape(cem_sampled_acs, (T, -1, *cem_sampled_acs.shape[3:]))
                all_pred_qvals = self._critic_network_1(observations, env_states, formatted_cem_sampled_acs, actions)
                all_pred_qvals = tf.reshape(all_pred_qvals, (T,B,self._omar_num_samples,N))
                all_pred_qvals = tf.transpose(all_pred_qvals, (0, 1, 3, 2))
                cem_sampled_acs = tf.transpose(cem_sampled_acs, (0, 1, 3, 4, 2))

                if iter_idx > 0:
                    all_pred_qvals = tf.concat((all_pred_qvals, last_top_k_qvals), axis=-1)
                    cem_sampled_acs = tf.concat((cem_sampled_acs, last_elite_acs), axis=-1)

                top_k_qvals, top_k_inds = tf.math.top_k(all_pred_qvals, self._omar_num_elites)
                elite_ac_inds = tf.stack([top_k_inds]*A, axis=-2)
                elite_acs = tf.gather(cem_sampled_acs, elite_ac_inds, batch_dims=-1)

                last_top_k_qvals, last_elite_acs = top_k_qvals, elite_acs

                updated_mu = tf.reduce_mean(elite_acs, axis=-1)
                updated_sigma = tf.math.reduce_std(elite_acs, axis=-1)

                omar_mu = updated_mu
                omar_sigma = updated_sigma

            top_qvals, top_inds = tf.math.top_k(all_pred_qvals, self._omar_num_elites)
            top_ac_inds = tf.stack([top_k_inds]*A, axis=-2)
            top_acs = tf.gather(cem_sampled_acs, top_ac_inds, batch_dims=-1)

            cem_qvals = top_qvals
            pol_qvals = pred_qvals
            cem_acs = top_acs
            pol_acs = tf.expand_dims(curr_pol_out, axis=-1)

            candidate_qvals = tf.concat([pol_qvals, cem_qvals], -1)
            candidate_acs = tf.concat([pol_acs, cem_acs], -1)

            max_inds = tf.argmax(candidate_qvals, axis=-1)
            max_ac_inds = tf.expand_dims(tf.stack([max_inds]*A, axis=-1), axis=-1)

            max_acs = tf.gather(candidate_acs, max_ac_inds, batch_dims=-1)
            max_acs = tf.stop_gradient(tf.reshape(max_acs, max_acs.shape[:-1]))

            def masked_mean(x):
                return tf.reduce_sum(x * zero_padding_mask) / tf.reduce_sum(zero_padding_mask)

            policy_loss = self._omar_coe * masked_mean(tf.reduce_mean((curr_pol_out - max_acs)**2, axis=-1)) - (1 - self._omar_coe) * masked_mean(tf.squeeze(pred_qvals)) + masked_mean(tf.reduce_mean(curr_pol_out ** 2,axis=-1)) * 1e-3

        # Train policy
        variables = (*self._policy_network.trainable_variables,)
        gradients = tape.gradient(policy_loss, variables)
        self._policy_optimizer.apply(gradients, variables)

        # Update target policy
        online_variables = (*self._policy_network.variables,)
        target_variables = (*self._target_policy_network.variables,)
        self._update_target_network(
            online_variables,
            target_variables,
        )

        return {"policy_loss": policy_loss}


    @tf.function(jit_compile=True)
    def _tf_train_step(self, train_step, batch):
        batch = batched_agents(self._environment.possible_agents, batch)

        # Unpack the batch
        observations = batch["observations"] # (B,T,N,O)
        actions = batch["actions"] # (B,T,N,A)
        env_states = batch["state"] # (B,T,S)
        rewards = batch["rewards"] # (B,T,N)
        truncations = batch["truncations"] # (B,T,N)
        terminals = batch["terminals"] # (B,T,N)
        zero_padding_mask = batch["mask"] # (B,T)

        # done = tf.cast(tf.logical_or(tf.cast(truncations, "bool"), tf.cast(terminals, "bool")), "float32")
        done = terminals

        # Get dims
        B, T, N = actions.shape[:3]

        # Maybe add agent ids to observation
        if self._add_agent_id_to_obs:
            observations = batch_concat_agent_id_to_obs(observations)
        
        # Make time-major
        observations = switch_two_leading_dims(observations)
        replay_actions = switch_two_leading_dims(actions)
        rewards = switch_two_leading_dims(rewards)
        done = switch_two_leading_dims(done)
        zero_padding_mask = switch_two_leading_dims(zero_padding_mask)
        env_states = switch_two_leading_dims(env_states)

        # Unroll target policy
        target_actions, _ = snt.static_unroll(
            self._target_policy_network,
            merge_batch_and_agent_dim_of_time_major_sequence(observations),
            self._target_policy_network.initial_state(B*N)
        )
        target_actions = expand_batch_and_agent_dim_of_time_major_sequence(target_actions, B, N)

        # Target critics
        target_qs_1 = self._target_critic_network_1(observations, env_states, target_actions, target_actions)
        target_qs_2 = self._target_critic_network_2(observations, env_states, target_actions, target_actions)

        # Take minimum between two target critics
        target_qs = tf.minimum(target_qs_1, target_qs_2)

        # Compute Bellman targets
        targets = rewards[:-1] + self._discount * (1-done[:-1]) * tf.squeeze(target_qs[1:], axis=-1)

        # Do forward passes through the networks and calculate the losses
        with tf.GradientTape(persistent=True) as tape:

            # Online critics
            qs_1 = tf.squeeze(self._critic_network_1(observations, env_states, replay_actions, replay_actions), axis=-1)
            qs_2 = tf.squeeze(self._critic_network_2(observations, env_states, replay_actions, replay_actions), axis=-1)

            # Squared TD-error
            critic_loss_1 = 0.5 * (targets - qs_1[:-1]) ** 2
            critic_loss_2 = 0.5 * (targets - qs_2[:-1]) ** 2

            ###########
            ### CQL ###
            ###########

            online_actions, _ = snt.static_unroll(
                self._policy_network,
                merge_batch_and_agent_dim_of_time_major_sequence(observations),
                self._policy_network.initial_state(B*N)
            )
            online_actions = expand_batch_and_agent_dim_of_time_major_sequence(online_actions, B, N)

            # Repeat all tensors num_ood_actions times andadd  next to batch dim
            repeat_observations = tf.stack([observations]*self._num_ood_actions, axis=2) # next to batch dim
            repeat_env_states = tf.stack([env_states]*self._num_ood_actions, axis=1) # next to batch dim
            repeat_online_actions = tf.stack([online_actions]*self._num_ood_actions, axis=1) # next to batch dim

            # Flatten into batch dim
            repeat_observations = tf.reshape(repeat_observations, (T, -1, *repeat_observations.shape[3:]))
            repeat_env_states = tf.reshape(repeat_env_states, (T, -1, *repeat_env_states.shape[3:]))
            repeat_online_actions = tf.reshape(repeat_online_actions, (T, -1, *repeat_online_actions.shape[3:]))

            # CQL Loss
            random_ood_actions = tf.random.uniform(
                            shape=repeat_online_actions.shape,
                            minval=-1.0,
                            maxval=1.0,
                            dtype=repeat_online_actions.dtype
            )
            random_ood_action_log_pi = tf.math.log(0.5 ** (random_ood_actions.shape[-1]))

            ood_qs_1 = self._critic_network_1(repeat_observations, repeat_env_states, random_ood_actions, random_ood_actions) - random_ood_action_log_pi
            ood_qs_2 = self._critic_network_2(repeat_observations, repeat_env_states, random_ood_actions, random_ood_actions) - random_ood_action_log_pi

            # # Actions near true actions
            mu = 0.0
            std = self._cql_sigma
            action_noise = tf.random.normal(
                                repeat_online_actions.shape,
                                mean=mu,
                                stddev=std,
                                dtype=repeat_online_actions.dtype,
                            )
            current_ood_actions = tf.clip_by_value(repeat_online_actions + action_noise, -1.0, 1.0)

            ood_actions_prob = (1 / (self._cql_sigma * tf.math.sqrt(2 * np.pi))) * tf.exp( - (action_noise - mu)**2 / (2 * self._cql_sigma**2) )
            ood_actions_log_prob = tf.math.log(tf.reduce_prod(ood_actions_prob, axis=-1, keepdims=True))

            current_ood_qs_1 = self._critic_network_1(repeat_observations, repeat_env_states, current_ood_actions, current_ood_actions) - ood_actions_log_prob
            current_ood_qs_2 = self._critic_network_2(repeat_observations, repeat_env_states, current_ood_actions, current_ood_actions) - ood_actions_log_prob

            # Reshape
            ood_qs_1 = tf.reshape(ood_qs_1, (T, B, self._num_ood_actions, N))
            ood_qs_2 = tf.reshape(ood_qs_2, (T, B, self._num_ood_actions, N))
            current_ood_qs_1 = tf.reshape(current_ood_qs_1, (T, B, self._num_ood_actions, N))
            current_ood_qs_2 = tf.reshape(current_ood_qs_2, (T, B, self._num_ood_actions, N))

            all_ood_qs_1 = tf.concat((ood_qs_1, current_ood_qs_1), axis=2)
            all_ood_qs_2 = tf.concat((ood_qs_2, current_ood_qs_2), axis=2)

            def masked_mean(x):
                return tf.reduce_sum(x * tf.expand_dims(zero_padding_mask,axis=-1)) / tf.reduce_sum(zero_padding_mask)

            cql_loss_1 = masked_mean(tf.reduce_logsumexp(all_ood_qs_1, axis=2, keepdims=False)) - masked_mean(qs_1)
            cql_loss_2 = masked_mean(tf.reduce_logsumexp(all_ood_qs_2, axis=2, keepdims=False)) - masked_mean(qs_2)

            cql_alpha = tf.exp(self._cql_log_alpha)
            cql_loss_1 = cql_alpha * (cql_loss_1 - self._target_action_gap)
            cql_loss_2 = cql_alpha * (cql_loss_2 - self._target_action_gap)

            critic_loss_1 += self._cql_weight * cql_loss_1
            critic_loss_2 += self._cql_weight * cql_loss_2

            cql_alpha_loss = (- cql_loss_1 - cql_loss_2) * 0.5

            ### END CQL ###

            critic_loss = (critic_loss_1 + critic_loss_2) / 2
            critic_loss = tf.reduce_sum(critic_loss * tf.expand_dims(zero_padding_mask[:-1],axis=-1)) / tf.reduce_sum(zero_padding_mask[:-1])

            # Policy Loss
            # Unroll online policy
            onlin_actions, _ = snt.static_unroll(
                self._policy_network,
                merge_batch_and_agent_dim_of_time_major_sequence(observations),
                self._policy_network.initial_state(B*N)
            )
            online_actions = expand_batch_and_agent_dim_of_time_major_sequence(onlin_actions, B, N)

            qs_1 = self._critic_network_1(observations, env_states, online_actions, replay_actions)
            qs_2 = self._critic_network_2(observations, env_states, online_actions,replay_actions)
            qs = tf.minimum(qs_1, qs_2)
            
            policy_loss = - tf.squeeze(qs, axis=-1)

            # Masked mean
            policy_mask = tf.squeeze(tf.stack([zero_padding_mask] * N, axis=2))
            policy_loss = tf.reduce_sum(policy_loss * policy_mask) / tf.reduce_sum(policy_mask)

        # Train critics
        variables = (
            *self._critic_network_1.trainable_variables,
            *self._critic_network_2.trainable_variables,
        )
        gradients = tape.gradient(critic_loss, variables)
        self._critic_optimizer.apply(gradients, variables)

        # Train policy
        variables = (
            *self._policy_network.trainable_variables,
        )
        gradients = tape.gradient(policy_loss, variables)
        self._policy_optimizer.apply(gradients, variables)

        # Optimise CQL alpha
        variables = [self._cql_log_alpha]
        gradients = tape.gradient(cql_alpha_loss, variables)
        self._cql_alpha_optimizer.apply(gradients, variables)

        # Update target networks
        online_variables = (
            *self._critic_network_1.variables,
            *self._critic_network_2.variables,
            *self._policy_network.variables
        )
        target_variables = (
            *self._target_critic_network_1.variables,
            *self._target_critic_network_2.variables,
            *self._target_policy_network.variables
        )   
        self._update_target_network(
            online_variables,
            target_variables,
        )

        del tape

        logs = {
            "Mean Q-values": tf.reduce_mean((qs_1 + qs_2) / 2),
            "Mean Critic Loss": critic_loss,
            "Policy Loss": policy_loss,
            "CQL Alpha Loss": cql_alpha_loss,
            "CQL Loss": cql_loss_1
        }

        return logs