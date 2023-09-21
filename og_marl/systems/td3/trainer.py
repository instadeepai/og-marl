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

import copy
import tensorflow as tf
import tensorflow_probability as tfp
import sonnet as snt

from og_marl.systems import TrainerBase
from og_marl.utils.trainer_utils import (
    sample_batch_agents,
    switch_two_leading_dims,
    merge_batch_and_agent_dim_of_time_major_sequence,
    expand_batch_and_agent_dim_of_time_major_sequence,
    batch_concat_agent_id_to_obs
)

class TD3Trainer(TrainerBase):
    def __init__(
        self,
        agents,
        dataset,
        logger,
        policy_network,
        critic_network,
        policy_optimizer,
        critic_optimizer,
        discount=0.99,
        gaussian_noise_network=None,
        target_update_rate=0.01,
        max_gradient_norm=20.0,
        add_agent_id_to_obs=False,
        max_trainer_steps=1e6,
    ):
        super().__init__(
            agents=agents,
            dataset=dataset,
            logger=logger,
            discount=discount,
            max_gradient_norm=max_gradient_norm,
            add_agent_id_to_obs=add_agent_id_to_obs,
            max_trainer_steps=max_trainer_steps,
        )

        self._policy_optimizer = policy_optimizer
        self._critic_optimizer_1 = critic_optimizer
        self._critic_optimizer_2 = snt.optimizers.Adam(5e-4)

        self._policy_network = policy_network
        self._critic_network_1 = critic_network
        self._critic_network_2 = copy.deepcopy(critic_network)

        # Change critic 2s variables 
        critic_1_variables = (
            *self._critic_network_1.variables,
        )
        critic_2_variables = (
            *self._critic_network_2.variables,
        )   
        for src, dest in zip(critic_1_variables, critic_2_variables):
            dest.assign(-1.0 * src)


        self._system_variables.update(
            {
                "policy_network": self._policy_network.variables,
            }
        )

        # Target networks
        self._target_critic_network_1 = copy.deepcopy(critic_network)
        self._target_critic_network_2 = copy.deepcopy(critic_network)
        self._target_policy_network = copy.deepcopy(self._policy_network)

        # Target update
        self._target_update_rate = target_update_rate

        # Gaussian noise network for target actions
        self._gaussian_noise_network = gaussian_noise_network

        # For logging
        self._policy_loss = tf.Variable(0.0, trainable=False, dtype="float32")

    @tf.function()
    def _train(self, sample, trainer_step):
        batch = sample_batch_agents(self._agents, sample, independent=True)

        # Get the relevant quantities
        observations = batch["observations"]
        replay_actions = batch["actions"]
        states = batch["states"]
        rewards = batch["rewards"]
        env_discounts = tf.cast(batch["discounts"], "float32")
        mask = tf.cast(batch["mask"], "float32")  # shape=(B,T)

        # Get dims
        B, T, N, A = replay_actions.shape[:4]

        # Maybe add agent ids to observation
        if self._add_agent_id_to_obs:
            observations = batch_concat_agent_id_to_obs(observations)
        
        # Make time-major
        observations = switch_two_leading_dims(observations)
        replay_actions = switch_two_leading_dims(replay_actions)
        rewards = switch_two_leading_dims(rewards)
        env_discounts = switch_two_leading_dims(env_discounts)
        mask = switch_two_leading_dims(mask)

        if states is not None:
            states = switch_two_leading_dims(states)

        # Do forward passes through the networks and calculate the losses
        with tf.GradientTape(persistent=True) as tape:

            # Unroll target policy
            target_actions, _ = snt.static_unroll(
                self._target_policy_network,
                merge_batch_and_agent_dim_of_time_major_sequence(observations),
                self._target_policy_network.initial_state(B*N)
            )
            target_actions = expand_batch_and_agent_dim_of_time_major_sequence(target_actions, B, N)

            if self._gaussian_noise_network:
                noisy_target_actions = self._gaussian_noise_network(target_actions)

            # Target critics
            target_qs_1 = self._target_critic_network_1(observations, states, noisy_target_actions, target_actions)
            target_qs_2 = self._target_critic_network_2(observations, states, noisy_target_actions, target_actions)

            # Take minimum between two critics
            target_qs = tf.squeeze(tf.minimum(target_qs_1, target_qs_2))

            # Compute Bellman targets
            targets = tf.stop_gradient(
                rewards[:-1]
                + self._discount * env_discounts[:-1] * target_qs[1:]
            )

            # Online critics
            qs_1 = tf.squeeze(self._critic_network_1(observations, states, replay_actions, replay_actions))
            qs_2 = tf.squeeze(self._critic_network_2(observations, states, replay_actions, replay_actions))

            # Squared TD-Error
            critic_loss_1 = 0.5 * (targets - qs_1[:-1]) ** 2
            critic_loss_2 = 0.5 * (targets - qs_2[:-1]) ** 2

            critic_loss_1, critic_loss_2, critic_extras = self._add_extras_to_critic_loss(critic_loss_1, critic_loss_2, qs_1, qs_2, observations, states, mask)

            # Masked mean
            critic_mask = tf.squeeze(tf.stack([mask[:-1]]*N, axis=2))
            critic_loss_1 = tf.reduce_sum(critic_loss_1 * critic_mask) / tf.reduce_sum(critic_mask)
            critic_loss_2 = tf.reduce_sum(critic_loss_2 * critic_mask) / tf.reduce_sum(critic_mask)

        # Train critic 1
        variables = (
            *self._critic_network_1.trainable_variables,
        )
        gradients = tape.gradient(critic_loss_1, variables)
        gradients = tf.clip_by_global_norm(gradients, self._max_gradient_norm)[0]
        self._critic_optimizer_1.apply(gradients, variables)

        # # Train critic 2
        variables = (
            *self._critic_network_2.trainable_variables,
        )
        gradients = tape.gradient(critic_loss_2, variables)
        gradients = tf.clip_by_global_norm(gradients, self._max_gradient_norm)[0]
        self._critic_optimizer_2.apply(gradients, variables)

        # Maybe apply extra step
        self._critic_apply_extras(tape, critic_extras)

        # Update target networks
        online_variables = (
            *self._critic_network_1.variables,
            *self._critic_network_2.variables,
        )
        target_variables = (
            *self._target_critic_network_1.variables,
            *self._target_critic_network_2.variables,
        )   
        self._update_target_network(
            online_variables,
            target_variables,
        )

        del tape # clear the gradient tape

        if trainer_step % 2 == 0:  # TD3 style delayed policy update

            # Compute policy loss
            with tf.GradientTape(persistent=True) as tape:
                
                # Unroll online policy
                onlin_actions, _ = snt.static_unroll(
                    self._policy_network,
                    merge_batch_and_agent_dim_of_time_major_sequence(observations),
                    self._policy_network.initial_state(B*N)
                )
                online_actions = expand_batch_and_agent_dim_of_time_major_sequence(onlin_actions, B, N)

                qs_1 = self._critic_network_1(observations, states, online_actions, replay_actions)
                qs_2 = self._critic_network_2(observations, states, online_actions,replay_actions)
                qs = tf.reduce_mean((qs_1, qs_2), axis=0)
                
                policy_loss = - tf.squeeze(qs)

                # Masked mean
                policy_mask = tf.squeeze(tf.stack([mask] * N, axis=2))
                mean_qs = tf.reduce_sum(tf.abs(policy_loss) * policy_mask) / tf.reduce_sum(policy_mask)
                policy_loss = self._add_extras_to_policy_loss(policy_loss, online_actions, replay_actions, mean_qs)

                policy_loss = tf.reduce_sum(policy_loss * policy_mask) / tf.reduce_sum(policy_mask)

            # Train policy
            variables = (*self._policy_network.trainable_variables,)
            gradients = tape.gradient(policy_loss, variables)
            gradients = tf.clip_by_global_norm(gradients, self._max_gradient_norm)[0]
            self._policy_optimizer.apply(gradients, variables)

            del tape # clear gradient tape

            # Update target policy
            online_variables = (*self._policy_network.variables,)
            target_variables = (*self._target_policy_network.variables,)
            self._update_target_network(
                online_variables,
                target_variables,
            )

            # For logging 
            self._policy_loss.assign(policy_loss)

        logs = {
            "Mean Q-values": tf.reduce_mean((qs_1 + qs_2) / 2),
            "Trainer Steps": trainer_step,
            "Critic Loss": (critic_loss_1 + critic_loss_2) / 2,
            "Policy Loss": self._policy_loss,
        }

        logs.update(critic_extras)

        return logs

    def _update_target_network(
        self, online_variables, target_variables
    ):
        """Update the target networks."""

        tau = self._target_update_rate
        for src, dest in zip(online_variables, target_variables):
            dest.assign(dest * (1.0 - tau) + src * tau)

    def _add_extras_to_critic_loss(self, critic_loss_1, critic_loss_2, qs_1, qs_2, observations, states, mask):
        return critic_loss_1, critic_loss_2, {} # extras

    def _critic_apply_extras(self, tape, critic_extras):
        return

    def _add_extras_to_policy_loss(self, policy_loss, online_actions, replay_actions, mean_qs):
        return policy_loss

    def after_train_step(self):
        info = {}
        return info


class TD3CQLTrainer(TD3Trainer):
    def __init__(
        self,
        agents,
        dataset,
        logger,
        policy_optimizer,
        critic_optimizer,
        policy_network,
        critic_network,
        num_ood_actions=10, # CQL
        cql_weight=10.0, # CQL
        discount=0.99,
        gaussian_noise_network=None,
        target_update_rate=0.01,
        max_gradient_norm=20.0,
        add_agent_id_to_obs=False,
        max_trainer_steps=1e6,
    ):
        super().__init__(
            agents=agents,
            dataset=dataset,
            logger=logger,
            policy_network=policy_network,
            critic_network=critic_network,
            policy_optimizer=policy_optimizer,
            critic_optimizer=critic_optimizer,
            discount=discount,
            gaussian_noise_network=gaussian_noise_network,
            target_update_rate=target_update_rate,
            max_gradient_norm=max_gradient_norm,
            add_agent_id_to_obs=add_agent_id_to_obs,
            max_trainer_steps=max_trainer_steps
        )

        self._cql_alpha_optimizer = snt.optimizers.Adam(5e-4)
        self._cql_weight = cql_weight
        self._cql_sigma = 0.1
        self._target_action_gap = 10.0
        self._num_ood_actions = num_ood_actions
        self._cql_log_alpha = tf.Variable(0.0, trainable=True)
        self._ctr = tf.Variable(0.0, trainable=False)

    def _add_extras_to_critic_loss(self, critic_loss_1, critic_loss_2, qs_1, qs_2, observations, states, mask):
        T, B, N = observations.shape[:3]

        # Unroll online policy
        online_actions, _ = snt.static_unroll(
            self._policy_network,
            merge_batch_and_agent_dim_of_time_major_sequence(observations),
            self._policy_network.initial_state(B*N)
        )
        online_actions = expand_batch_and_agent_dim_of_time_major_sequence(online_actions, B, N)

        # Repeat all tensors num_ood_actions times andadd  next to batch dim
        observations = tf.stack([observations]*self._num_ood_actions, axis=2) # next to batch dim
        states = tf.stack([states]*self._num_ood_actions, axis=1) # next to batch dim
        online_actions = tf.stack([online_actions]*self._num_ood_actions, axis=1) # next to batch dim

        # flatten into batch dim
        observations = tf.reshape(observations, (T, -1, *observations.shape[3:]))
        states = tf.reshape(states, (T, -1, *states.shape[3:]))
        online_actions = tf.reshape(online_actions, (T, -1, *online_actions.shape[3:]))

        # CQL Loss
        random_ood_actions = tf.random.uniform(
                        shape=online_actions.shape,
                        minval=-1.0,
                        maxval=1.0,
                        dtype=online_actions.dtype
        )
        # random_ood_action_log_pi = tf.math.log(0.5 ** (random_ood_actions.shape[-1]))

        ood_qs_1 = self._critic_network_1(observations, states, random_ood_actions, random_ood_actions) #- random_ood_action_log_pi
        ood_qs_2 = self._critic_network_2(observations, states, random_ood_actions, random_ood_actions) #- random_ood_action_log_pi

        # # Actions near true actions
        mu = 0.0
        std = self._cql_sigma
        action_noise = tf.random.normal(
                            online_actions.shape,
                            mean=mu,
                            stddev=std,
                            dtype=online_actions.dtype,
                        )
        current_ood_actions = tf.clip_by_value(online_actions + action_noise, -1.0, 1.0)

        # ood_actions_prob = (1 / (self._cql_sigma * tf.math.sqrt(2 * np.pi))) * tf.exp( - (action_noise - mu)**2 / (2 * self._cql_sigma**2) )
        # ood_actions_log_prob = tf.math.log(tf.reduce_prod(ood_actions_prob, axis=-1, keepdims=True))

        current_ood_qs_1 = self._critic_network_1(observations, states, current_ood_actions, current_ood_actions) #- ood_actions_log_prob
        current_ood_qs_2 = self._critic_network_2(observations, states, current_ood_actions, current_ood_actions) #- ood_actions_log_prob

        # Reshape
        ood_qs_1 = tf.reshape(ood_qs_1, (T, B, self._num_ood_actions, N))
        ood_qs_2 = tf.reshape(ood_qs_2, (T, B, self._num_ood_actions, N))
        current_ood_qs_1 = tf.reshape(current_ood_qs_1, (T, B, self._num_ood_actions, N))
        current_ood_qs_2 = tf.reshape(current_ood_qs_2, (T, B, self._num_ood_actions, N))

        all_ood_qs_1 = tf.concat((ood_qs_1, current_ood_qs_1), axis=2)
        all_ood_qs_2 = tf.concat((ood_qs_2, current_ood_qs_2), axis=2)

        def masked_mean(x):
            return tf.reduce_sum(x * mask) / tf.reduce_sum(mask)

        cql_loss_1 = masked_mean(tf.reduce_logsumexp(all_ood_qs_1, axis=2, keepdims=False)) - masked_mean(qs_1)
        cql_loss_2 = masked_mean(tf.reduce_logsumexp(all_ood_qs_2, axis=2, keepdims=False)) - masked_mean(qs_2)

        cql_alpha = tf.clip_by_value(tf.exp(self._cql_log_alpha), 0.0, 2.0)
        cql_loss_1 = cql_alpha * (cql_loss_1 - self._target_action_gap)
        cql_loss_2 = cql_alpha * (cql_loss_2 - self._target_action_gap)
        cql_alpha_loss = (- cql_loss_1 - cql_loss_2) * 0.5

        # For logging
        td_loss = tf.reduce_mean(critic_loss_1)

        critic_loss_1 += self._cql_weight * cql_loss_1
        critic_loss_2 += self._cql_weight * cql_loss_2

        extras = {
            "CQL Alpha Loss": cql_alpha_loss,
            "CQL Loss": tf.reduce_mean(cql_loss_1),
            "OOD Qs difference": tf.reduce_mean(all_ood_qs_1) - tf.reduce_mean(qs_1),
            "Alpha": cql_alpha,
            "TD Loss": td_loss

        }
        return critic_loss_1, critic_loss_2, extras # Noop

    def _critic_apply_extras(self, tape, extras):
        cql_alpha_loss = extras["CQL Alpha Loss"]

        # Optimise cql alpha
        variables = [self._cql_log_alpha]
        gradients = tape.gradient(cql_alpha_loss, variables)
        gradients = tf.clip_by_global_norm(gradients, self._max_gradient_norm)[0]
        self._cql_alpha_optimizer.apply(gradients, variables)
        return

class TD3BCTrainer(TD3Trainer):
    def __init__(
        self,
        agents,
        dataset,
        logger,
        policy_optimizer,
        critic_optimizer,
        policy_network,
        critic_network,
        bc_alpha=2.5, # BC
        discount=0.99,
        gaussian_noise_network=None,
        target_update_rate=0.01,
        max_gradient_norm=20.0,
        add_agent_id_to_obs=False,
        max_trainer_steps=1e6,
    ):
        super().__init__(
            agents=agents,
            dataset=dataset,
            logger=logger,
            policy_network=policy_network,
            critic_network=critic_network,
            policy_optimizer=policy_optimizer,
            critic_optimizer=critic_optimizer,
            discount=discount,
            gaussian_noise_network=gaussian_noise_network,
            target_update_rate=target_update_rate,
            max_gradient_norm=max_gradient_norm,
            add_agent_id_to_obs=add_agent_id_to_obs,
            max_trainer_steps=max_trainer_steps
        )
        self._bc_alpha = bc_alpha

    def _add_extras_to_policy_loss(self, policy_loss, online_actions, replay_actions, mean_qs):
        bc_loss = tf.reduce_mean((online_actions - replay_actions) ** 2) # mean over action dim
        q_weight = tf.stop_gradient(self._bc_alpha / mean_qs)
        return q_weight * policy_loss + bc_loss



class OMARTrainer(TD3CQLTrainer):
    def __init__(
        self,
        agents,
        dataset,
        logger,
        policy_optimizer,
        critic_optimizer,
        policy_network,
        critic_network,
        num_ood_actions=10, # CQL
        cql_weight=10.0, # CQL
        omar_iters=3, # OMAR
        omar_num_samples=20, # OMAR
        omar_num_elites=5, # OMAR
        omar_sigma=2.0, # OMAR
        omar_coe=0.7, # OMAR
        discount=0.99,
        gaussian_noise_network=None,
        target_update_rate=0.01,
        max_gradient_norm=20.0,
        add_agent_id_to_obs=False,
        max_trainer_steps=1e6,
    ):
        super().__init__(
            agents=agents,
            dataset=dataset,
            logger=logger,
            policy_network=policy_network,
            critic_network=critic_network,
            policy_optimizer=policy_optimizer,
            critic_optimizer=critic_optimizer,
            discount=discount,
            num_ood_actions=num_ood_actions,
            cql_weight=cql_weight,
            gaussian_noise_network=gaussian_noise_network,
            target_update_rate=target_update_rate,
            max_gradient_norm=max_gradient_norm,
            add_agent_id_to_obs=add_agent_id_to_obs,
            max_trainer_steps=max_trainer_steps
        )

        self.omar_coe = omar_coe
        self.omar_iters = omar_iters
        self.omar_num_samples = omar_num_samples
        self.init_omar_mu, self.init_omar_sigma = 0.0, omar_sigma
        self.omar_num_elites = omar_num_elites

    @tf.function
    def _train(self, sample, trainer_step):
        batch = sample_batch_agents(self._agents, sample, independent=True)

        # Get the relevant quantities
        observations = batch["observations"]
        replay_actions = batch["actions"]
        states = batch["states"]
        rewards = batch["rewards"]
        env_discounts = tf.cast(batch["discounts"], "float32")
        mask = tf.cast(batch["mask"], "float32")  # shape=(B,T)

        # Get dims
        B, T, N, A = replay_actions.shape[:4]

        # Maybe add agent ids to observation
        if self._add_agent_id_to_obs:
            observations = batch_concat_agent_id_to_obs(observations)
        
        # Make time-major
        observations = switch_two_leading_dims(observations)
        replay_actions = switch_two_leading_dims(replay_actions)
        rewards = switch_two_leading_dims(rewards)
        env_discounts = switch_two_leading_dims(env_discounts)
        mask = switch_two_leading_dims(mask)

        if states is not None:
            states = switch_two_leading_dims(states)

        # Do forward passes through the networks and calculate the losses
        with tf.GradientTape(persistent=True) as tape:

            # Unroll target policy
            target_actions, _ = snt.static_unroll(
                self._target_policy_network,
                merge_batch_and_agent_dim_of_time_major_sequence(observations),
                self._target_policy_network.initial_state(B*N)
            )
            target_actions = expand_batch_and_agent_dim_of_time_major_sequence(target_actions, B, N)

            if self._gaussian_noise_network:
                noisy_target_actions = self._gaussian_noise_network(target_actions)

            # Target critics
            target_qs_1 = self._target_critic_network_1(observations, states, noisy_target_actions, target_actions)
            target_qs_2 = self._target_critic_network_2(observations, states, noisy_target_actions, target_actions)

            # Take minimum between two critics
            target_qs = tf.squeeze(tf.minimum(target_qs_1, target_qs_2))

            # Compute Bellman targets
            targets = tf.stop_gradient(
                rewards[:-1]
                + self._discount * env_discounts[:-1] * target_qs[1:]
            )

            # Online critics
            qs_1 = tf.squeeze(self._critic_network_1(observations, states, replay_actions, replay_actions))
            qs_2 = tf.squeeze(self._critic_network_2(observations, states, replay_actions, replay_actions))

            # Squared TD-Error
            critic_loss_1 = 0.5 * (targets - qs_1[:-1]) ** 2
            critic_loss_2 = 0.5 * (targets - qs_2[:-1]) ** 2

            critic_loss_1, critic_loss_2, critic_extras = self._add_extras_to_critic_loss(critic_loss_1, critic_loss_2, qs_1, qs_2, observations, states, mask)

            # Masked mean
            critic_mask = tf.squeeze(tf.stack([mask[:-1]]*N, axis=2))
            critic_loss_1 = tf.reduce_sum(critic_loss_1 * critic_mask) / tf.reduce_sum(critic_mask)
            critic_loss_2 = tf.reduce_sum(critic_loss_2 * critic_mask) / tf.reduce_sum(critic_mask)

        # Train critic 1
        variables = (
            *self._critic_network_1.trainable_variables,
        )
        gradients = tape.gradient(critic_loss_1, variables)
        gradients = tf.clip_by_global_norm(gradients, self._max_gradient_norm)[0]
        self._critic_optimizer_1.apply(gradients, variables)

        # # Train critic 2
        variables = (
            *self._critic_network_2.trainable_variables,
        )
        gradients = tape.gradient(critic_loss_2, variables)
        gradients = tf.clip_by_global_norm(gradients, self._max_gradient_norm)[0]
        self._critic_optimizer_2.apply(gradients, variables)

        # Maybe apply extra step
        self._critic_apply_extras(tape, critic_extras)

        # Update target networks
        online_variables = (
            *self._critic_network_1.variables,
            *self._critic_network_2.variables,
        )
        target_variables = (
            *self._target_critic_network_1.variables,
            *self._target_critic_network_2.variables,
        )   
        self._update_target_network(
            online_variables,
            target_variables,
        )

        del tape # clear the gradient tape

        if trainer_step % 2 == 0:  # TD3 style delayed policy update

            # Compute policy loss
            with tf.GradientTape(persistent=True) as tape:

                ##### OMAR #####
                
                # Unroll online policy
                onlin_actions, _ = snt.static_unroll(
                    self._policy_network,
                    merge_batch_and_agent_dim_of_time_major_sequence(observations),
                    self._policy_network.initial_state(B*N)
                )
                curr_pol_out = expand_batch_and_agent_dim_of_time_major_sequence(onlin_actions, B, N)

                pred_qvals = self._critic_network_1(observations, states, curr_pol_out, replay_actions)

                omar_mu = tf.zeros_like(curr_pol_out) + self.init_omar_mu
                omar_sigma = tf.zeros_like(curr_pol_out) + self.init_omar_sigma

                # Repeat all tensors num_ood_actions times andadd  next to batch dim
                observations = tf.stack([observations]*self.omar_num_samples, axis=2) # next to batch dim
                states = tf.stack([states]*self.omar_num_samples, axis=2) # next to batch dim
                # online_actions = tf.stack([online_actions]*self._num_ood_actions, axis=1) # next to batch dim

                # Flatten into batch dim
                observations = tf.reshape(observations, (T, -1, *observations.shape[3:]))
                states = tf.reshape(states, (T, -1, *states.shape[3:]))
                # online_actions = tf.reshape(online_actions, (T, -1, *online_actions.shape[3:]))

                last_top_k_qvals, last_elite_acs = None, None
                for iter_idx in range(self.omar_iters):
                    dist = tfp.distributions.Normal(omar_mu, omar_sigma)
                    cem_sampled_acs = dist.sample((self.omar_num_samples,))
                    cem_sampled_acs = tf.transpose(cem_sampled_acs, (1, 2, 0, 3, 4))
                    cem_sampled_acs = tf.clip_by_value(cem_sampled_acs, -1.0, 1.0)

                    formatted_cem_sampled_acs = tf.reshape(cem_sampled_acs, (T, -1, *cem_sampled_acs.shape[3:]))
                    all_pred_qvals = self._critic_network_1(observations, states, formatted_cem_sampled_acs, replay_actions)
                    all_pred_qvals = tf.reshape(all_pred_qvals, (T,B,self.omar_num_samples,N))
                    all_pred_qvals = tf.transpose(all_pred_qvals, (0, 1, 3, 2))
                    cem_sampled_acs = tf.transpose(cem_sampled_acs, (0, 1, 3, 4, 2))

                    if iter_idx > 0:
                        all_pred_qvals = tf.concat((all_pred_qvals, last_top_k_qvals), axis=-1)
                        cem_sampled_acs = tf.concat((cem_sampled_acs, last_elite_acs), axis=-1)

                    top_k_qvals, top_k_inds = tf.math.top_k(all_pred_qvals, self.omar_num_elites)
                    elite_ac_inds = tf.stack([top_k_inds]*A, axis=-2)
                    elite_acs = tf.gather(cem_sampled_acs, elite_ac_inds, batch_dims=-1)

                    last_top_k_qvals, last_elite_acs = top_k_qvals, elite_acs

                    updated_mu = tf.reduce_mean(elite_acs, axis=-1)
                    updated_sigma = tf.math.reduce_std(elite_acs, axis=-1)

                    omar_mu = updated_mu
                    omar_sigma = updated_sigma

                top_qvals, top_inds = tf.math.top_k(all_pred_qvals, self.omar_num_elites)
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
                max_acs = tf.stop_gradient(tf.squeeze(max_acs))

                def masked_mean(x):
                    return tf.reduce_sum(x * mask) / tf.reduce_sum(mask)

                policy_loss = self.omar_coe * masked_mean(tf.reduce_mean((curr_pol_out - max_acs)**2, axis=-1)) - (1 - self.omar_coe) * masked_mean(tf.squeeze(pred_qvals)) + masked_mean(tf.reduce_mean(curr_pol_out ** 2,axis=-1)) * 1e-3

            # Train policy
            variables = (*self._policy_network.trainable_variables,)
            gradients = tape.gradient(policy_loss, variables)
            gradients = tf.clip_by_global_norm(gradients, self._max_gradient_norm)[0]
            self._policy_optimizer.apply(gradients, variables)

            del tape # clear gradient tape

            # Update target policy
            online_variables = (*self._policy_network.variables,)
            target_variables = (*self._target_policy_network.variables,)
            self._update_target_network(
                online_variables,
                target_variables,
            )

            # For logging 
            self._policy_loss.assign(policy_loss)

        logs = {
            "Mean Q-values": tf.reduce_mean((qs_1 + qs_2) / 2),
            "Trainer Steps": trainer_step,
            "Critic Loss": (critic_loss_1 + critic_loss_2) / 2,
            "Policy Loss": self._policy_loss,
        }

        logs.update(critic_extras)

        return logs