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

"""Implementation of TD3"""
import tensorflow as tf
import sonnet as snt
import numpy as np
import copy

from og_marl.tf2.systems.iddpg import IDDPGSystem
from og_marl.tf2.utils import (
    batch_concat_agent_id_to_obs,
    batched_agents,
    switch_two_leading_dims,
    merge_batch_and_agent_dim_of_time_major_sequence,
    expand_batch_and_agent_dim_of_time_major_sequence,
    unroll_rnn,
)

class QMixer(snt.Module):
    """QMIX mixing network."""

    def __init__(
        self, num_agents, embed_dim = 32, hypernet_embed = 64, preprocess_network = None, non_monotonic=False
    ) -> None:
        """Inialize QMIX mixing network

        Args:
            num_agents: Number of agents in the enviroment
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

    def __call__(self, agent_qs: tf.Tensor, states: tf.Tensor) -> tf.Tensor:
        """Forward method."""

        agent_qs = switch_two_leading_dims(agent_qs)
        states = switch_two_leading_dims(states)
        
        B = agent_qs.shape[0] # batch size
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

        q_tot = switch_two_leading_dims(q_tot)

        return q_tot
    
class ObservationsAndActionCritic(snt.Module):

    def __init__(self, num_agents, num_actions):
        self.N = num_agents
        self.A = num_actions

        self._critic_network = snt.DeepRNN(
            [
                snt.Linear(128),
                tf.keras.layers.ReLU(),
                snt.GRU(128),
                tf.keras.layers.ReLU(),
                snt.Linear(1)
            ]
        )

        super().__init__()

    def __call__(self, observations, states, agent_actions, other_actions, resets, stop_other_actions_gradient=True):
        """Forward pass of critic network.
        
        observations [T,B,N,O]
        states [T,B,S]
        agent_actions [T,B,N,A]: the actions the agent took.
        other_actions [T,B,N,A]: the actions the other agents took.
        """
        B,N = observations.shape[1:3]

        # Concat states and joint actions
        critic_input = tf.concat([observations, agent_actions], axis=-1)

        # Concat agent IDs to critic input
        critic_input = batch_concat_agent_id_to_obs(critic_input)

        critic_input = merge_batch_and_agent_dim_of_time_major_sequence(critic_input)
        resets = merge_batch_and_agent_dim_of_time_major_sequence(resets)

        q_values = unroll_rnn(
            self._critic_network,
            critic_input,
            resets
        )

        q_values = expand_batch_and_agent_dim_of_time_major_sequence(q_values, B, N)

        return q_values

class FACMACCQLSystem(IDDPGSystem):

    def __init__(
        self, 
        environment,
        logger,
        linear_layer_dim=64,
        recurrent_layer_dim=64,
        discount=0.99,
        target_update_rate=0.005,
        critic_learning_rate=1e-4,
        policy_learning_rate=1e-4,
        add_agent_id_to_obs=False,
        random_exploration_timesteps=0,
        num_ood_actions=10, # CQL
        cql_weight=5.0, # CQL  
        cql_sigma=0.2, # CQL
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
            random_exploration_timesteps=random_exploration_timesteps  
        )

        self._critic_network = ObservationsAndActionCritic(len(self._environment.possible_agents), self._environment._num_actions)

        # Target critic network
        self._target_critic_network = copy.deepcopy(self._critic_network)

        self._mixer = QMixer(len(self._environment.possible_agents))
        self._target_mixer = copy.deepcopy(self._mixer)

        self._num_ood_actions = num_ood_actions
        self._cql_weight = cql_weight
        self._cql_sigma = cql_sigma

    @tf.function(jit_compile=True) # NOTE: comment this out if using debugger
    def _tf_train_step(self, batch):
        batch = batched_agents(self._environment.possible_agents, batch)

        # Unpack the batch
        observations = batch["observations"] # (B,T,N,O)
        actions = batch["actions"] # (B,T,N,A)
        env_states = batch["state"] # (B,T,S)
        rewards = batch["rewards"] # (B,T,N)
        truncations = tf.cast(batch["truncations"], "float32") # (B,T,N)
        terminals = tf.cast(batch["terminals"], "float32") # (B,T,N)
        zero_padding_mask = batch["mask"] # (B,T)

        # When to reset the RNN hidden state
        resets = tf.maximum(terminals, truncations) # equivalent to logical 'or'

        # Get dims
        B, T, N = actions.shape[:3]

        # Maybe add agent ids to observation
        if self._add_agent_id_to_obs:
            observations = batch_concat_agent_id_to_obs(observations)
        
        # Make time-major
        observations = switch_two_leading_dims(observations)
        replay_actions = switch_two_leading_dims(actions)
        rewards = switch_two_leading_dims(rewards)
        terminals = switch_two_leading_dims(terminals)
        zero_padding_mask = switch_two_leading_dims(zero_padding_mask)
        env_states = switch_two_leading_dims(env_states)
        resets = switch_two_leading_dims(resets)


        # Unroll target policy
        target_actions = unroll_rnn(
            self._target_policy_network,
            merge_batch_and_agent_dim_of_time_major_sequence(observations),
            merge_batch_and_agent_dim_of_time_major_sequence(resets)
        )
        target_actions = expand_batch_and_agent_dim_of_time_major_sequence(target_actions, B, N)

        # Target critics
        target_qs = self._target_critic_network(observations, env_states, target_actions, target_actions, resets)
        target_qs = self._target_mixer(target_qs, env_states)

        # Compute Bellman targets
        targets = tf.reduce_mean(rewards[:-1], axis=2, keepdims=True) + self._discount * tf.reduce_mean((1-terminals[:-1]), axis=2, keepdims=True) * target_qs[1:]

        # Do forward passes through the networks and calculate the losses
        with tf.GradientTape(persistent=True) as tape:

            # Online critics
            qs = self._critic_network(observations, env_states, replay_actions, replay_actions, resets)
            qs = self._mixer(qs, env_states)

            # Squared TD-error
            critic_loss = 0.5 * (targets - qs[:-1]) ** 2

            ###########
            ### CQL ###
            ###########

            online_actions = unroll_rnn(
                self._policy_network,
                merge_batch_and_agent_dim_of_time_major_sequence(observations),
                merge_batch_and_agent_dim_of_time_major_sequence(resets)
            )
            online_actions = expand_batch_and_agent_dim_of_time_major_sequence(online_actions, B, N)

            # Repeat all tensors num_ood_actions times andadd  next to batch dim
            repeat_observations = tf.stack([observations]*self._num_ood_actions, axis=2) # next to batch dim
            repeat_env_states = tf.stack([env_states]*self._num_ood_actions, axis=2) # next to batch dim
            repeat_online_actions = tf.stack([online_actions]*self._num_ood_actions, axis=2) # next to batch dim
            repeat_resets = tf.stack([resets]*self._num_ood_actions, axis=2) # next to batch dim

            # Flatten into batch dim
            repeat_observations = tf.reshape(repeat_observations, (T, -1, *repeat_observations.shape[3:]))
            repeat_env_states = tf.reshape(repeat_env_states, (T, -1, *repeat_env_states.shape[3:]))
            repeat_online_actions = tf.reshape(repeat_online_actions, (T, -1, *repeat_online_actions.shape[3:]))
            repeat_resets = tf.reshape(repeat_resets, (T, -1, *repeat_resets.shape[3:]))

            # CQL Loss
            all_ood_qs = []
            random_ood_actions = tf.random.uniform(
                            shape=repeat_online_actions.shape,
                            minval=-1.0,
                            maxval=1.0,
                            dtype=repeat_online_actions.dtype
            )
            random_ood_action_log_pi = tf.math.log(0.5 ** (random_ood_actions.shape[-1]))

            ood_qs = self._critic_network(repeat_observations, repeat_env_states, random_ood_actions, random_ood_actions, repeat_resets)[:-1] - random_ood_action_log_pi
            ood_qs = self._mixer(ood_qs, repeat_env_states[:-1])
            all_ood_qs.append(ood_qs)

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

            ood_qs = self._critic_network(repeat_observations[:-1], repeat_env_states[:-1], current_ood_actions[:-1], current_ood_actions[:-1], repeat_resets[:-1]) - ood_actions_log_prob[:-1]
            ood_qs = self._mixer(ood_qs, repeat_env_states[:-1])
            all_ood_qs.append(ood_qs)

            ood_qs = self._critic_network(repeat_observations[:-1], repeat_env_states[:-1], current_ood_actions[1:], current_ood_actions[1:], repeat_resets[1:]) - ood_actions_log_prob[1:]
            ood_qs = self._mixer(ood_qs, repeat_env_states[:-1])
            all_ood_qs.append(ood_qs)

            # Reshape
            all_ood_qs = [tf.reshape(x, (T-1, B, self._num_ood_actions)) for x in all_ood_qs]
            all_ood_qs = tf.concat(all_ood_qs, axis=2)

            def masked_mean(x):
                return tf.reduce_sum(x * tf.expand_dims(zero_padding_mask[:-1], axis=-1)) / tf.reduce_sum(zero_padding_mask[:-1])

            cql_loss = masked_mean(tf.reduce_logsumexp(all_ood_qs, axis=2, keepdims=True)) - masked_mean(qs[:-1])

            critic_loss += self._cql_weight * cql_loss

            ### END CQL ###

            # Masked mean
            critic_loss = masked_mean(critic_loss)

            # Policy Loss
            qs = self._critic_network(observations, env_states, online_actions, online_actions, resets)
            qs = self._mixer(qs, env_states)
            
            policy_loss = -qs + 1e-3 * tf.reduce_mean(tf.square(online_actions))

            # Masked mean
            policy_loss = masked_mean(policy_loss[:-1])

        # Train critics
        variables = (
            *self._critic_network.trainable_variables,
            *self._mixer.trainable_variables
        )
        gradients = tape.gradient(critic_loss, variables)
        self._critic_optimizer.apply(gradients, variables)

        # Train policy
        variables = (
            *self._policy_network.trainable_variables,
        )
        gradients = tape.gradient(policy_loss, variables)
        self._policy_optimizer.apply(gradients, variables)

        # Update target networks
        online_variables = (
            *self._critic_network.variables,
            *self._policy_network.variables,
            *self._mixer.variables
        )
        target_variables = (
            *self._target_critic_network.variables,
            *self._target_policy_network.variables,
            *self._target_mixer.variables
        )   
        self._update_target_network(
            online_variables,
            target_variables,
        )

        del tape

        logs = {
            "Mean Q-values": tf.reduce_mean(qs),
            "Mean Critic Loss": critic_loss,
            "Policy Loss": policy_loss,
            # "CQL Alpha Loss": cql_alpha_loss,
            "CQL Loss": cql_loss
        }

        return logs