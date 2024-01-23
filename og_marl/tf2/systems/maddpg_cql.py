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
import numpy as np
import tensorflow as tf
import sonnet as snt

from og_marl.tf2.utils import (
    batched_agents
)

NUM_CQL_ACTIONS = 10
CQL_ALPHA = 5

class MADDPGCQLSystem:

    def __init__(
            self, 
            action_spaces, 
            observation_spaces,
            linear_layer_dims=(256, 256),
            policy_lr=1e-3,
            critic_lr=1e-3,
            target_update_rate=5e-3,
            discount_factor=0.99,
            cql_alpha=5,
            num_cql_actions=10
        ):

        self.action_spaces = action_spaces
        self.observation_spaces = observation_spaces

        # We assume all agents have the same action space
        action_dim = self.action_spaces["agent_0"].shape[0]

        # Shared parameters for policy network
        self.policy_network = snt.Sequential(
            [
                snt.Linear(linear_layer_dims[0]),
                tf.nn.relu,
                snt.Linear(linear_layer_dims[1]),
                tf.nn.relu,
                snt.Linear(action_dim),
                tf.nn.tanh # Assumes actions between -1 and 1
            ]
        )
        self.target_policy_network = copy.deepcopy(self.policy_network)

        # Shared parameters for critic network
        self.critic_network = snt.Sequential(
            [
                snt.Linear(linear_layer_dims[0]),
                tf.nn.relu,
                snt.Linear(linear_layer_dims[1]),
                tf.nn.relu,
                snt.Linear(1),
            ]
        )
        self.target_critic_network = copy.deepcopy(self.critic_network)

        self.policy_optimiser = snt.optimizers.RMSProp(policy_lr)
        self.critic_optimiser = snt.optimizers.RMSProp(critic_lr)

        # Hyper-parameters
        self.tau = target_update_rate
        self.gamma = discount_factor
        self.cql_alpha = cql_alpha
        self.num_cql_actions = num_cql_actions

    def select_actions(self, observations, legals=None):
        actions = self._select_actions(observations)
        actions = {agent: action.numpy()[0] for agent, action in actions.items()}
        return actions
    
    @tf.function(jit_compile=True)
    def _select_actions(self, observations):
        actions = {}
        agents = list(observations.keys())
        for i, agent in enumerate(agents):
            obs = tf.concat([observations[agent], tf.one_hot(i, len(agents))], axis=-1)
            obs = tf.expand_dims(obs, axis=0) # add batch dim
            action = self.policy_network(obs)

            # noise = tf.random.normal(action.shape, 0.0, 0.2) # TODO: make variable
            # action = tf.clip_by_value(action + noise, -1, 1)

            actions[agent] = action
        return actions
    
    @tf.function(jit_compile=True)
    def train(self, batch):
        agents = self._environment.possible_agents
        
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

        agent_ids = tf.stack([tf.eye(len(agents))]*B, axis=0)
        
        # agents have been batched along the 2nd dimension i.e. shape=(batch, num_agents, ...)
        observations = tf.concat([batch["observations"], agent_ids], axis=-1)
        states = batch["states"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        terminals = batch["terminals"]
        next_observations = tf.concat([batch["next_observations"], agent_ids], axis=-1)
        next_states = batch["next_states"]

        next_actions = self.target_policy_network(next_observations)
        target_critic_input = make_critic_input(next_states, next_actions, next_actions, agent_ids)
        next_value = self.target_critic_network(target_critic_input)
        next_value = tf.squeeze(next_value, axis=-1)
        target_value = rewards + self.gamma * (1 - tf.cast(terminals, "float32")) * next_value

        with tf.GradientTape(persistent=True) as tape:

            critic_input = make_critic_input(states, actions, actions, agent_ids)
            value = self.critic_network(critic_input)
            value = tf.squeeze(value, axis=-1)
            td_loss = tf.reduce_mean(tf.square(target_value - value))
            critic_loss = td_loss

            #### CQL ####
            cql_observations = tf.stack([observations] * NUM_CQL_ACTIONS, axis=0)
            cql_next_observations = tf.stack([next_observations] * NUM_CQL_ACTIONS, axis=0)
            cql_states = tf.stack([states] * NUM_CQL_ACTIONS, axis=0)

            on_policy_actions = self.policy_network(cql_observations)
            on_policy_next_actions = self.policy_network(cql_next_observations)
            random_actions = tf.random.uniform(on_policy_actions.shape, minval=-1, maxval=1)

            action_noise = tf.random.normal(on_policy_actions.shape, 0, 0.2)
            noisy_on_policy_actions = tf.clip_by_value(on_policy_actions + action_noise, -1, 1)
            noisy_on_policy_next_actions = tf.clip_by_value(on_policy_next_actions + action_noise, -1, 1)

            cql_critic_input = tf.stack([make_critic_input(cql_states[i], noisy_on_policy_actions[i], noisy_on_policy_actions[i], agent_ids) for i in range(NUM_CQL_ACTIONS)], axis=0)
            cql_next_critic_input = tf.stack([make_critic_input(cql_states[i], noisy_on_policy_next_actions[i], noisy_on_policy_next_actions[i], agent_ids) for i in range(NUM_CQL_ACTIONS)], axis=0)
            cql_random_critic_input = tf.stack([make_critic_input(cql_states[i], random_actions[i], random_actions[i], agent_ids) for i in range(NUM_CQL_ACTIONS)], axis=0)

            action_noise_prob = (1 / (0.2 * tf.math.sqrt(2 * np.pi))) * tf.exp( - (action_noise - 0.0)**2 / (2 * 0.2**2) )
            action_noise_logprob = tf.math.log(tf.reduce_prod(tf.reduce_prod(action_noise_prob, axis=-1, keepdims=True), axis=2, keepdims=True))
            random_action_logprob = tf.math.log(0.5**random_actions.shape[-1])
            cql_values = self.critic_network(cql_critic_input) - action_noise_logprob
            cql_next_values = self.critic_network(cql_next_critic_input) - action_noise_logprob
            cql_random_values = self.critic_network(cql_random_critic_input) - random_action_logprob

            ood_values = tf.concat([cql_values, cql_next_values, cql_random_values], axis=0)
            ood_values = tf.reduce_logsumexp(ood_values, axis=0)
            cql_loss = tf.reduce_mean(ood_values) - tf.reduce_mean(value)

            critic_loss = self.cql_alpha * cql_loss + td_loss

            #### End CQL ####

            policy_actions = self.policy_network(observations)
            target_policy_actions = self.target_policy_network(observations)
            critic_input = make_critic_input(states, policy_actions, target_policy_actions, agent_ids)
            policy_action_values = self.critic_network(critic_input)
            policy_loss = tf.reduce_mean(-policy_action_values + tf.square(policy_actions))

        critic_gradient = tape.gradient(critic_loss, self.critic_network.trainable_variables)
        policy_gradient = tape.gradient(policy_loss, self.policy_network.trainable_variables)

        self.critic_optimiser.apply(critic_gradient, self.critic_network.trainable_variables)
        self.policy_optimiser.apply(policy_gradient, self.policy_network.trainable_variables)

        # Update target networks
        online_variables = (
            *self.critic_network.variables,
            *self.policy_network.variables
        )
        target_variables = (
            *self.target_critic_network.variables,
            *self.target_policy_network.variables
        )
        for src, dest in zip(online_variables, target_variables):
            dest.assign(dest * (1.0 - self.tau) + src * self.tau)

        logs = {
            "policy_loss": policy_loss,
            "td_loss": td_loss,
            "cql_loss": cql_loss
        }

        del tape

        return logs

def make_critic_input(states, agent_actions, other_agent_actions, agent_ids):
    B,N,A = agent_actions.shape[:3] # (B,N,A)
    joint_actions = tf.reshape(agent_actions, (B,N,A))
    all_joint_actions = []
    for i in range(N):
        one_hot = tf.expand_dims(tf.cast(tf.stack([tf.one_hot(i, N)]*B,axis=0), "bool"), axis=-1)
        joint_action = tf.where(one_hot, agent_actions, other_agent_actions)
        joint_action = tf.reshape(joint_action, (B,N*A))
        all_joint_actions.append(joint_action)
    all_joint_actions = tf.stack(all_joint_actions, axis=1)
    states = tf.stack([states]*N, axis=1)
    # states = tf.stack([states]*N, axis=1)
    # critic_input = tf.concat([states, agent_actions, agent_ids], axis=-1)
    # critic_input = tf.expand_dims(critic_input, axis=1)
    return critic_input