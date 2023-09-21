"""Implementation of TD3"""
import copy
import tensorflow as tf
import sonnet as snt
import tree

from og_marl.systems.base import BaseMARLSystem
from og_marl.utils import (
    batch_concat_agent_id_to_obs,
    batched_agents,
    switch_two_leading_dims,
    merge_batch_and_agent_dim_of_time_major_sequence,
    expand_batch_and_agent_dim_of_time_major_sequence,
    set_growing_gpu_memory,
    concat_agent_id_to_obs
)

set_growing_gpu_memory()

class StateAndJointActionCritic(snt.Module):

    def __init__(self, num_agents, num_actions):
        self.N = num_agents
        self.A = num_actions

        self._critic_network = snt.Sequential(
            [
                snt.Linear(256),
                tf.keras.layers.ReLU(),
                snt.Linear(256),
                tf.keras.layers.ReLU(),
                snt.Linear(1)
            ]
        )

        super().__init__()

    def __call__(self, observations, states, agent_actions, other_actions, stop_other_actions_gradient=True):
        """Forward pass of critic network.

        observations [T,B,N,O]
        states [T,B,S]
        agent_actions [T,B,N,A]: the actions the agent took.
        other_actions [T,B,N,A]: the actions the other agents took.
        """
        if stop_other_actions_gradient:
            other_actions = tf.stop_gradient(other_actions)

        # Make joint action
        joint_actions = make_joint_action(agent_actions, other_actions)

        # Repeat states for each agent
        states = tf.stack([states]*self.N, axis=2) # [T,B,S] -> [T,B,N,S]

        # Concat states and joint actions
        critic_input = tf.concat([states, joint_actions], axis=-1)

        # Concat agent IDs to critic input
        critic_input = batch_concat_agent_id_to_obs(critic_input)

        q_values = self._critic_network(critic_input)
        return q_values


def make_joint_action(agent_actions, other_actions):
    """Method to construct the joint action.
    
    agent_actions [T,B,N,A]: tensor of actions the agent took. Usually
        the actions from the learnt policy network.
    other_actions [[T,B,N,A]]: tensor of actions the agent took. Usually
        the actions from the replay buffer.
    """
    N = agent_actions.shape[2]
    joint_actions = []
    for i in range(N):
        if N > 2:
            if i > 0 and i < N - 1:
                joint_action = tf.concat(
                    [
                        other_actions[:, :, :i],
                        tf.expand_dims(agent_actions[:, :, i], axis=2),
                        other_actions[:, :, i + 1 :],
                    ],
                    axis=2,  # along agent dim
                )
            elif i == 0:
                joint_action = tf.concat(
                    [
                        tf.expand_dims(agent_actions[:, :, i], axis=2),
                        other_actions[:, :, i + 1 :],
                    ],
                    axis=2,  # along agent dim
                )
            else:
                joint_action = tf.concat(
                    [
                        other_actions[:, :, :i],
                        tf.expand_dims(agent_actions[:, :, i], axis=2),
                    ],
                    axis=2,  # along agent dim
                )
        elif N == 2:
            if i == 0:
                joint_action = tf.concat(
                    [
                        tf.expand_dims(agent_actions[:, :, i], axis=2),
                        tf.expand_dims(other_actions[:, :, i + 1], axis=2),
                    ],
                    axis=2,  # along agent dim
                )
            else:
                joint_action = tf.concat(
                    [
                        tf.expand_dims(other_actions[:, :, i], axis=2),
                        tf.expand_dims(agent_actions[:, :, i], axis=2),
                    ],
                    axis=2,  # along agent dim
                )
        else:
            joint_action = agent_actions

        joint_action = tf.reshape(
            joint_action,
            (
                *joint_action.shape[:2],
                joint_action.shape[2] * joint_action.shape[3],
            ),
        )

        joint_actions.append(joint_action)
    joint_actions = tf.stack(joint_actions, axis=2)

    return joint_actions

class StateAndActionCritic(snt.Module):

    def __init__(self, num_agents, num_actions, preprocess_network=None):
        self.N = num_agents
        self.A = num_actions

        self._preprocess_network = preprocess_network

        self._critic_network = snt.Sequential(
            [
                snt.Linear(128),
                tf.keras.layers.ReLU(),
                snt.Linear(128),
                tf.keras.layers.ReLU(),
                snt.Linear(1)
            ]
        )

        super().__init__()

    def __call__(self, observations, states, agent_actions, other_actions, stop_other_actions_gradient=True):
        """Forward pass of critic network.
        
        observations [T,B,N,O]
        states [T,B,S]
        agent_actions [T,B,N,A]: the actions the agent took.
        other_actions [T,B,N,A]: the actions the other agents took.
        """
        if self._preprocess_network is not None:
            embeds = [] 
            for t in range(states.shape[0]):
                embeds.append(self._preprocess_network(states[t]))
            states =  tf.stack(embeds, axis=0) # stack along time

        # Repeat states for each agent
        states = tf.stack([states]*self.N, axis=2)

        # Concat states and joint actions
        critic_input = tf.concat([states, agent_actions], axis=-1)

        # Concat agent IDs to critic input
        critic_input = batch_concat_agent_id_to_obs(critic_input)

        q_values = self._critic_network(critic_input)
        return q_values

class IDDPGSystem(BaseMARLSystem):
    """Independent Deep Recurrent Q-Networs System"""

    def __init__(
        self,
        environment,
        logger,
        linear_layer_dim=100,
        recurrent_layer_dim=100,
        batch_size=64,
        discount=0.99,
        target_update_rate=0.005,
        critic_learning_rate=3e-4,
        policy_learning_rate=1e-3,
        add_agent_id_to_obs=True,
        random_exploration_timesteps=50_000
    ):

        super().__init__(
            environment,
            logger,
            add_agent_id_to_obs=add_agent_id_to_obs,
            batch_size=batch_size,
            discount=discount
        )

        self._linear_layer_dim = linear_layer_dim
        self._recurrent_layer_dim = recurrent_layer_dim

        # Policy network
        self._policy_network = snt.DeepRNN(
            [
                snt.Linear(linear_layer_dim),
                tf.nn.relu,
                snt.GRU(recurrent_layer_dim),
                tf.nn.relu,
                snt.Linear(self._environment._act_dim),
                tf.nn.tanh
            ]
        ) # shared network for all agents

        # Target policy network
        self._target_policy_network = copy.deepcopy(self._policy_network)

        # Critic network
        self._critic_network_1 = StateAndJointActionCritic(len(self._environment.possible_agents), self._environment._act_dim) # shared network for all agents
        self._critic_network_2 = copy.deepcopy(self._critic_network_1)

        # Target critic network
        self._target_critic_network_1 = copy.deepcopy(self._critic_network_1)
        self._target_critic_network_2 = copy.deepcopy(self._critic_network_1)
        self._target_update_rate = target_update_rate

        # Optimizers
        self._critic_optimizer=snt.optimizers.RMSProp(learning_rate=critic_learning_rate)
        self._policy_optimizer=snt.optimizers.RMSProp(learning_rate=policy_learning_rate)

        # Exploration
        self._random_exploration_timesteps = tf.Variable(random_exploration_timesteps)

        # Reset the recurrent neural network
        self._rnn_states = {agent: self._policy_network.initial_state(1) for agent in self._environment.possible_agents}

    def reset(self):
        """Called at the start of a new episode."""

        # Reset the recurrent neural network
        self._rnn_states = {agent: self._policy_network.initial_state(1) for agent in self._environment.possible_agents}

        return

    def select_actions(self, observations, legal_actions=None, explore=True):
        actions, next_rnn_states = self._tf_select_actions(observations, self._rnn_states, explore)
        self._rnn_states = next_rnn_states
        return tree.map_structure(lambda x: x[0].numpy(), actions) # convert to numpy and squeeze batch dim

    @tf.function()
    def _tf_select_actions(self, observations, rnn_states, explore=False):
        actions = {}
        next_rnn_states = {}
        for i, agent in enumerate(self._environment.possible_agents):
            agent_observation = observations[agent]
            if self._add_agent_id_to_obs:
                agent_observation = concat_agent_id_to_obs(agent_observation, i, len(self._environment.possible_agents))
            agent_observation = tf.expand_dims(agent_observation, axis=0) # add batch dimension
            action, next_rnn_states[agent] = self._policy_network(agent_observation, rnn_states[agent])

            # Add exploration noise
            if explore:
                if self._random_exploration_timesteps > 0:
                    action = tf.random.uniform(action.shape, -1, 1, dtype=action.dtype)
                    self._random_exploration_timesteps.assign_sub(1)
                else:            
                    noise = tf.random.normal(action.shape, 0.0, 0.3) # TODO: make variable
                    action = tf.clip_by_value(action + noise, -1, 1)

            # Store agent action
            actions[agent] = action

        return actions, next_rnn_states
    
    def train_step(self, batch):
        logs = self._tf_train_step(batch)
        return logs

    @tf.function(jit_compile=True) # NOTE: comment this out if using debugger
    def _tf_train_step(self, batch):
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

            # Masked mean
            critic_mask = tf.squeeze(tf.stack([zero_padding_mask[:-1]]*N, axis=2))
            critic_loss_1 = tf.reduce_sum(critic_loss_1 * critic_mask) / tf.reduce_sum(critic_mask)
            critic_loss_2 = tf.reduce_sum(critic_loss_2 * critic_mask) / tf.reduce_sum(critic_mask)
            critic_loss = (critic_loss_1 + critic_loss_2) / 2

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
            qs = tf.reduce_mean((qs_1, qs_2), axis=0)
            
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
            "Mean Critic Loss": (critic_loss),
            "Policy Loss": policy_loss,
        }

        return logs

    def _update_target_network(self, online_variables, target_variables):
        """Update the target networks."""
        tau = self._target_update_rate
        for src, dest in zip(online_variables, target_variables):
            dest.assign(dest * (1.0 - tau) + src * tau)