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

"""Independent and Centralised Critic Implementations for TD3 based systems."""
import tensorflow as tf
import sonnet as snt

from og_marl.utils.trainer_utils import batch_concat_agent_id_to_obs

class StateAndActionCritic(snt.Module):

    def __init__(self, num_agents, num_actions):
        self.N = num_agents
        self.A = num_actions

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

    def initialise(self, observation, state, action):
        """ A method to initialise the parameters in the critic network.

        observation: a dummy observation with no batch dimension. We assume
            all agent's observations have the same shape.
        state: a dummy environment state with no batch dimension.
        action: a dummy action with no batch dimension. We assume all agents have 
            the same action shape.
        """
        state = tf.reshape(state, (1,1) + state.shape) # add time and batch dim
        actions = tf.stack([action]*self.N, axis=0) # action for each agent
        actions = tf.reshape(actions, (1,1) + actions.shape) # add time and batch dim

        self(None, state, actions, actions) # __call__ with dummy inputs

    def __call__(self, observations, states, agent_actions, other_actions, stop_other_actions_gradient=True):
        """Forward pass of critic network.
        
        observations [T,B,N,O]
        states [T,B,S]
        agent_actions [T,B,N,A]: the actions the agent took.
        other_actions [T,B,N,A]: the actions the other agents took.
        """
        # Repeat states for each agent
        states = tf.stack([states]*self.N, axis=2)

        # Concat states and joint actions
        critic_input = tf.concat([states, agent_actions], axis=-1)

        # Concat agent IDs to critic input
        critic_input = batch_concat_agent_id_to_obs(critic_input)

        q_values = self._critic_network(critic_input)
        return q_values

class ObservationAndActionCritic(snt.Module):

    def __init__(self, num_agents, num_actions):
        self.N = num_agents
        self.A = num_actions

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

    def initialise(self, observation, state, action):
        """ A method to initialise the parameters in the critic network.

        observation: a dummy observation with no batch dimension. We assume
            all agent's observations have the same shape.
        state: a dummy environment state with no batch dimension.
        action: a dummy action with no batch dimension. We assume all agents have 
            the same action shape.
        """
        observation = tf.stack([observation]*self.N, axis=0) # observation for each agent
        observation = tf.reshape(observation, (1,1) + observation.shape) # add time and batch dim
        actions = tf.stack([action]*self.N, axis=0) # action for each agent
        actions = tf.reshape(actions, (1,1) + actions.shape) # add time and batch dim

        self(observation, state, actions, actions) # __call__ with dummy inputs

    def __call__(self, observations, states, agent_actions, other_actions, stop_other_actions_gradient=True):
        """Forward pass of critic network.
        
        observations [T,B,N,O]
        states [T,B,S]
        agent_actions [T,B,N,A]: the actions the agent took.
        other_actions [T,B,N,A]: the actions the other agents took.
        """
        # Concat states and joint actions
        critic_input = tf.concat([observations, agent_actions], axis=-1)

        # Concat agent IDs to critic input
        critic_input = batch_concat_agent_id_to_obs(critic_input)

        q_values = self._critic_network(critic_input)
        return q_values


class StateAndJointActionCritic(snt.Module):

    def __init__(self, num_agents, num_actions):
        self.N = num_agents
        self.A = num_actions

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

    def initialise(self, observation, state, action):
        """ A method to initialise the parameters in the critic network.

        observation: a dummy observation with no batch dimension. We assume
            all agent's observations have the same shape.
        state: a dummy environment state with no batch dimension.
        action: a dummy action with no batch dimension. We assume all agents have 
            the same action shape.
        """
        state = tf.reshape(state, (1,1) + state.shape) # add time and batch dim
        actions = tf.stack([action]*self.N, axis=0) # action for each agent
        actions = tf.reshape(actions, (1,1) + actions.shape) # add time and batch dim

        self(None, state, actions, actions) # __call__ with dummy inputs

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