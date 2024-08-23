from typing import Sequence

import tensorflow as tf
from tensorflow import Tensor
import sonnet as snt

from og_marl.tf2.utils import batch_concat_agent_id_to_obs


class StateAndJointActionCritic(snt.Module):
    def __init__(self, num_agents: int, num_actions: int):
        self.N = num_agents
        self.A = num_actions

        self._critic_network = snt.Sequential(
            [
                snt.Linear(128),
                tf.nn.relu,
                snt.Linear(128),
                tf.nn.relu,
                snt.Linear(1),
            ]
        )

        super().__init__()

    def __call__(
        self,
        states: Tensor,
        agent_actions: Tensor,
        other_actions: Tensor,
        stop_other_actions_gradient: bool = True,
    ) -> Tensor:
        """Forward pass of critic network.

        observations [T,B,N,O]
        states [T,B,S]
        agent_actions [T,B,N,A]: the actions the agent took.
        other_actions [T,B,N,A]: the actions the other agents took.
        """
        if stop_other_actions_gradient:
            other_actions = tf.stop_gradient(other_actions)

        # Make joint action
        joint_actions = self.make_joint_action(agent_actions, other_actions)

        # Repeat states for each agent
        states = tf.stack([states] * self.N, axis=2)  # [T,B,S] -> [T,B,N,S]

        # Concat states and joint actions
        critic_input = tf.concat([states, joint_actions], axis=-1)

        # Concat agent IDs to critic input
        # critic_input = batch_concat_agent_id_to_obs(critic_input)

        q_values: Tensor = self._critic_network(critic_input)

        return q_values

    def make_joint_action(self, agent_actions: Tensor, other_actions: Tensor) -> Tensor:
        """Method to construct the joint action.

        agent_actions [T,B,N,A]: tensor of actions the agent took. Usually
            the actions from the learnt policy network.
        other_actions [[T,B,N,A]]: tensor of actions the agent took. Usually
            the actions from the replay buffer.
        """
        T, B, N, A = agent_actions.shape[:4]  # (B,N,A)
        all_joint_actions = []
        for i in range(N):  # type: ignore
            one_hot = tf.expand_dims(
                tf.cast(tf.stack([tf.stack([tf.one_hot(i, N)] * B, axis=0)] * T, axis=0), "bool"),  # type: ignore
                axis=-1,
            )
            joint_action = tf.where(one_hot, agent_actions, other_actions)
            joint_action = tf.reshape(joint_action, (T, B, N * A))  # type: ignore
            all_joint_actions.append(joint_action)
        all_joint_actions: Tensor = tf.stack(all_joint_actions, axis=2)

        return all_joint_actions


class StateAndActionCritic(snt.Module):
    def __init__(self, num_agents: int, num_actions: int, add_agent_id: bool = True):
        self.N = num_agents
        self.A = num_actions
        self.add_agent_id = add_agent_id

        self._critic_network = snt.Sequential(
            [
                snt.Linear(128),
                tf.nn.relu,
                snt.Linear(128),
                tf.nn.relu,
                snt.Linear(1),
            ]
        )

        super().__init__()

    def __call__(
        self,
        states: Tensor,
        agent_actions: Tensor,
    ) -> Tensor:
        """Forward pass of critic network.

        states [T,B,S]
        agent_actions [T,B,N,A]: the actions the agent took.
        """
        # Repeat states for each agent
        states = tf.stack([states] * self.N, axis=2)

        # Concat states and joint actions
        critic_input = tf.concat([states, agent_actions], axis=-1)

        # Concat agent IDs to critic input
        if self.add_agent_id:
            critic_input = batch_concat_agent_id_to_obs(critic_input)

        q_values: Tensor = self._critic_network(critic_input)

        return q_values


class QMixer(snt.Module):

    """QMIX mixing network."""

    def __init__(
        self,
        num_agents: int,
        embed_dim: int = 32,
        hypernet_embed: int = 64,
        non_monotonic: bool = False,
    ):
        """Initialise QMIX mixing network

        Args:
        ----
            num_agents: Number of agents in the environment
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

    def __call__(self, agent_qs: Tensor, states: Tensor) -> Tensor:
        """Forward method."""
        B = agent_qs.shape[0]  # batch size
        state_dim = states.shape[2:]

        agent_qs = tf.reshape(agent_qs, (-1, 1, self.num_agents))

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

        return q_tot

    def k(self, states: Tensor) -> Tensor:
        """Method used by MAICQ."""
        B, T = states.shape[:2]

        w1 = tf.math.abs(self.hyper_w_1(states))
        w_final = tf.math.abs(self.hyper_w_final(states))
        w1 = tf.reshape(w1, shape=(-1, self.num_agents, self.embed_dim))
        w_final = tf.reshape(w_final, shape=(-1, self.embed_dim, 1))
        k = tf.matmul(w1, w_final)
        k = tf.reshape(k, shape=(B, -1, self.num_agents))
        k = k / (tf.reduce_sum(k, axis=2, keepdims=True) + 1e-10)
        return k


@snt.allow_empty_variables
class IdentityNetwork(snt.Module):
    def __init__(self) -> None:
        super().__init__()
        return

    def __call__(self, x: Tensor) -> Tensor:
        return x


class CNNEmbeddingNetwork(snt.Module):
    def __init__(
        self, output_channels: Sequence[int] = (8, 16), kernel_sizes: Sequence[int] = (3, 2)
    ) -> None:
        super().__init__()
        assert len(output_channels) == len(kernel_sizes)

        layers = []
        for layer_i in range(len(output_channels)):
            layers.append(snt.Conv2D(output_channels[layer_i], kernel_sizes[layer_i]))
            layers.append(tf.nn.relu)
        layers.append(tf.keras.layers.Flatten())

        self.conv_net = snt.Sequential(layers)

    def __call__(self, x: Tensor) -> Tensor:
        """Embed a pixel-styled input into a vector using a conv net.

        We assume the input has trailing dims
        being the width, height and channel dimensions of the input.

        The output shape is then given as (B,T,N,Embed)
        """
        leading_dims = x.shape[:-3]
        trailing_dims = x.shape[-3:]  # W,H,C

        x = tf.reshape(x, shape=(-1, *trailing_dims))
        embed = self.conv_net(x)
        embed = tf.reshape(embed, shape=(*leading_dims, -1))
        return embed
