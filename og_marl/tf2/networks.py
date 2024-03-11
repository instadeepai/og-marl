from typing import Sequence

import tensorflow as tf
from tensorflow import Tensor
import sonnet as snt


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
