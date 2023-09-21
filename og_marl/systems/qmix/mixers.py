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

import tensorflow as tf
import sonnet as snt

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

        if preprocess_network is not None:
            self.preprocess_network = preprocess_network
        else: 
            self.preprocess_network = snt.Sequential([tf.identity])

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
        
        B = agent_qs.shape[0] # batch size
        state_dim = states.shape[2:]

        agent_qs = tf.reshape(agent_qs, (-1, 1, self.num_agents))
        states = tf.reshape(states, (-1, *state_dim))

        # Pass states through preprocess network
        states = self.preprocess_network(states)

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

    def k(self, states):
        """Method used by MAICQ."""

        is_pixel_state = len(states.shape) > 4
        if is_pixel_state:
            B, T = states.shape[:2]
            trailing_dims = states.shape[2:]
            states = tf.reshape(states, (B*T, *trailing_dims))
        else:
            B, T = states.shape[:2]

        # Pass states through preprocess network
        states = self.preprocess_network(states)

        if is_pixel_state:
            states = tf.reshape(states, (B,T,-1))

        w1 = tf.math.abs(self.hyper_w_1(states))
        w_final = tf.math.abs(self.hyper_w_final(states))
        w1 = tf.reshape(w1, shape=(-1, self.num_agents, self.embed_dim))
        w_final = tf.reshape(w_final, shape=(-1, self.embed_dim, 1))
        k = tf.matmul(w1,w_final)
        k = tf.reshape(k, shape=(B, -1, self.num_agents))
        k = k / tf.reduce_sum(k, axis=2, keepdims=True)
        return k