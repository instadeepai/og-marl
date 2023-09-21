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

@snt.allow_empty_variables
class IdentityNetwork(snt.Module):

    def __init__(self):
        super().__init__()

    def __call__(self, x):
        return x

@snt.allow_empty_variables
class LocalObservationStateCriticNetwork(snt.Module):

    def __init__(self, local_observation_network: snt.Module, state_network: snt.Module, output_network: snt.Module,
    obs_is_pixel_based=False, state_is_pixel_based=False):
        super().__init__()

        self._local_observation_network = local_observation_network
        self._state_network = state_network
        self._output_network = output_network

        self._obs_is_pixel_bsed = obs_is_pixel_based
        self._state_is_pixel_based = state_is_pixel_based

    def __call__(self, observations: tf.Tensor, states: tf.Tensor) -> tf.Tensor:
        local_observation_embed = self._local_observation_network(observations)
        state_embed = self._state_network(states)

        if self._obs_is_pixel_bsed:
            leading_dims = local_observation_embed.shape[:-3]
            local_observation_embed = tf.reshape(local_observation_embed, (*leading_dims,-1))

        if self._state_is_pixel_based:
            leading_dims = state_embed.shape[:-3]
            state_embed = tf.reshape(state_embed, (*leading_dims,-1))

        embed = tf.concat([local_observation_embed, state_embed], axis=-1)

        output = self._output_network(embed)

        return output