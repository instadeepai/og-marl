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

import numpy as np

class Dtype:

    def __init__(self, environment, dtype):

        self._environment = environment
        self._dtype = dtype

    def reset(self):
        observations = self._environment.reset()

        if isinstance(observations, tuple):
            observations, infos = observations
        else:
            infos = {}

        for agent, observation in observations.items():
            observations[agent] = observation.astype(self._dtype)

        return observations, infos
    
    def step(self, actions):
        next_observations, rewards, terminals, truncations, infos = self._environment.step(actions)

        for agent, observation in next_observations.items():
            next_observations[agent] = observation.astype(self._dtype)

        return next_observations, rewards, terminals, truncations, infos
    
    def __getattr__(self, name: str):
        """Expose any other attributes of the underlying environment."""
        if hasattr(self.__class__, name):
            return self.__getattribute__(name)
        else:
            return getattr(self._environment, name)

class PadObsandActs:

    def __init__(self, environment):

        self._environment = environment

        self._obs_dim = 0
        self._act_dim = 0

        for agent in self._environment.possible_agents:
            act_dim = self._environment.action_spaces[agent].shape[0]
            obs_dim = self._environment.observation_spaces[agent].shape[0]

            if act_dim > self._act_dim:
                self._act_dim = act_dim

            if obs_dim > self._obs_dim:
                self._obs_dim = obs_dim

    def reset(self):
        observations = self._environment.reset()

        if isinstance(observations, tuple):
            observations, infos = observations
        else:
            infos = {}

        for agent, observation in observations.items():
            if observation.shape[0] < self._obs_dim:
                missing_dim = self._obs_dim - observation.shape[0]
                observations[agent] = np.concatenate((observation, np.zeros((missing_dim,), observation.dtype)))

        return observations, infos
    
    def step(self, actions):
        actions = {agent: action[:self._environment.action_spaces[agent].shape[0]] for agent, action in actions.items()}
        next_observations, rewards, terminals, truncations, infos = self._environment.step(actions)

        for agent, observation in next_observations.items():
            if observation.shape[0] < self._obs_dim:
                missing_dim = self._obs_dim - observation.shape[0]
                next_observations[agent] = np.concatenate((observation, np.zeros((missing_dim,), observation.dtype)))

        return next_observations, rewards, terminals, truncations, infos
    
    def __getattr__(self, name: str):
        """Expose any other attributes of the underlying environment."""
        if hasattr(self.__class__, name):
            return self.__getattribute__(name)
        else:
            return getattr(self._environment, name)

