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
import cpprb
import tree

class SequenceCPPRB:

    def __init__(self, environment, sequence_length=20, max_size=10_000, batch_size=32):
        self._environment = environment
        self._sequence_length = sequence_length
        self._max_size = max_size
        self._batch_size = batch_size
        self._info_spec = self._environment.info_spec

        cpprb_env_dict = {}
        sequence_buffer = {}
        for agent in environment.possible_agents:
            obs_shape = self._environment.observation_spaces[agent].shape
            act_shape = self._environment.action_spaces[agent].shape

            cpprb_env_dict[f"{agent}_observations"] = {"shape": (sequence_length, *obs_shape)}
            cpprb_env_dict[f"{agent}_actions"] = {"shape": (sequence_length, *act_shape)}
            cpprb_env_dict[f"{agent}_rewards"] = {"shape": (sequence_length,)}
            cpprb_env_dict[f"{agent}_terminals"] = {"shape": (sequence_length,)}
            cpprb_env_dict[f"{agent}_truncations"] = {"shape": (sequence_length,)}

            sequence_buffer[f"{agent}_observations"] = np.zeros((sequence_length, *obs_shape), "float32")
            sequence_buffer[f"{agent}_actions"] = np.zeros((sequence_length, *act_shape), "float32")
            sequence_buffer[f"{agent}_rewards"] = np.zeros((sequence_length,), "float32")
            sequence_buffer[f"{agent}_terminals"] = np.zeros((sequence_length,), "float32")
            sequence_buffer[f"{agent}_truncations"] = np.zeros((sequence_length,), "float32")

            if "legals" in self._info_spec:
                legals_shape = self._info_spec["legals"][agent].shape
                cpprb_env_dict[f"{agent}_legals"] = {"shape": (sequence_length, *legals_shape)}
                sequence_buffer[f"{agent}_legals"] = np.zeros((sequence_length, *legals_shape), "float32")
        
        cpprb_env_dict["mask"] = {"shape": (sequence_length,)}
        sequence_buffer["mask"] = np.zeros((sequence_length,), "float32")

        if "state" in self._info_spec:
            state_shape = self._info_spec["state"].shape

            cpprb_env_dict["state"] = {"shape": (sequence_length, *state_shape)}
            sequence_buffer["state"] = np.zeros((sequence_length, *state_shape), "float32")

        self._cpprb = cpprb.ReplayBuffer(
            max_size,
            env_dict =cpprb_env_dict,
            default_dtype=np.float32
        )

        self._sequence_buffer = sequence_buffer

        self._t = 0

    def add(self, observations, actions, rewards, terminals, truncations, infos):

        for agent in self._environment.possible_agents:
            self._sequence_buffer[f"{agent}_observations"][self._t] = np.array(observations[agent], "float32")
            self._sequence_buffer[f"{agent}_actions"][self._t] = np.array(actions[agent], "float32")
            self._sequence_buffer[f"{agent}_rewards"][self._t] = np.array(rewards[agent], "float32")
            self._sequence_buffer[f"{agent}_terminals"][self._t] = np.array(terminals[agent], "float32")
            self._sequence_buffer[f"{agent}_truncations"][self._t] = np.array(truncations[agent], "float32")

            if "legals" in infos:
                self._sequence_buffer[f"{agent}_legals"][self._t] = np.array(infos["legals"][agent], "float32")

        self._sequence_buffer["mask"][self._t] = np.array(1, "float32")

        if "state" in infos:
            self._sequence_buffer["state"][self._t] = np.array(infos["state"], "float32")

        self._t += 1

        if self._t == self._sequence_length:
            self._push_to_cpprb()
            self._t = 0

    def end_of_episode(self):
        if self._t > 0:
            self._zero_pad()
            self._push_to_cpprb()

        self._cpprb.on_episode_end()

        self._t = 0

    def populate_from_dataset(self, dataset):
        dataset = dataset.batch(128)
        for batch in dataset:
            batch = tree.map_structure(lambda x: x.numpy(), batch)
            self._cpprb.add(**batch)
        print("Done")

    def _push_to_cpprb(self):
        self._cpprb.add(**self._sequence_buffer)

    def _zero_pad(self):
        for key, value in self._sequence_buffer.items():
            trailing_dims = value.shape[1:]
            zero_pad = np.zeros((self._sequence_length - self._t, *trailing_dims), "float32")
            self._sequence_buffer[key][self._t:] = zero_pad

    def __iter__(self):
        return self
    
    def __next__(self):
        cpprb_sample = self._cpprb.sample(self._batch_size)

        return cpprb_sample