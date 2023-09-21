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
import tensorflow_io
from acme.tf import utils as tf2_utils
import launchpad as lp
import abc


class TrainerBase(abc.ABC):
    def __init__(
        self,
        agents,
        dataset,
        logger,
        discount=0.99,
        max_gradient_norm=20.0,
        add_agent_id_to_obs=False,
        max_trainer_steps=1e6,
    ):
        # Agent keys in the environment
        self._agents = agents

        # Dataset used for training
        self._dataset_iter = iter(dataset)

        # Logger
        self._logger = logger

        # Dict to store variables the the executor fetches
        self._system_variables = {}

        # Trainer step counter
        self._trainer_step_counter = 0

        # Hyper-params
        self._discount = discount
        self._max_gradient_norm = max_gradient_norm

        # Add agent id to obs
        self._add_agent_id_to_obs = add_agent_id_to_obs

        # Max trainer steps until termination
        self._max_trainer_steps = max_trainer_steps

    def get_steps(self):
        return self._trainer_step_counter

    def step(self):
        self.before_train_step()

        if (
            self._max_trainer_steps
            and self._trainer_step_counter >= self._max_trainer_steps
        ):
            lp.stop()

        # Increment trainer step counter
        self._trainer_step_counter += 1

        # Sample dataset
        sample = next(self._dataset_iter)

        # Pass sample to _train method
        logs = self._train(
            sample, trainer_step=tf.convert_to_tensor(self._trainer_step_counter)
        )

        after_logs = self.after_train_step()

        logs.update(after_logs)

        # Write logs
        self._logger.write(logs)

        return logs

    def before_train_step(self):
        return {}

    def after_train_step(self):
        return {}

    def run(self):
        while True:
            logs = self.step()

    def get_variables(self, names):
        return [tf2_utils.to_numpy(self._system_variables[name]) for name in names]

    @abc.abstractmethod
    @tf.function
    def _train(self, sample, trainer_step):
        pass
