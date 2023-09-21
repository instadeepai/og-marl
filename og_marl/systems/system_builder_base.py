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

import sonnet as snt
from mava import specs as mava_specs


class SystemBuilderBase:
    def __init__(
        self,
        environment_factory,
        logger_factory,
        max_gradient_norm=20.0,
        discount=0.99,
        variable_update_period=1,
        add_agent_id_to_obs=False,
        max_trainer_steps=1e6,
        checkpoint_subpath="",
        must_checkpoint=False,
    ):

        self._environment_factory = environment_factory
        self._environment_spec = mava_specs.MAEnvironmentSpec(environment_factory())
        self._agents = self._environment_spec.get_agent_ids()

        self._logger_factory = logger_factory

        self._max_gradient_norm = max_gradient_norm
        self._discount = discount
        self._variable_update_period = variable_update_period
        self._add_agent_id_to_obs = add_agent_id_to_obs
        self._max_trainer_steps = max_trainer_steps

        self._must_checkpoint = must_checkpoint
        self._checkpoint_subpath = checkpoint_subpath

    def evaluator(self, trainer, *args, **kwargs):
        raise NotImplementedError

    def trainer(self, *args, **kwargs):
        raise NotImplementedError

    def run_in_parallel(self, *args, **kwargs):
        """Method to train the system online using LaunchPad (see Mava)."""
        raise NotImplementedError

    def run_offline(self, *args, **kwargs):
        """Method to train the system offline."""
        raise NotImplementedError
