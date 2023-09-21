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

"""Behaviour cloning implementation."""
import copy
import tensorflow as tf
import sonnet as snt
from acme.tf import variable_utils

from og_marl.offline_tools.offline_dataset import MAOfflineDataset
from og_marl.environment_loop import EvaluationEnvironmentLoop
from og_marl.systems.bc.executor import (
    DiscreteBCExecutor,
    ContinuousBCExecutor,
) 
from og_marl.systems.bc.trainer import (
    DiscreteBCTrainer,
    ContinuousBCTrainer,
)
from og_marl.systems.system_builder_base import SystemBuilderBase
from og_marl.utils.executor_utils import concat_agent_id_to_obs


class BCSystemBuilder(SystemBuilderBase):
    """Muti-Agent Behaviour Cloning."""

    def __init__(
        self,
        environment_factory,
        logger_factory,
        behaviour_cloning_network: snt.Module,
        discrete_actions=True,
        batch_size=32,
        variable_update_period=1,
        max_gradient_norm=20.0,
        optimizer=snt.optimizers.Adam(learning_rate=1e-3),
        add_agent_id_to_obs=False,
        max_trainer_steps=50_000,
        must_checkpoint=True,
        evaluation_period=1000,
        evaluation_episodes=32,
        checkpoint_subpath=".",
    ):

        super().__init__(
            environment_factory,
            logger_factory,
            max_gradient_norm=max_gradient_norm,
            discount=None,
            variable_update_period=variable_update_period,
            add_agent_id_to_obs=add_agent_id_to_obs,
            max_trainer_steps=max_trainer_steps,
            must_checkpoint=must_checkpoint,
            checkpoint_subpath=checkpoint_subpath,
        )

        # Hyper-params
        self._batch_size = batch_size
        self._optimizer = optimizer

        # Network
        self._behaviour_cloning_network = behaviour_cloning_network

        # Trainer and Executor functions
        self._trainer_fn = DiscreteBCTrainer if discrete_actions else ContinuousBCTrainer
        self._executor_fn = DiscreteBCExecutor if discrete_actions else ContinuousBCExecutor

        # Evaluation
        self._evaluation_period=evaluation_period
        self._evaluation_episodes=evaluation_episodes

    def evaluator(self, trainer):
        # Create the environment.
        environment = self._environment_factory()

        # Create logger
        logger = self._logger_factory("evaluator")

        # Create the executor.
        executor = self._build_executor(trainer)

        # Create the loop to connect environment and executor
        executor_environment_loop = EvaluationEnvironmentLoop(
            environment,
            executor,
            trainer,
            logger=logger,
        )

        return executor_environment_loop

    def trainer(self):
        # Create logger
        logger = self._logger_factory("trainer")

        # Create environment for the offline dataset
        environment = self._environment_factory()

        # Build offline dataset
        dataset = MAOfflineDataset(
            environment=environment,
            logdir=self._offline_dataset_dir,
            batch_size=self._batch_size,
            shuffle_buffer_size=self._shuffle_buffer_size,
            s3=self._s3,
        )

        trainer = self._build_trainer(dataset, logger)

        return trainer

    def run_offline(
        self,
        dataset_dir,
        shuffle_buffer_size=5000
    ):
        # Make logger
        logger = self._logger_factory("trainer")

        # Create environment for the offline dataset
        environment = self._environment_factory()

        # Build offline dataset
        dataset = MAOfflineDataset(
            environment=environment,
            logdir=dataset_dir,
            batch_size=self._batch_size,
            shuffle_buffer_size=shuffle_buffer_size,
        )

        trainer = self._build_trainer(dataset, logger)

        evaluator = self.evaluator(trainer)

        trainer_steps = 0
        while trainer_steps < self._max_trainer_steps:

            trainer_logs = trainer.step()  # logging done in trainer

            if trainer_steps % self._evaluation_period == 0:
                evaluator_logs = evaluator.run_evaluation(
                    trainer_steps, self._evaluation_episodes
                )  # logging done in evaluator

            trainer_steps += 1

        # Final evaluation
        evaluator_logs = evaluator.run_evaluation(
            trainer_steps,
            num_evaluation_episodes=10*self._evaluation_episodes,
            use_best_checkpoint=True,
        )

    def _build_executor(self, trainer):

        # Initialise networks
        behaviour_cloning_network = self._initialise_networks()["bc_network"]

        # Variable client
        variables = {
            "behaviour_cloning_network": behaviour_cloning_network.variables,
        }
        variable_client = variable_utils.VariableClient(
            client=trainer,
            variables=variables,
            update_period=self._variable_update_period,
        )
        variable_client.update_and_wait()

        # Executor
        executor = self._executor_fn(
            agents=self._agents,
            variable_client=variable_client,
            behaviour_cloning_network=behaviour_cloning_network,
            add_agent_id_to_obs=self._add_agent_id_to_obs,
            must_checkpoint=self._must_checkpoint,
            checkpoint_subpath=self._checkpoint_subpath,
        )

        return executor

    def _build_trainer(self, dataset, logger):

        # Initialise networks
        behaviour_cloning_network = self._initialise_networks()["bc_network"]

        trainer = self._trainer_fn(
            self._agents,
            dataset=dataset,
            logger=logger,
            behaviour_cloning_network=behaviour_cloning_network,
            optimizer=self._optimizer,
            max_gradient_norm=self._max_gradient_norm,
            add_agent_id_to_obs=self._add_agent_id_to_obs,
        )

        return trainer

    def _initialise_networks(self):
        behaviour_cloning_network = copy.deepcopy(self._behaviour_cloning_network)

        spec = list(self._environment_spec.get_agent_specs().values())[0]

        dummy_observation = tf.zeros_like(spec.observations.observation)

        if self._add_agent_id_to_obs:
            dummy_observation = concat_agent_id_to_obs(
                dummy_observation, 0, len(self._agents)
            )

        dummy_observation = tf.expand_dims(
            dummy_observation, axis=0
        )  # add dummy batch dim

        # Initialize behaviour cloning network
        dummy_core_state = behaviour_cloning_network.initial_state(
            1
        )  # Dummy recurent core state
        behaviour_cloning_network(dummy_observation, dummy_core_state)

        return {"bc_network": behaviour_cloning_network}