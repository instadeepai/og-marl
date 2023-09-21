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

"""MAICQ implementation."""
import copy
import tensorflow as tf
import sonnet as snt
from acme.tf import variable_utils

from og_marl.offline_tools.offline_dataset import MAOfflineDataset
from og_marl.environment_loop import EvaluationEnvironmentLoop
from og_marl.systems.maicq.executor import MAICQExecutor
from og_marl.systems.maicq.trainer import MAICQTrainer
from og_marl.systems.system_builder_base import SystemBuilderBase
from og_marl.utils.executor_utils import concat_agent_id_to_obs


class MAICQSystemBuilder(SystemBuilderBase):
    """Muti-Agent Implicit Constrained Q-Learning.
    
    Offline Algorithm only. Can not be run online.
    """

    def __init__(
        self,
        environment_factory,
        logger_factory,
        policy_network,
        critic_network,
        mixer,
        batch_size=32,
        add_agent_id_to_obs=True,
        variable_update_period=1,
        max_gradient_norm=20.0,
        discount=0.99,
        lambda_=0.6,
        target_update_period=600,  # Every 600 trainer steps
        critic_optimizer=snt.optimizers.Adam(learning_rate=1e-3),
        policy_optimizer=snt.optimizers.Adam(learning_rate=1e-3),
        max_trainer_steps=50_001,
        must_checkpoint=False,
        checkpoint_subpath="",
        evaluation_period=100,  # ~ every 100 trainer steps
        evaluation_episodes=32,
    ):
        super().__init__(
            environment_factory,
            logger_factory,
            max_gradient_norm=max_gradient_norm,
            discount=discount,
            variable_update_period=variable_update_period,
            add_agent_id_to_obs=add_agent_id_to_obs,
            max_trainer_steps=max_trainer_steps,
            must_checkpoint=must_checkpoint,
            checkpoint_subpath=checkpoint_subpath,
        )

        # Hyper-params
        self._batch_size = batch_size

        # Critic, mixer and policy network
        self._critic_network = critic_network
        self._mixer = mixer
        self._policy_network = policy_network

        # Optimizers
        self._policy_optimizer = policy_optimizer
        self._critic_optimizer = critic_optimizer

        # Hyper-parameters
        self._lambda = lambda_
        self._target_update_period = target_update_period

        # Trainer and Executor functions
        self._trainer_fn = MAICQTrainer
        self._executor_fn = MAICQExecutor

        # Evaluation
        self._evaluation_episodes = evaluation_episodes
        self._evaluation_period = evaluation_period

    def evaluator(self, trainer):
        # Create the environment.
        environment = self._environment_factory()

        # Create logger
        logger = self._logger_factory(f"evaluator")

        # Create the executor.
        executor = self._build_executor(
            trainer,
        )

        # Create the loop to connect environment and executor
        executor_environment_loop = EvaluationEnvironmentLoop(
            environment,
            executor,
            trainer,
            logger=logger,
        )

        return executor_environment_loop

    def run_offline(
        self,
        dataset_dir,
        shuffle_buffer_size=5000
    ):
        # Create logger
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
            num_evaluation_episodes=10 * self._evaluation_episodes,
            use_best_checkpoint=True,
        )

    def _build_executor(self, trainer):

        # Initialise networks
        (
            policy_network,
            critic_network,
        ) = self._initialise_networks()

        # Variable client
        variables = {
            "policy_network": policy_network.variables,
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
            policy_network=policy_network,
            add_agent_id_to_obs=self._add_agent_id_to_obs,
        )

        return executor

    def _build_trainer(self, dataset, logger):

        # Initialise networks
        (
            policy_network,
            critic_network,
        ) = self._initialise_networks()

        trainer = self._trainer_fn(
            self._agents,
            dataset=dataset,
            logger=logger,
            policy_network=policy_network,
            critic_network=critic_network,
            mixer=self._mixer,
            policy_optimizer=self._policy_optimizer,
            critic_optimizer=self._critic_optimizer,
            discount=self._discount,
            target_update_period=self._target_update_period,
            lambda_=self._lambda,
            max_gradient_norm=self._max_gradient_norm,
            add_agent_id_to_obs=self._add_agent_id_to_obs,
        )

        return trainer

    def _initialise_networks(self):
        policy_network = copy.deepcopy(self._policy_network)
        critic_network = copy.deepcopy(self._critic_network)

        spec = list(self._environment_spec.get_agent_specs().values())[0]
        dummy_observation = tf.zeros_like(spec.observations.observation)

        # Dummy state
        state_spec = self._environment_spec.get_extra_specs()["s_t"]
        dummy_state = tf.zeros_like(state_spec)

        if self._add_agent_id_to_obs:
            dummy_observation = concat_agent_id_to_obs(dummy_observation, 1, len(self._agents))

        dummy_observation = tf.expand_dims(dummy_observation, axis=0)
        dummy_state = tf.expand_dims(dummy_state, axis=0)


        # Initialize observation networks
        critic_network(dummy_observation, dummy_state)

        # Dummy recurent core state
        dummy_core_state = policy_network.initial_state(1)

        # Initialize policy network variables
        policy_network(dummy_observation, dummy_core_state)

        return (
            policy_network,
            critic_network,
        )
