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

"""Implementation of independent Q-learning (DRQN style)"""
import copy
import tensorflow as tf
import numpy as np
import sonnet as snt
import launchpad as lp
from acme.tf import variable_utils
from acme import datasets
from mava.adders import reverb as reverb_adders
from mava.utils import lp_utils
import reverb

from og_marl.offline_tools.offline_dataset import MAOfflineDataset
from og_marl.offline_tools.offline_environment_logger import (
    MAOfflineEnvironmentSequenceLogger,
)
from og_marl.environment_loop import EnvironmentLoop, EvaluationEnvironmentLoop
from og_marl.systems import SystemBuilderBase
from og_marl.systems.iql.executor import IQLExecutor
from og_marl.systems.iql.trainer import IQLTrainer
from og_marl.utils.executor_utils import concat_agent_id_to_obs


class IQLSystemBuilder(SystemBuilderBase):
    def __init__(
        self,
        environment_factory,
        logger_factory,
        q_network,
        batch_size=64,
        min_replay_size=64,
        max_replay_size=5000,  # num episodes in buffer (online)
        sequence_length=20,
        period=10,
        samples_per_insert=None,
        eps_start=1.0,
        eps_min=0.05,
        eps_dec=1e-5,
        variable_update_period=1,
        max_gradient_norm=20.0,
        discount=0.99,
        lambda_=0.6,
        target_update_rate=0.001,
        optimizer=snt.optimizers.Adam(learning_rate=1e-4),
        offline_environment_logging: bool = False,
        trajectories_per_file=100,  # Number of sequences per file in offline dataset
        offline_env_log_dir=None,
        add_agent_id_to_obs=False,
        record_evaluator_every=None,
        record_executor_every=None,
        evaluation_period=100,  # ~ every 100 trainer steps
        evaluation_episodes=32,
        max_trainer_steps=1e6,
        checkpoint_subpath="",
        must_checkpoint=False,
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
        # Q-Network
        self._q_network = q_network

        # Extra hyper-parameters
        self._lambda = lambda_
        self._target_update_rate = target_update_rate

        # Optimizer
        self._optimizer = optimizer

        # Replay buffer config
        self._min_replay_size = min_replay_size
        self._max_replay_size = max_replay_size
        self._samples_per_insert = samples_per_insert
        self._sequence_length = sequence_length
        self._period = period
        self._batch_size = batch_size

        # Exploration
        self._eps_start = eps_start
        self._eps_min = eps_min
        self._eps_dec = eps_dec

        # Executor and trainer
        self._trainer_fn = IQLTrainer
        self._executor_fn = IQLExecutor

        # Wheter to log experience into an offline dataset
        self._offline_environment_logging = offline_environment_logging
        self._trajectories_per_file = trajectories_per_file
        self._offline_env_log_dir = offline_env_log_dir

        # Recording and evaluation
        self._record_evaluator_every = record_evaluator_every
        self._evaluation_episodes = evaluation_episodes
        self._evaluation_period = evaluation_period
        self._record_executor_every = record_executor_every

    def replay(self, *args, **kwargs):
        adder_signiture = reverb_adders.ParallelSequenceAdder.signature(
            self._environment_spec, self._sequence_length, self._get_extra_spec()
        )
        if self._samples_per_insert is None:
            # We will take a samples_per_insert ratio of None to mean that there is
            # no limit, i.e. this only implies a min size limit.
            rate_limiter = reverb.rate_limiters.MinSize(self._min_replay_size)
        else:
            # Create enough of an error buffer to give a 10% tolerance in rate.
            samples_per_insert_tolerance = 0.1 * self._samples_per_insert
            error_buffer = self._min_replay_size * samples_per_insert_tolerance

            rate_limiter = reverb.rate_limiters.SampleToInsertRatio(
                min_size_to_sample=self._min_replay_size,
                samples_per_insert=self._samples_per_insert,
                error_buffer=error_buffer,
            )

        replay_table = reverb.Table(
            name="priority_table",
            sampler=reverb.selectors.Uniform(),
            remover=reverb.selectors.Fifo(),
            max_size=self._max_replay_size,
            rate_limiter=rate_limiter,
            signature=adder_signiture,
        )

        return [replay_table]


    def executor(self, trainer, replay_client, id=0):
        # Create the environment.
        environment = self._environment_factory()

        # Create logger
        logger = self._logger_factory(f"executor_{id}")

        # Setup offline environment logging
        if self._offline_environment_logging:

            environment = MAOfflineEnvironmentSequenceLogger(
                environment=environment,
                sequence_length=self._sequence_length,
                period=self._period,
                logdir=self._offline_env_log_dir
                if self._offline_env_log_dir
                else logger._path("offline_logs"),
                min_sequences_per_file=self._trajectories_per_file,
                label=f"executor_{id}",
            )

        # Create replay adder
        adder = self._build_adder(replay_client=replay_client)

        # Create the executor.
        executor = self._build_executor(
            logger,
            trainer,
            adder=adder,
            eps_dec=self._eps_dec,
            eps_min=self._eps_min,
            eps_start=self._eps_start,
            variable_update_period=self._variable_update_period,
        )

        # Create the loop to connect environment and executor
        executor_environment_loop = EnvironmentLoop(
            environment,
            executor,
            logger=logger,
            record_every=None,  # evaluator records during training
        )

        return executor_environment_loop

    def evaluator(self, trainer):

        # Create the environment.
        environment = self._environment_factory()

        # Create logger
        logger = self._logger_factory("evaluator")

        # Setup offline environment logging
        if self._offline_environment_logging:

            environment = MAOfflineEnvironmentSequenceLogger(
                environment=environment,
                sequence_length=self._sequence_length,
                period=self._period,
                logdir=self._offline_env_log_dir
                if self._offline_env_log_dir
                else logger._path("offline_logs"),
                min_sequences_per_file=self._trajectories_per_file,
                label="evaluator",
            )

        # Create the executor.
        executor = self._build_executor(
            logger,
            trainer,
            adder=None,
            eps_dec=0.0,
            eps_min=0.0,
            eps_start=0.0,
            variable_update_period=1,
        )

        # Create the loop to connect environment and executor
        executor_environment_loop = EvaluationEnvironmentLoop(
            environment,
            executor,
            trainer,
            logger=logger,
            record_every=self._record_evaluator_every,
            evaluation_episodes=self._evaluation_episodes,
            evaluation_period=self._evaluation_period,
        )

        return executor_environment_loop

    def trainer(self, replay_client, *args, **kwargs):
        # Create logger
        logger = self._logger_factory("trainer")

        # Build dataset
        dataset = datasets.make_reverb_dataset(
            table="priority_table",
            server_address=replay_client.server_address,
            batch_size=self._batch_size,
            prefetch_size=4,
            sequence_length=self._sequence_length,
        )

        trainer = self._build_trainer(
            dataset,
            logger,
        )

        return trainer
    
    def run_sequentially(
        self,
        max_executor_episodes=1_000_000,
        trainer_steps_per_episode=1,
        min_episodes_before_train=10,
        evaluator_period=64,
        evaluator_episodes=32,
    ):

        self._samples_per_insert = (
            None  # set to None so that reverb does not deadlock system
        )
        self._min_replay_size = min(min_episodes_before_train, self._min_replay_size)
        replay_tables = self.replay()
        replay_server = reverb.Server(tables=replay_tables)
        replay_client = reverb.Client(f"localhost:{replay_server.port}")

        trainer = self.trainer(replay_client)

        executor = self.executor(trainer, replay_client)

        evaluator = self.evaluator(trainer)

        executor_episodes = 0
        trainer_steps = 0
        while executor_episodes < max_executor_episodes:

            executor_logs = executor.run_episode()

            if executor_episodes >= min_episodes_before_train:
                for _ in range(trainer_steps_per_episode):
                    trainer_logs = trainer.step()  # logging done in trainer
                    trainer_steps += 1

            if executor_episodes % evaluator_period == 0:
                evaluator_logs = evaluator.run_evaluation(
                    trainer_steps, evaluator_episodes
                )

            executor_episodes += 1

    def run_in_parallel(
        self,
        num_executors=4,
        use_evaluator=True,
        terminal="current_terminal",
        nodes_on_gpu=["trainer"],
    ):
        """Run online multi-process system."""

        program = lp.Program(name="launchpad_program")

        with program.group("replay"):
            replay = program.add_node(lp.ReverbNode(self.replay))

        with program.group("trainer"):
            # Add trainer
            trainer = program.add_node(
                lp.CourierNode(self.trainer, replay_client=replay)
            )

        if use_evaluator:
            with program.group("evaluator"):
                # Add evaluator
                program.add_node(lp.CourierNode(self.evaluator, trainer=trainer))

        with program.group("executor"):

            # Add executors
            for id in range(num_executors):
                program.add_node(
                    lp.CourierNode(
                        self.executor, replay_client=replay, trainer=trainer, id=id
                    )
                )

        local_resources = lp_utils.to_device(
            program_nodes=program.groups.keys(), nodes_on_gpu=nodes_on_gpu
        )

        # Launch
        lp.launch(
            program,
            lp.LaunchType.LOCAL_MULTI_PROCESSING,
            terminal=terminal,
            local_resources=local_resources,
        )

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
            10 * self._evaluation_episodes,
            use_best_checkpoint=True,
        )

    def _get_extra_spec(self):
        q_network = self._initialise_networks()["q_network"]

        core_state_specs = {}
        for agent in self._agents:
            core_state_specs[agent] = q_network.initial_state(1)[0][
                0
            ]  # squeeze out batch dim
        return {"zero_padding_mask": np.array(1)}

    def _build_adder(self, replay_client):

        adder = reverb_adders.ParallelSequenceAdder(
            priority_fns=None,
            client=replay_client,
            sequence_length=self._sequence_length,
            period=self._period,
        )

        return adder

    def _build_executor(
        self,
        logger,
        trainer=None,
        adder=None,
        eps_start=0.0,
        eps_min=0.0,
        eps_dec=0.0,
        variable_update_period=1,
    ):

        # Initialise Q-network
        q_network = self._initialise_networks()["q_network"]

        # Variable client
        variables = {
            "q_network": q_network.variables,
        }
        if trainer is not None:
            variable_client = variable_utils.VariableClient(
                client=trainer,
                variables=variables,
                update_period=variable_update_period,
            )
            variable_client.update_and_wait()
        else:
            variable_client = None

        # Executor
        executor = self._executor_fn(
            agents=self._agents,
            variable_client=variable_client,
            q_network=q_network,
            adder=adder,
            eps_start=eps_start,
            eps_min=eps_min,
            eps_dec=eps_dec,
            add_agent_id_to_obs=self._add_agent_id_to_obs,
            checkpoint_subpath=self._checkpoint_subpath
            if self._checkpoint_subpath
            else logger._path("checkpoints"),
            must_checkpoint=self._must_checkpoint,
        )

        return executor

    def _build_trainer(
        self,
        dataset,
        logger,
    ):

        # Initialise Q-network
        q_network = self._initialise_networks()

        trainer = self._trainer_fn(
            agents=self._agents,
            q_network=q_network,
            optimizer=self._optimizer,
            discount=self._discount,
            target_update_rate=self._target_update_rate,
            lambda_=self._lambda,
            dataset=dataset,
            max_gradient_norm=self._max_gradient_norm,
            logger=logger,
            add_agent_id_to_obs=self._add_agent_id_to_obs,
            max_trainer_steps=self._max_trainer_steps,
        )

        return trainer

    def _initialise_networks(self):
        q_network = copy.deepcopy(self._q_network)

        spec = list(self._environment_spec.get_agent_specs().values())[0]

        dummy_observation = tf.zeros_like(spec.observations.observation)

        if self._add_agent_id_to_obs:
            dummy_observation = concat_agent_id_to_obs(
                dummy_observation, 0, len(self._agents)
            )

        dummy_observation = tf.expand_dims(
            dummy_observation, axis=0
        )  # add dummy batch dim

        # Initialize Q-network variables
        q_network(dummy_observation, q_network.initial_state(1))

        return {"q_network": q_network}
