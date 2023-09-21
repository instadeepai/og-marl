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

"""QMIX implementation."""
import copy
import tensorflow as tf
import sonnet as snt
from mava.utils.loggers import logger_utils
import functools
from og_marl.systems.qmix.mixers import QMixer

from og_marl.systems.qmix.trainer import QmixBcqTrainer, QmixCqlTrainer, QmixTrainer
from og_marl.systems.iql import IQLSystemBuilder
from og_marl.utils.executor_utils import concat_agent_id_to_obs

class QMIXSystemBuilder(IQLSystemBuilder):
    def __init__(
        self,
        environment_factory,
        logger_factory,
        q_network,
        mixer,
        batch_size=64,
        min_replay_size=64,
        max_replay_size=5000,  # num episodes in buffer
        sequence_length=20,
        period=10,
        samples_per_insert=None,
        eps_start=1.0,
        eps_min=0.05,
        eps_dec=1e-5,
        variable_update_period=3,  # Update varibles every 3 episodes
        max_gradient_norm=20.0,
        discount=0.99,
        lambda_=0.6,
        target_update_rate=0.01,
        optimizer=snt.optimizers.Adam(learning_rate=1e-4),
        offline_environment_logging=False,
        trajectories_per_file=100,
        add_agent_id_to_obs=False,
        offline_env_log_dir=None,
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
            q_network,
            optimizer=optimizer,
            max_gradient_norm=max_gradient_norm,
            discount=discount,
            variable_update_period=variable_update_period,
            batch_size=batch_size,
            min_replay_size=min_replay_size,
            max_replay_size=max_replay_size,
            sequence_length=sequence_length,
            period=period,
            samples_per_insert=samples_per_insert,
            eps_start=eps_start,
            eps_min=eps_min,
            eps_dec=eps_dec,
            lambda_=lambda_,
            offline_environment_logging=offline_environment_logging,
            trajectories_per_file=trajectories_per_file,
            add_agent_id_to_obs=add_agent_id_to_obs,
            offline_env_log_dir=offline_env_log_dir,
            record_evaluator_every=record_evaluator_every,
            record_executor_every=record_executor_every,
            evaluation_period=evaluation_period,
            evaluation_episodes=evaluation_episodes,
            target_update_rate=target_update_rate,
            max_trainer_steps=max_trainer_steps,
            must_checkpoint=must_checkpoint,
            checkpoint_subpath=checkpoint_subpath,
        )

        self._mixer = mixer
        self._trainer_fn = QmixTrainer

    def _build_trainer(self, dataset, logger):

        # Initialise networks
        q_network = self._initialise_networks()["q_network"]

        trainer = self._trainer_fn(
            agents=self._agents,
            q_network=q_network,
            mixer=self._mixer,
            optimizer=self._optimizer,
            discount=self._discount,
            target_update_rate=self._target_update_rate,
            lambda_=self._lambda,
            dataset=dataset,
            max_gradient_norm=self._max_gradient_norm,
            logger=logger,
            max_trainer_steps=self._max_trainer_steps,
            add_agent_id_to_obs=self._add_agent_id_to_obs,
        )

        return trainer

    def _initialise_networks(self):
        q_network = copy.deepcopy(self._q_network)

        spec = list(self._environment_spec.get_agent_specs().values())[0]
        dummy_observation = tf.zeros_like(spec.observations.observation)

        if self._add_agent_id_to_obs:
            dummy_observation = concat_agent_id_to_obs(
                dummy_observation, 1, len(self._agents)
            )

        dummy_observation = tf.expand_dims(
            dummy_observation, axis=0
        )  # add dummy batch dim

        # Initialise q-network
        dummy_core_state = q_network.initial_state(1)  # Dummy recurent core state
        q_network(dummy_observation, dummy_core_state)

        return {"q_network": q_network}


class QMIXBCQSystemBuilder(QMIXSystemBuilder):
    """Offline QMIX+BCQ"""

    def __init__(
        self,
        environment_factory,
        logger_factory,
        q_network,
        mixer,
        behaviour_cloning_network,  # BCQ
        threshold=0.3,  # BCQ
        batch_size=64,
        min_replay_size=64,
        max_replay_size=5000,  # num episodes in buffer
        sequence_length=20,
        period=10,
        samples_per_insert=None,
        eps_start=1.0,
        eps_min=0.05,
        eps_dec=1e-5,
        variable_update_period=3,  # Update varibles every 3 episodes
        max_gradient_norm=20.0,
        discount=0.99,
        lambda_=0.6,
        target_update_rate=None,
        optimizer=snt.optimizers.Adam(learning_rate=1e-4),
        offline_environment_logging=False,
        trajectories_per_file=100,
        add_agent_id_to_obs=False,
        offline_env_log_dir=None,
        record_evaluator_every=None,
        record_executor_every=None,
        evaluation_period=100,  # ~ every 100 trainer steps
        evaluation_episodes=32,
        max_trainer_steps=1e6,
        checkpoint_subpath="",
        must_checkpoint=False,
    ):

        super().__init__(
            environment_factory=environment_factory,
            logger_factory=logger_factory,
            q_network=q_network,
            mixer=mixer,
            batch_size=batch_size,
            min_replay_size=min_replay_size,
            max_replay_size=max_replay_size,  # num episodes in buffer
            sequence_length=sequence_length,
            period=period,
            samples_per_insert=samples_per_insert,
            eps_start=eps_start,
            eps_min=eps_min,
            eps_dec=eps_dec,
            variable_update_period=variable_update_period, 
            max_gradient_norm=max_gradient_norm,
            discount=discount,
            lambda_=lambda_,
            target_update_rate=target_update_rate,
            optimizer=optimizer,
            offline_environment_logging=offline_environment_logging,
            trajectories_per_file=trajectories_per_file,
            add_agent_id_to_obs=add_agent_id_to_obs,
            offline_env_log_dir=offline_env_log_dir,
            record_evaluator_every=record_evaluator_every,
            record_executor_every=record_executor_every,
            evaluation_period=evaluation_period,  # ~ every 100 trainer steps
            evaluation_episodes=evaluation_episodes,
            max_trainer_steps=max_trainer_steps,
            checkpoint_subpath=checkpoint_subpath,
            must_checkpoint=must_checkpoint,
        )

        # BCQ
        self._behaviour_cloning_network = behaviour_cloning_network  # BCQ
        self._threshold = threshold

        # Use BCQ trainer
        self._trainer_fn = QmixBcqTrainer

    def _build_trainer(self, dataset, logger):

        # Initialise networks
        networks = self._initialise_networks()
        q_network = networks["q_network"]
        bc_network = networks["bc_network"]

        trainer = self._trainer_fn(
            self._agents,
            dataset=dataset,
            logger=logger,
            q_network=q_network,
            mixer=self._mixer,
            behaviour_cloning_network=bc_network,
            threshold=self._threshold,
            optimizer=self._optimizer,
            discount=self._discount,
            target_update_rate=self._target_update_rate,
            lambda_=self._lambda,
            max_gradient_norm=self._max_gradient_norm,
            add_agent_id_to_obs=self._add_agent_id_to_obs,
        )

        return trainer

    def _initialise_networks(self):
        q_network = copy.deepcopy(self._q_network)
        bc_network = copy.deepcopy(self._behaviour_cloning_network)

        spec = list(self._environment_spec.get_agent_specs().values())[0]
        dummy_observation = tf.zeros_like(spec.observations.observation)

        if self._add_agent_id_to_obs:
            dummy_observation = concat_agent_id_to_obs(
                dummy_observation, 1, len(self._agents)
            )

        dummy_observation = tf.expand_dims(
            dummy_observation, axis=0
        )  # add dummy batch dim

        # Initialise q-network
        dummy_core_state = q_network.initial_state(1)  # Dummy recurent core state
        q_network(dummy_observation, dummy_core_state)

        # Initialize bc-network
        dummy_core_state = bc_network.initial_state(1)  # Dummy recurent core state
        bc_network(dummy_observation, dummy_core_state)

        return {"q_network": q_network, "bc_network": bc_network}


class QMIXCQLSystemBuilder(QMIXSystemBuilder):
    """Offline QMIX+CQL"""

    def __init__(
        self,
        environment_factory,
        logger_factory,
        q_network,
        mixer,
        num_ood_actions=20,  # CQL
        cql_weight=2.0,  # CQL
        batch_size=64,
        min_replay_size=64,
        max_replay_size=5000,  # num episodes in buffer
        sequence_length=20,
        period=10,
        samples_per_insert=None,
        eps_start=1.0,
        eps_min=0.05,
        eps_dec=1e-5,
        variable_update_period=3,
        max_gradient_norm=20.0,
        discount=0.99,
        lambda_=0.6,
        target_update_rate=None,
        optimizer=snt.optimizers.Adam(learning_rate=1e-4),
        offline_environment_logging=False,
        trajectories_per_file=100,
        add_agent_id_to_obs=False,
        offline_env_log_dir=None,
        record_evaluator_every=None,
        record_executor_every=None,
        evaluation_period=100,
        evaluation_episodes=32,
        max_trainer_steps=1e6,
        checkpoint_subpath="",
        must_checkpoint=False,
    ):

        super().__init__(
            environment_factory=environment_factory,
            logger_factory=logger_factory,
            q_network=q_network,
            mixer=mixer,
            batch_size=batch_size,
            min_replay_size=min_replay_size,
            max_replay_size=max_replay_size,  # num episodes in buffer
            sequence_length=sequence_length,
            period=period,
            samples_per_insert=samples_per_insert,
            eps_start=eps_start,
            eps_min=eps_min,
            eps_dec=eps_dec,
            variable_update_period=variable_update_period, 
            max_gradient_norm=max_gradient_norm,
            discount=discount,
            lambda_=lambda_,
            target_update_rate=target_update_rate,
            optimizer=optimizer,
            offline_environment_logging=offline_environment_logging,
            trajectories_per_file=trajectories_per_file,
            add_agent_id_to_obs=add_agent_id_to_obs,
            offline_env_log_dir=offline_env_log_dir,
            record_evaluator_every=record_evaluator_every,
            record_executor_every=record_executor_every,
            evaluation_period=evaluation_period,  # ~ every 100 trainer steps
            evaluation_episodes=evaluation_episodes,
            max_trainer_steps=max_trainer_steps,
            checkpoint_subpath=checkpoint_subpath,
            must_checkpoint=must_checkpoint,
        )

        # CQL
        self._num_ood_actions = num_ood_actions
        self._cql_weight = cql_weight

        # Use BCQ trainer
        self._trainer_fn = QmixCqlTrainer

    def _build_trainer(self, dataset, logger):

        # Initialise networks
        networks = self._initialise_networks()
        q_network = networks["q_network"]

        trainer = self._trainer_fn(
            self._agents,
            dataset=dataset,
            logger=logger,
            q_network=q_network,
            mixer=self._mixer,
            cql_weight=self._cql_weight,  # CQL
            num_ood_actions=self._num_ood_actions,  # CQL
            optimizer=self._optimizer,
            discount=self._discount,
            target_update_rate=self._target_update_rate,
            lambda_=self._lambda,
            max_gradient_norm=self._max_gradient_norm,
            add_agent_id_to_obs=self._add_agent_id_to_obs,
        )

        return trainer


class QMIX:
    """Example of setting up a QMIX system using the System Builder."""

    def __init__(self, env):

        self.env = env

        # env factory
        env_factory = lambda: env

        # Logger factory
        logger_factory = functools.partial(
            logger_utils.make_logger,
            directory="logs",
            to_terminal=True,
            to_tensorboard=True,
            time_stamp="code_snippet",
            time_delta=1,  # log every 1 sec
        )

        num_actions = env.num_actions
        num_agents = env.num_agents

        q_network=snt.DeepRNN(
            [
                snt.Linear(64),
                tf.nn.relu,
                snt.GRU(64),
                tf.nn.relu,
                snt.Linear(num_actions),
            ]
        )

        mixer=QMixer(
            num_agents=num_agents,
            embed_dim=32,
            hypernet_embed=64,
        )

        self.system = QMIXSystemBuilder(env_factory, logger_factory, q_network, mixer, max_trainer_steps=2000)

    def run_online(self):
        self.system._initialise_networks()
        self.system.run_sequentially(max_executor_episodes=100)
        
    def run_offline(self, dataset):
        dataset_dir = f"datasets/{self.env.environment_label}/Good"
        self.system.run_offline(dataset_dir)
        