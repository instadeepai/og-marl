"""Example running offline Systems on SMAC"""
from mava.utils.training_utils import set_growing_gpu_memory
set_growing_gpu_memory()

import functools
from datetime import datetime
from absl import app, flags
import tensorflow as tf
import sonnet as snt 
from mava.utils.loggers import logger_utils

from og_marl.systems.maicq import MAICQSystemBuilder, IdentityNetwork, LocalObservationStateCriticNetwork
from og_marl.systems.bc import BCSystemBuilder 
from og_marl.systems.qmix import QMIXSystemBuilder, QMIXCQLSystemBuilder, QMIXBCQSystemBuilder, QMixer
from og_marl.environments.smac import SMAC
from og_marl.utils.loggers import WandbLogger


"""This script can be used to re-produce the results reported in the OG-MARL paper.

To run the script make sure you follow the OG-MARL instalation instructions for
SMAC in the README. After that you can run the script as follows:

`python examples/baselines/benchmark_smac.py --algo_name=qmix --dataset_quality=Good --env_name=3m`

    --algo_name can be used to change the algorithm you want to run (bc, itd3, itd3+cql, itd3+bc, omar)
    --dataset_quality is used to change wich dataset type to run (Good, Medium and Poor)
    --env_name is used to change the scenario (2ant, 4ant, 2halfcheetah)

You will need to make sure you download the datasets from the OG-MARL website.
https://sites.google.com/view/og-marl

Make sure the unzip the dataset and add it to the path 
`datasets/smac/<env_name>/<dataset_quality>/`
"""

FLAGS = flags.FLAGS
flags.DEFINE_string("id", str(datetime.now()), "tensorboard, wandb")
flags.DEFINE_string("logger", "tensorboard", "tensorboard or wandb")
flags.DEFINE_string("base_log_dir", "logs", "Base dir to store experiment logs.")
flags.DEFINE_string("base_dataset_dir", "datasets/smac_v1", "Directory with tfrecord files.")
flags.DEFINE_string("env_name", "3m", "E.g. 3m, 8m")
flags.DEFINE_string("dataset_quality", "Medium", "E.g. Good, Medium, Poor")
flags.DEFINE_string("algo_name", "qmix", "E.g. qmix, qmix+cql, qmix+bcq, bc")
flags.DEFINE_string("max_trainer_steps", "50_001", "Max number of trainer steps.")
flags.DEFINE_string("seed", "0", "Random Seed.")

### SYSTEM BUILD FUNCTIONS ###

def build_bc_system(num_actions, environment_factory, logger_factory):
    system = BCSystemBuilder(
        environment_factory=environment_factory,
        logger_factory=logger_factory,
        behaviour_cloning_network=snt.DeepRNN(
            [
                snt.Linear(64),
                tf.nn.relu,
                snt.GRU(64),
                tf.nn.relu,
                snt.Linear(num_actions),
                tf.nn.softmax,
            ]
        ),
        discrete_actions=True,
        optimizer=snt.optimizers.Adam(1e-3),
        batch_size=32,
        add_agent_id_to_obs=True,
        max_trainer_steps=int(FLAGS.max_trainer_steps),
        must_checkpoint=True,
        checkpoint_subpath=f"{FLAGS.base_log_dir}/{FLAGS.id}",
        evaluation_episodes=10,
        evaluation_period=1000
    )
    return system

def build_qmix_system(num_agents, num_actions, environment_factory, logger_factory):
    system = QMIXSystemBuilder(
        environment_factory=environment_factory,
        logger_factory=logger_factory,
        q_network=snt.DeepRNN(
            [
                snt.Linear(64),
                tf.nn.relu,
                snt.GRU(64),
                tf.nn.relu,
                snt.Linear(num_actions),
            ]
        ),
        mixer=QMixer(
            num_agents=num_agents,
            embed_dim=32,
            hypernet_embed=64,
        ),
        optimizer=snt.optimizers.Adam(1e-3),
        target_update_rate=0.01,
        batch_size=32,
        add_agent_id_to_obs=True,
        max_trainer_steps=int(FLAGS.max_trainer_steps),
        must_checkpoint=True,
        checkpoint_subpath=f"{FLAGS.base_log_dir}/{FLAGS.id}",
        evaluation_episodes=10,
        evaluation_period=1000
    )
    return system

def build_maicq_system(num_agents, num_actions, environment_factory, logger_factory):
    system = MAICQSystemBuilder(
        environment_factory=environment_factory,
        logger_factory=logger_factory,
        policy_network=snt.DeepRNN(
            [
                snt.Linear(64),
                tf.keras.layers.ReLU(),
                snt.GRU(64),
                tf.keras.layers.ReLU(),
                snt.Linear(num_actions),
            ]
        ),
        critic_network=LocalObservationStateCriticNetwork(
            local_observation_network=IdentityNetwork(),
            state_network=IdentityNetwork(),
            output_network=snt.Sequential(
                [
                    snt.Linear(64),
                    tf.keras.layers.ReLU(),
                    snt.Linear(64),
                    tf.keras.layers.ReLU(),
                    snt.Linear(num_actions),
                ]
            )
        ),
        critic_optimizer=snt.optimizers.Adam(1e-3),
        policy_optimizer=snt.optimizers.Adam(1e-3),
        mixer=QMixer(
            num_agents=num_agents,
            embed_dim=32,
            hypernet_embed=64,
        ),
        batch_size=32,
        target_update_period=600, # from maicq paper
        lambda_=0.6, # Q-lambda
        max_gradient_norm=20.0,
        add_agent_id_to_obs=True,
        evaluation_episodes=10,
        evaluation_period=1000
    )
    return system

def build_qmix_bcq_system(num_agents, num_actions, environment_factory, logger_factory):
    system = QMIXBCQSystemBuilder(
        environment_factory=environment_factory,
        logger_factory=logger_factory,
        q_network=snt.DeepRNN(
            [
                snt.Linear(64),
                tf.nn.relu,
                snt.GRU(64),
                tf.nn.relu,
                snt.Linear(num_actions),
            ]
        ),
        mixer=QMixer(
            num_agents=num_agents,
            embed_dim=32,
            hypernet_embed=64,
        ),
        behaviour_cloning_network=snt.DeepRNN(
            [
                snt.Linear(64),
                tf.nn.relu,
                snt.GRU(64),
                tf.nn.relu,
                snt.Linear(num_actions),
                tf.nn.softmax,
            ]
        ),
        optimizer=snt.optimizers.Adam(1e-3),
        target_update_rate=0.01,
        threshold=0.4, # BC threshhold
        batch_size=32,
        add_agent_id_to_obs=True,
        max_trainer_steps=int(FLAGS.max_trainer_steps),
        must_checkpoint=True,
        checkpoint_subpath=f"{FLAGS.base_log_dir}/{FLAGS.id}",
        evaluation_episodes=10,
        evaluation_period=1000
    )
    return system

def build_qmix_cql_system(num_agents, num_actions, environment_factory, logger_factory):
    system = QMIXCQLSystemBuilder(
        environment_factory=environment_factory,
        logger_factory=logger_factory,
        q_network=snt.DeepRNN(
            [
                snt.Linear(64),
                tf.nn.relu,
                snt.GRU(64),
                tf.nn.relu,
                snt.Linear(num_actions),
            ]
        ),
        mixer=QMixer(
            num_agents=num_agents,
            embed_dim=32,
            hypernet_embed=64,
        ),
        optimizer=snt.optimizers.Adam(1e-3),
        target_update_rate=0.01,
        cql_weight=2.0,
        num_ood_actions=20,
        batch_size=32,
        add_agent_id_to_obs=True,
        max_trainer_steps=int(FLAGS.max_trainer_steps),
        must_checkpoint=True,
        checkpoint_subpath=f"{FLAGS.base_log_dir}/{FLAGS.id}",
        evaluation_episodes=10,
        evaluation_period=1000
    )
    return system

### MAIN ###
def main(_):

    # Logger factory
    logger_factory = functools.partial(
        logger_utils.make_logger,
        directory=FLAGS.base_log_dir,
        to_terminal=True,
        to_tensorboard=FLAGS.logger == "tensorboard",
        time_stamp=FLAGS.id,
        time_delta=1,  # log every 1 sec
        external_logger = WandbLogger if FLAGS.logger == "wandb" else None,
    )

    # Environment factory
    environment_factory = functools.partial(SMAC, FLAGS.env_name)

    # Get info from environment
    tmp_env = environment_factory()
    num_agents = tmp_env.num_agents
    num_actions = tmp_env.num_actions
    tmp_env.close()
    del tmp_env

    # Instantiate offline system
    if FLAGS.algo_name == "bc":
        print("RUNNING BC")
        system = build_bc_system(num_actions, environment_factory, logger_factory)

    elif FLAGS.algo_name == "maicq":
        print("RUNNING MAICQ")
        system = build_maicq_system(num_agents, num_actions, environment_factory, logger_factory)

    elif FLAGS.algo_name == "qmix":
        print("RUNNING QMIX")
        system = build_qmix_system(num_agents, num_actions, environment_factory, logger_factory)

    elif FLAGS.algo_name == "qmix+bcq":
        print("RUNNING QMIX+BCQ")
        system = build_qmix_bcq_system(num_agents, num_actions, environment_factory, logger_factory)

    elif FLAGS.algo_name == "qmix+cql":
        print("RUNNING QMIX+CQL")
        system = build_qmix_cql_system(num_agents, num_actions, environment_factory, logger_factory)

    else:
        raise ValueError("Unrecognised algorithm.")

    # Run system
    system.run_offline(
        f"{FLAGS.base_dataset_dir}/{FLAGS.env_name}/{FLAGS.dataset_quality}",
        shuffle_buffer_size=5000
    )

if __name__ == "__main__":
    app.run(main)