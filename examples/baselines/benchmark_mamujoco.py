from mava.utils.training_utils import set_growing_gpu_memory

from og_marl.systems.td3.critics import StateAndActionCritic, StateAndJointActionCritic
from og_marl.systems.td3.trainer import OMARTrainer, TD3BCTrainer, TD3CQLTrainer, TD3Trainer
set_growing_gpu_memory()

import tensorflow as tf
import functools, os
from datetime import datetime
from absl import app, flags
import sonnet as snt
from mava.utils.loggers import logger_utils

from og_marl.utils.loggers import WandbLogger
from og_marl.systems.bc.system_builder import BCSystemBuilder
from og_marl.systems.td3.system_builder import TD3SystemBuilder
from og_marl.environments.mamujoco import Mujoco

"""This script can be used to re-produce the results reported in the OG-MARL paper.

To run the script make sure you follow the OG-MARL instalation instructions for
MAMuJoCo in the README. After that you can run the script as follows:

`python examples/baselines/benchmark_mamujoco.py --algo_name=itd3 --dataset_quality=Good --env_name=2halfcheetah`

    --algo_name can be used to change the algorithm you want to run (bc, itd3, itd3+cql, itd3+bc, omar)
    --dataset_quality is used to change wich dataset type to run (Good, Medium and Poor)
    --env_name is used to change the scenario (2ant, 4ant, 2halfcheetah)

You will need to make sure you download the datasets you want to use from the OG-MARL website.
https://sites.google.com/view/og-marl

Make sure the unzip the dataset and add it to the path 
`datasets/mamujoco/<env_name>/<dataset_quality>/`
"""

FLAGS = flags.FLAGS
flags.DEFINE_string("id", str(datetime.now()), "tensorboard, wandb")
flags.DEFINE_string("logger", "tensorboard", "tensorboard, neptune or wandb")
flags.DEFINE_string("base_log_dir", "logs", "Base dir to store experiments.")
flags.DEFINE_string("base_dataset_dir", "datasets/mamujoco", "Directory with tfrecord files.")
flags.DEFINE_string("dataset_quality", "Tiny", "E.g. Good, Medium, Poor")
flags.DEFINE_string("env_name", "2halfcheetah", "E.g. 2ant, 4ant, 2halfcheetah")
flags.DEFINE_string("algo_name", "bc", "itd3cql, itd3bc, itd3cql, itd3")
flags.DEFINE_integer("max_trainer_steps", "1_000_001", "Max number of trainer steps")
flags.DEFINE_string("seed", "0", "Random Seed.")
 
### SYSTEM BUILD FUNCTIONS ###


def build_bc_system(num_actions, environment_factory, logger_factory):
    system = BCSystemBuilder(
        environment_factory=environment_factory,
        logger_factory=logger_factory,
        discrete_actions=False,
        behaviour_cloning_network=snt.DeepRNN(
            [
                snt.Linear(128),
                tf.nn.relu,
                snt.GRU(128),
                tf.nn.relu,
                snt.Linear(num_actions),
                tf.keras.activations.tanh,
            ]
        ),
        evaluation_period=5000,
        evaluation_episodes=10,
        max_trainer_steps=FLAGS.max_trainer_steps,
        optimizer=snt.optimizers.Adam(5e-4),
        batch_size=32,
        max_gradient_norm=10.0,
        variable_update_period=1,
        add_agent_id_to_obs=True,
        must_checkpoint=True,
        checkpoint_subpath=f"{FLAGS.base_log_dir}/{FLAGS.id}",
    )

    return system

def build_matd3bc_system(num_agents, num_actions, environment_factory, logger_factory):
    policy_network = snt.DeepRNN(
        [
            snt.Linear(128),
            tf.keras.layers.ReLU(),
            snt.GRU(128),
            tf.keras.layers.ReLU(),
            snt.Linear(num_actions),
            tf.keras.activations.tanh,
        ]
    )

    critic_network = StateAndJointActionCritic(num_agents, num_actions)

    # Distributed program
    system = TD3SystemBuilder(
        environment_factory=environment_factory,
        logger_factory=logger_factory,
        policy_network=policy_network,
        critic_network=critic_network,
        trainer_fn=TD3BCTrainer,
        policy_optimizer=snt.optimizers.Adam(5e-4),
        critic_optimizer=snt.optimizers.Adam(5e-4),
        batch_size=32,
        trainer_sigma=0.01,
        target_update_rate=0.01,
        max_gradient_norm=10.0,
        add_agent_id_to_obs=True,
        evaluation_period=5000,
        evaluation_episodes=10,
        max_trainer_steps=FLAGS.max_trainer_steps,
    )

    return system

def build_itd3bc_system(num_agents, num_actions, environment_factory, logger_factory):
    policy_network = snt.DeepRNN(
        [
            snt.Linear(128),
            tf.keras.layers.ReLU(),
            snt.GRU(128),
            tf.keras.layers.ReLU(),
            snt.Linear(num_actions),
            tf.keras.activations.tanh,
        ]
    )

    critic_network = StateAndJointActionCritic(num_agents, num_actions)

    # Distributed program
    system = TD3SystemBuilder(
        environment_factory=environment_factory,
        logger_factory=logger_factory,
        policy_network=policy_network,
        critic_network=critic_network,
        trainer_fn=TD3BCTrainer,
        policy_optimizer=snt.optimizers.Adam(5e-4),
        critic_optimizer=snt.optimizers.Adam(5e-4),
        batch_size=32,
        trainer_sigma=0.01,
        target_update_rate=0.01,
        max_gradient_norm=10.0,
        add_agent_id_to_obs=True,
        evaluation_period=5000,
        evaluation_episodes=10,
        max_trainer_steps=FLAGS.max_trainer_steps,
    )

    return system


def build_matd3_system(num_agents, num_actions, environment_factory, logger_factory):

    policy_network = snt.DeepRNN(
        [
            snt.Linear(128),
            tf.keras.layers.ReLU(),
            snt.GRU(128),
            tf.keras.layers.ReLU(),
            snt.Linear(num_actions),
            tf.keras.activations.tanh,
        ]
    )

    critic_network = StateAndActionCritic(num_agents, num_actions)

    # Distributed program
    system = TD3SystemBuilder(
        environment_factory=environment_factory,
        logger_factory=logger_factory,
        policy_network=policy_network,
        critic_network=critic_network,
        trainer_fn=TD3Trainer,
        policy_optimizer=snt.optimizers.Adam(5e-4),
        critic_optimizer=snt.optimizers.Adam(5e-4),
        batch_size=32,
        trainer_sigma=0.01,
        target_update_rate=0.01,
        max_gradient_norm=10.0,
        add_agent_id_to_obs=True,  # important in MAMUJOCO
        evaluation_period=5000,
        evaluation_episodes=10,
        max_trainer_steps=FLAGS.max_trainer_steps,
    )

    return system

def build_itd3_system(num_agents, num_actions, environment_factory, logger_factory):

    policy_network = snt.DeepRNN(
        [
            snt.Linear(200),
            tf.keras.layers.ReLU(),
            snt.GRU(100),
            tf.keras.layers.ReLU(),
            snt.Linear(num_actions),
            tf.keras.activations.tanh,
        ]
    )

    critic_network = StateAndActionCritic(num_agents, num_actions)

    # Distributed program
    system = TD3SystemBuilder(
        environment_factory=environment_factory,
        logger_factory=logger_factory,
        policy_network=policy_network,
        critic_network=critic_network,
        trainer_fn=TD3Trainer,
        policy_optimizer=snt.optimizers.Adam(5e-4),
        critic_optimizer=snt.optimizers.Adam(5e-4),
        batch_size=32,
        trainer_sigma=0.01,
        target_update_rate=0.01,
        max_gradient_norm=10.0,
        add_agent_id_to_obs=True,
        evaluation_period=5000,
        evaluation_episodes=10,
        max_trainer_steps=FLAGS.max_trainer_steps,
    )

    return system

def build_matd3cql_system(num_agents, num_actions, environment_factory, logger_factory):

    policy_network = snt.DeepRNN(
        [
            snt.Linear(200),
            tf.keras.layers.ReLU(),
            snt.GRU(100),
            tf.keras.layers.ReLU(),
            snt.Linear(num_actions),
            tf.keras.activations.tanh,
        ]
    )

    critic_network = StateAndJointActionCritic(num_agents, num_actions)

    # Distributed program
    system = TD3SystemBuilder(
        environment_factory=environment_factory,
        logger_factory=logger_factory,
        policy_network=policy_network,
        critic_network=critic_network,
        trainer_fn=TD3CQLTrainer,
        policy_optimizer=snt.optimizers.Adam(5e-4),
        critic_optimizer=snt.optimizers.Adam(5e-4),
        batch_size=32,
        trainer_sigma=0.01,
        target_update_rate=0.01,
        max_gradient_norm=10.0,
        add_agent_id_to_obs=True,  # important in MAMUJOCO
        evaluation_period=5000,
        evaluation_episodes=10,
        max_trainer_steps=FLAGS.max_trainer_steps,
    )

    return system

def build_itd3cql_system(num_agents, num_actions, environment_factory, logger_factory):

    policy_network = snt.DeepRNN(
        [
            snt.Linear(200),
            tf.keras.layers.ReLU(),
            snt.GRU(100),
            tf.keras.layers.ReLU(),
            snt.Linear(num_actions),
            tf.keras.activations.tanh,
        ]
    )

    critic_network = StateAndActionCritic(num_agents, num_actions)

    # Distributed program
    system = TD3SystemBuilder(
        environment_factory=environment_factory,
        logger_factory=logger_factory,
        policy_network=policy_network,
        critic_network=critic_network,
        trainer_fn=TD3CQLTrainer,
        policy_optimizer=snt.optimizers.Adam(5e-4),
        critic_optimizer=snt.optimizers.Adam(5e-4),
        batch_size=32,
        trainer_sigma=0.01,
        target_update_rate=0.01,
        max_gradient_norm=10.0,
        add_agent_id_to_obs=True,
        evaluation_period=5000,
        evaluation_episodes=10,
        max_trainer_steps=FLAGS.max_trainer_steps,
    )

    return system

def build_omar_system(num_agents, num_actions, environment_factory, logger_factory):

    policy_network = snt.DeepRNN(
        [
            snt.Linear(200),
            tf.keras.layers.ReLU(),
            snt.GRU(100),
            tf.keras.layers.ReLU(),
            snt.Linear(num_actions),
            tf.keras.activations.tanh,
        ]
    )

    critic_network = StateAndActionCritic(num_agents, num_actions)

    # Distributed program
    system = TD3SystemBuilder(
        environment_factory=environment_factory,
        logger_factory=logger_factory,
        policy_network=policy_network,
        critic_network=critic_network,
        trainer_fn=OMARTrainer,
        policy_optimizer=snt.optimizers.Adam(5e-4),
        critic_optimizer=snt.optimizers.Adam(5e-4),
        batch_size=32,
        trainer_sigma=0.01,
        target_update_rate=0.01,
        max_gradient_norm=10.0,
        add_agent_id_to_obs=True,
        evaluation_period=5000,
        evaluation_episodes=10,
        max_trainer_steps=FLAGS.max_trainer_steps,
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
        time_stamp=str(datetime.now()),
        time_delta=1,  # log every 1 sec
        external_logger= WandbLogger if FLAGS.logger == "wandb" else None,
    )

    # Environment factory
    if FLAGS.env_name in ["4ant", "2ant", "2halfcheetah"]:  # is a mamujoco scenario

        environment_factory = functools.partial(Mujoco, FLAGS.env_name)

        # Get info from environment
        tmp_env = environment_factory()
        num_agents = tmp_env.num_agents
        num_actions = tmp_env.num_actions
        tmp_env.close()
        del tmp_env
    else:
        raise ValueError("Unrecognised environment.")

    # Offline system
    if FLAGS.algo_name == "bc":
        print("RUNNING Bc")
        system = build_bc_system(
            num_actions, environment_factory, logger_factory
        )
    elif FLAGS.algo_name == "matd3":
        print("RUNNING MaTd3")
        system = build_matd3_system(
            num_agents, num_actions, environment_factory, logger_factory
        )
    elif FLAGS.algo_name == "matd3bc":
        print("RUNNING MaTd3Bc")
        system = build_matd3bc_system(
            num_agents, num_actions, environment_factory, logger_factory
        )
    elif FLAGS.algo_name == "matd3cql":
        print("RUNNING MaTd3Cql")
        system = build_matd3cql_system(
            num_agents, num_actions, environment_factory, logger_factory
        )
    elif FLAGS.algo_name == "itd3bc":
        print("RUNNING ITd3Bc")
        system = build_itd3bc_system(
            num_agents, num_actions, environment_factory, logger_factory
        )
    elif FLAGS.algo_name == "itd3cql":
        print("RUNNING ITd3Cql")
        system = build_itd3cql_system(
            num_agents, num_actions, environment_factory, logger_factory
        )
    elif FLAGS.algo_name == "itd3":
        print("RUNNING iTd3")
        system = build_itd3_system(
            num_agents, num_actions, environment_factory, logger_factory
        )
    elif FLAGS.algo_name == "omar":
        print("RUNNING OMAR")
        system = build_omar_system(
            num_agents, num_actions, environment_factory, logger_factory
        )
    else:
        raise ValueError("Unrecognised algorithm.")

    # Run System
    system.run_offline(
        dataset_dir=f"{FLAGS.base_dataset_dir}/{FLAGS.env_name}/{FLAGS.dataset_quality}",
        shuffle_buffer_size=5000
    )

if __name__ == "__main__":
    app.run(main)
