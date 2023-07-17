import functools
from absl import app, flags
import tensorflow as tf
import sonnet as snt
import datetime
from mava.utils.loggers import logger_utils

from og_marl.systems.qmix import QMixer, QMIXSystemBuilder, QMIXBCQSystemBuilder, QMIXCQLSystemBuilder
from og_marl.systems.maicq import IdentityNetwork, LocalObservationStateCriticNetwork, MAICQSystemBuilder
from og_marl.systems.bc import BCSystemBuilder

from double_cartpole import DoubleCartPole

"""PART 6: This is part 6 of the tutorial on how to use OG-MARL. Make 
sure you did parts 1-5 in `example/quickstart/generate_dataset.py` before 
doing this.

Run this script on your new dataset by typing 

`python example/quickstart/train_offline_algos.py --algo_name=qmix+bcq`

You can set --algo_name to any one of bc, qmix, qmix+bcq, qmix+cql and maicq.
"""

FLAGS = flags.FLAGS
flags.DEFINE_string("base_log_dir", "logs", "Base dir. to store experiments.")
flags.DEFINE_string("algo_name", "qmix+bcq", "qmix, qmix+cql, qmix+bcq, bc")
flags.DEFINE_string("max_trainer_steps", "10_001", "Max number of trainer steps.")

### SYSTEM BUILD FUNCTIONS ###

def build_mabc_system(num_agents, num_actions, environment_factory, logger_factory):
    system = BCSystemBuilder(
        environment_factory=environment_factory,
        logger_factory=logger_factory,
        behaviour_cloning_network=snt.DeepRNN(
            [
                snt.Linear(32),
                tf.nn.relu,
                snt.GRU(32),
                tf.nn.relu,
                snt.Linear(num_actions),
                tf.nn.softmax,
            ]
        ),
        optimizer=snt.optimizers.Adam(1e-3),
        batch_size=32,
        add_agent_id_to_obs=True,
    )
    return system

def build_qmix_system(num_agents, num_actions, environment_factory, logger_factory):
    system = QMIXSystemBuilder(
        environment_factory=environment_factory,
        logger_factory=logger_factory,
        q_network=snt.DeepRNN(
            [
                snt.Linear(32),
                tf.nn.relu,
                snt.GRU(32),
                tf.nn.relu,
                snt.Linear(num_actions)
            ]
        ),
        mixer=QMixer(
            num_agents=num_agents,
            embed_dim=32,
            hypernet_embed=32,
        ),
        optimizer=snt.optimizers.Adam(1e-3),
        target_update_rate=0.01,
        batch_size=32,
        add_agent_id_to_obs=True,
    )
    return system

def build_maicq_system(num_agents, num_actions, environment_factory, logger_factory):
    system = MAICQSystemBuilder(
        environment_factory=environment_factory,
        logger_factory=logger_factory,
        policy_network=snt.DeepRNN(
            [
                snt.Linear(32),
                tf.nn.relu,
                snt.GRU(32),
                tf.nn.relu,
                snt.Linear(num_actions) 
            ]
        ),
        critic_network=LocalObservationStateCriticNetwork(
            local_observation_network=IdentityNetwork(),
            state_network=IdentityNetwork(),
            output_network=snt.Sequential(
                [
                    snt.Linear(32),
                    tf.keras.layers.ReLU(),
                    snt.Linear(32),
                    tf.keras.layers.ReLU(),
                    snt.Linear(num_actions),
                ]
            )
        ),
        critic_optimizer=snt.optimizers.Adam(1e-4),
        policy_optimizer=snt.optimizers.Adam(1e-4),
        mixer=QMixer(
            num_agents=num_agents,
            embed_dim=32,
            hypernet_embed=32,
        ),
        batch_size=32,
        target_update_period=600,
        lambda_=0.6,
        max_gradient_norm=10.0,
        add_agent_id_to_obs=True,
    )
    return system

def build_bcq_system(num_agents, num_actions, environment_factory, logger_factory):
    system = QMIXBCQSystemBuilder(
        environment_factory=environment_factory,
        logger_factory=logger_factory,
        q_network=snt.DeepRNN(
            [
                snt.Linear(32),
                tf.nn.relu,
                snt.GRU(32),
                tf.nn.relu,
                snt.Linear(num_actions)
            ]
        ),
        mixer=QMixer(
            num_agents=num_agents,
            embed_dim=32,
            hypernet_embed=32,
        ),
        behaviour_cloning_network=snt.DeepRNN(
            [
                snt.Linear(32),
                tf.nn.relu,
                snt.GRU(32),
                tf.nn.relu,
                snt.Linear(num_actions),
                tf.nn.softmax,
            ]
        ),
        optimizer=snt.optimizers.Adam(1e-3),
        target_update_rate=0.01,
        threshold=0.4,
        batch_size=32,
        add_agent_id_to_obs=True,
    )
    return system

def build_cql_system(num_agents, num_actions, environment_factory, logger_factory):
    system = QMIXCQLSystemBuilder(
        environment_factory=environment_factory,
        logger_factory=logger_factory,
        q_network=snt.DeepRNN(
            [
                snt.Linear(32),
                tf.nn.relu,
                snt.GRU(32),
                tf.nn.relu,
                snt.Linear(num_actions) # five actions
            ]
        ),
        mixer=QMixer(
            num_agents=num_agents,
            embed_dim=32,
            hypernet_embed=32,
        ),
        optimizer=snt.optimizers.Adam(1e-3),
        target_update_rate=0.01,
        cql_weight=2.0,
        num_ood_actions=20,
        batch_size=32,
        add_agent_id_to_obs=True,
    )
    return system

### MAIN ###
def main(_):

    # Logger factory
    logger_factory = functools.partial(
        logger_utils.make_logger,
        directory=FLAGS.base_log_dir,
        to_terminal=True,
        to_tensorboard=True,
        time_stamp=str(datetime.datetime.now()),
        time_delta=1,  # log every 1 sec
    )

    # Environment factory
    environment_factory = functools.partial(DoubleCartPole)

    env = environment_factory()
    num_agents = len(env.agents)
    num_actions = env.num_actions
    env.close()
    del env


    # Offline system
    if FLAGS.algo_name == "bc":
        print("RUNNING MABC")
        system = build_mabc_system(num_agents, num_actions, environment_factory, logger_factory)
    elif FLAGS.algo_name == "maicq":
        print("RUNNING MAICQ")
        system = build_maicq_system(num_agents, num_actions, environment_factory, logger_factory)
    elif FLAGS.algo_name == "qmix":
        print("RUNNING QMIX")
        system = build_qmix_system(num_agents, num_actions, environment_factory, logger_factory)
    elif FLAGS.algo_name == "qmix+bcq":
        print("RUNNING QMIX+BCQ")
        system = build_bcq_system(num_agents, num_actions, environment_factory, logger_factory)
    elif FLAGS.algo_name == "qmix+cql":
        print("RUNNING QMIX+CQL")
        system = build_cql_system(num_agents, num_actions, environment_factory, logger_factory)
    else:
        raise ValueError("Unrecognised algorithm.")

    # Run System
    system.run_offline(
        f"./datasets/double_cartpole/",
        shuffle_buffer_size=1000
    )

if __name__ == "__main__":
    app.run(main)