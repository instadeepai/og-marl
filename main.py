from absl import app, flags

from environment_wrappers import get_environment
from utils.loggers import WandbLogger
from utils.offline_dataset import download_and_unzip_vault
from utils.replay_buffers import FlashbaxReplayBuffer
from systems import get_system
from utils.utils import set_growing_gpu_memory

set_growing_gpu_memory()

FLAGS = flags.FLAGS
flags.DEFINE_string("env", "mamujoco_omar", "Environment name.")
flags.DEFINE_string("scenario", "2halfcheetah", "Environment scenario name.")
flags.DEFINE_string("dataset", "expert", "Dataset type.")
flags.DEFINE_string("system", "iddpg+bc", "System name.")
flags.DEFINE_integer("seed", 42, "Seed.")
flags.DEFINE_float("trainer_steps", 5e5, "Number of training steps.")


def main(_):
    config = {
        "env": FLAGS.env,
        "scenaro": FLAGS.scenario,
        "dataset": FLAGS.dataset,
        "system": FLAGS.system,
        "seed": FLAGS.seed
    }

    # Logger
    logger = WandbLogger(project="offline-marl-baselines", config=config)

    # Make environment for evaluation
    env = get_environment(FLAGS.env, FLAGS.scenario)

    # Replay buffer that samples sequences of 20 timesteps
    buffer = FlashbaxReplayBuffer(sequence_length=20, sample_period=1, seed=FLAGS.seed)

    # Download dataset, if not present
    download_and_unzip_vault(FLAGS.env, FLAGS.scenario)

    # Load offline data into replay buffer
    buffer.populate_from_vault(FLAGS.env, FLAGS.scenario, FLAGS.dataset)

    system_kwargs = {
        "add_agent_id_to_obs_in_trainer": not (FLAGS.env.split("_")[-1] == "omiga" and FLAGS.env.split("_")[0] == "mamujoco"),
        "add_agent_id_to_obs_in_action_selection": True,
    }

    if FLAGS.env.split("_")[-1] == "omiga" and FLAGS.env.split("_")[0] == "mamujoco":
        system_kwargs["is_omiga"] = True

    print(system_kwargs)

    system = get_system(FLAGS.system, env, logger, **system_kwargs)

    system.train_offline(
        buffer,
        max_trainer_steps=FLAGS.trainer_steps,
        evaluate_every=5000,
        num_eval_episodes=32,
    )


if __name__ == "__main__":
    app.run(main)