from absl import app, flags

from mamujoco_wrapper import MAMuJoCo
from utils import WandbLogger, set_growing_gpu_memory, download_and_unzip_vault, download_and_unzip_vault
from replay_buffers import FlashbaxReplayBuffer, PrioritisedFlashbaxReplayBuffer
from maddpg_cql import MADDPGCQLSystem

set_growing_gpu_memory()

FLAGS = flags.FLAGS
flags.DEFINE_string("dataset", "Good", "Use either 'Good' or 'GoodMedium'")
flags.DEFINE_string("system", "maddpg+cql", "Use either 'maddpg+cql' or 'maddpg+cql+pjap'")
flags.DEFINE_float("trainer_steps", 3e5, "Number of training steps.")
flags.DEFINE_float("gaussian_steepness", 4, "Parameter to control relationship between distance and priority.")
flags.DEFINE_float("min_priority", 0.001, "Minimum priority.")
flags.DEFINE_integer("seed", 42, "Seed.")


def main(_):
    env = MAMuJoCo("2halfcheetah")

    if FLAGS.system == "maddpg+cql+pjap":
        buffer = PrioritisedFlashbaxReplayBuffer(
            batch_size=64,
            sequence_length=20,
            sample_period=10,
            seed=FLAGS.seed,
            priority_exponent=0.99,
        )
    else:
        buffer = FlashbaxReplayBuffer(
            sequence_length=20, sample_period=10, batch_size=64, seed=FLAGS.seed
        )

    download_and_unzip_vault("./vaults")

    is_vault_loaded = buffer.populate_from_vault(
        FLAGS.dataset,
    )
    if not is_vault_loaded:
        print("Vault not found. Exiting.")
        return

    logger = WandbLogger(
        entity="Username", project="pjap"
    )

    system_kwargs = {
        "add_agent_id_to_obs": True, 
        "gaussian_steepness": FLAGS.gaussian_steepness,
        "min_priority": FLAGS.min_priority
    }

    system = MADDPGCQLSystem(env, logger, **system_kwargs)

    system.train_offline(
        buffer, max_trainer_steps=FLAGS.trainer_steps, evaluate_every=5000
    )


if __name__ == "__main__":
    app.run(main)
