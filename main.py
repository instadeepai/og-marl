from absl import app, flags

from utils import WandbLogger, set_growing_gpu_memory, download_and_unzip_vault, download_and_unzip_vault
from replay_buffers import FlashbaxReplayBuffer, PrioritisedFlashbaxReplayBuffer
from maddpg_cql import MADDPGCQLSystem

set_growing_gpu_memory()

FLAGS = flags.FLAGS
flags.DEFINE_string("dataset", "Expert", "Use either 'Good' or 'GoodMedium'")
flags.DEFINE_string("scenario", "3hopper", "Use either '3hopper' or '2halfcheetah'")
flags.DEFINE_string("system", "maddpg+cql+pjap", "Use either 'maddpg+cql' or 'maddpg+cql+pjap'")
flags.DEFINE_float("trainer_steps", 3e5, "Number of training steps.")
flags.DEFINE_float("gaussian_steepness", 4.0, "Parameter to control relationship between distance and priority.")
flags.DEFINE_float("min_priority", 0.0001, "Minimum priority.")
flags.DEFINE_integer("seed", 42, "Seed.")


def main(_):
    config = {
        "dataset": FLAGS.dataset,
        "scenario": FLAGS.scenario
    }

    ########## TODO
    ###### Fix the issue when downloading datasets

    if FLAGS.scenario == "2halfcheetah":
        from mamujoco_wrapper import MAMuJoCo
        env = MAMuJoCo("2halfcheetah")
    elif FLAGS.scenario in ["6halfcheetah", "2ant", "3hopper"]:
        from omiga_mamujoco_wrapper import OmigaMAMuJoCo
        env = OmigaMAMuJoCo(FLAGS.scenario)
    else:
        raise ValueError("Scenario not recognised.")

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

    download_and_unzip_vault(FLAGS.scenario, "./vaults")

    is_vault_loaded = buffer.populate_from_vault(
        FLAGS.scenario,
        FLAGS.dataset,
    )
    if not is_vault_loaded:
        print("Vault not found. Exiting.")
        return

    logger = WandbLogger(
        entity="claude_formanek", project="pjap", config=config
    )

    system_kwargs = {
        "add_agent_id_to_obs": True, 
        "gaussian_steepness": FLAGS.gaussian_steepness,
        "min_priority": FLAGS.min_priority,
        "is_omiga": FLAGS.scenario in ["3hopper", "2ant", "6halfcheetah"]
    }

    system = MADDPGCQLSystem(env, logger, **system_kwargs)

    system.train_offline(
        buffer, max_trainer_steps=FLAGS.trainer_steps, evaluate_every=5000
    )


if __name__ == "__main__":
    app.run(main)