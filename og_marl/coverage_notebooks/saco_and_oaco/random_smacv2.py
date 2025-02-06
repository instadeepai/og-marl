from og_marl.environments.smacv2 import SMACv2
from og_marl.replay_buffers import FlashbaxReplayBuffer
from og_marl.tf2.systems.qmix import QMIXSystem
from og_marl.environments.wrappers import ExperienceRecorder

from absl import app, flags

from og_marl.loggers import WandbLogger
from og_marl.tf2.networks import CNNEmbeddingNetwork
from og_marl.tf2.systems import get_system
from og_marl.tf2.utils import set_growing_gpu_memory


set_growing_gpu_memory()

FLAGS = flags.FLAGS
flags.DEFINE_string("scenario", "terran_5_vs_5", "Environment scenario name.")
flags.DEFINE_string("system", "idrqn", "System name.")
flags.DEFINE_integer("seed", 50, "Seed.")


def main(_):
    config = {
        "scenario": FLAGS.scenario,
        "system": FLAGS.system,
        "backend": "tf2",
    }
    

    env = ExperienceRecorder(SMACv2(FLAGS.scenario),f"{FLAGS.system}_{FLAGS.scenario}_{FLAGS.seed}")

    logger = WandbLogger(project="og-marl-baselines", config=config)

    system_kwargs = {"add_agent_id_to_obs": True}
    system = get_system(FLAGS.system, env, logger, **system_kwargs)

    rb = FlashbaxReplayBuffer(sequence_length=20, batch_size=64)

    system.train_online(rb, max_env_steps=int(3e6))


if __name__ == "__main__":
    app.run(main)
