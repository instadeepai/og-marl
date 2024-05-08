from og_marl.environments.smacv1 import SMACv1
from og_marl.loggers import WandbLogger
from og_marl.replay_buffers import FlashbaxReplayBuffer
from og_marl.tf2.systems.qmix import QMIXSystem
from og_marl.environments.wrappers import ExperienceRecorder

env = ExperienceRecorder(SMACv1("2s3z"),"qmix_2s3z")

logger = WandbLogger()

# TODO
system = QMIXSystem(env, logger, add_agent_id_to_obs=False)  # noqa: F821

rb = FlashbaxReplayBuffer(sequence_length=20, batch_size=128)

system.train_online(rb)
