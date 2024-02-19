from og_marl.environments.smacv1 import SMACv1
from og_marl.loggers import WandbLogger
from og_marl.replay_buffers import FlashbaxReplayBuffer
from og_marl.tf2.systems.qmix import QMIXSystem

env = SMACv1("3m")

logger = WandbLogger(entity="claude_formanek")

system = QMIXSystsem(env, logger, add_agent_id_to_obs=False)

rb = FlashbaxReplayBuffer(sequence_length=20, batch_size=128)

system.train_online(rb)
