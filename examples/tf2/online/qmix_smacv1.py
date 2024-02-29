from og_marl.environments.smacv1 import SMACv1
from og_marl.loggers import WandbLogger
from og_marl.replay_buffers import FlashbaxReplayBuffer

env = SMACv1("3m")

logger = WandbLogger(entity="claude_formanek")

# TODO
system = QMIXSystsem(env, logger, add_agent_id_to_obs=False)  # noqa: F821

rb = FlashbaxReplayBuffer(sequence_length=20, batch_size=128)

system.train_online(rb)
