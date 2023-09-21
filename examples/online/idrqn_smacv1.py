from og_marl.loggers import WandbLogger
from og_marl.systems.idrqn import IDRQNSystem
from og_marl.environments.smacv1 import SMACv1
from og_marl.replay_buffers import SequenceCPPRB

env = SMACv1("3m")

logger = WandbLogger(entity="claude_formanek")

system = IDRQNSystem(env, logger)

rb = SequenceCPPRB(env, batch_size=64)

system.train_online(rb)