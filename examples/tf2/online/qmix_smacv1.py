from og_marl.loggers import WandbLogger
from og_marl.tf2.systems.qmix import QMIXSystem
from og_marl.environments.smacv1 import SMACv1
from og_marl.replay_buffers import SequenceCPPRB

env = SMACv1("3m")

logger = WandbLogger(entity="claude_formanek")

system = QMIXSystem(env, logger, add_agent_id_to_obs=False)

rb = SequenceCPPRB(env, batch_size=128)

system.train_online(rb)