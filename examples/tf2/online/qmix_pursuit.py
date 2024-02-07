from og_marl.environments.pursuit import Pursuit
from og_marl.loggers import WandbLogger
from og_marl.replay_buffers import SequenceCPPRB
from og_marl.tf2.systems.qmix import QMIXSystem

env = Pursuit()

logger = WandbLogger(entity="claude_formanek")

system = QMIXSystem(env, logger, add_agent_id_to_obs=True, target_update_rate=0.00005)

rb = SequenceCPPRB(env, batch_size=64)

system.train_online(rb)
