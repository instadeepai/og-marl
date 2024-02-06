from og_marl.environments.smacv2 import SMACv2
from og_marl.loggers import WandbLogger
from og_marl.replay_buffers import SequenceCPPRB
from og_marl.tf2.systems.qmix import QMIXSystem

env = SMACv2("terran_5_vs_5")

logger = WandbLogger(entity="claude_formanek")

system = QMIXSystem(
    env, logger, add_agent_id_to_obs=True, learning_rate=0.0005, eps_decay_timesteps=100_000
)

rb = SequenceCPPRB(env, batch_size=128)

system.train_online(rb, max_env_steps=1e7)
