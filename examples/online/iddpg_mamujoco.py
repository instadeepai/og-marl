from og_marl.environments.wrappers import PadObsandActs, Dtype
from og_marl.loggers import WandbLogger
from og_marl.systems.iddpg import IDDPGSystem
from og_marl.environments.mamujoco import MAMuJoCo
from og_marl.replay_buffers import SequenceCPPRB

env = MAMuJoCo("4ant")

env = PadObsandActs(env)

env = Dtype(env, "float32")

logger = WandbLogger(entity="claude_formanek")

system = IDDPGSystem(env, logger)

rb = SequenceCPPRB(env)

system.train_online(rb, 3e6)