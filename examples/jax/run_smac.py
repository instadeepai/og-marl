from og_marl.jax.systems.maicq import train_maicq
from og_marl.environments.utils import get_environment

eval_env = get_environment("smac_v1", "8m")
dataset_path = ".experience/Good_8m"

train_maicq(eval_env, dataset_path, batch_size=256, num_epochs=10, num_episodes_per_evaluation=4)