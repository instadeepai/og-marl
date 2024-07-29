import os
from absl import app, flags

from og_marl.tf2.utils import set_growing_gpu_memory

set_growing_gpu_memory()

FLAGS = flags.FLAGS


SCRIPT = "og_marl/tf2/systems/maddpg_cql.py"
EXPERIMENT_CONFIGS = [
    "maddpg_cql/mamujoco/2halfcheetah/Lemon",
    "maddpg_cql/mamujoco/2halfcheetah/Cherry",
]  # , "idrqn_cql/smac_v1/5m_vs_6m/Medium_OG_MARL.yaml"]
WANDB_PROJECT = "lemon-cherry-experiments"
SEEDS = [1, 2]


def main(_):
    for seed in SEEDS:
        for config in EXPERIMENT_CONFIGS:
            os.system(
                f"python {SCRIPT} +task={config} task.seed={seed} task.wandb_project={WANDB_PROJECT}"
            )


if __name__ == "__main__":
    app.run(main)
