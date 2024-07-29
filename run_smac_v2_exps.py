import os
from absl import app, flags

from og_marl.tf2.utils import set_growing_gpu_memory

set_growing_gpu_memory()

FLAGS = flags.FLAGS
flags.DEFINE_integer("num_seeds", 10, "Number of seeds to use.")

SCRIPT = "og_marl/tf2/systems/idrqn_cql.py"
EXPERIMENT_CONFIGS = ["idrqn_cql/smac_v2/all.yaml"] #, "idrqn_cql/smac_v1/5m_vs_6m/Medium_OG_MARL.yaml"]
WANDB_PROJECT = "smac-v2-benchmark"

SCENARIOS = ["terran_5_vs_5", "zerg_5_vs_5"]

DATASETS = ["Replay"]

def main(_):

    SEEDS = list(range(FLAGS.num_seeds))

    for seed in SEEDS:
        for config in EXPERIMENT_CONFIGS:
            for scenario in SCENARIOS:
                for dataset in DATASETS:
                    os.system(f"python {SCRIPT} +task={config} task.seed={seed} task.wandb_project={WANDB_PROJECT} task.scenario={scenario} task.dataset={dataset}")

if __name__ == "__main__":
    app.run(main)