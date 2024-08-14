import os

from og_marl.tf2.utils import set_growing_gpu_memory

set_growing_gpu_memory()

WANDB_PROJECT = "og-marl-smacv1-baselines"

SCRIPTS = [
    "og_marl/tf2/systems/iql_cql.py",
    "og_marl/tf2/systems/qmix_cql.py",
    "og_marl/tf2/systems/maicq.py",
    "og_marl/tf2/systems/iql_bcq.py",
    "og_marl/tf2/systems/qmix_bcq.py",
    "og_marl/tf2/systems/discrete_bc.py",
]

TASK = "task.scenario=5m_vs_6m task.dataset=Good"

SEEDS = [3,4]

def main():

    for seed in SEEDS:
        for script in SCRIPTS:
            os.system(f"python {script} {TASK} wandb_project={WANDB_PROJECT} seed={seed}")

if __name__ == "__main__":
    main()