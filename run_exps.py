# import os

# from og_marl.tf2.utils import set_growing_gpu_memory

# set_growing_gpu_memory()

# WANDB_PROJECT = "og-marl-smacv1-baselines"

# SCRIPTS = [
#     # "og_marl/tf2/systems/iql_cql.py",
#     "og_marl/tf2/systems/qmix_cql.py",
#     # "og_marl/tf2/systems/maicq.py",
#     # "og_marl/tf2/systems/iql_bcq.py",
#     # "og_marl/tf2/systems/qmix_bcq.py",
#     # "og_marl/tf2/systems/discrete_bc.py",
# ]

# TASK = "task.scenario=5m_vs_6m task.dataset=Good"

# SEEDS = [1,2,3,4,5]

# def main():

#     for seed in SEEDS:
#         for script in SCRIPTS:
#             os.system(f"python {script} {TASK} wandb_project={WANDB_PROJECT} seed={seed}")

# if __name__ == "__main__":
#     main()


import os

from og_marl.tf2.utils import set_growing_gpu_memory

set_growing_gpu_memory()

WANDB_PROJECT = "haddpg+cql-vs-maddpg+cql-2walker"

SCRIPTS = [
    # "og_marl/tf2/systems/iddpg_cql.py",
    # "og_marl/tf2/systems/iddpg_bc.py",
    "og_marl/tf2/systems/omar.py",
    # "og_marl/tf2/offline/haddpg_cql.py",
    # "og_marl/tf2/offline/maddpg_cql.py",
    # "og_marl/tf2/systems/continuous_bc.py",
    # "og_marl/tf2/systems/haddpg_cql.py",
    # "og_marl/tf2/systems/maddpg_cql_non_shared.py",
    #    "og_marl/tf2/systems/maddpg_cql_cpg.py"
]

TASKS = [
    "task.env=gymnasium_mamujoco task.scenario=2walker task.dataset=Replay",
    "task.env=gymnasium_mamujoco task.scenario=6halfcheetah task.dataset=Replay1400",
]

training_steps = int(5e5)


SEEDS = [1]


def main():
    for task in TASKS:
        for seed in SEEDS:
            for script in SCRIPTS:
                os.system(
                    f"python {script} {task} wandb_project={WANDB_PROJECT} seed={seed} training_steps={training_steps}"
                )


if __name__ == "__main__":
    main()
