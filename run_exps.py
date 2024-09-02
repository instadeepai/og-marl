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

WANDB_PROJECT = "bc-systems-on-mamujoco-1m"

TRAINING_STEPS = int(2.5e5)

SCRIPTS = [
    # "og_marl/tf2/systems/iddpg_cql.py",
    # "og_marl/tf2/systems/iddpg_bc.py",
    # "og_marl/tf2/systems/omar.py",
    "og_marl/tf2/systems/maddpg_bc.py",
    "og_marl/tf2/systems/iddpg_bc.py",
    "og_marl/tf2/systems/continuous_bc.py",
    # "og_marl/tf2/systems/hacql.py",
    # "og_marl/tf2/systems/maddpg_cql_non_shared.py",
#    "og_marl/tf2/systems/maddpg_cql_cpg.py"
]

# TASKS = [
#     "task.source=omiga task.env=mamujoco task.scenario=2ant task.dataset=Expert system.bc_alpha=0.5",
#     "task.source=omiga task.env=mamujoco task.scenario=2ant task.dataset=Medium system.bc_alpha=0.1",
#     "task.source=omiga task.env=mamujoco task.scenario=2ant task.dataset=Medium-Expert system.bc_alpha=0.1",
#     "task.source=omiga task.env=mamujoco task.scenario=2ant task.dataset=Medium-Replay system.bc_alpha=0.1",
# ]

TASKS = [
    "task.source=omiga task.env=mamujoco task.scenario=3hopper task.dataset=Expert system.bc_alpha=5.0",
    "task.source=omiga task.env=mamujoco task.scenario=3hopper task.dataset=Medium system.bc_alpha=5.0",
    "task.source=omiga task.env=mamujoco task.scenario=3hopper task.dataset=Medium-Expert system.bc_alpha=5.0",
    "task.source=omiga task.env=mamujoco task.scenario=3hopper task.dataset=Medium-Replay system.bc_alpha=5.0",
]

SEEDS = [4] 

def main():

    for task in TASKS:
        for seed in SEEDS:
            for script in SCRIPTS:
                os.system(f"python {script} {task} wandb_project={WANDB_PROJECT} seed={seed} training_steps={TRAINING_STEPS}")

if __name__ == "__main__":
    main()
