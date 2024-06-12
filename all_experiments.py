import os
from absl import app, flags

from utils.utils import set_growing_gpu_memory

set_growing_gpu_memory()

FLAGS = flags.FLAGS
flags.DEFINE_string("env", "mamujoco_omiga", "Environment name.")
flags.DEFINE_integer("num_seeds", 10, "Number of seeds to use.")

EXPERIMENT_CONFIGS = {
    "smac_v1": { # From OG-MARL
        "scenarios": ["5m_vs_6m", "8m", "2s3z"],
        "datasets": ["Good", "Medium", "Poor"],
        "systems": ["bc", "iql+cql"],
        "trainer_steps": 100000
    },
    "smac_v1_omiga": {
        "scenarios": ["corridor", "5m_vs_6m", "2c_vs_64zg", "6h_vs_8z"],
        "datasets": ["Good", "Medium", "Poor"],
        "systems": ["bc", "iql+cql"],
        "trainer_steps": 100000
    },
    "smac_v1_cfcql": {
        "scenarios": ["2s3z", "5m_vs_6m", "3s_vs_5z", "6h_vs_8z"],
        "datasets": ["Expert", "Mixed", "Medium", "Medium-Replay"],
        "systems": ["bc", "iql+cql"],
        "trainer_steps": 100000
    },
    "mamujoco": { # From OG-MARL
        "scenarios": ["2halfcheetah", "2ant", "4ant"],
        "datasets": ["Good", "Medium", "Poor"],
        "systems": ["iddpg+bc", "maddpg+cql"],
        "trainer_steps": 200000
    },
    "mamujoco_omar": {
        "scenarios": ["2halfcheetah"],
        "datasets": ["Random", "Medium-Replay", "Medium", "Expert"],
        "systems": ["iddpg+bc", "maddpg+cql"],
        "trainer_steps": 200000
    },
    "mamujoco_omiga": {
        "scenarios": ["6halfcheetah", "3hopper", "2ant"],
        "datasets": ["Medium-Expert", "Medium-Replay", "Medium", "Expert"],
        "systems": ["iddpg+bc", "maddpg+cql"],
        "trainer_steps": 200000
    },
    "mpe_omar": {
        "scenarios": ["simple_spread"],
        "datasets": ["Random", "Medium-Replay", "Medium", "Expert"],
        "systems": ["iddpg+bc", "maddpg+cql"],
        "trainer_steps": 50000
    }
}

def main(_):

    SEEDS = list(range(FLAGS.num_seeds))

    for seed in SEEDS:
        for env, config in EXPERIMENT_CONFIGS.items():
            if FLAGS.env != env:
                continue
            for scenario in config["scenarios"]:
                for dataset in config["datasets"]:
                    for system in config["systems"]:
                        os.system(f"python main.py --env={env} --scenario={scenario} --dataset={dataset} --system={system} --trainer_steps={config["trainer_steps"]} --seed={seed}")


if __name__ == "__main__":
    app.run(main)
