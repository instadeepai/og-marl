import os

SEEDS = [0,1,2,3,4,5,6,7,8,9]

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
        "scenarios": ["corridor", "5m_vs_6m", "2c_vs_64zg", "6h_vs_8z"],
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

the_env = "smac_v1_omiga"

if __name__ == "__main__":
    for seed in SEEDS:
        for env, config in EXPERIMENT_CONFIGS.items():
            if env != the_env:
                continue
            for scenario in config["scenarios"]:
                for dataset in config["datasets"]:
                    for system in config["systems"]:
                        trainer_steps = 100 # config["trainer_steps"]
                        if scenario.split("_")[0] == "mamujoco" and len(scenario.split("_")[0]) > 1:
                            os.environ["LD_LIBRARY_PATH"] = "$LD_LIBRARY_PATH:/root/.mujoco/mujoco200/bin:/usr/lib/nvidia"
                            python = "/root/miniconda3/envs/baselines200/bin/python"
                        else:
                            os.environ["LD_LIBRARY_PATH"] = "$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin:/usr/lib/nvidia"
                            python = "/root/miniconda3/envs/baselines210/bin/python"
                        
                        os.system(f"{python} main.py --env={env} --scenario={scenario} --dataset={dataset} --system={system} --trainer_steps={trainer_steps} --seed={seed}")
