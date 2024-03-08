import os

# import module
import traceback

from og_marl.environments import get_environment
from og_marl.loggers import TerminalLogger, JsonWriter
from og_marl.replay_buffers import FlashbaxReplayBuffer
from og_marl.tf2.networks import CNNEmbeddingNetwork
from og_marl.tf2.systems import get_system
from og_marl.tf2.utils import set_growing_gpu_memory

set_growing_gpu_memory()

# For MAMuJoCo
os.environ["SUPPRESS_GR_PROMPT"] = "1"

scenario_system_configs = {
    "smac_v1": {
        "3m": {
            "systems": ["idrqn", "idrqn+cql", "idrqn+bcq", "qmix+cql", "qmix+bcq", "maicq", "dbc"],
            "datasets": ["Good"],
            "trainer_steps": 2000,
            "evaluate_every": 1000,
        },
    },
    "pettingzoo": {
        "pursuit": {
            "systems": ["idrqn", "idrqn+cql", "idrqn+bcq", "qmix+cql", "qmix+bcq", "maicq", "dbc"],
            "datasets": ["Good"],
            "trainer_steps": 2000,
            "evaluate_every": 1000,
        },
    },
    # "mamujoco": {
    #     "2halfcheetah": {
    #         "systems": ["iddpg", "iddpg+cql", "maddpg+cql", "maddpg", "omar"],
    #         "datasets": ["Good"],
    #         "trainer_steps": 3000,
    #         "evaluate_every": 1000,
    #     },
    # },
}

seeds = [42]

for seed in seeds:
    for env_name in scenario_system_configs.keys():
        for scenario_name in scenario_system_configs[env_name].keys():
            for dataset_name in scenario_system_configs[env_name][scenario_name]["datasets"]:
                for system_name in scenario_system_configs[env_name][scenario_name]["systems"]:
                    try:
                        config = {
                            "env": env_name,
                            "scenario": scenario_name,
                            "dataset": dataset_name,
                            "system": env_name,
                            "seed": seed,
                        }
                        logger = TerminalLogger()
                        env = get_environment(env_name, scenario_name)

                        buffer = FlashbaxReplayBuffer(sequence_length=20, sample_period=1)
                        is_vault_loaded = buffer.populate_from_vault(
                            env_name, scenario_name, dataset_name
                        )
                        if not is_vault_loaded:
                            raise ValueError("Vault not found. Exiting.")

                        json_writer = JsonWriter(
                            "test_all_baselines",
                            system_name,
                            f"{scenario_name}_{dataset_name}",
                            env_name,
                            seed,
                        )

                        system_kwargs = {"add_agent_id_to_obs": True}

                        if scenario_name == "pursuit":
                            system_kwargs["observation_embedding_network"] = CNNEmbeddingNetwork()

                        system = get_system(system_name, env, logger, **system_kwargs)

                        trainer_steps = scenario_system_configs[env_name][scenario_name][
                            "trainer_steps"
                        ]
                        evaluate_every = scenario_system_configs[env_name][scenario_name][
                            "evaluate_every"
                        ]
                        system.train_offline(
                            buffer,
                            max_trainer_steps=trainer_steps,
                            evaluate_every=evaluate_every,
                            json_writer=json_writer,
                        )
                    except:  # noqa: E722
                        logger.close()
                        print("BROKEN:", env_name, scenario_name, system_name)
                        traceback.print_exc()
                        print()
                        continue
