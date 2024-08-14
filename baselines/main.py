# Copyright 2023 InstaDeep Ltd. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from absl import app, flags

from og_marl.environments import get_environment
from og_marl.loggers import JsonWriter, WandbLogger
from og_marl.offline_dataset import download_and_unzip_vault
from og_marl.replay_buffers import FlashbaxReplayBuffer
from og_marl.tf2.networks import CNNEmbeddingNetwork
from og_marl.tf2.systems import get_system
from og_marl.tf2.utils import set_growing_gpu_memory

set_growing_gpu_memory()

FLAGS = flags.FLAGS
flags.DEFINE_string("env", "pettingzoo", "Environment name.")
flags.DEFINE_string("scenario", "pursuit", "Environment scenario name.")
flags.DEFINE_string("dataset", "Good", "Dataset type.: 'Good', 'Medium', 'Poor' or 'Replay' ")
flags.DEFINE_string("system", "qmix", "System name.")
flags.DEFINE_integer("seed", 42, "Seed.")
flags.DEFINE_float("trainer_steps", 5e4, "Number of training steps.")
flags.DEFINE_integer("batch_size", 64, "Number of training steps.")


def main(_):
    config = {
        "env": FLAGS.env,
        "scenario": FLAGS.scenario,
        "dataset": FLAGS.dataset,
        "system": FLAGS.system,
        "backend": "tf2",
    }

    env = get_environment(FLAGS.env, FLAGS.scenario)

    buffer = FlashbaxReplayBuffer(sequence_length=20, sample_period=1)

    download_and_unzip_vault(FLAGS.env, FLAGS.scenario)

    is_vault_loaded = buffer.populate_from_vault(FLAGS.env, FLAGS.scenario, FLAGS.dataset)
    if not is_vault_loaded:
        print("Vault not found. Exiting.")
        return

    logger = WandbLogger(project="og-marl-baselines", config=config)

    json_writer = JsonWriter(
        "logs",
        f"{FLAGS.system}",
        f"{FLAGS.scenario}_{FLAGS.dataset}",
        FLAGS.env,
        FLAGS.seed,
        file_name=f"{FLAGS.scenario}_{FLAGS.dataset}_{FLAGS.seed}.json",
        save_to_wandb=True,
    )

    system_kwargs = {"add_agent_id_to_obs": True}
    if FLAGS.scenario == "pursuit":
        system_kwargs["observation_embedding_network"] = CNNEmbeddingNetwork()

    system = get_system(FLAGS.system, env, logger, **system_kwargs)

    system.train_offline(buffer, max_trainer_steps=FLAGS.trainer_steps, json_writer=json_writer)


if __name__ == "__main__":
    app.run(main)
