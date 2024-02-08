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

from og_marl.environments.utils import get_environment
from og_marl.replay_buffers import FlashbaxReplayBuffer, SequenceCPPRB
from og_marl.tf2.utils import get_system, set_growing_gpu_memory
from og_marl.loggers import JsonWriter, WandbLogger
from og_marl.offline_dataset import OfflineMARLDataset
from og_marl.replay_buffers import SequenceCPPRB
from og_marl.tf2.utils import get_system, set_growing_gpu_memory

set_growing_gpu_memory()

FLAGS = flags.FLAGS
flags.DEFINE_string("env", "mamujoco", "Environment name.")
flags.DEFINE_string("scenario", "2halfcheetah", "Environment scenario name.")
flags.DEFINE_string(
    "dataset", "Good", "Dataset type.: 'Good', 'Medium', 'Poor' or '' for combined. "
)
flags.DEFINE_string("system", "iddpg+cql", "System name.")
flags.DEFINE_integer("seed", 42, "Seed.")
flags.DEFINE_float("trainer_steps", 1e5, "Number of training steps.")
flags.DEFINE_integer("batch_size", 32, "Number of training steps.")
flags.DEFINE_integer(
    "num_offline_sequences",
    10_000,
    "Number of sequences to load from the offline dataset into the replay buffer.",
)


def main(_):
    config = {
        "env": FLAGS.env,
        "scenario": FLAGS.scenario,
        "dataset": FLAGS.dataset if FLAGS.dataset != "" else "All",
        "system": FLAGS.system,
        "backend": "tf2",
    }

    env = get_environment(FLAGS.env, FLAGS.scenario)
    
    buffer = FlashbaxReplayBuffer(sequence_length=20, sample_period=10, max_size=200_000)

    buffer.populate_from_vault(f"{FLAGS.scenario}_{FLAGS.dataset}.vlt", "0")

    logger = WandbLogger(project="tf2-og-marl", config=config)

    json_writer = JsonWriter(
        "logs", f"tf2+{FLAGS.system}", f"{FLAGS.scenario}_{FLAGS.dataset}", FLAGS.env, FLAGS.seed
    )

    system_kwargs = {"add_agent_id_to_obs": True}
    system = get_system(FLAGS.system, env, logger, **system_kwargs)

    system.train_offline(
        buffer, 
        max_trainer_steps=FLAGS.trainer_steps,
        json_writer=json_writer
    )


if __name__ == "__main__":
    app.run(main)
