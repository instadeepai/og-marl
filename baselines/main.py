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
from og_marl.loggers import WandbLogger
from og_marl.offline_dataset import download_and_unzip_vault
from og_marl.replay_buffers import FlashbaxReplayBuffer, PrioritisedFlashbaxReplayBuffer
from og_marl.tf2.systems import get_system
from og_marl.tf2.utils import set_growing_gpu_memory
from og_marl.tf2.systems.rec_maddpg_cql import MADDPGCQLSystem

set_growing_gpu_memory()

FLAGS = flags.FLAGS
flags.DEFINE_string("env", "mpe", "Environment name.")
flags.DEFINE_string("scenario", "simple_spread", "Environment scenario name.")
flags.DEFINE_string("dataset", "medium-replay", "Dataset type.: 'Good', 'Medium', 'Poor' or 'Replay' ")
flags.DEFINE_string("system", "maddpg+cql", "System name.")
flags.DEFINE_integer("seed", 42, "Seed.")
flags.DEFINE_float("trainer_steps", 3e5, "Number of training steps.")
flags.DEFINE_integer("batch_size", 64, "Number of training steps.")
flags.DEFINE_float("priority_exponent", 0.99, "Number of training steps.")
flags.DEFINE_float("gaussian_steepness", 4, "")


def main(_):
    config = {
        "env": FLAGS.env,
        "scenario": FLAGS.scenario,
        "dataset": FLAGS.dataset,
        "system": FLAGS.system,
        "backend": "tf2",
    }

    env = get_environment(FLAGS.env, FLAGS.scenario)

    if FLAGS.system == "maddpg+cql+per":
        buffer = PrioritisedFlashbaxReplayBuffer(
            batch_size=FLAGS.batch_size,
            sequence_length=20,
            sample_period=10,
            seed=FLAGS.seed,
            priority_exponent=FLAGS.priority_exponent,
        )
    else:
        buffer = FlashbaxReplayBuffer(
            sequence_length=20, sample_period=10, batch_size=FLAGS.batch_size, seed=FLAGS.seed
        )

    download_and_unzip_vault(FLAGS.env, FLAGS.scenario)

    is_vault_loaded = buffer.populate_from_vault(
        FLAGS.env,
        FLAGS.scenario,
        FLAGS.dataset,
    )
    if not is_vault_loaded:
        print("Vault not found. Exiting.")
        return

    logger = WandbLogger(
        entity="off-the-grid-marl-team", project="rec_maddpg", config=config
    )

    system_kwargs = {"add_agent_id_to_obs": True, "gaussian_steepness": FLAGS.gaussian_steepness}

    system = MADDPGCQLSystem(env, logger, **system_kwargs)

    system.train_offline(
        buffer, max_trainer_steps=FLAGS.trainer_steps, evaluate_every=5000
    )


if __name__ == "__main__":
    app.run(main)
