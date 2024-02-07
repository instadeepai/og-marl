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
from og_marl.jax.utils import get_system
from og_marl.loggers import JsonWriter, WandbLogger

FLAGS = flags.FLAGS
flags.DEFINE_string("env", "smac_v1", "Environment name.")
flags.DEFINE_string("scenario", "8m", "Environment scenario name.")
flags.DEFINE_string(
    "dataset", "Good", "Dataset type.: 'Good', 'Medium', 'Poor' or '' for combined. "
)
flags.DEFINE_string("system", "maicq", "System name.")
flags.DEFINE_integer("seed", 42, "Seed.")
flags.DEFINE_integer("num_epochs", 10, "Number of training steps.")
flags.DEFINE_integer("batch_size", 256, "Number of training steps.")


def main(_):
    config = {
        "env": FLAGS.env,
        "scenario": FLAGS.scenario,
        "dataset": FLAGS.dataset,
        "system": FLAGS.system,
        "backend": "jax",
    }

    env = get_environment(FLAGS.env, FLAGS.scenario)

    logger = WandbLogger(project="benchmark-jax-og-marl", config=config)

    train_system_fn = get_system(FLAGS.system, env, logger)

    json_writer = JsonWriter(
        "logs", f"jax+{FLAGS.system}", f"{FLAGS.scenario}_{FLAGS.dataset}", FLAGS.env, FLAGS.seed
    )

    dataset_path = f"datasets/flashbax/{FLAGS.env}/{FLAGS.scenario}/{FLAGS.dataset}"

    train_system_fn(
        dataset_path,
        num_epochs=FLAGS.num_epochs,
        json_writer=json_writer,
        batch_size=FLAGS.batch_size,
    )


if __name__ == "__main__":
    app.run(main)
