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

from absl import flags, app

from og_marl.environments.utils import get_environment
from og_marl.replay_buffers import SequenceCPPRB
from og_marl.tf2.utils import get_system, set_growing_gpu_memory
from og_marl.loggers import WandbLogger
from og_marl.offline_dataset import OfflineMARLDataset

set_growing_gpu_memory()

FLAGS = flags.FLAGS
flags.DEFINE_string("env", "smac_v1", "Environment name.")
flags.DEFINE_string("scenario", "8m", "Environment scenario name.")
flags.DEFINE_string("dataset", "Good", "Dataset type. 'Good', 'Medium', 'Poor' or '' for combined. ")
flags.DEFINE_string("system", "maicq", "System name.")

def main(_):
    config = {
        "env": FLAGS.env,
        "scenario": FLAGS.scenario,
        "dataset": FLAGS.dataset if FLAGS.dataset != "" else "All",
        "system": FLAGS.system,
        "backend": "tf2"
    }

    env = get_environment(FLAGS.env, FLAGS.scenario)

    dataset = OfflineMARLDataset(env, f"datasets/{FLAGS.env}/{FLAGS.scenario}/{FLAGS.dataset}")

    batched_dataset = SequenceCPPRB(env, max_size=100_000, batch_size=256)
    batched_dataset.populate_from_dataset(dataset)

    logger = WandbLogger(project="tf2-og-marl", config=config)

    system_kwargs = {
        "add_agent_id_to_obs": True
    }
    system = get_system(FLAGS.system, env, logger, **system_kwargs)

    system.train_offline(
        batched_dataset, 
        max_trainer_steps=5e4,
        evaluate_every=1000,
        num_eval_episodes=4,
        batch_size=256,
    )

if __name__ == "__main__":
    app.run(main)