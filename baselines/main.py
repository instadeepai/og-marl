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
from og_marl.replay_buffers import FlashbaxReplayBuffer, MixedBuffer
from og_marl.tf2.networks import CNNEmbeddingNetwork
from og_marl.tf2.systems import get_system
from og_marl.tf2.utils import set_growing_gpu_memory

set_growing_gpu_memory()

FLAGS = flags.FLAGS

flags.DEFINE_string("env", "smac_v1", "Environment name.")
flags.DEFINE_string("scenario", "2s3z", "Environment scenario name.")
flags.DEFINE_string("dataset", "Poor", "Dataset type.: 'Good', 'Medium', 'Poor' or 'Replay' ") # Usually just use the good data, except in an ablation study over data types
flags.DEFINE_string("system", "qmix+cql", "System name.") # NOTE: just keep on qmix+cql, even for online QMIX, experiment with IDRQN+CQL
flags.DEFINE_integer("seed", 42, "Seed.")

# Offline params to experiment with
flags.DEFINE_float("offline_training_steps", 1000, "Number of offline training steps.") # 1000, 2000, 3000, 4000, 5000
flags.DEFINE_float("offline_cql_weight", 2, "CQL Weight during offline training.") # NOTE: no need to change this

# Online params to experiment with
flags.DEFINE_string("online_buffer", "mixed", "Set the online buffer to be 'mixed' or 'online'.") # online, mixed 
flags.DEFINE_float("online_cql_weight", 0, "CQL Weight during online training.") # 0, 1, 2

# NOTE: Keep fixed for now
flags.DEFINE_float("eps_decay_steps", 1000, "Timesteps during which to decay epsilon for online training.") # Probably just keep fixed for online and pre training exps

flags.DEFINE_string("suffix", "", "Additional information to be contained in the WandB project name.")

###
# NOTE: to run online QMIX, set offline_training_steps to zero, online_cql_weight to zero and online_buffer to "online"
###

def main(_):
    config = {
        "env": FLAGS.env,
        "scenario": FLAGS.scenario,
        "dataset": FLAGS.dataset,
        "system": FLAGS.system,
        "offline_cql_weight": FLAGS.offline_cql_weight,
        "online_cql_weight": FLAGS.online_cql_weight,
    }

    if FLAGS.suffix != "":
        FLAGS.suffix = "_("+FLAGS.suffix+")"
    project_name = "+"+FLAGS.system+"_"+str(FLAGS.offline_training_steps)+"_"+FLAGS.scenario+"_"+FLAGS.dataset+FLAGS.suffix
    logger = WandbLogger(project=project_name, config=config)

    system_kwargs = {
        "add_agent_id_to_obs": True,
        "eps_decay_timesteps": FLAGS.eps_decay_steps, # NOTE: lets try keep this fixed at 1000 for offline and online experiments
        "learning_rate": 3e-4, # NOTE: keep fixed
        "eps_min": 0.05 # NOTE: keep fixed
    }

    env = get_environment(FLAGS.env, FLAGS.scenario)

    system = get_system(FLAGS.system, env, logger, **system_kwargs)

    # Setup Offline Replay Buffer
    r2go = "calql" in FLAGS.system # NOTE ignore calql for now, focus on CQL 
    offline_buffer = FlashbaxReplayBuffer(sequence_length=10, sample_period=1, batch_size=64, rewards_to_go=r2go, seed=FLAGS.seed)

    download_and_unzip_vault(FLAGS.env, FLAGS.scenario)

    is_vault_loaded = offline_buffer.populate_from_vault(FLAGS.env, FLAGS.scenario, FLAGS.dataset, discount=0.99)
    if not is_vault_loaded:
        print("Vault not found. Exiting.")
        return

    # Offline Pre-training
    system._cql_weight.assign(FLAGS.offline_cql_weight)
    system.train_offline(offline_buffer, max_trainer_steps=FLAGS.offline_training_steps, evaluate_every=100, num_eval_episodes=4)


    # Setup online Replay buffer
    online_buffer = FlashbaxReplayBuffer(sequence_length=10, sample_period=1, batch_size=64, max_size=10000, rewards_to_go=r2go, seed=FLAGS.seed) # NOTE: keep max_size fixed

    if FLAGS.online_buffer == "mixed":
        online_buffer = MixedBuffer(online_buffer, offline_buffer) # setup mixed buffer 50/50 online/offline batches
    elif FLAGS.online_buffer != "online":
        raise ValueError(f"Unrecognised online buffer type - {FLAGS.online_buffer}")

    # Online training
    system._env_step_ctr = 0.0
    system._cql_weight.assign(FLAGS.online_cql_weight)
    system.train_online(online_buffer, max_env_steps=100000, train_period=20) # NOTE: keep environment steps fixed and train period

if __name__ == "__main__":
    app.run(main)
