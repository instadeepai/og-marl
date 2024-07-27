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
import hydra
from omegaconf import DictConfig, OmegaConf

from og_marl.environments import get_environment
from og_marl.loggers import WandbLogger
from og_marl.offline_dataset import download_and_unzip_vault
from og_marl.replay_buffers import FlashbaxReplayBuffer
from og_marl.tf2.systems import get_system
from og_marl.tf2.utils import set_growing_gpu_memory

set_growing_gpu_memory()

@hydra.main(version_base=None, config_path="og_marl/tf2/systems/conf", config_name="base")
def run_experiment(cfg : DictConfig) -> None:
    env = get_environment(cfg["env"], cfg["scenario"], seed=cfg["seed"])

    # Remove sequence_length and sample_period from params before passing params to system
    sequence_length, sample_period = cfg["params"].pop("sequence_length"), cfg["params"].pop("sample_period")
    buffer = FlashbaxReplayBuffer(sequence_length=sequence_length, sample_period=sample_period)

    download_and_unzip_vault(cfg["env"], cfg["scenario"])

    buffer.populate_from_vault(cfg["env"], cfg["scenario"], cfg["dataset"])

    logger = WandbLogger(project=cfg["project"], config={})

    trainer_steps = cfg["params"].pop("trainer_steps")
    system = get_system(cfg["system"], **cfg["params"])
    system.train_offline(buffer, max_trainer_steps=trainer_steps)


if __name__=="__main__":
    run_experiment()