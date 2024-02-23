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

# from og_marl.environments.gymnasium_mamujoco import MAMuJoCo
# from og_marl.environments.wrappers import Dtype, PadObsandActs
from og_marl.environments.old_mamujoco import MAMuJoCo
from og_marl.loggers import WandbLogger
from og_marl.replay_buffers import FlashbaxReplayBuffer
from og_marl.tf2.systems.maddpg import MADDPGSystem

env = MAMuJoCo("4ant")

# env = PadObsandActs(env)

# env = Dtype(env, "float32")

logger = WandbLogger()

# system = IDDPGSystem(env, logger)
system = MADDPGSystem(env, logger)

rb = FlashbaxReplayBuffer(sequence_length=20, max_size=50_000)

system.train_online(rb, 10e6)
