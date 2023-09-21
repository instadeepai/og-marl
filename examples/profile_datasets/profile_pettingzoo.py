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

from og_marl.environments import cooperative_pong
from og_marl.environments import pursuit
from og_marl.environments import pistonball
from og_marl.utils.dataset_utils import profile_dataset

scenario = "coop_pong" # Change Scenario Here 

if scenario == "pursuit":   
    env = pursuit.Pursuit()
elif scenario == "pistonball":
    env = pistonball.Pistonball()
elif scenario == "coop_pong":
    env = cooperative_pong.CooperativePong()
else:
    raise NotImplementedError

dataset = env.get_dataset("Poor") # Change Dataset Type Here

stats = profile_dataset(dataset)

print("\DATASET STATS")
print(stats)

print("\DATASET SAMPLE")
dataset = iter(dataset)
sample = next(dataset)

print()
print("!!! Note that samples are sequences of consecutive timesteps. So the leading dimension is the time dimension !!!")
print()

print(f"Agent_0 Observation Shape: {sample.observations['agent_0'].observation.shape}")
print(f"Agent_0 Action Shape: {sample.actions['agent_0'].shape}")
print(f"Agent_0 Reward Shape: {sample.rewards['agent_0'].shape}")
print(f"Agent_0 Reward Shape: {sample.rewards['agent_0'].shape}")
print(f"Agent_0 Discount Shape: {sample.discounts['agent_0'].shape}")