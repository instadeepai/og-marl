from og_marl.environments import smacv2
from og_marl.utils.dataset_utils import profile_dataset

env = smacv2.SMACv2("zerg_5_vs_5") # Change SMACv2 Scenario Here

dataset = env.get_dataset("Replay") # Change Dataset Type Here

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