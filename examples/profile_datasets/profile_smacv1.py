from og_marl.environments import smac
from og_marl.utils.dataset_utils import profile_dataset

env = smac.SMAC("27m_vs_30m") # Change SMAC Scenario Here

dataset = env.get_dataset("Good") # Change Dataset Type Here

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