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