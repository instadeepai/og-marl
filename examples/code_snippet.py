from og_marl.environments.smac import SMAC
from og_marl.systems.qmix import QMIX
from og_marl.offline_tools import OfflineLogger

# Instantiate Environment
env = SMAC("3m")

# Wrap env in offline logger
env = OfflineLogger(env)

# Make multi-agent system
system = QMIX(env)

# Collect data
print("Online Training")
system.run_online()

# Load dataset
print("Loading Dataset")
dataset = env.get_dataset("Good")

# Train offline
print("Offline Training")
system.run_offline(dataset)