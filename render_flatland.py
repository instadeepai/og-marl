from og_marl.wrapped_environments.flatland_wrapper import Flatland
import numpy as np
import matplotlib.pyplot as plt

env = Flatland("50trains")

env.reset()

# x = env.render()

for i in range(20):
    action = {f"{i}": np.array([2]) for i in range(50)}
    env.step(action)
    x = env.render()

plt.imshow(x)
plt.savefig("flatland_render.png")

