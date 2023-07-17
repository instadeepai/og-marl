import numpy as np
import pandas as pd
import seaborn
import matplotlib.pyplot as plt

seaborn.set()

def profile_dataset(dataset):
    all_returns = []
    for item in dataset:
        if "episode_return" in item.extras:
            all_returns.append(item.extras["episode_return"].numpy())
    dataset_stats = pd.Series(all_returns).describe().to_dict()
    dataset_stats["mode"] = max(set(all_returns), key=all_returns.count)
    seaborn.violinplot(all_returns)
    plt.savefig("violin_plot")
    return dataset_stats