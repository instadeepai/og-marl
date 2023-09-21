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