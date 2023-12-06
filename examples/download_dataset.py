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

# from og_marl.offline_dataset import download_and_unzip_dataset

# download_and_unzip_dataset("smac_v1", "3m", dataset_base_dir="test_dataset")

import wandb
api = wandb.Api()
from og_marl.loggers import JsonWriter

project_name = "Off-The-Grid MARL"
runs_to_export = ["revived-sweep-153", "serene-sweep-152", "eternal-sweep-154"]

runs = api.runs(f"{project_name}", filters={"config.algo_name": "maicq", "config.env_name": "8m", "config.dataset_quality": "Poor"})

i=0
for run in runs:

    if run.name in runs_to_export:
        json_writer = JsonWriter("logs", f"original+tf2+maicq", f"8m_Poor", "smac_v1", 42+i)
        i+=1

        for j in range(11):
            if j == 10:
                json_writer.write(1000*j, "absolute/episode_return", run.history()[j]["evaluator_episode_return"], evaluation_step=j)
            else:
                json_writer.write(1000*j, "evaluator/episode_return", run.history()[j]["evaluator_episode_return"], evaluation_step=j)
        

    



