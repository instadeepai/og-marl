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

import sys
from pathlib import Path
import tensorflow as tf
import tree
import zipfile
import os
import requests

DATASET_INFO = {
    "smac_v1": {
        "3m": {
            "url": "https://tinyurl.com/3m-dataset",
            "sequence_length": 20,
            "period": 10
        },
        "8m": {
            "url": "https://tinyurl.com/8m-dataset",
            "sequence_length": 20,
            "period": 10
        },
        "5m_vs_6m": {
            "url": "https://tinyurl.com/5m-vs-6m-dataset",
            "sequence_length": 20,
            "period": 10
        },
        "2s3z": {
            "url": "https://tinyurl.com/2s3z-dataset",
            "sequence_length": 20,
            "period": 10
        },
        "3s5z_vs_3s6z": {
            "url": "https://tinyurl.com/3s5z-vs-3s6z-dataset3",
            "sequence_length": 20,
            "period": 10
        },
        "2c_vs_64zg": {
            "url": "https://tinyurl.com/2c-vs-64zg-dataset",
            "sequence_length": 20,
            "period": 10
        },
        "27m_vs_30m": {
            "url": "https://tinyurl.com/27m-vs-30m-dataset",
            "sequence_length": 20,
            "period": 10
        },
    },
    "smac_v2": {
        "terran_5_vs_5": {
            "url": "https://tinyurl.com/terran-5-vs-5-dataset",
            "sequence_length": 20,
            "period": 10
        },
        "zerg_5_vs_5": {
            "url": "https://tinyurl.com/zerg-5-vs-5-dataset",
            "sequence_length": 20,
            "period": 10
        },
        "terran_10_vs_10": {
            "url": "https://tinyurl.com/terran-10-vs-10-dataset",
            "sequence_length": 20,
            "period": 10
        },
    },
    "flatland": {
        "3_trains": {
            "url": "https://tinyurl.com/3trains-dataset",
            "sequence_length": 20, #TODO
            "period": 10
        },
        "5_trains": {
            "url": "https://tinyurl.com/5trains-dataset",
            "sequence_length": 20, #TODO
            "period": 10
        },
    },
    "mamujoco": {
        "2halfcheetah": {
            "url": "https://tinyurl.com/2halfcheetah-dataset",
            "sequence_length": 20,
            "period": 10
        },
        "2ant": {
            "url": "https://tinyurl.com/2ant-dataset",
            "sequence_length": 20,
            "period": 10
        },
        "4ant": {
            "url": "https://tinyurl.com/4ant-dataset",
            "sequence_length": 20,
            "period": 10
        },
    },
    "voltage_control": {
        "case33_3min_final": {
            "url": "https://tinyurl.com/case33-3min-final-dataset",
            "sequence_length": 20,
            "period": 10
        },
    }
}

def get_schema_dtypes(environment):
    act_type = list(environment.action_spaces.values())[0].dtype
    schema = {}
    for agent in environment.possible_agents:
        schema[agent + "_observations"] = tf.float32
        schema[agent + "_legal_actions"] = tf.float32
        schema[agent + "_actions"] = act_type
        schema[agent + "_rewards"] = tf.float32
        schema[agent + "_discounts"] = tf.float32

    ## Extras
    # Zero-padding mask
    schema["zero_padding_mask"] = tf.float32

    # Env state
    schema["env_state"] = tf.float32

    # Episode return
    schema["episode_return"] = tf.float32

    return schema

class OfflineMARLDataset:
    def __init__(
        self,
        environment,
        env_name,
        scenario_name,
        dataset_type,
        base_dataset_dir="./datasets",
    ):
        self._environment = environment
        self._schema = get_schema_dtypes(environment)
        self._agents = environment.possible_agents

        path_to_dataset = f"{base_dataset_dir}/{env_name}/{scenario_name}/{dataset_type}"

        file_path = Path(path_to_dataset)
        sub_dir_to_idx = {}
        idx = 0
        for subdir in os.listdir(file_path):
            if file_path.joinpath(subdir).is_dir():
                sub_dir_to_idx[subdir] = idx
                idx += 1

        def get_fname_idx(file_name):
            dir_idx = sub_dir_to_idx[file_name.split("/")[-2]] * 1000
            return dir_idx + int(file_name.split("log_")[-1].split(".")[0])

        filenames = [str(file_name) for file_name in file_path.glob("**/*.tfrecord")]
        filenames = sorted(filenames, key=get_fname_idx)

        filename_dataset = tf.data.Dataset.from_tensor_slices(filenames)
        self.raw_dataset = filename_dataset.flat_map(
            lambda x: tf.data.TFRecordDataset(x, compression_type="GZIP").map(
                self._decode_fn
            )
        )

        self.period = DATASET_INFO[env_name][scenario_name]["period"]
        self.sequence_length = DATASET_INFO[env_name][scenario_name]["sequence_length"]
        self.max_episode_length = environment.max_episode_length

    def _decode_fn(self, record_bytes):
        example = tf.io.parse_single_example(
            record_bytes,
            tree.map_structure(
                lambda x: tf.io.FixedLenFeature([], dtype=tf.string), self._schema
            ),
        )

        for key, dtype in self._schema.items():
            example[key] = tf.io.parse_tensor(example[key], dtype)

        sample = {"observations": {}, "actions": {}, "rewards": {}, "terminals": {}, "truncations": {}, "infos": {"legals":{}}}
        for agent in self._agents:
            sample["observations"][agent] = example[f"{agent}_observations"]
            sample["actions"][agent] = example[f"{agent}_actions"]
            sample["rewards"][agent] = example[f"{agent}_rewards"]
            sample["terminals"][agent] = 1 - example[f"{agent}_discounts"]
            sample["truncations"][agent] = tf.zeros_like(example[f"{agent}_discounts"])
            sample["infos"]["legals"][agent] = example[f"{agent}_legal_actions"]
            
        sample["infos"]["mask"] = example["zero_padding_mask"]
        sample["infos"]["state"] = example["env_state"]
        sample["infos"]["episode_return"] = example["episode_return"]

        return sample

    def __getattr__(self, name):
        """Expose any other attributes of the underlying environment.

        Args:
            name (str): attribute.

        Returns:
            Any: return attribute from env or underlying env.
        """
        if hasattr(self.__class__, name):
            return self.__getattribute__(name)
        else:
            return getattr(self._tf_dataset, name)
        

def download_and_unzip_dataset(env_name, scenario_name, dataset_base_dir="./datasets"):
    dataset_download_url = DATASET_INFO[env_name][scenario_name]["url"]

    # TODO add check to see if dataset exists already.

    os.makedirs(f'{dataset_base_dir}/tmp/', exist_ok=True)
    os.makedirs(f'{dataset_base_dir}/{env_name}/', exist_ok=True)

    zip_file_path = f'{dataset_base_dir}/tmp/tmp_dataset.zip'

    extraction_path = f'{dataset_base_dir}/{env_name}'

    response = requests.get(dataset_download_url, stream=True)
    total_length = response.headers.get('content-length')

    with open(zip_file_path, 'wb') as file:
        if total_length is None: # no content length header
            file.write(response.content)
        else:
            dl = 0
            total_length = int(total_length)
            for data in response.iter_content(chunk_size=4096):
                dl += len(data)
                file.write(data)
                done = int(50 * dl / total_length)
                sys.stdout.write("\r[%s%s]" % ('=' * done, ' ' * (50-done)) )    
                sys.stdout.flush()

    # Step 2: Unzip the file
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extraction_path)

    # Optionally, delete the zip file after extraction
    os.remove(zip_file_path)