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
from collections import namedtuple
import tree
import zipfile
import os
import requests

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
        path_to_dataset,
        num_parallel_calls=None
    ):
        self._environment = environment
        self._schema = get_schema_dtypes(environment)
        self._agents = environment.possible_agents

        file_path = Path(path_to_dataset)
        filenames = [
            str(file_name) for file_name in file_path.glob("**/*.tfrecord")
        ]
        filename_dataset = tf.data.Dataset.from_tensor_slices(filenames)
        self._tf_dataset = filename_dataset.interleave(
            lambda x: tf.data.TFRecordDataset(x, compression_type="GZIP").map(
                self._decode_fn
            ),
            cycle_length=None,
            num_parallel_calls=num_parallel_calls,
            deterministic=False,
            block_length=None,
        )

    def get_sequence_length(self):
        for sample in self._tf_dataset:
            T = sample["mask"].shape[0]
            break
        return T

    def _decode_fn(self, record_bytes):
        example = tf.io.parse_single_example(
            record_bytes,
            tree.map_structure(
                lambda x: tf.io.FixedLenFeature([], dtype=tf.string), self._schema
            ),
        )

        for key, dtype in self._schema.items():
            example[key] = tf.io.parse_tensor(example[key], dtype)

        sample = {}
        for agent in self._agents:
            sample[f"{agent}_observations"] = example[f"{agent}_observations"]
            sample[f"{agent}_actions"] = example[f"{agent}_actions"]
            sample[f"{agent}_rewards"] = example[f"{agent}_rewards"]
            sample[f"{agent}_terminals"] = 1 - example[f"{agent}_discounts"]
            sample[f"{agent}_truncations"] = tf.zeros_like(example[f"{agent}_discounts"])
            sample[f"{agent}_legals"] = example[f"{agent}_legal_actions"]
            
        sample["mask"] = example["zero_padding_mask"]
        sample["state"] = example["env_state"]
        sample["episode_return"] = example["episode_return"]

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
        

DATASET_URLS = {
    "smac_v1": {
        "3m": "https://tinyurl.com/3m-dataset",
        "8m": "https://tinyurl.com/8m-dataset",
        "5m_vs_6m": "https://tinyurl.com/5m-vs-6m-dataset",
        "2s3z": "https://tinyurl.com/2s3z-dataset",
        "3s5z_vs_3s6z": "https://tinyurl.com/3s5z-vs-3s6z-dataset3",
        "2c_vs_64zg": "https://tinyurl.com/2c-vs-64zg-dataset",
        "27m_vs_30m": "https://tinyurl.com/27m-vs-30m-dataset"
    },
    "smac_v2": {
        "terran_5_vs_5": "https://tinyurl.com/terran-5-vs-5-dataset",
        "zerg_5_vs_5": "https://tinyurl.com/zerg-5-vs-5-dataset",
        "terran_10_vs_10": "https://tinyurl.com/terran-10-vs-10-dataset"
    },
    "flatland": {
        "3_trains": "https://tinyurl.com/3trains-dataset",
        "5_trains": "https://tinyurl.com/5trains-dataset"
    },
    "mamujoco": {
        "2_halfcheetah": "",
        "2_ant": "",
        "4_ant": ""
    },
    "voltage_control": {
        "case33_3min_final": "https://tinyurl.com/case33-3min-final-dataset",
    }
}
def download_and_unzip_dataset(env_name, scenario_name, dataset_base_dir="./datasets"):
    dataset_download_url = DATASET_URLS[env_name][scenario_name]

    os.makedirs(f'{dataset_base_dir}/tmp/', exist_ok=True)
    os.makedirs(f'{dataset_base_dir}/{env_name}', exist_ok=True)

    zip_file_path = f'{dataset_base_dir}/tmp/tmp_dataset.zip'

    extraction_path = f'{dataset_base_dir}/{env_name}/'

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

FLASHBAX_DATASET_URLS = {
    "smac_v1": {
        "8m": "https://s3.kao.instadeep.io/offline-marl-dataset/flashbax_8m.zip"
    }
}
def download_flashbax_dataset(env_name, scenario_name, base_dir="./datasets/flashbax"):
    dataset_download_url = FLASHBAX_DATASET_URLS[env_name][scenario_name]

    os.makedirs(f'{base_dir}/tmp/', exist_ok=True)
    os.makedirs(f'{base_dir}/{env_name}/{scenario_name}', exist_ok=True)

    zip_file_path = f'{base_dir}/tmp/tmp_dataset.zip'

    extraction_path = f'{base_dir}/{env_name}/{scenario_name}'

    print("Downloading dataset. This could take a few minutes.")
    
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