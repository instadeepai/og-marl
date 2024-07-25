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

import os
import sys
import zipfile
from typing import Dict, List
import pprint

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import requests  # type: ignore
import seaborn as sns
from chex import Array
from flashbax.vault import Vault
from git import Optional


VAULT_INFO = {
    "og_marl": {
        "smac_v1": {
            "3m": {
                "url": "https://huggingface.co/datasets/InstaDeepAI/og-marl/resolve/main/core/smac_v1/3m.zip"
            },
            "8m": {
                "url": "https://huggingface.co/datasets/InstaDeepAI/og-marl/resolve/main/core/smac_v1/8m.zip"
            },
            "5m_vs_6m": {
                "url": "https://huggingface.co/datasets/InstaDeepAI/og-marl/resolve/main/core/smac_v1/5m_vs_6m.zip"
            },
            "2s3z": {
                "url": "https://huggingface.co/datasets/InstaDeepAI/og-marl/resolve/main/core/smac_v1/2s3z.zip"
            },
            "3s5z_vs_3s6z": {
                "url": "https://huggingface.co/datasets/InstaDeepAI/og-marl/resolve/main/core/smac_v1/3s5z_vs_3s6z.zip"
            },
        },
        "smac_v2": {
            "terran_5_vs_5": {
                "url": "https://huggingface.co/datasets/InstaDeepAI/og-marl/resolve/main/core/smac_v2/terran_5_vs_5.zip"
            },
            "zerg_5_vs_5": {
                "url": "https://huggingface.co/datasets/InstaDeepAI/og-marl/resolve/main/core/smac_v2/zerg_5_vs_5.zip"
            },
        },
        "mamujoco": {
            "2halfcheetah": {
                "url": "https://huggingface.co/datasets/InstaDeepAI/og-marl/resolve/main/core/mamujoco/2halfcheetah.zip"
            },
        },
    },
    "cfcql": {
        "smac_v1": {
            "6h_vs_8z": {
                "url": "https://huggingface.co/datasets/InstaDeepAI/og-marl/resolve/main/prior_work/cfcql/smac_v1/6h_vs_8z.zip"
            },
            "3s_vs_5z": {
                "url": "https://huggingface.co/datasets/InstaDeepAI/og-marl/resolve/main/prior_work/cfcql/smac_v1/3s_vs_5z.zip"
            },
            "5m_vs_6m": {
                "url": "https://huggingface.co/datasets/InstaDeepAI/og-marl/resolve/main/prior_work/cfcql/smac_v1/5m_vs_6m.zip"
            },
            "2s3z": {
                "url": "https://huggingface.co/datasets/InstaDeepAI/og-marl/resolve/main/prior_work/cfcql/smac_v1/2s3z.zip"
            },
        },
    },
}

def print_dataset_options():
    pprint.pprint(VAULT_INFO, depth=2)
    return

def download_and_unzip_vault(
    dataset_source: str,
    env_name: str,
    scenario_name: str,
    dataset_base_dir: str = "./vaults",
) -> None:
    
    
    

    if check_directory_exists_and_not_empty(f"{dataset_base_dir}/{env_name}/{scenario_name}.vlt"):
        print(f"Vault '{dataset_base_dir}/{env_name}/{scenario_name}' already exists.")
        return

    dataset_download_url = VAULT_INFO[env_name][scenario_name]["url"]

    os.makedirs(f"{dataset_base_dir}/tmp/", exist_ok=True)
    os.makedirs(f"{dataset_base_dir}/{env_name}/", exist_ok=True)

    zip_file_path = f"{dataset_base_dir}/tmp/tmp_dataset.zip"

    extraction_path = f"{dataset_base_dir}/{env_name}"

    response = requests.get(dataset_download_url, stream=True)
    total_length = response.headers.get("content-length")

    with open(zip_file_path, "wb") as file:
        if total_length is None:  # no content length header
            file.write(response.content)
        else:
            dl = 0
            total_length = int(total_length)  # type: ignore
            for data in response.iter_content(chunk_size=4096):
                dl += len(data)
                file.write(data)
                done = int(50 * dl / total_length)  # type: ignore
                sys.stdout.write("\r[%s%s]" % ("=" * done, " " * (50 - done)))
                sys.stdout.flush()

    # Step 2: Unzip the file
    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        zip_ref.extractall(extraction_path)

    # Optionally, delete the zip file after extraction
    os.remove(zip_file_path)


def check_directory_exists_and_not_empty(path: str) -> bool:
    # Check if the directory exists
    if os.path.exists(path) and os.path.isdir(path):
        # Check if the directory is not empty
        if not os.listdir(path):  # This will return an empty list if the directory is empty
            return False  # Directory exists but is empty
        else:
            return True  # Directory exists and is not empty
    else:
        return False  # Directory does not exist
