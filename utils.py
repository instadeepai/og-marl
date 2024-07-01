from typing import Dict, List, Optional, Any

import os
import sys
import time
import requests
import zipfile

import wandb
import tensorflow as tf
from tensorflow import Module, Tensor

def set_growing_gpu_memory() -> None:
    """Solve gpu mem issues."""
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    physical_devices = tf.config.list_physical_devices("GPU")
    if physical_devices:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)

class WandbLogger:
    def __init__(
        self,
        config: Dict = {},  # noqa: B006
        project: str = "default_project",
        notes: str = "",
        tags: List = ["default"],  # noqa: B006
        entity: Optional[str] = None,
        log_every: int = 2,  # seconds
    ):
        wandb.init(project=project, notes=notes, tags=tags, entity=entity, config=config)

        self._log_every = log_every
        self._ctr = 0
        self._last_log = time.time()

    def write(self, logs: Dict[str, Any], force: bool = False) -> None:
        if time.time() - self._last_log > self._log_every or force:
            wandb.log(logs)

            for key, log in logs.items():
                print(f"{key}: {float(log)} |", end=" ")
            print()

            if not force:
                self._last_log = time.time()

        self._ctr += 1

    def close(self) -> None:
        wandb.finish()

def download_and_unzip_vault(scenario, dataset_base_dir: str = "./vaults") -> None:
    if check_directory_exists_and_not_empty(f"{dataset_base_dir}/mamujoco/{scenario}.vlt"):
        print(f"Vault already exists.")
        return

    dataset_download_urls = {
        "2halfcheetah": "https://s3.kao.instadeep.io/offline-marl-dataset/vaults/2halfcheetah2.zip",
        "2ant": "https://s3.kao.instadeep.io/offline-marl-dataset/omiga/2ant.zip",
        "6halfcheetah": "https://s3.kao.instadeep.io/offline-marl-dataset/omiga/6halfcheetah.zip",
        "3hopper": "https://s3.kao.instadeep.io/offline-marl-dataset/omiga/3hopper.zip"
    }

    dataset_download_url = dataset_download_urls[scenario]

    os.makedirs(f"{dataset_base_dir}/tmp/", exist_ok=True)
    os.makedirs(f"{dataset_base_dir}/mamujoco/", exist_ok=True)

    zip_file_path = f"{dataset_base_dir}/tmp/tmp_dataset.zip"

    if scenario == "2halfcheetah":
        extraction_path = f"{dataset_base_dir}/mamujoco/{scenario}.vlt"
    else:
        extraction_path = f"{dataset_base_dir}/mamujoco/"

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
    
def gather(values: Tensor, indices: Tensor, axis: int = -1, keepdims: bool = False) -> Tensor:
    one_hot_indices = tf.one_hot(indices, depth=values.shape[axis])
    if len(values.shape) > 4:  # we have extra dim for distributional q-learning
        one_hot_indices = tf.expand_dims(one_hot_indices, axis=-1)
    gathered_values = tf.reduce_sum(values * one_hot_indices, axis=axis, keepdims=keepdims)
    return gathered_values


def switch_two_leading_dims(x: Tensor) -> Tensor:
    trailing_perm = []
    for i in range(2, len(x.shape)):
        trailing_perm.append(i)
    x: Tensor = tf.transpose(x, perm=[1, 0, *trailing_perm])
    return x


def merge_batch_and_agent_dim_of_time_major_sequence(x: Tensor) -> Tensor:
    T, B, N = x.shape[:3]  # assume time major
    trailing_dims = x.shape[3:]
    x = tf.reshape(x, shape=(T, B * N, *trailing_dims))  # type: ignore
    return x


def merge_time_batch_and_agent_dim(x: Tensor) -> Tensor:
    T, B, N = x.shape[:3]  # assume time major
    trailing_dims = x.shape[3:]
    x = tf.reshape(x, shape=(T * B * N, *trailing_dims))  # type: ignore
    return x


def expand_time_batch_and_agent_dim_of_time_major_sequence(
    x: Tensor, T: int, B: int, N: int
) -> Tensor:
    TNB = x.shape[:1]  # assume time major
    assert TNB == T * B * N  # type: ignore
    trailing_dims = x.shape[1:]
    x = tf.reshape(x, shape=(T, B, N, *trailing_dims))  # type: ignore
    return x


def expand_batch_and_agent_dim_of_time_major_sequence(x: Tensor, B: int, N: int) -> Tensor:
    T, NB = x.shape[:2]  # assume time major
    assert NB == B * N
    trailing_dims = x.shape[2:]
    x = tf.reshape(x, shape=(T, B, N, *trailing_dims))  # type: ignore
    return x


def concat_agent_id_to_obs(obs: Tensor, agent_id: int, N: int, at_back=False) -> Tensor:
    is_vector_obs = len(obs.shape) == 1

    if is_vector_obs:
        agent_id = tf.one_hot(agent_id, depth=N)
    else:
        h, w = obs.shape[:2]
        agent_id = tf.zeros((h, w, 1), "float32") + (agent_id / N) + 1 / (2 * N)

    if not is_vector_obs and len(obs.shape) == 2:  # if no channel dim
        obs = tf.expand_dims(obs, axis=-1)

    if at_back:
        obs: Tensor = tf.concat([obs, agent_id], axis=-1)
    else:    
        obs: Tensor = tf.concat([agent_id, obs], axis=-1)

    return obs


def unroll_rnn(rnn_network: Module, inputs: Tensor, resets: Tensor) -> Tensor:
    T, B = inputs.shape[:2]

    outputs = []
    hidden_state = rnn_network.initial_state(B)  # type: ignore
    for i in range(T):  # type: ignore
        output, hidden_state = rnn_network(inputs[i], hidden_state)  # type: ignore
        outputs.append(output)

        hidden_state = (
            tf.where(
                tf.cast(tf.expand_dims(resets[i], axis=-1), "bool"),
                rnn_network.initial_state(B)[0],  # type: ignore
                hidden_state[0],
            ),
        )  # hidden state wrapped in tuple

    return tf.stack(outputs, axis=0)  # type: ignore


def batch_concat_agent_id_to_obs(obs: Tensor) -> Tensor:
    B, T, N = obs.shape[:3]  # batch size, timedim, num_agents
    is_vector_obs = len(obs.shape) == 4

    agent_ids = []
    for i in range(N):  # type: ignore
        if is_vector_obs:
            agent_id = tf.one_hot(i, depth=N)
        else:
            h, w = obs.shape[3:5]
            agent_id = tf.zeros((h, w, 1), "float32") + (i / N) + 1 / (2 * N)  # type: ignore
        agent_ids.append(agent_id)
    agent_ids = tf.stack(agent_ids, axis=0)

    # Repeat along time dim
    agent_ids = tf.stack([agent_ids] * T, axis=0)  # type: ignore

    # Repeat along batch dim
    agent_ids = tf.stack([agent_ids] * B, axis=0)  # type: ignore

    if not is_vector_obs and len(obs.shape) == 5:  # if no channel dim
        obs = tf.expand_dims(obs, axis=-1)

    obs = tf.concat([agent_ids, obs], axis=-1)

    return obs