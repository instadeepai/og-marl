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

import json
import os
import time
from typing import Any, Dict, List, Optional

from chex import Numeric

import wandb


class BaseLogger:
    def write(self, logs: Dict[str, Numeric], force: bool = False) -> None:
        raise NotImplementedError

    def close(self) -> None:
        return


class TerminalLogger(BaseLogger):
    def __init__(
        self,
        log_every: int = 2,  # seconds
    ):
        self._log_every = log_every
        self._ctr = 0
        self._last_log = time.time()

    def write(self, logs: Dict[str, Numeric], force: bool = False) -> None:
        if time.time() - self._last_log > self._log_every or force:
            for key, log in logs.items():
                print(f"{key}: {float(log)} |", end=" ")
            print()

            if not force:
                self._last_log = time.time()

        self._ctr += 1


class WandbLogger(BaseLogger):
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

    def write(self, logs: Dict[str, Numeric], force: bool = False) -> None:
        if time.time() - self._last_log > self._log_every or force:
            wandb.log(logs)

            # for key, log in logs.items():
            #     print(f"{key}: {float(log)} |", end=" ")
            # print()

            if not force:
                self._last_log = time.time()

        self._ctr += 1

        if self._ctr % 1000 == 0:
            print(self._ctr)

    def close(self) -> None:
        wandb.finish()


class JsonWriter:

    """Writer to create json files for reporting experiment results according to marl-eval

    Follows conventions from https://github.com/instadeepai/marl-eval/tree/main#usage-
    This writer was adapted from the implementation found in BenchMARL. For the original
    implementation please see https://tinyurl.com/2t6fy548

    Args:
    ----
        path (str): where to write the file
        algorithm_name (str): algorithm name
        task_name (str): task name
        environment_name (str): environment name
        seed (int): random seed of the experiment

    """

    def __init__(
        self,
        path: str,
        algorithm_name: str,
        task_name: str,
        environment_name: str,
        seed: int,
        file_name: str = "metrics.json",
        save_to_wandb: bool = False,
    ):
        self.path = path
        self.file_name = file_name
        self.run_data: Dict[str, Any] = {"absolute_metrics": {}}
        self._save_to_wandb = save_to_wandb

        # If the file already exists, load it
        if os.path.isfile(f"{self.path}/{self.file_name}"):
            with open(f"{self.path}/{self.file_name}", "r") as f:
                data = json.load(f)
        else:
            # Create the logging directory if it doesn't exist
            os.makedirs(self.path, exist_ok=True)
            data = {}

        # Merge the existing data with the new data
        self.data = data
        if environment_name not in self.data:
            self.data[environment_name] = {}
        if task_name not in self.data[environment_name]:
            self.data[environment_name][task_name] = {}
        if algorithm_name not in self.data[environment_name][task_name]:
            self.data[environment_name][task_name][algorithm_name] = {}
        self.data[environment_name][task_name][algorithm_name][f"seed_{seed}"] = self.run_data

        with open(f"{self.path}/{self.file_name}", "w") as f:
            json.dump(self.data, f, indent=4)

    def write(
        self,
        timestep: int,
        key: str,
        value: float,
        evaluation_step: Optional[int] = None,
    ) -> None:
        """Writes a step to the json reporting file

        Args:
        ----
            timestep (int): the current environment timestep
            key (str): the metric that should be logged
            value (str): the value of the metric that should be logged
            evaluation_step (int): the evaluation step

        """
        logging_prefix, *metric_key = key.split("/")
        metric_key = "/".join(metric_key)

        metrics = {metric_key: [value]}

        if logging_prefix == "evaluator":
            step_metrics = {"step_count": timestep}
            step_metrics.update(metrics)  # type: ignore
            step_str = f"step_{evaluation_step}"
            if step_str in self.run_data:
                self.run_data[step_str].update(step_metrics)
            else:
                self.run_data[step_str] = step_metrics

        # Store the absolute metrics
        if logging_prefix == "absolute":
            self.run_data["absolute_metrics"].update(metrics)

        with open(f"{self.path}/{self.file_name}", "w") as f:
            json.dump(self.data, f, indent=4)

    def close(self) -> None:
        if self._save_to_wandb:
            wandb.save(f"{self.path}/{self.file_name}")
