# Copyright 2022 InstaDeep Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import abc
import logging
import os
import zipfile
from datetime import datetime
from enum import Enum
from typing import ClassVar, Dict, List, Union

import jax
import neptune
import numpy as np
from colorama import Fore, Style 
from jax import tree
from jax.typing import ArrayLike
# from marl_eval.json_tools import JsonLogger as MarlEvalJsonLogger
from neptune.utils import stringify_unsupported
from omegaconf import DictConfig
from pandas.io.json._normalize import _simple_json_normalize as flatten_dict
# from tensorboard_logger import configure, log_value


class LogEvent(Enum):
    ACT = "actor"
    TRAIN = "trainer"
    EVAL = "evaluator"
    ABSOLUTE = "absolute"
    MISC = "misc"


class MavaLogger:
    """The main logger for Mava systems.

    Thin wrapper around the MultiLogger that is able to describe arrays of metrics
    and calculate environment specific metrics if required (e.g winrate).
    """

    def __init__(self, config: DictConfig) -> None:
        self.logger: BaseLogger = _make_multi_logger(config)
        self.cfg = config

    def log(self, metrics: Dict, t: int, t_eval: int, event: LogEvent) -> None:
        """Log a dictionary metrics at a given timestep.

        Args:
        ----
            metrics (Dict): dictionary of metrics to log.
            t (int): the current timestep.
            t_eval (int): the number of previous evaluations.
            event (LogEvent): the event that the metrics are associated with.

        """
        # Ideally we want to avoid special metrics like this as much as possible.
        # Might be better to calculate this outside as we want to keep the number of these
        # if statements to a minimum.
        if "won_episode" in metrics:
            metrics = self.calc_winrate(metrics, event)

        if event == LogEvent.TRAIN:
            # We only want to log mean losses, max/min/std don't matter.
            metrics = tree.map(np.mean, metrics)
        else:
            # {metric1_name: [metrics], metric2_name: ...} ->
            # {metric1_name: {mean: metric, max: metric, ...}, metric2_name: ...}
            metrics = tree.map(describe, metrics)

        self.logger.log_dict(metrics, t, t_eval, event)

    def calc_winrate(self, episode_metrics: Dict, event: LogEvent) -> Dict:
        """Log the win rate of the environment's episodes."""
        # Get the number of episodes used to evaluate.
        if event == LogEvent.ABSOLUTE:
            # To measure the absolute metric, we evaluate the best policy
            # found across training over 10 times the evaluation episodes.
            # For more details on the absolute metric please see:
            # https://arxiv.org/abs/2209.10485.
            n_episodes = self.cfg.system.num_eval_episodes * 10
        else:
            n_episodes = self.cfg.system.num_eval_episodes

        # Calculate the win rate.
        n_won_episodes: int = np.sum(episode_metrics["won_episode"])
        win_rate = (n_won_episodes / n_episodes) * 100

        episode_metrics["win_rate"] = win_rate
        episode_metrics.pop("won_episode")

        return episode_metrics

    def stop(self) -> None:
        """Stop the logger."""
        self.logger.stop()


class BaseLogger(abc.ABC):
    @abc.abstractmethod
    def __init__(self, cfg: DictConfig, unique_token: str) -> None:
        pass

    @abc.abstractmethod
    def log_stat(self, key: str, value: float, step: int, eval_step: int, event: LogEvent) -> None:
        """Log a single metric."""
        raise NotImplementedError

    def log_dict(self, data: Dict, step: int, eval_step: int, event: LogEvent) -> None:
        """Log a dictionary of metrics."""
        # in case the dict is nested, flatten it.
        data = flatten_dict(data, sep="/")

        for key, value in data.items():
            self.log_stat(key, value, step, eval_step, event)

    def stop(self) -> None:
        """Stop the logger."""
        return None


class MultiLogger(BaseLogger):
    """Logger that can log to multiple loggers at oncce."""

    def __init__(self, loggers: List[BaseLogger]) -> None:
        self.loggers = loggers

    def log_stat(self, key: str, value: float, step: int, eval_step: int, event: LogEvent) -> None:
        for logger in self.loggers:
            logger.log_stat(key, value, step, eval_step, event)

    def log_dict(self, data: Dict, step: int, eval_step: int, event: LogEvent) -> None:
        for logger in self.loggers:
            logger.log_dict(data, step, eval_step, event)

    def stop(self) -> None:
        for logger in self.loggers:
            logger.stop()


class NeptuneLogger(BaseLogger):
    """Logger for neptune.ai."""

    def __init__(self, cfg: DictConfig, unique_token: str) -> None:
        tags = list(cfg.logger.kwargs.neptune_tag)
        project = cfg.logger.kwargs.neptune_project
        mode = (
            "async" if cfg.arch.architecture_name == "anakin" else "sync"
        )  # async logging leads to deadlocks in sebulba

        self.logger = neptune.init_run(project=project, tags=tags, mode=mode)

        self.logger["config"] = stringify_unsupported(cfg)
        self.detailed_logging = cfg.logger.kwargs.detailed_neptune_logging

        # Store json path for uploading json data to Neptune.
        json_exp_path = get_logger_path(cfg, "json")
        self.json_file_path = os.path.join(
            cfg.logger.base_exp_path, f"{json_exp_path}/{unique_token}/metrics.json"
        )
        self.unique_token = unique_token
        self.upload_json_data = cfg.logger.kwargs.upload_json_data
        self.upload_checkpoint = cfg.logger.checkpointing.upload_model
        self.model_name = cfg.logger.system_name
        self.checkpoint_uid = cfg.logger.checkpointing.save_args.checkpoint_uid

    def log_stat(self, key: str, value: float, step: int, eval_step: int, event: LogEvent) -> None:
        # Main metric if it's the mean of a list of metrics (ends with '/mean')
        # or it's a single metric doesn't contain a '/'.
        is_main_metric = "/" not in key or key.endswith("/mean")
        # If we're not detailed logging (logging everything) then make sure it's a main metric.
        if not self.detailed_logging and not is_main_metric:
            return

        value = value.item() if isinstance(value, (jax.Array, np.ndarray)) else value
        self.logger[f"{event.value}/{key}"].log(value, step=step)

    def stop(self) -> None:
        if self.upload_json_data:
            self._zip_and_upload_json()
        if self.upload_checkpoint:
            self._upload_checkpoint_to_neptune(
                checkpoint_rel_dir="checkpoints",
                model_name=self.model_name,
                checkpoint_uid=self.checkpoint_uid,
            )
        self.logger.stop()

    def _zip_and_upload_json(self) -> None:
        # Create the zip file path by replacing '.json' with '.zip'
        zip_file_path = self.json_file_path.rsplit(".json", 1)[0] + ".zip"

        # Create a zip file containing the specified JSON file
        with zipfile.ZipFile(zip_file_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(self.json_file_path)

        self.logger[f"metrics/metrics_{self.unique_token}"].upload(zip_file_path)

    def _upload_checkpoint_to_neptune(
        self, checkpoint_rel_dir: str, model_name: str, checkpoint_uid: str
    ) -> None:
        base_checkpoint_directory = os.path.join(os.getcwd(), checkpoint_rel_dir, model_name)

        # If there is no uid set, get the most recent checkpoint in base_checkpoint_directory
        if not checkpoint_uid:
            checkpoint_uid = sorted(os.listdir(base_checkpoint_directory), reverse=True)[0]

        checkpoint_folder_path = os.path.join(base_checkpoint_directory, checkpoint_uid)

        # zip the entire checkpoint folder under checkpoint_folder_path
        checkpoint_zip_file_path = checkpoint_folder_path + ".zip"
        with zipfile.ZipFile(checkpoint_zip_file_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(checkpoint_folder_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    # Compute the archive name by removing the base directory path
                    # This includes the 'checkpoint_uid' directory in the arcname
                    arcname = os.path.relpath(file_path, start=base_checkpoint_directory)
                    zipf.write(file_path, arcname=arcname)

        self.logger["model_checkpoint"].upload(checkpoint_zip_file_path, wait=True)
        os.remove(checkpoint_zip_file_path)

    @staticmethod
    def download_checkpoint_from_neptune(
        checkpoint_rel_dir: str, model_name: str, neptune_run_name: str
    ) -> None:
        # Construct the base directory where the checkpoint will be saved
        base_checkpoint_directory = os.path.join(os.getcwd(), checkpoint_rel_dir, model_name)

        # Ensure the base directory exists
        os.makedirs(base_checkpoint_directory, exist_ok=True)

        # Initialize the Neptune run in read-only mode
        with neptune.init_run(
            project="Instadeep/Offline-Sable", with_id=neptune_run_name, mode="read-only"
        ) as run:
            # Download the model checkpoint zip file to the base directory
            run["model_checkpoint"].download(destination=base_checkpoint_directory)

            # Iterate over all files in the base directory to find zip files
            for root, _, files in os.walk(base_checkpoint_directory):
                for file in files:
                    if file.endswith(".zip"):
                        zip_file_path = os.path.join(root, file)

                        # Unzip the file into the current directory
                        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
                            zip_ref.extractall(root)

                        # Optionally, delete the zip file after extraction
                        os.remove(zip_file_path)

class ConsoleLogger(BaseLogger):
    """Logger for writing to stdout."""

    _EVENT_COLOURS: ClassVar[Dict[LogEvent, str]] = {
        LogEvent.TRAIN: Fore.MAGENTA,
        LogEvent.EVAL: Fore.GREEN,
        LogEvent.ABSOLUTE: Fore.BLUE,
        LogEvent.ACT: Fore.CYAN,
        LogEvent.MISC: Fore.YELLOW,
    }

    def __init__(self, cfg: DictConfig, unique_token: str) -> None:
        self.logger = logging.getLogger()

        self.logger.handlers = []

        ch = logging.StreamHandler()
        formatter = logging.Formatter(f"{Fore.CYAN}{Style.BRIGHT}%(message)s", "%H:%M:%S")
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        # Set to info to suppress debug outputs.
        self.logger.setLevel("INFO")

    def log_stat(self, key: str, value: float, step: int, eval_step: int, event: LogEvent) -> None:
        colour = self._EVENT_COLOURS[event]

        # Replace underscores with spaces and capitalise keys.
        key = key.replace("_", " ").capitalize()
        self.logger.info(
            f"{colour}{Style.BRIGHT}{event.value.upper()} - {key}: {value:.3f}{Style.RESET_ALL}"
        )

    def log_dict(self, data: Dict, step: int, eval_step: int, event: LogEvent) -> None:
        # in case the dict is nested, flatten it.
        data = flatten_dict(data, sep=" ")

        colour = self._EVENT_COLOURS[event]
        # Replace underscores with spaces and capitalise keys.
        keys = [k.replace("_", " ").capitalize() for k in data.keys()]
        # Round values to 3 decimal places if they are floats.
        values = []
        for value in data.values():
            value = value.item() if isinstance(value, jax.Array) else value
            values.append(f"{value:.3f}" if isinstance(value, float) else str(value))
        log_str = " | ".join([f"{k}: {v}" for k, v in zip(keys, values, strict=True)])

        self.logger.info(
            f"{colour}{Style.BRIGHT}{event.value.upper()} - {log_str}{Style.RESET_ALL}"
        )


def _make_multi_logger(cfg: DictConfig) -> BaseLogger:
    """Creates a MultiLogger given a config"""
    loggers: List[BaseLogger] = []
    unique_token = datetime.now().strftime("%Y%m%d%H%M%S")

    if (
        cfg.logger.use_neptune
        and cfg.logger.use_json
        and cfg.logger.kwargs.upload_json_data
        and cfg.logger.kwargs.json_path
    ):
        raise ValueError(
            "Cannot upload json data to Neptune when `json_path` is set in the base logger config. "
            "This is because each subsequent run will create a larger json file which will use "
            "unnecessary storage. Either set `upload_json_data: false` if you don't want to "
            "upload your json data but store a large file locally or set `json_path: ~` in "
            "the base logger config."
        )

    if cfg.logger.use_neptune:
        loggers.append(NeptuneLogger(cfg, unique_token))
    if cfg.logger.use_console:
        loggers.append(ConsoleLogger(cfg, unique_token))

    return MultiLogger(loggers)


def get_logger_path(config: DictConfig, logger_type: str) -> str:
    """Helper function to create the experiment path."""
    return f"{logger_type}/{config.logger.system_name}"


def describe(x: ArrayLike) -> Union[Dict[str, ArrayLike], ArrayLike]:
    """Generate summary statistics for an array of metrics (mean, std, min, max)."""
    if not isinstance(x, (jax.Array, np.ndarray)) or x.ndim == 0:
        return x

    # np instead of jnp because we don't jit here
    return {"mean": np.mean(x), "std": np.std(x), "min": np.min(x), "max": np.max(x)}
