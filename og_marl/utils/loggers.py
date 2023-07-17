from acme.utils.loggers.base import Logger
import wandb
import neptune.new as neptune
import os
import warnings
from typing import Dict, List, Mapping, Any
import numpy as np
from datetime import datetime

LoggingData = Mapping[str, Any]


class WandbSweppLogger(Logger):
    def __init__(self, label):

        self._label = label

        wandb.init()

    def write(self, logs):

        if self._label == "evaluator":
            wandb.log(logs)

    def close(self):
        if self._label == "evaluator":
            wandb.finish()


class WandbLogger(Logger):
    def __init__(
        self,
        label="default",
        project="default_project",
        notes="",
        tags=["default"],
        entity="arkalim",
    ):
        self._label = label
        if label == "evaluator":
            name = str(datetime.now())
            wandb.init(
                name=name, project=project, notes=notes, tags=tags, entity=entity
            )

    def write(self, logs):
        if self._label == "evaluator":
            wandb.log(logs)

    def close(self):
        if self._label == "evaluator":
            wandb.finish()


def format_key(key: str) -> str:
    """Internal function for formatting keys in Tensorboard format."""
    return key.title().replace("_", "")


class NeptuneLogger(Logger):
    def __init__(
        self,
        label: str,
        project: str,
        name: str,
        tag: str,
        exp_params: Dict = {},
        # Logging hardware metrics fails with nvidia migs
        capture_hardware_metrics: bool = True,
    ):
        self._label = label
        self._name = name
        self._exp_params = exp_params
        self._api_token = os.getenv("NEPTUNE_API_TOKEN")
        self._project = project
        self._tag = tag
        self._run = neptune.init(
            name=self._name,
            monitoring_namespace=f"monitoring/{self._label}",
            api_token=self._api_token,
            project=self._project,
            tags=self._tag,
            capture_hardware_metrics=capture_hardware_metrics,
        )
        self._run["params"] = self._exp_params  # type: ignore

    def write(self, values: LoggingData) -> None:
        try:
            if isinstance(values, dict):
                for key, value in values.items():
                    is_scalar_array = hasattr(value, "shape") and (
                        value.shape == [1] or value.shape == 1 or value.shape == ()
                    )
                    if np.isscalar(value) or is_scalar_array:
                        self.scalar_summary(key, value)
                    elif hasattr(value, "shape"):
                        self.histogram_summary(key, value)
                    elif isinstance(value, dict):
                        flatten_dict = self._flatten_dict(
                            parent_key=key, dict_info=value
                        )
                        self.write(flatten_dict)
                    elif isinstance(value, tuple) or isinstance(value, list):
                        for index, elements in enumerate(value):
                            self.write({f"{key}_info_{index}": elements})
                    else:
                        warnings.warn(
                            f"Unable to log: {key}, unknown type: {type(value)}"
                        )
            elif isinstance(values, tuple) or isinstance(value, list):
                for elements in values:
                    self.write(elements)
            else:
                warnings.warn(f"Unable to log: {values}, unknown type: {type(values)}")
        except Exception as ex:
            warnings.warn(
                f"Unable to log: {key}, type: {type(value)} , value: {value}"
                + f"ex: {ex}"
            )

    def scalar_summary(self, key: str, value: float) -> None:
        if self._run:
            self._run[f"{self._label}/{format_key(key)}"].log(value)

    def dict_summary(self, key: str, value: Dict) -> None:
        dict_info = self._flatten_dict(parent_key=key, dict_info=value)
        for (k, v) in dict_info.items():
            self.scalar_summary(k, v)

    def histogram_summary(self, key: str, value: np.ndarray) -> None:
        return

    # Flatten dict, adapted from
    # https://stackoverflow.com/questions/6027558/flatten-nested-dictionaries-compressing-keys
    # Converts {'agent_0': {'critic_loss': 0.1, 'policy_loss': 0.2},...}
    #   to  {'agent_0_critic_loss':0.1,'agent_0_policy_loss':0.1 ,...}
    def _flatten_dict(
        self, parent_key: str, dict_info: Dict, sep: str = "_"
    ) -> Dict[str, float]:
        items: List = []
        for k, v in dict_info.items():
            k = str(k)
            if parent_key:
                new_key = parent_key + sep + k
            else:
                new_key = k
            if isinstance(v, dict):
                items.extend(
                    self._flatten_dict(parent_key=new_key, dict_info=v, sep=sep).items()
                )
            else:
                items.append((new_key, v))
        return dict(items)

    def close(self) -> None:
        self._run = None


def make_logger_base(logger, env_name, project="", tags=[]):
    now = datetime.now()
    date_time = now.strftime("%d-%m-%Y %H-%M-%S")
    if logger.lower() == "wandb":
        external_logger = WandbLogger
        tags.append(env_name)
        external_logger_kwargs = {
            "project": project,
            "notes": "",
            "entity": "off-the-grid-marl-team",
            "tags": tags,
        }
        to_tensorboard = False
    elif logger.lower() == "neptune":
        external_logger = NeptuneLogger
        external_logger_kwargs = {
            "name": date_time,
            "project": "Instadeep/offline-marl",
            "tag": f"{date_time} {tags} {env_name}",
        }
        to_tensorboard = False
    else:  # tensorboard
        external_logger = None
        external_logger_kwargs = {}
        to_tensorboard = True
    return external_logger, external_logger_kwargs, to_tensorboard
