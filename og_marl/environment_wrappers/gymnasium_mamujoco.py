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

from typing import Any, Dict

import gymnasium_robotics
import numpy as np

from og_marl.environment_wrappers.base import ResetReturn, StepReturn, BaseEnvironment
from og_marl.environment_wrappers.wrappers import Dtype, PadObsandActs


def get_env_config(scenario: str) -> Dict[str, Any]:
    """Helper method to get env_args."""
    env_args: Dict[str, Any] = {
        "agent_obsk": 1,
    }
    if scenario.lower() == "halfcheetah":
        env_args["scenario"] = "HalfCheetah"
        env_args["agent_conf"] = None
    elif scenario.lower() == "2halfcheetah":
        env_args["scenario"] = "HalfCheetah"
        env_args["agent_conf"] = "2x3"
    elif scenario.lower() == "6halfcheetah":
        env_args["scenario"] = "HalfCheetah"
        env_args["agent_conf"] = "6x1"
    elif scenario.lower() == "2ant":
        env_args["scenario"] = "Ant"
        env_args["agent_conf"] = "2x4"
    elif scenario.lower() == "2humanoid":
        env_args["scenario"] = "Humanoid"
        env_args["agent_conf"] = "9|8"
    elif scenario.lower() == "4ant":
        env_args["scenario"] = "Ant"
        env_args["agent_conf"] = "4x2"
    elif scenario.lower() == "3hopper":
        env_args["scenario"] = "Hopper"
        env_args["agent_conf"] = "3x1"
    elif scenario.lower() == "2walker":
        env_args["scenario"] = "Walker2d"
        env_args["agent_conf"] = "2x3"
    elif scenario.lower() == "2walker":
        env_args["scenario"] = "Reacher"
        env_args["agent_conf"] = "2x1"
    return env_args


class WrappedGymnasiumMAMuJoCo(BaseEnvironment):
    def __init__(self, scenario: str, seed=None):
        self.environment = GymnasiumMAMuJoCo(scenario, seed)
        self.wrapped_environment = PadObsandActs(Dtype(self.environment, "float32"))
        self.num_actions = self.wrapped_environment._num_actions
        self.agents = self.environment.agents

    def reset(self) -> ResetReturn:
        return self.wrapped_environment.reset()

    def step(self, actions: Dict[str, np.ndarray]) -> StepReturn:
        return self.wrapped_environment.step(actions)


class GymnasiumMAMuJoCo(BaseEnvironment):

    """Environment wrapper Multi-Agent MuJoCo."""

    def __init__(self, scenario: str, seed=None):
        env_config = get_env_config(scenario)
        self.environment = gymnasium_robotics.mamujoco_v0.parallel_env(**env_config)
        self.agents = self.environment.agents
        self.num_actions = list(self.environment.action_spaces.values())[0].shape[0]

    def reset(self) -> ResetReturn:
        observations, _ = self.environment.reset()

        info = {"state": self.environment.state().astype("float32")}

        return observations, info

    def step(self, actions: Dict[str, np.ndarray]) -> StepReturn:
        observations, rewards, terminals, trunctations, _ = self.environment.step(actions)

        info = {"state": self.environment.state().astype("float32")}

        return observations, rewards, terminals, trunctations, info

    def __getattr__(self, name: str) -> Any:
        """Expose any other attributes of the underlying environment."""
        if hasattr(self.__class__, name):
            return self.__getattribute__(name)
        else:
            return getattr(self.environment, name)
