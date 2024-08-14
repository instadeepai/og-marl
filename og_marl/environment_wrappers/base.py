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

"""Base class for OG-MARL Environment Wrappers."""


from typing import Any, Dict, Tuple

import numpy as np

Observations = Dict[str, np.ndarray]
NextObservations = Observations
Rewards = Dict[str, np.ndarray]
Terminals = Dict[str, np.ndarray]
Truncations = Dict[str, np.ndarray]
Info = Dict[str, Any]

ResetReturn = Tuple[Observations, Info]
StepReturn = Tuple[NextObservations, Rewards, Terminals, Truncations, Info]


class BaseEnvironment:
    """Base environment class for OG-MARL."""

    def __init__(self) -> None:
        """Constructor."""
        pass

    def reset(self) -> ResetReturn:
        raise NotImplementedError

    def step(self, actions: Dict[str, np.ndarray]) -> StepReturn:
        raise NotImplementedError

    def get_stats(self) -> Dict:
        """Return extra stats to be logged.

        Returns:
        -------
            extra stats to be logged.

        """
        return {}

    def render(self) -> Any:
        """Return frame for rendering"""
        return np.zeros((10, 10, 3), "float32")

    def __getattr__(self, name: str) -> Any:
        """Expose any other attributes of the underlying environment.

        Args:
        ----
            name (str): attribute.

        Returns:
        -------
            Any: return attribute from env or underlying env.

        """
        if hasattr(self.__class__, name):
            return self.__getattribute__(name)
        else:
            return getattr(self._environment, name)
