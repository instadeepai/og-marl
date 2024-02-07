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
from functools import partial


def get_system(system_name, environment, logger, **kwargs):
    if system_name == "bc":
        from og_marl.jax.systems.bc import train_bc_system

        return partial(train_bc_system, environment, logger)
    elif system_name == "maicq":
        from og_marl.jax.systems.maicq import train_maicq_system

        return partial(train_maicq_system, environment, logger)
    else:
        raise ValueError("System name not recognised.")
