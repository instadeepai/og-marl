# type: ignore

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


from og_marl.environments.base import BaseEnvironment


def get_environment(env_name: str, scenario: str) -> BaseEnvironment:  # noqa: C901
    if env_name == "smac_v1":
        from og_marl.environments.smacv1 import SMACv1

        return SMACv1(scenario)
    elif env_name == "smac_v2":
        from og_marl.environments.smacv2 import SMACv2

        return SMACv2(scenario)
    elif env_name == "mamujoco":
        from og_marl.environments.old_mamujoco import MAMuJoCo

        return MAMuJoCo(scenario)
    elif scenario == "pursuit":
        from og_marl.environments.pursuit import Pursuit

        return Pursuit()
    elif scenario == "coop_pong":
        from og_marl.environments.coop_pong import CooperativePong

        return CooperativePong()
    elif env_name == "gymnasium_mamujoco":
        from og_marl.environments.gymnasium_mamujoco import MAMuJoCo

        return MAMuJoCo(scenario)
    elif env_name == "flatland":
        from og_marl.environments.flatland_wrapper import Flatland

        return Flatland(scenario)
    elif env_name == "voltage_control":
        from og_marl.environments.voltage_control import VoltageControlEnv

        return VoltageControlEnv()
    elif env_name == "smax":
        from og_marl.environments.jaxmarl_smax import SMAX

        return SMAX(scenario)
    elif env_name == "lbf":
        from og_marl.environments.jumanji_lbf import JumanjiLBF

        return JumanjiLBF(scenario)
    elif env_name == "rware":
        from og_marl.environments.jumanji_rware import JumanjiRware

        return JumanjiRware(scenario)
    else:
        raise ValueError("Environment not recognised.")
