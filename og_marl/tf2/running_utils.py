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

def get_environment(env_name, scenario):
    if env_name == "smac_v1":
        from og_marl.environments.smacv1 import SMACv1
        return SMACv1(scenario)
    elif env_name == "smac_v2":
        from og_marl.environments.smacv2 import SMACv2
        return SMACv2(scenario)
    elif env_name == "mamujoco":
        from og_marl.environments.mamujoco import MAMuJoCo
        return MAMuJoCo(scenario)
    elif env_name == "gymnasium_mamujoco":
        from og_marl.environments.gymnasium_mamujoco import MAMuJoCo
        return MAMuJoCo(scenario)
    
def get_system(system_name, environment, logger, **kwargs) :
    if system_name == "idrqn":
        from og_marl.tf2.systems.idrqn import IDRQNSystem
        return IDRQNSystem(environment, logger, **kwargs)
    elif system_name == "idrqn+cql":
        from og_marl.tf2.systems.idrqn_cql import IDRQNCQLSystem
        return IDRQNCQLSystem(environment, logger, **kwargs)
    elif system_name == "idrqn+bcq":
        from og_marl.tf2.systems.idrqn_bcq import IDRQNBCQSystem
        return IDRQNBCQSystem(environment, logger, **kwargs)
    elif system_name == "qmix":
        from og_marl.tf2.systems.qmix import QMIXSystem
        return QMIXSystem(environment, logger, **kwargs)
    elif system_name == "qmix+cql":
        from og_marl.tf2.systems.qmix_cql import QMIXCQLSystem
        return QMIXCQLSystem(environment, logger, **kwargs)
    elif system_name == "qmix+bcq":
        from og_marl.tf2.systems.qmix_bcq import QMIXBCQSystem
        return QMIXBCQSystem(environment, logger, **kwargs)
    elif system_name == "omar":
        from og_marl.tf2.systems.omar import OMARSystem
        return OMARSystem(environment, logger, **kwargs)