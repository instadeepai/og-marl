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

from setuptools import find_packages, setup

dataset_requirements = [
    "tensorflow~=2.8.0",
    "tensorflow-io==0.27.0",
    "dm_env",
    "pandas",
    "seaborn",
    "protobuf==3.20.*"
]

baseline_requirements = [
    "id-mava[reverb,tf]==0.1.3",
    "neptune-client==0.16.2",
    "wandb"
]

setup(
    name='OG-MARL',
    version='0.0.1',
    author='Anon Anonymous',
    author_email='anon@anonymous.com',
    packages=find_packages(),
    url='',
    license='',
    description='Off-the-Grid MARL: a Framework for Dataset Generation with Baselines for Cooperative Offline Multi-Agent Reinforcement Learning',
    long_description="",
    install_requires=[],
    extras_require={
        "datasets": dataset_requirements,
        "baselines": baseline_requirements
        
    }
)