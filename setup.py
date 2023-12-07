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

setup(
    name='OG-MARL',
    version='0.0.2',
    author='Claude Formanek',
    author_email='c.formanek@instadeep.com',
    packages=find_packages(),
    url='https://sites.google.com/view/og-marl',
    license='',
    description='Off-the-Grid MARL: a Framework for Dataset \
        Generation with Baselines for Cooperative Offline \
        Multi-Agent Reinforcement Learning',
    long_description="",
    install_requires=[
        "numpy",
        "dm_tree",
        "tensorflow==2.8.*",
        "tensorflow_io",
        "tensorflow_probability==0.16.*",
        "dm_sonnet",
        "wandb",
        "cpprb",
        "absl-py",
        "gymnasium",
        "requests"
    ],
    extras_require={
        'jax': ['flashbax', 'optax', "jax", "flax", "orbax-checkpoint"],
    }

)
