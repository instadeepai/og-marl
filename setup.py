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
    name="OG-MARL",
    version="0.0.2",
    author="Claude Formanek",
    author_email="c.formanek@instadeep.com",
    packages=find_packages(),
    url="https://sites.google.com/view/og-marl",
    license="",
    description="Off-the-Grid MARL: Datasets and Baselines for Offline \
        Multi-Agent Reinforcement Learning",
    long_description="",
    install_requires=[
        "numpy",
        "dm_tree",
        "tensorflow[and-cuda]",
        "tensorflow_io",
        "tensorflow-probability[tf]",
        "dm_sonnet",
        "wandb",
        "absl-py",
        "gymnasium",
        "requests",
        "jax",
        "matplotlib",
        "seaborn",
        # "flashbax==0.1.2", # install post
    ],
    extras_require={
        "jax": ["flashbax==0.1.2", "optax", "jax", "flax", "orbax-checkpoint"],
    },
)
