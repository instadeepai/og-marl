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