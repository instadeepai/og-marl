from typing import Tuple

import gymnasium
import gymnasium as gym
import gymnasium.vector
import gymnasium.wrappers
import jaxmarl
import jax.numpy as jnp
import jumanji
# import matrax
from gigastep import ScenarioBuilder
from jaxmarl.environments.smax import Scenario
from jumanji.environments.routing.cleaner.generator import (
    RandomGenerator as CleanerRandomGenerator,
)
from jumanji.environments.routing.connector.generator import (
    RandomWalkGenerator as ConnectorRandomGenerator,
)
from jumanji.environments.routing.lbf.generator import (
    RandomGenerator as LbfRandomGenerator,
)
from jumanji.environments.routing.robot_warehouse.generator import (
    RandomGenerator as RwareRandomGenerator,
)
from omegaconf import DictConfig

from og_marl.baselines.jax_systems.env_types import MarlEnv
from og_marl.baselines.jax_systems.wrappers import (
    AgentIDWrapper,
    AutoResetWrapper,
    CleanerWrapper,
    ConnectorWrapper,
    GymAgentIDWrapper,
    GymRecordEpisodeMetrics,
    GymToJumanji,
    LbfWrapper,
    MabraxWrapper,
    MPEWrapper,
    RecordEpisodeMetrics,
    RwareWrapper,
    SmacWrapper,
    SmaxWrapper,
    UoeWrapper,
    VectorConnectorWrapper,
    async_multiagent_worker,
)

from og_marl.baselines.jax_systems.utils.t_maze import TMaze

# Registry mapping environment names to their generator and wrapper classes.
_jumanji_registry = {
    "RobotWarehouse": {"generator": RwareRandomGenerator, "wrapper": RwareWrapper},
    "LevelBasedForaging": {"generator": LbfRandomGenerator, "wrapper": LbfWrapper},
    "MaConnector": {"generator": ConnectorRandomGenerator, "wrapper": VectorConnectorWrapper},
    "VectorMaConnector": {
        "generator": ConnectorRandomGenerator,
        "wrapper": VectorConnectorWrapper,
    },
    "Cleaner": {"generator": CleanerRandomGenerator, "wrapper": CleanerWrapper},
}

# Registry mapping environment names directly to the corresponding wrapper classes.
# _matrax_registry = {"Matrax": MatraxWrapper}
_jaxmarl_registry = {"Smax": SmaxWrapper, "MaBrax": MabraxWrapper, "MPE": MPEWrapper}

_tmaze_registry = {"TMAZE": TMaze}

_gym_registry = {
    "RobotWarehouse": UoeWrapper,
    "LevelBasedForaging": UoeWrapper,
    "SMACLite": SmacWrapper,
}

MAP_NAME_TO_SCENARIO = {
    # name: (unit_types, n_allies, n_enemies, SMACv2 position generation, SMACv2 unit generation)
    "3m": Scenario(jnp.zeros((6,), dtype=jnp.uint8), 3, 3, False, False),
    "2s3z": Scenario(
        jnp.array([2, 2, 3, 3, 3] * 2, dtype=jnp.uint8), 5, 5, False, False
    ),
    "25m": Scenario(jnp.zeros((50,), dtype=jnp.uint8), 25, 25, False, False),
    "3s5z": Scenario(
        jnp.array(
            [
                2,
                2,
                2,
                3,
                3,
                3,
                3,
                3,
            ]
            * 2,
            dtype=jnp.uint8,
        ),
        8,
        8,
        False,
        False,
    ),
    "8m": Scenario(jnp.zeros((16,), dtype=jnp.uint8), 8, 8, False, False),
    "5m_vs_6m": Scenario(jnp.zeros((11,), dtype=jnp.uint8), 5, 6, False, False),
    "10m_vs_11m": Scenario(jnp.zeros((21,), dtype=jnp.uint8), 10, 11, False, False),
    "27m_vs_30m": Scenario(jnp.zeros((57,), dtype=jnp.uint8), 27, 30, False, False),
    "3s5z_vs_3s6z": Scenario(
        jnp.concatenate(
            [
                jnp.array([2, 2, 2, 3, 3, 3, 3, 3], dtype=jnp.uint8),
                jnp.array([2, 2, 2, 3, 3, 3, 3, 3, 3], dtype=jnp.uint8),
            ]
        ),
        8,
        9,
        False,
        False,
    ),
    "3s_vs_5z": Scenario(
        jnp.array([2, 2, 2, 3, 3, 3, 3, 3], dtype=jnp.uint8), 3, 5, False, False
    ),
    "6h_vs_8z": Scenario(
        jnp.array([5, 5, 5, 5, 5, 5, 3, 3, 3, 3, 3, 3, 3, 3], dtype=jnp.uint8),
        6,
        8,
        False,
        False,
    ),
    "smacv2_5_units": Scenario(jnp.zeros((10,), dtype=jnp.uint8), 5, 5, True, True),
    "smacv2_10_units": Scenario(jnp.zeros((20,), dtype=jnp.uint8), 10, 10, True, True),
    "smacv2_20_units": Scenario(jnp.zeros((40,), dtype=jnp.uint8), 20, 20, True, True),
    "smacv2_30_units": Scenario(jnp.zeros((60,), dtype=jnp.uint8), 30, 30, True, True),
    "smacv2_40_units": Scenario(jnp.zeros((80,), dtype=jnp.uint8), 40, 40, True, True),
    "smacv2_50_units": Scenario(jnp.zeros((100,), dtype=jnp.uint8), 50, 50, True, True),
}


def map_name_to_scenario(map_name):
    """maps from smac map names to a scenario array"""
    return MAP_NAME_TO_SCENARIO[map_name]

def add_extra_wrappers(
    train_env: MarlEnv, eval_env: MarlEnv, config: DictConfig
) -> Tuple[MarlEnv, MarlEnv]:
    # Disable the AgentID wrapper if the environment has implicit agent IDs.

    eval_env = AgentIDWrapper(eval_env)
    train_env = AgentIDWrapper(train_env)
    train_env = AutoResetWrapper(train_env)
    train_env = RecordEpisodeMetrics(train_env)
    eval_env = RecordEpisodeMetrics(eval_env)

    return train_env, eval_env

def make_tmaze(config: DictConfig, add_global_state: bool = False) -> Tuple[MarlEnv, MarlEnv]:
    train_env = TMaze(length=5, width=2, time_limit=20)
    eval_env = TMaze(length=5, width=2, time_limit=20)
    train_env, eval_env = add_extra_wrappers(train_env, eval_env, config)

    return train_env, eval_env


def make_jumanji_env(config: DictConfig, add_global_state: bool = False) -> Tuple[MarlEnv, MarlEnv]:
    """
    Create a Jumanji environments for training and evaluation.

    Args:
    ----
        env_name (str): The name of the environment to create.
        config (Dict): The configuration of the environment.
        add_global_state (bool): Whether to add the global state to the observation.

    Returns:
    -------
        A tuple of the environments.

    """
    # Config generator and select the wrapper.
    generator = _jumanji_registry[config.env.env_name]["generator"]
    generator = generator(**config.env.scenario.task_config)
    wrapper = _jumanji_registry[config.env.env_name]["wrapper"]

    # Create envs.
    env_config = {**config.env.kwargs, **config.env.scenario.env_kwargs}
    train_env = jumanji.make(config.env.scenario.name, generator=generator, **env_config)
    eval_env = jumanji.make(config.env.scenario.name, generator=generator, **env_config)
    train_env = wrapper(train_env, add_global_state=add_global_state)
    eval_env = wrapper(eval_env, add_global_state=add_global_state)

    train_env, eval_env = add_extra_wrappers(train_env, eval_env, config)
    return train_env, eval_env


def make_jaxmarl_env(config: DictConfig, add_global_state: bool = False) -> Tuple[MarlEnv, MarlEnv]:
    """
     Create a JAXMARL environment.

    Args:
    ----
        env_name (str): The name of the environment to create.
        config (Dict): The configuration of the environment.
        add_global_state (bool): Whether to add the global state to the observation.

    Returns:
    -------
        A JAXMARL environment.

    """
    kwargs = dict(config.env.kwargs)
    if "smax" in config.env.env_name.lower():
        kwargs["scenario"] = map_name_to_scenario(config.env.scenario.task_name)
    elif "mpe" in config.env.env_name.lower():
        kwargs.update(config.env.scenario.task_config)

    # Create jaxmarl envs.
    train_env: MarlEnv = _jaxmarl_registry[config.env.env_name](
        jaxmarl.make(config.env.scenario.name, **kwargs),
        add_global_state,
    )
    eval_env: MarlEnv = _jaxmarl_registry[config.env.env_name](
        jaxmarl.make(config.env.scenario.name, **kwargs),
        add_global_state,
    )

    train_env, eval_env = add_extra_wrappers(train_env, eval_env, config)

    return train_env, eval_env


# def make_matrax_env(config: DictConfig, add_global_state: bool = False) -> Tuple[MarlEnv, MarlEnv]:
#     """
#     Creates Matrax environments for training and evaluation.

#     Args:
#     ----
#         env_name: The name of the environment to create.
#         config: The configuration of the environment.
#         add_global_state: Whether to add the global state to the observation.

#     Returns:
#     -------
#         A tuple containing a train and evaluation Matrax environment.

#     """
#     # Select the Matrax wrapper.
#     # wrapper = _matrax_registry[config.env.scenario.name]

#     # Create envs.
#     task_name = config["env"]["scenario"]["task_name"]
#     # train_env = matrax.make(task_name, **config.env.kwargs)
#     # eval_env = matrax.make(task_name, **config.env.kwargs)
#     train_env = wrapper(train_env, add_global_state)
#     eval_env = wrapper(eval_env, add_global_state)

#     train_env, eval_env = add_extra_wrappers(train_env, eval_env, config)
#     return train_env, eval_env


def make_gym_env(
    config: DictConfig,
    num_env: int,
    add_global_state: bool = False,
) -> GymToJumanji:
    """
     Create a gymnasium environment.

    Args:
        config (Dict): The configuration of the environment.
        num_env (int) : The number of parallel envs to create.
        add_global_state (bool): Whether to add the global state to the observation. Default False.

    Returns:
        Async environments.
    """
    wrapper = _gym_registry[config.env.env_name]
    config.system.add_agent_id = config.system.add_agent_id & (~config.env.implicit_agent_id)

    def create_gym_env(config: DictConfig, add_global_state: bool = False) -> gymnasium.Env:
        registered_name = f"{config.env.scenario.name}:{config.env.scenario.task_name}"
        env = gym.make(registered_name, disable_env_checker=True, **config.env.kwargs)
        wrapped_env = wrapper(env, config.env.use_shared_rewards, add_global_state)
        if config.system.add_agent_id:
            wrapped_env = GymAgentIDWrapper(wrapped_env)
        wrapped_env = GymRecordEpisodeMetrics(wrapped_env)
        return wrapped_env

    envs = gymnasium.vector.AsyncVectorEnv(
        [lambda: create_gym_env(config, add_global_state) for _ in range(num_env)],
        worker=async_multiagent_worker,
    )

    envs = GymToJumanji(envs)

    return envs


def make(config: DictConfig, add_global_state: bool = False) -> Tuple[MarlEnv, MarlEnv]:
    """
    Create environments for training and evaluation.

    Args:
    ----
        config (Dict): The configuration of the environment.
        add_global_state (bool): Whether to add the global state to the observation.

    Returns:
    -------
        A tuple of the environments.

    """
    env_name = config.env.env_name

    if env_name in _jumanji_registry:
        return make_jumanji_env(config, add_global_state)
    elif env_name in _jaxmarl_registry:
        return make_jaxmarl_env(config, add_global_state)
    elif env_name in _tmaze_registry:
        return make_tmaze(config, add_global_state)
    else:
        raise ValueError(f"{env_name} is not a supported environment.")
