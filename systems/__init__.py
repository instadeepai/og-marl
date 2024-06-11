from typing import Any

from environment_wrappers.base import BaseEnvironment
from utils.loggers import BaseLogger
from systems.base import BaseMARLSystem


def get_system(  # noqa: C901
    system_name: str,
    environment: BaseEnvironment,
    logger: BaseLogger,
    **kwargs: Any,
) -> BaseMARLSystem:
    if system_name == "iql+cql":
        from .idrqn_cql import IDRQNCQLSystem

        return IDRQNCQLSystem(environment, logger, **kwargs)
    elif system_name == "bc":
        from .bc import DicreteActionBehaviourCloning

        return DicreteActionBehaviourCloning(environment, logger, **kwargs)
    elif system_name == "maddpg+cql":
        from .maddpg_cql import MADDPGCQLSystem

        return MADDPGCQLSystem(environment, logger, **kwargs)
    elif system_name == "iddpg+bc":
        from .iddpg_bc import IDDPGBCSystem

        return IDDPGBCSystem(environment, logger, **kwargs)
    else:
        raise ValueError(f"Unknown system name: {system_name}")
