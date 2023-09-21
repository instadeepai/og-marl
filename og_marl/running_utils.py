from og_marl.systems.base import BaseMARLSystem

def get_environment(env_name, scenario):
    if env_name == "smac_v1":
        from og_marl.environments.smacv1 import SMACv1
        return SMACv1(scenario)
    elif env_name == "smac_v2":
        from og_marl.environments.smacv2 import SMACv2
        return SMACv2(scenario)
    
def get_system(system_name, environment, logger, kwargs) -> BaseMARLSystem :
    if system_name == "idrqn":
        from og_marl.systems.idrqn import IDRQNSystem
        return IDRQNSystem(environment, logger, **kwargs)
    elif system_name == "idrqn+cql":
        from og_marl.systems.idrqn_cql import IDRQNCQLSystem
        return IDRQNCQLSystem(environment, logger, **kwargs)
    elif system_name == "idrqn+bcq":
        from og_marl.systems.idrqn_bcq import IDRQNBCQSystem
        return IDRQNBCQSystem(environment, logger, **kwargs)
    elif system_name == "qmix":
        from og_marl.systems.qmix import QMIXSystem
        return QMIXSystem(environment, logger, **kwargs)
    elif system_name == "qmix+cql":
        from og_marl.systems.qmix_cql import QMIXCQLSystem
        return QMIXCQLSystem(environment, logger, **kwargs)
    elif system_name == "qmix+bcq":
        from og_marl.systems.qmix_bcq import QMIXBCQSystem
        return QMIXBCQSystem(environment, logger, **kwargs)