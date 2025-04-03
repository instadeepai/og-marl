from og_marl.wrapped_environments.base import BaseEnvironment


def get_environment(source: str, env_name: str, scenario: str, seed: int = 42) -> BaseEnvironment:  # noqa: C901
    if env_name=="smac_v1" and source != "omiga":
        from og_marl.wrapped_environments.smacv1 import SMACv1

        return SMACv1(scenario, seed=seed)
    elif env_name == "smac_v1" and source=="omiga":
        from og_marl.wrapped_environments.smacv1_omiga import SMACv1OMIGA

        return SMACv1OMIGA(scenario, seed=seed)
    elif env_name == "smac_v2":
        from og_marl.wrapped_environments.smacv2 import SMACv2

        return SMACv2(scenario, seed=seed)
    elif env_name == "mpe" and source=="omar":
        from og_marl.wrapped_environments.mpe_omar import MPEOMAR

        return MPEOMAR(scenario, seed=seed)
    elif env_name == "mamujoco" and source == "og_marl":
        from og_marl.wrapped_environments.mamujoco import MAMuJoCo

        return MAMuJoCo(scenario, seed=seed)
    elif env_name == "gymnasium_mamujoco":
        from og_marl.wrapped_environments.gymnasium_mamujoco import WrappedGymnasiumMAMuJoCo

        return WrappedGymnasiumMAMuJoCo(scenario, seed=seed)
    elif env_name == "mamujoco" and source=="omiga":
        from og_marl.wrapped_environments.mamujoco_omiga import MAMuJoCoOMIGA

        return MAMuJoCoOMIGA(scenario, seed=seed)
    elif env_name == "flatland":
        from og_marl.wrapped_environments.flatland_wrapper import Flatland

        return Flatland(scenario)
    elif env_name == "rware" and source=="alberdice":
        from og_marl.wrapped_environments.rware_alberdice import RWAREAlberDICE

        return RWAREAlberDICE(scenario)
    else:
        raise ValueError("Environment not recognised.")
