from og_marl.environment_wrappers.base import BaseEnvironment


def get_environment(env_name: str, scenario: str, seed: int = 42) -> BaseEnvironment:  # noqa: C901
    if env_name in ["smac_v1", "smac_v1_cfcql"]:
        from og_marl.environment_wrappers.smacv1 import SMACv1

        return SMACv1(scenario, seed=seed)
    elif env_name == "smac_v1_omiga":
        from og_marl.environment_wrappers.smacv1_omiga import SMACv1OMIGA

        return SMACv1OMIGA(scenario, seed=seed)
    elif env_name == "smac_v2":
        from og_marl.environment_wrappers.smacv2 import SMACv2

        return SMACv2(scenario, seed=seed)
    elif env_name == "mamujoco":
        from og_marl.environment_wrappers.old_mamujoco import MAMuJoCo

        return MAMuJoCo(scenario, seed=seed)
    elif env_name == "mamujoco_omiga":
        from og_marl.environment_wrappers.mamujoco_omiga import MAMuJoCoOMIGA

        return MAMuJoCoOMIGA(scenario, seed=seed)
    elif env_name == "flatland":
        from og_marl.environment_wrappers.flatland_wrapper import Flatland

        return Flatland(scenario)
    else:
        raise ValueError("Environment not recognised.")
