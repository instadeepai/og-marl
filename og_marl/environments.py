from og_marl.environment_wrappers.base import BaseEnvironment


def get_environment(source: str, env_name: str, scenario: str, seed: int = 42) -> BaseEnvironment:  # noqa: C901
    if env_name=="smac_v1" and source in ["cfcql", "og_marl"]:
        from og_marl.environment_wrappers.smacv1 import SMACv1

        return SMACv1(scenario, seed=seed)
    elif env_name == "smac_v1" and source=="omiga":
        from og_marl.environment_wrappers.smacv1_omiga import SMACv1OMIGA

        return SMACv1OMIGA(scenario, seed=seed)
    elif env_name == "smac_v2":
        from og_marl.environment_wrappers.smacv2 import SMACv2

        return SMACv2(scenario, seed=seed)
    elif env_name == "mpe" and source=="omar":
        from og_marl.environment_wrappers.mpe_omar import MPEOMAR

        return MPEOMAR(scenario, seed=seed)
    elif env_name == "mamujoco" and source == "og_marl":
        from og_marl.environment_wrappers.mamujoco import MAMuJoCo

        return MAMuJoCo(scenario, seed=seed)
    elif env_name == "mamujoco" and source=="omiga":
        from og_marl.environment_wrappers.mamujoco_omiga import MAMuJoCoOMIGA

        return MAMuJoCoOMIGA(scenario, seed=seed)
    elif env_name == "flatland":
        from og_marl.environment_wrappers.flatland_wrapper import Flatland

        return Flatland(scenario)
    else:
        raise ValueError("Environment not recognised.")
