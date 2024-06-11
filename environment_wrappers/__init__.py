from environment_wrappers.base import BaseEnvironment


def get_environment(env_name: str, scenario: str) -> BaseEnvironment:  # noqa: C901
    if env_name == "smac_v1":
        from environment_wrappers.smacv1 import SMACv1

        return SMACv1(scenario)
    elif env_name == "smac_v1_omiga":
        from environment_wrappers.smacv1_omiga import SMACv1

        return SMACv1(scenario)
    elif env_name == "smac_v1_cfcql":
        from environment_wrappers.smacv1 import SMACv1

        return SMACv1(scenario)
    elif env_name == "mamujoco":
        from environment_wrappers.mamujoco import MAMuJoCo

        return MAMuJoCo(scenario)
    elif env_name == "mamujoco_omar":
        from environment_wrappers.mamujoco_omar import MAMuJoCo

        return MAMuJoCo(scenario)
    elif env_name == "mamujoco_omiga":
        from environment_wrappers.mamujoco_omiga import MAMuJoCo

        return MAMuJoCo(scenario)
    elif env_name == "mpe_omar":
        from environment_wrappers.mpe_omar import MPE

        return MPE()
    else:
        raise ValueError("Environment not recognised.")
