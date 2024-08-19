from og_marl.environment_wrappers.base import BaseEnvironment


def get_environment(env_name: str, scenario: str, seed: int = 42) -> BaseEnvironment:  # noqa: C901
    if env_name in ["smac_v1", "smac_v1_cfcql"]:
        from og_marl.environment_wrappers.smacv1 import SMACv1

        return SMACv1(scenario, seed=seed)
    elif env_name == "smac_v2":
        from og_marl.environment_wrappers.smacv2 import SMACv2

        return SMACv2(scenario, seed=seed)
    elif env_name == "mamujoco":
        from og_marl.environment_wrappers.old_mamujoco import MAMuJoCo

        return MAMuJoCo(scenario, seed=seed)
    elif scenario == "pursuit":
        from og_marl.environment_wrappers.pursuit import Pursuit

        return Pursuit()
    elif scenario == "coop_pong":
        from og_marl.environment_wrappers.coop_pong import CooperativePong

        return CooperativePong()
    elif env_name == "gymnasium_mamujoco":
        from og_marl.environment_wrappers.gymnasium_mamujoco import MAMuJoCoGymnasium

        return MAMuJoCoGymnasium(scenario)
    elif env_name == "flatland":
        from og_marl.environment_wrappers.flatland_wrapper import Flatland

        return Flatland(scenario)
    elif env_name == "voltage_control":
        from og_marl.environment_wrappers.voltage_control import VoltageControlEnv

        return VoltageControlEnv()
    # elif env_name == "smax":
    #     from og_marl.environment_wrappers.jaxmarl_smax import SMAX

    #     return SMAX(scenario)
    elif env_name == "lbf":
        from og_marl.environment_wrappers.jumanji_lbf import JumanjiLBF

        return JumanjiLBF(scenario)
    elif env_name == "rware":
        from og_marl.environment_wrappers.jumanji_rware import JumanjiRware

        return JumanjiRware(scenario)
    else:
        raise ValueError("Environment not recognised.")
