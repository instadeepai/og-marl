def get_system(system_name, environment, logger, **kwargs):  # noqa: C901
    # TODO: Fix the cognitive complexity here

    if system_name == "idrqn":
        from og_marl.tf2.systems.idrqn import IDRQNSystem

        return IDRQNSystem(environment, logger, **kwargs)
    elif system_name == "idrqn+cql":
        from og_marl.tf2.systems.idrqn_cql import IDRQNCQLSystem

        return IDRQNCQLSystem(environment, logger, **kwargs)
    elif system_name == "idrqn+bcq":
        from og_marl.tf2.systems.idrqn_bcq import IDRQNBCQSystem

        return IDRQNBCQSystem(environment, logger, **kwargs)
    elif system_name == "qmix":
        from og_marl.tf2.systems.qmix import QMIXSystem

        return QMIXSystem(environment, logger, **kwargs)
    elif system_name == "qmix+cql":
        from og_marl.tf2.systems.qmix_cql import QMIXCQLSystem

        return QMIXCQLSystem(environment, logger, **kwargs)
    elif system_name == "maicq":
        from og_marl.tf2.systems.maicq import MAICQSystem

        return MAICQSystem(environment, logger, **kwargs)
    elif system_name == "qmix+bcq":
        from og_marl.tf2.systems.qmix_bcq import QMIXBCQSystem

        return QMIXBCQSystem(environment, logger, **kwargs)
    elif system_name == "iddpg":
        from og_marl.tf2.systems.iddpg import IDDPGSystem

        return IDDPGSystem(environment, logger, **kwargs)
    elif system_name == "iddpg+cql":
        from og_marl.tf2.systems.iddpg_cql import IDDPGCQLSystem

        return IDDPGCQLSystem(environment, logger, **kwargs)
    elif system_name == "facmac+cql":
        from og_marl.tf2.systems.facmac_cql import FACMACCQLSystem

        return FACMACCQLSystem(environment, logger, **kwargs)
    elif system_name == "omar":
        from og_marl.tf2.systems.omar import OMARSystem

        return OMARSystem(environment, logger, **kwargs)
    elif system_name == "maddpg":
        from og_marl.tf2.systems.maddpg import MADDPGSystem

        return MADDPGSystem(environment, logger, **kwargs)
    elif system_name == "maddpg+cql":
        from og_marl.tf2.systems.maddpg_cql import MADDPGCQLSystem

        return MADDPGCQLSystem(environment, logger, **kwargs)
    elif system_name == "facmac+cql":
        from og_marl.tf2.systems.facmac_cql import FACMACCQLSystem

        return FACMACCQLSystem(environment, logger, **kwargs)
