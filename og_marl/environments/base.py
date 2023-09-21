"""Base class for OG-MARL Environment Wrappers."""

class BaseEnvironment:
    """Base environment class for OG-MARL."""

    def __init__(self):
        """Constructor."""
        self._environment = None
        self._agents = None

        self.num_actions = None
        self.num_agents = None

    def reset(self):
        """Resets the env.

        Returns:
            Dict: observations
        """
        raise NotImplementedError
        

    def step(self, actions):
        """Steps in env.

        Args:
            actions (Dict[str, np.ndarray]): actions per agent.

        Returns:
            observations, rewards, done
        """
        raise NotImplementedError


    def get_stats(self):
        """Return extra stats to be logged.

        Returns:
            extra stats to be logged.
        """
        return {}

    def __getattr__(self, name: str):
        """Expose any other attributes of the underlying environment.

        Args:
            name (str): attribute.

        Returns:
            Any: return attribute from env or underlying env.
        """
        if hasattr(self.__class__, name):
            return self.__getattribute__(name)
        else:
            return getattr(self._environment, name)