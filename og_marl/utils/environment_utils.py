from typing import Dict
import dm_env

class MAEnvironmentSpec:
    def __init__(
        self,
        environment: dm_env.Environment,
        specs: Dict = None,
        extra_specs: Dict = None,
    ):
        if not specs:
            specs = self._make_ma_environment_spec(environment)
        else:
            self.extra_specs = extra_specs

        self._specs = {key: specs[key] for key in self._keys}

    def _make_ma_environment_spec(
        self, environment: dm_env.Environment
    ) -> Dict:
        """Returns an `EnvironmentSpec` describing values used by
        an environment for each agent."""
        specs = {}
        observation_specs = environment.observation_spec()
        action_specs = environment.action_spec()
        reward_specs = environment.reward_spec()
        discount_specs = environment.discount_spec()
        self.extra_specs = environment.extra_spec()
        for agent in environment.possible_agents:
            specs[agent] = EnvironmentSpec(
                observations=observation_specs[agent],
                actions=action_specs[agent],
                rewards=reward_specs[agent],
                discounts=discount_specs[agent],
            )
        return specs

    def get_extra_specs(self):
        return self.extra_specs  # type: ignore

    def get_agent_specs(self):
        return self._specs

    def get_agent_type_specs(self):
        specs = {}
        agent_types = list({agent.split("_")[0] for agent in self._keys})
        for agent_type in agent_types:
            specs[agent_type] = self._specs[f"{agent_type}_0"]
        return specs

    def get_agent_ids(self):
        return self._keys

    def get_agent_types(self) -> List[str]:
        return list({agent.split("_")[0] for agent in self._keys})

    def get_agents_by_type(self) -> Dict[str, List[str]]:
        agents_by_type: Dict[str, List[str]] = {}
        agents_ids = self.get_agent_ids()
        agent_types = self.get_agent_types()
        for agent_type in agent_types:
            agents_by_type[agent_type] = []
            for agent in agents_ids:
                if agent_type in agent:
                    agents_by_type[agent_type].append(agent)
        return agents_by_type