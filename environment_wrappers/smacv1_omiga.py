from typing import Any, Dict, List

import numpy as np
from dataclasses import dataclass
from environments.starcraft2.StarCraft2_Env import StarCraft2Env

from .base import BaseEnvironment, ResetReturn, StepReturn


@dataclass
class OmigaConf:
    map_name: str
    add_move_state: bool = False
    add_local_obs: bool = False
    add_distance_state: bool = False
    add_enemy_action_state: bool = False
    add_visible_state: bool = False
    add_xy_state: bool = False
    add_agent_id: bool = True
    use_state_agent: bool = True
    use_mustalive: bool = True
    add_center_xy: bool = True
    use_stacked_frames: bool = False
    stacked_frames: int = 1
    use_obs_instead_of_state: bool = False


class SMACv1(BaseEnvironment):

    """Environment wrapper for the version SMACv1 used in the OMIGA paper."""

    def __init__(
        self,
        map_name: str,
        seed: int = 0,
    ):
        self._environment = StarCraft2Env(
            args=OmigaConf(map_name=map_name),
            seed=seed,
        )
        self.possible_agents = [f"agent_{n}" for n in range(self._environment.n_agents)]

        self._num_agents = len(self.possible_agents)
        self._num_actions = self._environment.n_actions
        self._obs_dim = self._environment.get_obs_size()[0]

        self.max_episode_length = self._environment.episode_limit

    def reset(self) -> ResetReturn:
        """Resets the env."""
        # Reset the environment
        self._environment.reset()
        self._done = False

        # Get observation from env
        observations = self._environment.get_obs()
        observations = {agent: observations[i] for i, agent in enumerate(self.possible_agents)}

        legal_actions = self._get_legal_actions()
        legals = {agent: legal_actions[i] for i, agent in enumerate(self.possible_agents)}

        env_state = self._environment.get_state(agent_id=0).astype("float32")

        info = {"legals": legals, "state": env_state}

        return observations, info

    def step(self, actions: Dict[str, np.ndarray]) -> StepReturn:
        """Step in env."""
        # Convert dict of actions to list for SMAC
        smac_actions = []
        for agent in self.possible_agents:
            smac_actions.append(actions[agent])

        o, g, r, d, i, ava = self._environment.step(smac_actions)

        observations = {agent: o[i] for i, agent in enumerate(self.possible_agents)}
        rewards = {
            agent: np.array(r[i], "float32").squeeze(-1)
            for i, agent in enumerate(self.possible_agents)
        }
        terminals = {agent: np.array(d[i]) for i, agent in enumerate(self.possible_agents)}
        truncations = {agent: np.array(False) for agent in self.possible_agents}
        legals = {agent: np.array(ava[i], "int32") for i, agent in enumerate(self.possible_agents)}
        agent_states = {
            agent: np.array(g[i], "float32") for i, agent in enumerate(self.possible_agents)
        }
        env_state = agent_states["agent_0"]
        info = {"legals": legals, "state": env_state}

        return observations, rewards, terminals, truncations, info

    def _get_legal_actions(self) -> List[np.ndarray]:
        """Get legal actions from the environment."""
        legal_actions = []
        for i, _ in enumerate(self.possible_agents):
            legal_actions.append(
                np.array(self._environment.get_avail_agent_actions(i), dtype="float32")
            )
        return legal_actions

    def get_stats(self) -> Any:
        """Return extra stats to be logged."""
        return self._environment.get_stats()