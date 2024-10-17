# Copyright 2023 InstaDeep Ltd. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Wrapper for Flatland."""
from typing import Any, Dict, Tuple

import numpy as np
from flatland.core.grid.grid4_utils import get_new_position
from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.observations import Node, TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from gymnasium.spaces import Box, Discrete

from og_marl.environments.base import BaseEnvironment, Observations, ResetReturn, StepReturn

FLATLAND_MAP_CONFIGS = {
    "3trains": {
        "num_trains": 3,
        "num_cities": 2,
        "width": 25,
        "height": 25,
        "max_episode_len": 80,
    },
    "5trains": {
        "num_trains": 5,
        "num_cities": 2,
        "width": 25,
        "height": 25,
        "max_episode_len": 100,
    },
}


class Flatland(BaseEnvironment):
    def __init__(self, map_name: str = "5_trains"):
        map_config = FLATLAND_MAP_CONFIGS[map_name]

        self._num_actions = 5
        self.num_agents = map_config["num_trains"]
        self._num_cities = map_config["num_cities"]
        self._map_width = map_config["width"]
        self._map_height = map_config["height"]
        self._tree_depth = 2

        self.possible_agents = [f"{i}" for i in range(self.num_agents)]

        self.rail_generator = sparse_rail_generator(max_num_cities=self._num_cities)

        self.obs_builder = TreeObsForRailEnv(
            max_depth=self._tree_depth,
            predictor=ShortestPathPredictorForRailEnv(max_depth=20),
        )

        # Initialize the properties of the environment
        self._environment = RailEnv(
            width=self._map_width,
            height=self._map_height,
            number_of_agents=self.num_agents,
            rail_generator=self.rail_generator,
            line_generator=sparse_line_generator(),
            obs_builder_object=self.obs_builder,
        )

        self._obs_dim = 11 * sum(4**i for i in range(self._tree_depth + 1)) + 7

        self.action_spaces = {agent: Discrete(self._num_actions) for agent in self.possible_agents}
        self.observation_spaces = {
            agent: Box(-np.inf, np.inf, (self._obs_dim,)) for agent in self.possible_agents
        }

        self.info_spec = {
            "state": np.zeros((11 * self.num_agents,), "float32"),
            "legals": {
                agent: np.zeros((self._num_actions,), "int64") for agent in self.possible_agents
            },
        }

        self.max_episode_length = map_config["max_episode_len"]

    def reset(self) -> ResetReturn:
        self._done = False

        observations, info = self._environment.reset()

        legal_actions = self._get_legal_actions()

        observations = self._convert_observations(observations, info)

        state = self._make_state_representation()

        info = {"state": state, "legals": legal_actions}

        return observations, info

    def step(self, actions: Dict[str, np.ndarray]) -> StepReturn:
        actions = {int(agent): action.item() for agent, action in actions.items()}

        # Step the Flatland environment
        next_observations, all_rewards, all_dones, info = self._environment.step(actions)

        # Team done flag
        self._done = all(list(all_dones.values()))

        # Rewards
        rewards = {
            agent: np.array(all_rewards[int(agent)], dtype="float32")
            for agent in self.possible_agents
        }

        # Legal actions
        legal_actions = self._get_legal_actions()

        # Observations
        next_observations = self._convert_observations(next_observations, info)

        # Make extras
        state = self._make_state_representation()

        info = {"state": state, "legals": legal_actions}

        terminals = {agent: np.array(self._done) for agent in self.possible_agents}
        truncations = {agent: np.array(False) for agent in self.possible_agents}

        return next_observations, rewards, terminals, truncations, info

    def _get_legal_actions(self) -> Dict[str, np.ndarray]:
        legal_actions = {}
        for agent in self.possible_agents:
            agent_id = int(agent)
            flatland_agent = self._environment.agents[agent_id]

            if not self._environment.action_required(
                flatland_agent.state, flatland_agent.speed_counter.is_cell_entry
            ):
                legals = np.zeros(self._num_actions, "float32")
                legals[0] = 1  # can only do nothng
            else:
                legals = np.ones(5, "float32")

            legal_actions[agent] = legals

        return legal_actions

    def _make_state_representation(self) -> np.ndarray:
        state = []
        for i, _ in enumerate(self.possible_agents):
            agent = self._environment.agents[i]
            state.append(np.array(agent.target, "float32"))

            pos = agent.position
            if pos is None:
                pos = (-1, -1)
            if agent.state == 7:
                pos = agent.target  # agent is done

            state.append(np.array(pos, "float32"))

            one_hot_state = np.zeros((7,), "float32")
            one_hot_state[agent.state] = 1
            state.append(one_hot_state)
        state = np.concatenate(state)
        return state  # type: ignore

    def _convert_observations(
        self,
        observations: Dict[int, np.ndarray],
        info: Dict[str, Dict[int, np.ndarray]],
    ) -> Observations:
        new_observations = {}
        for i, agent in enumerate(self.possible_agents):
            agent_id = i
            norm_observation = normalize_observation(
                observations[agent_id],
                tree_depth=self._tree_depth,
            )
            state = info["state"][agent_id]  # train state
            one_hot_state = np.zeros((7,), "float32")
            one_hot_state[state] = 1
            obs = np.concatenate([one_hot_state, norm_observation], axis=-1)
            new_observations[agent] = obs
        return new_observations


### RailEnv Wrappers from: https://gitlab.aicrowd.com/flatland/flatland/-/blob/master/flatland/contrib/wrappers/flatland_wrappers.py


def find_all_cells_where_agent_can_choose(env: RailEnv):  # type: ignore
    """input: a RailEnv (or something which behaves similarly, e.g. a wrapped RailEnv),

    WHICH HAS BEEN RESET ALREADY!
    (o.w., we call env.rail, which is None before reset(), and crash.)
    """
    switches = []
    switches_neighbors = []
    directions = list(range(4))
    for h in range(env.height):
        for w in range(env.width):
            pos = (h, w)

            is_switch = False
            # Check for switch: if there is more than one outgoing transition
            for orientation in directions:
                possible_transitions = env.rail.get_transitions(*pos, orientation)
                num_transitions = np.count_nonzero(possible_transitions)
                if num_transitions > 1:
                    switches.append(pos)
                    is_switch = True
                    break
            if is_switch:
                # Add all neighbouring rails, if pos is a switch
                for orientation in directions:
                    possible_transitions = env.rail.get_transitions(*pos, orientation)
                    for movement in directions:
                        if possible_transitions[movement]:
                            switches_neighbors.append(get_new_position(pos, movement))

    decision_cells = switches + switches_neighbors
    return tuple(map(set, (switches, switches_neighbors, decision_cells)))


# The block of code below is obtained from the flatland starter-kit
# at https://gitlab.aicrowd.com/flatland/flatland-starter-kit/-/blob/master/
# utils/observation_utils.py
# this is done just to obtain the normalize_observation function that would
# serve as the default preprocessor for the Tree obs builder.


def max_lt(seq: np.ndarray, val: Any) -> Any:  # type: ignore
    """Get max in sequence.

    Return greatest item in seq for which item < val applies.
    None is returned if seq was empty or all items in seq were >= val.
    """
    max_val = 0
    idx = len(seq) - 1
    while idx >= 0:
        if seq[idx] < val and seq[idx] >= 0 and seq[idx] > max_val:
            max_val = seq[idx]  # type: ignore
        idx -= 1
    return max_val


def min_gt(seq: np.ndarray, val: Any) -> Any:  # type: ignore
    """Gets min in a sequence.

    Return smallest item in seq for which item > val applies.
    None is returned if seq was empty or all items in seq were >= val.
    """
    min_val = np.inf
    idx = len(seq) - 1
    while idx >= 0:
        if seq[idx] >= val and seq[idx] < min_val:
            min_val = seq[idx]  # type: ignore
        idx -= 1
    return min_val


def norm_obs_clip(  # type: ignore
    obs: np.ndarray,
    clip_min: int = -1,
    clip_max: int = 1,
    fixed_radius: int = 0,
    normalize_to_range: bool = False,
) -> np.ndarray:
    """Normalize observation.

    This function returns the difference between min and max value of an observation
    :param obs: Observation that should be normalized
    :param clip_min: min value where observation will be clipped
    :param clip_max: max value where observation will be clipped
    :return: returns normalized and clipped observation
    """
    if fixed_radius > 0:
        max_obs = fixed_radius
    else:
        max_obs = max(1, max_lt(obs, 1000)) + 1

    min_obs = 0  # min(max_obs, min_gt(obs, 0))
    if normalize_to_range:
        min_obs = min_gt(obs, 0)
    if min_obs > max_obs:
        min_obs = max_obs
    if max_obs == min_obs:
        return np.clip(np.array(obs) / max_obs, clip_min, clip_max)  # type: ignore
    norm = np.abs(max_obs - min_obs)
    return np.clip((np.array(obs) - min_obs) / norm, clip_min, clip_max)  # type: ignore


def _split_node_into_feature_groups(  # type: ignore
    node: Node,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Splits node into features."""
    data = np.zeros(6)
    distance = np.zeros(1)
    agent_data = np.zeros(4)

    data[0] = node.dist_own_target_encountered
    data[1] = node.dist_other_target_encountered
    data[2] = node.dist_other_agent_encountered
    data[3] = node.dist_potential_conflict
    data[4] = node.dist_unusable_switch
    data[5] = node.dist_to_next_branch

    distance[0] = node.dist_min_to_target

    agent_data[0] = node.num_agents_same_direction
    agent_data[1] = node.num_agents_opposite_direction
    agent_data[2] = node.num_agents_malfunctioning
    agent_data[3] = node.speed_min_fractional

    return data, distance, agent_data


def _split_subtree_into_feature_groups(  # type: ignore
    node: Node, current_tree_depth: int, max_tree_depth: int
) -> Tuple:
    """Split subtree."""
    if node == -np.inf:
        remaining_depth = max_tree_depth - current_tree_depth
        # reference:
        # https://stackoverflow.com/questions/515214/total-number-of-nodes-in-a-tree-data-structure
        num_remaining_nodes = int((4 ** (remaining_depth + 1) - 1) / (4 - 1))
        return (
            [-np.inf] * num_remaining_nodes * 6,
            [-np.inf] * num_remaining_nodes,
            [-np.inf] * num_remaining_nodes * 4,
        )

    data, distance, agent_data = _split_node_into_feature_groups(node)

    if not node.childs:
        return data, distance, agent_data

    for direction in TreeObsForRailEnv.tree_explored_actions_char:
        sub_data, sub_distance, sub_agent_data = _split_subtree_into_feature_groups(
            node.childs[direction], current_tree_depth + 1, max_tree_depth
        )
        data = np.concatenate((data, sub_data))
        distance = np.concatenate((distance, sub_distance))
        agent_data = np.concatenate((agent_data, sub_agent_data))

    return data, distance, agent_data


def split_tree_into_feature_groups(  # type: ignore
    tree: Node, max_tree_depth: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """This function splits the tree into three difference arrays."""
    data, distance, agent_data = _split_node_into_feature_groups(tree)

    for direction in TreeObsForRailEnv.tree_explored_actions_char:
        sub_data, sub_distance, sub_agent_data = _split_subtree_into_feature_groups(
            tree.childs[direction], 1, max_tree_depth
        )
        data = np.concatenate((data, sub_data))
        distance = np.concatenate((distance, sub_distance))
        agent_data = np.concatenate((agent_data, sub_agent_data))

    return data, distance, agent_data


def normalize_observation(  # type: ignore
    observation: Node, tree_depth: int, observation_radius: int = 0
) -> np.ndarray:
    """This function normalizes the observation used by the RL algorithm."""
    if observation is None:
        return np.zeros(  # type: ignore
            11 * sum(np.power(4, i) for i in range(tree_depth + 1)),
            dtype=np.float32,
        )
    data, distance, agent_data = split_tree_into_feature_groups(observation, tree_depth)

    data = norm_obs_clip(data, fixed_radius=observation_radius)
    distance = norm_obs_clip(distance, normalize_to_range=True)
    agent_data = np.clip(agent_data, -1, 1)
    normalized_obs = np.array(
        np.concatenate((np.concatenate((data, distance)), agent_data)),
        dtype=np.float32,
    )
    return normalized_obs  # type: ignore
