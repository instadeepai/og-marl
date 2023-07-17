from typing import List, Dict, Tuple, Any

import numpy as np
import dm_env
from og_marl.environments.base import parameterized_restart, BaseEnvironment, OLT
from dm_env import specs

from flatland.core.env import Environment
from flatland.envs.observations import TreeObsForRailEnv, Node
from flatland.core.env_observation_builder import ObservationBuilder
from flatland.core.env_prediction_builder import PredictionBuilder
from flatland.core.grid.grid4_utils import get_new_position
from flatland.core.grid.grid_utils import coordinate_to_position
from flatland.envs.agent_utils import EnvAgent
from flatland.envs.rail_env import RailEnv, RailEnvActions
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.utils.rendertools import AgentRenderVariant, RenderTool
from flatland.envs.step_utils.states import TrainState


def get_config_key(num_agents):
    return str(num_agents) + "_trains"


FLATLAND_MAP_CONFIGS = {
    "3_trains": {
        "num_trains": 3,
        "num_cities": 2,
        "width": 25,
        "height": 25,
        "max_episode_len": 80,
    },
    "5_trains": {
        "num_trains": 5,
        "num_cities": 2,
        "width": 25,
        "height": 25,
        "max_episode_len": 100,
    },
}

class Flatland(BaseEnvironment):
    def __init__(self, map_name="5_trains", joint_obs_as_global_state=False):

        map_config = FLATLAND_MAP_CONFIGS[map_name]

        self.num_actions = 5
        self.num_agents = map_config["num_trains"]
        self._tree_depth = 2
        self._num_cities = map_config["num_cities"]
        self._map_width = map_config["width"]
        self._map_height = map_config["height"]
        self.max_episode_len = map_config["max_episode_len"]

        self.environment_label = f"flatland/{map_name}"

        self._joint_obs_as_global_state = joint_obs_as_global_state
        self._agents = [str(handle) for handle in range(self.num_agents)]

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

        self._env_renderer = RenderTool(
            self._environment,
            agent_render_variant=AgentRenderVariant.ONE_STEP_BEHIND,
            show_debug=False,
            screen_height=600,  # Adjust these parameters to fit your resolution
            screen_width=600,
        )  # Adjust these parameters to fit your resolution

    def _make_state_representation(self):
        state = []
        for i, agent_id in enumerate(self._agents):
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
        return state

    def reset(self):
        self._done = False
        self._reset_next_step = False
        self._step_type = dm_env.StepType.FIRST

        # Call reset() to initialize the environment
        episode_len = 10000000
        while episode_len > self.max_episode_len:
            observations, info = self._environment.reset()
            episode_len = self._environment._max_episode_steps

        self.reset_cells()

        # Reset the rendering sytem
        self._env_renderer.reset()

        legal_actions = self._get_legal_actions(observations, info)
        observations = self._convert_observations(
            observations, legal_actions, self._done, info
        )

        # Make extras
        if self._joint_obs_as_global_state:
            joint_observation = self._make_joint_observation(observations)
            extras = {"s_t": joint_observation}
        else:
            state = self._make_state_representation()
            extras = {"s_t": state}

        # Set env discount to 1 for all agents
        self._discounts = {agent: np.array(1, "float32") for agent in self._agents}

        # Set reward to zero for all agents
        rewards = {agent: np.array(0, "float32") for agent in self._agents}

        return parameterized_restart(rewards, self._discounts, observations), extras

    def step(self, actions):

        # Possibly reset the environment
        if self._reset_next_step:
            return self.reset()

        actions = {int(agent): action.item() for agent, action in actions.items()}

        # Step the Flatland environment
        next_observations, all_rewards, all_dones, info = self._environment.step(
            actions
        )

        # Team done flag
        self._done = all(list(all_dones.values()))

        # Rewards
        rewards = {
            agent: np.array(all_rewards[int(agent)], dtype="float32")
            for agent in self._agents
        }

        # Legal actions
        legal_actions = self._get_legal_actions(next_observations, info)

        # Observations
        next_observations = self._convert_observations(
            next_observations, legal_actions, self._done, info
        )

        # Make extras
        if self._joint_obs_as_global_state:
            joint_observation = self._make_joint_observation(next_observations)
            extras = {"s_t": joint_observation}
        else:
            state = self._make_state_representation()
            extras = {"s_t": state}

        if self._done:
            self._step_type = dm_env.StepType.LAST
            self._reset_next_step = True

            # Discount on last timestep set to zero
            self._discounts = {agent: np.array(0, "float32") for agent in self._agents}

            # Update extra stats eg. completion rate
            self._update_stats(info, rewards)
        else:
            self._step_type = dm_env.StepType.MID

        # Create timestep object
        timestep = dm_env.TimeStep(
            observation=next_observations,
            reward=rewards,
            discount=self._discounts,
            step_type=self._step_type,
        )

        return timestep, extras

    def _get_legal_actions(self, observations, info):
        legal_actions = {}
        for agent in self._agents:
            flatland_agent = self._environment.agents[int(agent)]

            if not self._environment.action_required(flatland_agent):
                legals = np.zeros(self.num_actions, "float32")
                legals[0] = 1  # can only do nothng
            else:
                legals = np.ones(5, "float32")

            legal_actions[agent] = legals

        return legal_actions

    def _make_joint_observation(self, observations):
        joint_observation = []
        for agent in self._agents:
            joint_observation.append(observations[agent].observation)
        joint_observation = np.concatenate(joint_observation, axis=-1)
        return joint_observation

    def _convert_observations(self, observations, legal_actions, done, info):
        olt_observations = {}
        for agent in self._agents:
            norm_observation = normalize_observation(
                observations[int(agent)], tree_depth=self._tree_depth
            )
            state = info["state"][int(agent)]
            one_hot_state = np.zeros((7,), "float32")
            one_hot_state[state] = 1
            obs = np.concatenate([one_hot_state, norm_observation], axis=-1)
            olt_observations[agent] = OLT(
                observation=obs,
                legal_actions=legal_actions[agent],
                terminal=np.asarray([done], dtype=np.float32),
            )

        return olt_observations

    def _update_stats(self, info: Dict, rewards: Dict) -> None:
        """Update flatland stats."""
        episode_return = sum(list(rewards.values()))
        tasks_finished = sum(
            [1 if state == TrainState.DONE else 0 for state in info["state"].values()]
        )
        completion = tasks_finished / len(self._agents)
        normalized_score = episode_return / (
            self._environment._max_episode_steps * len(self._agents)
        )

        self._latest_score = normalized_score
        self._latest_completion = completion

    def get_stats(self) -> Dict:
        """Get flatland specific stats."""
        if self._latest_completion is not None and self._latest_score is not None:
            return {
                "score": self._latest_score,
                "completion": self._latest_completion,
            }
        else:
            return {}

    def render(self, mode: str = "human") -> np.ndarray:
        """Renders the environment."""
        if mode == "human":
            show = True
        else:
            show = False

        return self._env_renderer.render_env(
            show=show,
            show_observations=False,
            show_predictions=False,
            return_image=True,
        )

    def env_done(self) -> bool:
        """Check if env is done.

        Returns:
            bool: bool indicating if env is done.
        """
        return self._done

    def extra_spec(self) -> Dict[str, specs.BoundedArray]:
        """Function returns extra spec (format) of the env.

        Returns:
            Dict[str, specs.BoundedArray]: extra spec.
        """
        if self._joint_obs_as_global_state:
            observation_spec = np.zeros(
                11 * sum(4**i for i in range(self._tree_depth + 1)) + 7,
                dtype=np.float32,
            )
            extras = {
                "s_t": np.concatenate([observation_spec] * len(self._agents), axis=-1)
            }
        else:
            extras = {"s_t": np.zeros((55,), "float32")}
        return extras

    def observation_spec(self) -> Dict:
        """Observation spec.

        Returns:
            types.Observation: spec for environment.
        """
        observation_spec = np.zeros(
            11 * sum(4**i for i in range(self._tree_depth + 1)) + 7,
            dtype=np.float32,
        )
        legal_actions_spec = np.zeros(self.num_actions, "float32")

        observation_specs = {}
        for i, agent in enumerate(self._agents):

            observation_specs[agent] = OLT(
                observation=observation_spec,
                legal_actions=legal_actions_spec,
                terminal=np.asarray([True], dtype=np.float32),
            )

        return observation_specs

    def action_spec(
        self,
    ) -> Dict[str, specs.DiscreteArray]:
        """Action spec.

        Returns:
            spec for actions.
        """
        action_specs = {}
        for agent in self._agents:
            action_specs[agent] = specs.DiscreteArray(
                num_values=self.num_actions, dtype=int
            )
        return action_specs

### RailEnv Wrappers from: https://gitlab.aicrowd.com/flatland/flatland/-/blob/master/flatland/contrib/wrappers/flatland_wrappers.py


def find_all_cells_where_agent_can_choose(env: RailEnv):
    """
    input: a RailEnv (or something which behaves similarly, e.g. a wrapped RailEnv),
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


def max_lt(seq: np.ndarray, val: Any) -> Any:
    """Get max in sequence.
    Return greatest item in seq for which item < val applies.
    None is returned if seq was empty or all items in seq were >= val.
    """
    max = 0
    idx = len(seq) - 1
    while idx >= 0:
        if seq[idx] < val and seq[idx] >= 0 and seq[idx] > max:
            max = seq[idx]
        idx -= 1
    return max


def min_gt(seq: np.ndarray, val: Any) -> Any:
    """Gets min in a sequence.
    Return smallest item in seq for which item > val applies.
    None is returned if seq was empty or all items in seq were >= val.
    """
    min = np.inf
    idx = len(seq) - 1
    while idx >= 0:
        if seq[idx] >= val and seq[idx] < min:
            min = seq[idx]
        idx -= 1
    return min


def norm_obs_clip(
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
    :return: returnes normalized and clipped observatoin
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
        return np.clip(np.array(obs) / max_obs, clip_min, clip_max)
    norm = np.abs(max_obs - min_obs)
    return np.clip((np.array(obs) - min_obs) / norm, clip_min, clip_max)


def _split_node_into_feature_groups(
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


def _split_subtree_into_feature_groups(
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


def split_tree_into_feature_groups(
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


def normalize_observation(
    observation: Node, tree_depth: int, observation_radius: int = 0
) -> np.ndarray:
    """This function normalizes the observation used by the RL algorithm."""
    if observation is None:
        return np.zeros(
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
    return normalized_obs
