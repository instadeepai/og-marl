"""Base class for OG-MARL Environment Wrappers."""
from typing import Any, Dict, List, NamedTuple
import os
import tree
from pathlib import Path
import dm_env
from dm_env import specs
import numpy as np
from gym import spaces
import tensorflow as tf

class Step(NamedTuple):
    """Step class used internally for reverb adders."""

    observations: Dict
    actions: Dict
    rewards: Dict
    discounts: Dict
    start_of_episode: bool
    extras: Dict

def get_schema(environment):
    schema = {}
    for agent in environment.possible_agents:

        schema[agent + "_observations"] = environment.observation_spec()[agent].observation
        schema[agent + "_legal_actions"] = environment.observation_spec()[agent].legal_actions
        schema[agent + "_actions"] = environment.action_spec()[agent]
        schema[agent + "_rewards"] = environment.reward_spec()[agent]
        schema[agent + "_discounts"] = environment.discount_spec()[agent]

    ## Extras
    # Zero-padding mask
    schema["zero_padding_mask"] = np.array(1, dtype=np.float32)

    # Global env state
    extras_spec = environment.extra_spec()
    if "s_t" in extras_spec:
        schema["env_state"] = extras_spec["s_t"]

    schema["episode_return"] = np.array(0, dtype="float32")

    return schema

def parameterized_restart(rewards, discounts, observations):
    return dm_env.TimeStep(dm_env.StepType.FIRST, rewards, discounts, observations)

class OLT(NamedTuple):
    """Container for (observation, legal_actions, terminal) tuples."""

    observation: Any
    legal_actions: Any
    terminal: Any

def convert_space_to_spec(space):
    """Converts an OpenAI Gym space to a dm_env spec."""
    if isinstance(space, spaces.Discrete):
        return specs.DiscreteArray(num_values=space.n, dtype=space.dtype)

    elif isinstance(space, spaces.Box):
        return specs.BoundedArray(
            shape=space.shape,
            dtype=space.dtype,
            minimum=space.low,
            maximum=space.high,
            )
    else:
        raise NotImplementedError

class BaseEnvironment:
    """Base environment class for OG-MARL."""

    def __init__(self):
        """Constructor."""
        self._environment = None
        self._agents = None

        self.num_actions = None
        self.action_dim = None
        self.max_trajectory_length = None
        self.environment_label = None

        self._reset_next_step = True
        self._done = False

    def get_dataset(self, dataset_type, datasets_base_dir="datasets"):
        dataset_dir = f"{datasets_base_dir}/{self.environment_label}/{dataset_type}"
        if os.path.exists(dataset_dir):
            file_path = Path(dataset_dir)
            filenames = [
                str(file_name) for file_name in file_path.glob("**/*.tfrecord")
            ]
            filename_dataset = tf.data.Dataset.from_tensor_slices(filenames)
            dataset = filename_dataset.interleave(
                lambda x: tf.data.TFRecordDataset(x, compression_type="GZIP").map(
                    self._decode_fn
                ),
                cycle_length=None,
                num_parallel_calls=2,
                deterministic=False,
                block_length=None,
            )
        else:
            raise FileNotFoundError(f"Dataset not found. Please download it and place files in the correct directory. We checked {dataset_dir}")
        return dataset

    def reset(self) -> dm_env.TimeStep:
        """Resets the env.

        Returns:
            dm_env.TimeStep: dm timestep.
        """
        raise NotImplementedError

    def step(self, actions: Dict[str, np.ndarray]) -> dm_env.TimeStep:
        """Steps in env.

        Args:
            actions (Dict[str, np.ndarray]): actions per agent.

        Returns:
            dm_env.TimeStep: dm timestep, extras: Dict
        """
        raise NotImplementedError

    def render(self, mode):
        return self._environment.render()

    def env_done(self) -> bool:
        """Check if env is done.

        Returns:
            bool: bool indicating if env is done.
        """
        return self._done

    def extra_spec(self) -> Dict:
        raise NotImplementedError

    def observation_spec(self) -> Dict:
        """Observation spec.

        Returns:
            types.Observation: spec for environment.
        """
        raise NotImplementedError

    def action_spec(self) -> Dict:
        """Action spec.

        Returns:
            spec for actions.
        """
        raise NotImplementedError

    def reward_spec(self) -> Dict:
        """Reward spec.

        Returns:
            Dict[str, specs.Array]: spec for rewards.
        """
        reward_specs = {}
        for agent in self._agents:
            reward_specs[agent] = np.array(1, "float32")
        return reward_specs

    def discount_spec(self) -> Dict:
        """Discount spec.

        Returns:
            Dict[str, specs.BoundedArray]: spec for discounts.
        """
        discount_specs = {}
        for agent in self._agents:
            discount_specs[agent] = np.array(1, "float32")
        return discount_specs

    @property
    def agents(self) -> List:
        """Agents still alive in env (not done).

        Returns:
            List: alive agents in env.
        """
        return self._agents

    @property
    def possible_agents(self) -> List:
        """All possible agents in env.

        Returns:
            List: all possible agents in env.
        """
        return self._agents

    @property
    def environment(self):
        """Returns the wrapped environment.

        Returns:
            ParallelEnv: parallel env.
        """
        return self._environment

    def get_stats(self):
        """Return extra stats to be logged.

        Returns:
            extra stats to be logged.
        """
        return {}

    def __getattr__(self, name: str) -> Any:
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
        
    def _decode_fn(self, record_bytes):
        schema = get_schema(self)

        example = tf.io.parse_single_example(
            record_bytes,
            tree.map_structure(
                lambda x: tf.io.FixedLenFeature([], dtype=tf.string), schema
            ),
        )

        for key, item in schema.items():
            example[key] = tf.io.parse_tensor(example[key], item.dtype)

        observations = {}
        actions = {}
        rewards = {}
        discounts = {}
        legal_actions = {}
        extras = {}
        for agent in self._agents:
            observations[agent] = example[agent + "_observations"]
            legal_actions[agent] = example[agent + "_legal_actions"]
            actions[agent] = example[agent + "_actions"]
            rewards[agent] = example[agent + "_rewards"]
            discounts[agent] = example[agent + "_discounts"]

        # Make OLTs
        for agent in self._agents:
            observations[agent] = OLT(
                observation=observations[agent],
                legal_actions=legal_actions[agent],
                terminal=tf.zeros(
                    1, dtype="float32"
                ),  # TODO only a place holder for now
            )

        ## Extras
        # Zero padding
        zero_padding_mask = example["zero_padding_mask"]
        extras["zero_padding_mask"] = zero_padding_mask
        # Global env state
        if "env_state" in example:
            extras["s_t"] = example["env_state"]

        # Start of episode
        start_of_episode = tf.zeros(
            1, dtype="float32"
        )  # TODO only a place holder for now

        # If "episode return" in example
        extras["episode_return"] = example["episode_return"]

        # Pack into Step
        step = Step(
            observations=observations,
            actions=actions,
            rewards=rewards,
            discounts=discounts,
            start_of_episode=start_of_episode,
            extras=extras,
        )

        return step
