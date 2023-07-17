from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
import reverb
import tensorflow as tf
import tree
import pandas as pd
import seaborn as sb
from mava.specs import MAEnvironmentSpec
from mava.types import OLT
from mava.adders.reverb.base import Step

from og_marl.offline_tools.offline_environment_logger import get_schema

class MAOfflineDataset:
    def __init__(
        self,
        environment,
        logdir,
        batch_size=32,
        shuffle_buffer_size=1000,
        return_pytorch_tensors=False,
    ):
        self._environment = environment
        self._schema = get_schema(environment)
        self._spec = MAEnvironmentSpec(environment)
        self._agents = self._spec.get_agent_ids()
        self._return_pytorch_tensors = return_pytorch_tensors

        file_path = Path(logdir)
        filenames = [
            str(file_name) for file_name in file_path.glob("**/*.tfrecord")
        ]
        filename_dataset = tf.data.Dataset.from_tensor_slices(filenames)
        self._no_repeat_dataset = filename_dataset.interleave(
            lambda x: tf.data.TFRecordDataset(x, compression_type="GZIP").map(
                self._decode_fn
            ),
            cycle_length=None,
            num_parallel_calls=2,
            deterministic=False,
            block_length=None,
        )

        self._dataset = (
            self._no_repeat_dataset.shuffle(
                buffer_size=shuffle_buffer_size, reshuffle_each_iteration=False
            )
            .batch(batch_size)
            .repeat()
        )
        self._batch_size = batch_size

        self._dataset = iter(self._dataset)

    def _decode_fn(self, record_bytes):
        example = tf.io.parse_single_example(
            record_bytes,
            tree.map_structure(
                lambda x: tf.io.FixedLenFeature([], dtype=tf.string), self._schema
            ),
        )

        for key, item in self._schema.items():
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
        reverb_sample_data = Step(
            observations=observations,
            actions=actions,
            rewards=rewards,
            discounts=discounts,
            start_of_episode=start_of_episode,
            extras=extras,
        )

        # Make reverb sample so that interface same as in online algos
        reverb_sample_info = reverb.SampleInfo(
            key=-1, probability=-1.0, table_size=-1, priority=-1.0
        )  # TODO only a place holder for now

        # Rever sample
        reverb_sample = reverb.ReplaySample(
            info=reverb_sample_info, data=reverb_sample_data
        )

        return reverb_sample

    def __iter__(self):
        return self

    def __next__(self):
        sample = next(self._dataset)

        while list(sample.data.rewards.values())[0].shape[0] < self._batch_size:
            sample = next(self._dataset)

        return sample

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
            return getattr(self._dataset, name)

    def profile(self, filename):
        sb.set()
        plt.clf()
        plt.tight_layout()
        all_returns = []
        for item in self._no_repeat_dataset:
            if "episode_return" in item.data.extras:
                all_returns.append(item.data.extras["episode_return"].numpy())
            else:
                rewards = list(item.data.rewards.values())[
                    0
                ]  # Assume all agents have the same reward
                undiscounted_return = tf.reduce_sum(rewards)
                all_returns.append(undiscounted_return.numpy())
        plt.xlabel("Episode Returns")
        plt.ylabel("Count")
        num_bins = 50
        # plt.margins(x=0)
        # plt.xticks(np.arange(num_bins + 1))
        plt.hist(all_returns, num_bins)
        plt.savefig(filename)
        dataset_stats = pd.Series(all_returns).describe().to_dict()
        dataset_stats["mode"] = max(set(all_returns), key=all_returns.count)
        return dataset_stats, all_returns
