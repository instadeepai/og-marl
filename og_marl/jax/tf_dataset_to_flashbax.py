import chex
import jax
import os
import jax.numpy as jnp
import flashbax as fbx
from flashbax.buffers.trajectory_buffer import TrajectoryBufferState
from pathlib import Path
import tensorflow as tf
import tree
import orbax.checkpoint
from og_marl.environments.utils import get_environment
from og_marl.tf2.utils import set_growing_gpu_memory

class FlashbaxBufferStore:
    def __init__(
        self,
        dataset_path: str,
    ) -> None:
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer() 
        options = orbax.checkpoint.CheckpointManagerOptions(
            max_to_keep=1,
            create=True,
        )
        self._manager = orbax.checkpoint.CheckpointManager(
            os.path.join(os.getcwd(), dataset_path),
            orbax_checkpointer,
            options,
            metadata=None,
        )

    def save(self, t, buffer_state):
        return self._manager.save(step=t, items=buffer_state)

    def restore_state(self):
        raw_restored = self._manager.restore(self._manager.latest_step())
        return TrajectoryBufferState(
            experience=jax.tree_util.tree_map(jnp.asarray, raw_restored['experience']),
            current_index=jnp.asarray(raw_restored['current_index']),
            is_full=jnp.asarray(raw_restored['is_full']),
        )

def get_schema_dtypes(environment):
    schema = {}
    for agent in environment.possible_agents:
        schema[agent + "_observations"] = tf.float32
        schema[agent + "_legal_actions"] = tf.float32
        schema[agent + "_actions"] = tf.int64
        schema[agent + "_rewards"] = tf.float32
        schema[agent + "_discounts"] = tf.float32

    ## Extras
    # Zero-padding mask
    schema["zero_padding_mask"] = tf.float32

    # Env state
    schema["env_state"] = tf.float32

    # Episode return
    schema["episode_return"] = tf.float32

    return schema


def make_decode_fn(schema, agents):
    def _decode_fn(record_bytes):
        example = tf.io.parse_single_example(
            record_bytes,
            tree.map_structure(
                lambda x: tf.io.FixedLenFeature([], dtype=tf.string), schema
            ),
        )

        for key, dtype in schema.items():
            example[key] = tf.io.parse_tensor(example[key], dtype)

        sample = {}
        for agent in agents:
            sample[f"{agent}_observations"] = example[f"{agent}_observations"]
            sample[f"{agent}_actions"] = example[f"{agent}_actions"]
            sample[f"{agent}_rewards"] = example[f"{agent}_rewards"]
            sample[f"{agent}_done"] = 1 - example[f"{agent}_discounts"]
            sample[f"{agent}_legals"] = example[f"{agent}_legal_actions"]
            
        sample["mask"] = example["zero_padding_mask"]
        sample["state"] = example["env_state"]
        sample["episode_return"] = tf.repeat(example["episode_return"], len(sample["state"]))

        return sample
    return _decode_fn

if __name__=="__main__":
    SCENARIO = "8m"
    DATASET = "Good"

    # set_growing_gpu_memory()

    tf.config.experimental.set_visible_devices([], "GPU")
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


    environment = get_environment("smac_v1", SCENARIO)

    # First define hyper-parameters of the buffer.
    max_length_time_axis = 20000 * 20 # Maximum length of the buffer along the time axis. 
    min_length_time_axis = 16 # Minimum length across the time axis before we can sample.
    sample_batch_size = 4 # Batch size of trajectories sampled from the buffer.
    add_batch_size = 1 # Batch size of trajectories added to the buffer.
    sample_sequence_length = 20 # Sequence length of trajectories sampled from the buffer.
    add_sequence_length = 20 # Sequence length of trajectories added to the buffer.
    period = 20 # Period at which we sample trajectories from the buffer.

    # Instantiate the trajectory buffer, which is a NamedTuple of pure functions.
    buffer = fbx.make_trajectory_buffer(
        max_length_time_axis=max_length_time_axis,
        min_length_time_axis=min_length_time_axis,
        sample_batch_size=sample_batch_size,
        add_batch_size=add_batch_size,
        sample_sequence_length=sample_sequence_length,
        period=period
    )

    store = FlashbaxBufferStore(f"{DATASET}_{SCENARIO}")

    schema = get_schema_dtypes(environment)
    agents = environment.possible_agents
    decode_fn = make_decode_fn(schema, agents)

    path_to_dataset = f"datasets/smac_v1/{SCENARIO}/{DATASET}"
    contents = os.listdir(path_to_dataset)
    directories = [content for content in contents if os.path.isdir(os.path.join(path_to_dataset, content))]

    first_sample = True
    for dir in directories:
        filenames = Path(os.path.join(path_to_dataset, dir)).glob("**/*.tfrecord")
        filenames = list(filenames)
        filenames.sort(key=lambda x: int(str(x).split("executor_sequence_log_")[-1][:-9]))
    
        for filename in filenames:
            print(filename)
            tf_record_dataset = tf.data.TFRecordDataset(filename, compression_type="GZIP").map(
                decode_fn
            )
            for sample in tf_record_dataset:
                sample = tree.map_structure(lambda x: jnp.array(x.numpy()), sample)

                if first_sample:
                    first_sample = False

                    init_sample = tree.map_structure(lambda x: jnp.array(x[0]), sample)
                    state = buffer.init(init_sample)
                
                add_sample = tree.map_structure(lambda x: jnp.expand_dims(x, axis=0), sample)
                state = buffer.add(state, add_sample)

                if (state.current_index % 1000) == 0:
                    print(round(state.current_index / max_length_time_axis, 4)*100)

                if (state.current_index % 100_000) == 0:
                    t = state.current_index // 100_000
                    store.save(t, state)

                if state.is_full:
                    break
            if state.is_full:
                break
        if state.is_full:
            break
    
        store.save(t, state)

    rng_key = jax.random.PRNGKey(0)
    batch = buffer.sample(state, rng_key)
    print("Done")