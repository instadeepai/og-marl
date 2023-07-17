import tensorflow as tf


def gather(values, indices, axis=-1, keepdims=False):
    one_hot_indices = tf.one_hot(indices, depth=values.shape[axis])
    if len(values.shape) > 4:  # we have extra dim for distributional q-learning
        one_hot_indices = tf.expand_dims(one_hot_indices, axis=-1)
    gathered_values = tf.reduce_sum(
        values * one_hot_indices, axis=axis, keepdims=keepdims
    )
    return gathered_values

def switch_two_leading_dims(x):
    trailing_perm = []
    for i in range(2, len(x.shape)):
        trailing_perm.append(i)
    x = tf.transpose(x, perm=[1, 0, *trailing_perm])
    return x


def merge_batch_and_agent_dim_of_time_major_sequence(x):
    T, B, N = x.shape[:3]  # assume time major
    trailing_dims = x.shape[3:]
    x = tf.reshape(x, shape=(T, B * N, *trailing_dims))
    return x


def merge_time_batch_and_agent_dim(x):
    T, B, N = x.shape[:3]  # assume time major
    trailing_dims = x.shape[3:]
    x = tf.reshape(x, shape=(T * B * N, *trailing_dims))
    return x


def expand_time_batch_and_agent_dim_of_time_major_sequence(x, T, B, N):
    TNB = x.shape[:1]  # assume time major
    assert TNB == T * B * N
    trailing_dims = x.shape[1:]
    x = tf.reshape(x, shape=(T, B, N, *trailing_dims))
    return x


def expand_batch_and_agent_dim_of_time_major_sequence(x, B, N):
    T, NB = x.shape[:2]  # assume time major
    assert NB == B * N
    trailing_dims = x.shape[2:]
    x = tf.reshape(x, shape=(T, B, N, *trailing_dims))
    return x


def batch_concat_agent_id_to_obs(obs):
    B, T, N = obs.shape[:3]  # batch size, timedim, num_agents
    is_vector_obs = len(obs.shape) == 4

    agent_ids = []
    for i in range(N):
        if is_vector_obs:
            agent_id = tf.one_hot(i, depth=N)
        else:
            h, w = obs.shape[3:5]
            agent_id = tf.zeros((h, w, 1), "float32") + (i / N) + 1 / (2 * N)
        agent_ids.append(agent_id)
    agent_ids = tf.stack(agent_ids, axis=0)

    # Repeat along time dim
    agent_ids = tf.stack([agent_ids] * T, axis=0)

    # Repeat along batch dim
    agent_ids = tf.stack([agent_ids] * B, axis=0)

    if not is_vector_obs and len(obs.shape) == 5:  # if no channel dim
        obs = tf.expand_dims(obs, axis=-1)

    obs = tf.concat([agent_ids, obs], axis=-1)

    return obs


def sample_batch_agents(agents, sample, independent=False):
    # Unpack sample
    data = sample.data
    observations, actions, rewards, discounts, _, extras = (
        data.observations,
        data.actions,
        data.rewards,
        data.discounts,
        data.start_of_episode,
        data.extras,
    )

    all_observations = []
    all_legals = []
    all_actions = []
    all_rewards = []
    all_discounts = []
    all_logprobs = []
    for agent in agents:
        all_observations.append(observations[agent].observation)
        all_legals.append(observations[agent].legal_actions)
        all_actions.append(actions[agent])
        all_rewards.append(rewards[agent])
        all_discounts.append(discounts[agent])

        if "logprobs" in extras:
            all_logprobs.append(extras["logprobs"][agent])

    all_observations = tf.stack(all_observations, axis=2)  # (B,T,N,O)
    all_legals = tf.stack(all_legals, axis=2)  # (B,T,N,A)
    all_actions = tf.stack(all_actions, axis=2)  # (B,T,N,Act)
    all_rewards = tf.stack(all_rewards, axis=-1)  # (B,T,N)
    all_discounts = tf.stack(all_discounts, axis=-1)  # (B,T,N)

    if "logprobs" in extras:
        all_logprobs = tf.stack(all_logprobs, axis=2)  # (B,T,N,O)

    if not independent:
        all_rewards = tf.reduce_mean(all_rewards, axis=-1, keepdims=True)  # (B,T,1)
        all_discounts = tf.reduce_mean(all_discounts, axis=-1, keepdims=True)  # (B,T,1)

    # Cast legals to bool
    all_legals = tf.cast(all_legals, "bool")

    mask = tf.expand_dims(extras["zero_padding_mask"], axis=-1)  # (B,T,1)

    states = extras["s_t"] if "s_t" in extras else None  # (B,T,N,S)

    batch = {
        "observations": all_observations,
        "actions": all_actions,
        "rewards": all_rewards,
        "discounts": all_discounts,
        "legals": all_legals,
        "mask": mask,
        "states": states,
    }

    if "logprobs" in extras:
        batch.update({"logprobs": all_logprobs})

    return batch