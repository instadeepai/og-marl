import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from acme.wrappers.video import make_animation

def concat_agent_id_to_obs(obs, agent_id, N):
    is_vector_obs = len(obs.shape) == 1

    if is_vector_obs:
        agent_id = tf.one_hot(agent_id, depth=N)
    else:
        h, w = obs.shape[:2]
        agent_id = tf.zeros((h, w, 1), "float32") + (agent_id / N) + 1 / (2 * N)

    if not is_vector_obs and len(obs.shape) == 2:  # if no channel dim
        obs = tf.expand_dims(obs, axis=-1)

    obs = tf.concat([agent_id, obs], axis=-1)

    return obs

def epsilon_greedy_action_selection(action_values=None, logits=None, legal_actions=None, epsilon=0.0):
    # assert (action_values and not logits) or (logits and not action_values)

    if legal_actions is None:
        legal_actions = tf.ones_like(action_values) # All actions legal
        
    legal_actions = tf.cast(legal_actions, dtype=tf.float32)

    # Dithering action distribution.
    dither_probs = (
        1
        / tf.reduce_sum(legal_actions, axis=-1, keepdims=True)
        * legal_actions
    )

    if action_values is not None:
        masked_values = tf.where(
            tf.equal(legal_actions, 1),
            action_values,
            tf.fill(tf.shape(action_values), -np.inf),
        )
        # Greedy action distribution, breaking ties uniformly at random.
        # Max value considers only valid/masked action values
        max_value = tf.reduce_max(masked_values, axis=-1, keepdims=True)
        greedy_probs = tf.cast(
            tf.equal(masked_values, max_value),
            action_values.dtype,
        )
        greedy_probs /= tf.reduce_sum(greedy_probs, axis=-1, keepdims=True)
    else:
        greedy_probs = tf.nn.softmax(logits, axis=-1)

    # Epsilon-greedy action distribution.
    probs = epsilon * dither_probs + (1 - epsilon) * greedy_probs

    # Masked probs
    masked_probs = probs * legal_actions
    masked_probs = masked_probs / tf.reduce_sum(masked_probs, axis=-1, keepdims=True)

    # Sample action from distribution
    dist = tfp.distributions.Categorical(probs=masked_probs)
    action = dist.sample()

    # Return sampled action.
    return tf.cast(action, "int64"), dist

def save_video(path: str, frames, fps, fig_size, episode_counter) -> None:
    video = make_animation(frames, fps, fig_size).to_html5_video()
    with open(f"{path}/episode_{episode_counter}.html", "w") as f:
        f.write(video)