from typing import Tuple

import chex
import jax.numpy as jnp
from flax import linen as nn

# General shapes legend:
# B: batch size
# S: sequence length
# C: number of agents per chunk of sequence


def train_encoder_fn(
    encoder: nn.Module,
    obs: chex.Array,
    hstate: chex.Array,
    dones: chex.Array,
    step_count: chex.Array,
    chunk_size: int,
) -> Tuple[chex.Array, chex.Array, chex.Array]:
    """Chunkwise encoding for discrete action spaces."""
    B, S = obs.shape[:2]
    obs_rep = jnp.zeros((B, S, encoder.net_config.embed_dim))

    # Apply the encoder per chunk
    num_chunks = S // chunk_size
    assert num_chunks > 0
    for chunk_id in range(0, num_chunks):
        start_idx = chunk_id * chunk_size
        end_idx = (chunk_id + 1) * chunk_size
        # Chunk obs, dones, and step_count
        chunk_obs = obs[:, start_idx:end_idx]
        chunk_dones = dones[:, start_idx:end_idx]
        chunk_step_count = step_count[:, start_idx:end_idx]
        chunk_obs_rep, hstate = encoder(
            chunk_obs, hstate, chunk_dones, chunk_step_count
        )
        obs_rep = obs_rep.at[:, start_idx:end_idx].set(chunk_obs_rep)

    return obs_rep, hstate


def act_encoder_fn(
    encoder: nn.Module,
    obs: chex.Array,
    decayed_hstate: chex.Array,
    step_count: chex.Array,
    chunk_size: int,
) -> Tuple[chex.Array, chex.Array, chex.Array]:
    """Chunkwise encoding for ff-Sable and for discrete action spaces."""
    B, C = obs.shape[:2]
    obs_rep = jnp.zeros((B, C, encoder.net_config.embed_dim))

    # Apply the encoder per chunk
    num_chunks = C // chunk_size
    for chunk_id in range(0, num_chunks):
        start_idx = chunk_id * chunk_size
        end_idx = (chunk_id + 1) * chunk_size
        # Chunk obs and step_count
        chunk_obs = obs[:, start_idx:end_idx]
        chunk_step_count = step_count[:, start_idx:end_idx]
        chunk_obs_rep, decayed_hstate = encoder.recurrent(
            chunk_obs, decayed_hstate, chunk_step_count
        )
        obs_rep = obs_rep.at[:, start_idx:end_idx].set(chunk_obs_rep)

    return obs_rep, decayed_hstate
