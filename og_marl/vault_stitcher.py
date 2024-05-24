import os
import jax
import jax.numpy as jnp
import numpy as np
import flashbax as fbx
from flashbax.vault import Vault

# just gathering good, medium and poor vaults in a list of unread vaults
def _get_all_vaults(rel_dir, vault_name, vault_uids):
    vlts = []
    for quality in vault_uids:
        vlts.append(Vault(rel_dir=rel_dir, vault_name=vault_name, vault_uid=quality))
    return vlts

# cumulative summing per-episode
def _get_episode_returns_and_term_idxes(vlt):
    all_data = vlt.read()
    offline_data = all_data.experience

    rewards = offline_data['rewards'][0, :, 0]
    terminal_flag = offline_data['terminals'][0, :, ...].all(axis=-1)

    assert terminal_flag[-1] == True

    def scan_cumsum(return_so_far,prev_term_reward):
        term, reward = prev_term_reward
        return_so_far = return_so_far*(1-term)+ reward
        return return_so_far, return_so_far

    xs = (terminal_flag[:-1], rewards[1:])

    _, cumsums = jax.lax.scan(scan_cumsum, rewards[0],xs)

    term_idxes = np.argwhere(terminal_flag)

    # shift back as we did for the loop
    return cumsums[term_idxes-1], term_idxes

# first store indices of episodes, then sort by episode return.
# outputs return, start, end and vault index in vault list
def _sort_concat(returns, eps_ends, ids):

    episode_start_idxes = eps_ends[:-1]+1
    episode_start_idxes = jnp.insert(episode_start_idxes,0,0).reshape(-1,1)
    sorting_idxes = jnp.lexsort(jnp.array([returns[:,0]]), axis=-1)
    # print(sorting_idxes)

    return_start_end = jnp.concatenate([returns,episode_start_idxes.reshape(-1,1),eps_ends,ids],axis=-1)

    # return, start, end sorted by return value ascending
    sorted_return_start_end = return_start_end[sorting_idxes]
    return sorted_return_start_end

def preprocess(rel_dir, vault_name, vault_uids = ["Good","Medium","Poor"]):
    vlts = _get_all_vaults(rel_dir, vault_name, vault_uids)

    # get returns, term idxes for each episode per vault
    returns_list = []
    episode_end_list = []
    vault_ids = []
    for j,vault in enumerate(vlts):
        # print(j)
        returns, episode_end_idxes = _get_episode_returns_and_term_idxes(vault)
        returns_list.append(returns)
        episode_end_list.append(episode_end_idxes)
        vault_ids.append(jnp.zeros_like(returns)+j)

    # make np compatible
    all_returns = jnp.concatenate(returns_list)
    all_episode_end_idxes = jnp.concatenate(episode_end_list)
    all_vault_ids = jnp.concatenate(vault_ids)

    # concatenate then sort all results
    all_sorted_return_start_end = _sort_concat(all_returns,all_episode_end_idxes, all_vault_ids)
    return vlts, all_sorted_return_start_end

def _stitch_vault_from_sampled_episodes(
    vlts,
    all_sorted_return_start_end,
    vault_name,
    stitched_uid,
    rel_dir,
    top_n_samples,
):
    return_start_end_sample = all_sorted_return_start_end[-top_n_samples:, ...]

    all_data = vlts[0].read()
    offline_data = all_data.experience

    dest_buffer = fbx.make_trajectory_buffer(
        # Sampling parameters
        sample_batch_size=1,
        sample_sequence_length=1,
        period=1,
        max_length_time_axis=10_000_000,  # TODO
        min_length_time_axis=1,
        add_batch_size=1,
    )

    dummy_experience = jax.tree_map(lambda x: x[0, 0, ...], all_data.experience)
    del offline_data
    del all_data

    dest_state = dest_buffer.init(dummy_experience)
    buffer_add = jax.jit(dest_buffer.add, donate_argnums=0)
    dest_vault = Vault(
        experience_structure=dest_state.experience,
        vault_name=vault_name,
        vault_uid=stitched_uid,
        rel_dir=rel_dir,
    )

    for vault_id, vlt in enumerate(vlts):
        samples_frm_this_vault = return_start_end_sample[np.where(return_start_end_sample[:,-1]==vault_id)]
        starts = samples_frm_this_vault[:,1]
        ends = samples_frm_this_vault[:,2]

        all_data = vlt.read()
        offline_data = all_data.experience

        for start, end in zip(starts,ends):
            sample_experience = jax.tree_util.tree_map(lambda x: x[:,int(start):int(end+2),...],offline_data)
            dest_state = buffer_add(dest_state, sample_experience)
        timesteps_written = dest_vault.write(dest_state)

        # print(timesteps_written)
        del offline_data
        del all_data


def create_top_n_vault(vault_name, vault_uids, rel_dir, top_n_percent, stitched_uid = None):
    vlts, all_sorted_return_start_end = preprocess(
        rel_dir=rel_dir,
        vault_name=vault_name,
        vault_uids=vault_uids,
    )
    all_n_count = all_sorted_return_start_end.shape[0]
    top_n_samples = int(all_n_count*top_n_percent/100)

    if stitched_uid is None:
        stitched_uid = f"{vault_name}_top-{top_n_percent}"

    # Only do this if the stitched vault doesn't already exist
    if not os.path.exists(f"{rel_dir}/{vault_name}/{stitched_uid}"):
        _stitch_vault_from_sampled_episodes(
            vlts=vlts,
            all_sorted_return_start_end=all_sorted_return_start_end,
            vault_name=vault_name,
            rel_dir=rel_dir,
            stitched_uid=stitched_uid,
            top_n_samples=top_n_samples,
        )

    return stitched_uid
