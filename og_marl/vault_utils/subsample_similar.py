import jax
import jax.numpy as jnp
import numpy as np
import flashbax as fbx
from flashbax.vault import Vault
from scipy.stats import norm
import matplotlib.pyplot as plt
import copy
import flashbax
from flashbax.buffers.trajectory_buffer import TrajectoryBufferState


# just gathering good, medium and poor vaults in a list of unread vaults
def get_all_vaults(rel_dir,vault_name,vault_uids=["Good","Medium","Poor"]):
    vlts = []
    for quality in vault_uids:
        vlts.append(Vault(rel_dir=rel_dir, vault_name=vault_name, vault_uid=quality))
    return vlts

# cumulative summing per-episode
def get_episode_returns_and_term_idxes(vlt):
    all_data = vlt.read()
    offline_data = all_data.experience

    rewards = offline_data['rewards'][0, :, 0]
    terminal_flag = offline_data['terminals'][0, :, ...].all(axis=-1)

    # assert terminal_flag[-1]==True

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
def sort_concat(returns,eps_ends, ids):

    episode_start_idxes = eps_ends[:-1]+1
    episode_start_idxes = jnp.insert(episode_start_idxes,0,0).reshape(-1,1)
    sorting_idxes = jnp.lexsort(jnp.array([returns[:,0]]), axis=-1)
    # print(sorting_idxes)

    return_start_end = jnp.concatenate([returns,episode_start_idxes.reshape(-1,1),eps_ends,ids],axis=-1)

    # return, start, end sorted by return value ascending
    sorted_return_start_end = return_start_end[sorting_idxes]
    return sorted_return_start_end

# given bin edges and a sorted array of values, get the bin number per value
def get_bin_numbers(sorted_values,bin_edges):
    bin_numbers = np.zeros_like(sorted_values)
    
    def get_bin_number(bin_num,value):
        is_overflowing = value>bin_edges[bin_num]

        if is_overflowing:
            bin_num += 1
            is_doubly_overflowing = value>bin_edges[bin_num]
            if is_doubly_overflowing:
                bin_num = get_bin_number(bin_num,value)
            
        return bin_num
    
    bin_bookmark = 0

    for i,val in enumerate(sorted_values):
        bin_bookmark = get_bin_number(bin_bookmark,val)
        bin_numbers[i] = bin_bookmark

    return bin_numbers

# sample from pdf according to heights
# BIG NOTE: CHECK THE DISPARITY, OTHERWISE YOUR DISTRIBUTION WILL BE TOO MUCH
def episode_idxes_sampled_from_pdf(pdf,bar_heights):
    num_to_sample = np.round(pdf).astype(int)
    sample_range_edges = np.concatenate([[0],np.cumsum(bar_heights)])

    assert num_to_sample.shape==bar_heights.shape

    target_sample_idxes = []
    for i,n_sample in enumerate(num_to_sample):
            sample_base = np.arange(sample_range_edges[i],sample_range_edges[i+1])
            print(sample_base)
            if n_sample<=0:
                 pass
            # if we sample more than all in the bar
            else:
                if n_sample>=bar_heights[i]:
                    sample_rest = np.random.choice(sample_base,n_sample-bar_heights[i],replace=True)
                    sample = np.concatenate([sample_base,sample_rest])
                else:
                    sample = np.random.choice(sample_base,n_sample,replace=True) #make false for no replace
                target_sample_idxes = target_sample_idxes+list(np.sort(sample))
    return target_sample_idxes

def preprocess(rel_dir,vault_name):
    vlts = get_all_vaults(rel_dir, vault_name)

    # get returns, term idxes for each episode per vault
    returns_list = []
    episode_end_list = []
    vault_ids = []
    for j,vault in enumerate(vlts):
        print(j)
        returns, episode_end_idxes = get_episode_returns_and_term_idxes(vault)
        returns_list.append(returns)
        episode_end_list.append(episode_end_idxes)
        vault_ids.append(jnp.zeros_like(returns)+j)

    # make np compatible
    all_returns = jnp.concatenate(returns_list)
    all_episode_end_idxes = jnp.concatenate(episode_end_list)
    all_vault_ids = jnp.concatenate(vault_ids)

    # concatenate then sort all results
    all_sorted_return_start_end = sort_concat(all_returns,all_episode_end_idxes, all_vault_ids)
    return vlts, all_sorted_return_start_end

def preprocess_vault(rel_dir,vault_name, vault_uid):
    vault = Vault(rel_dir=rel_dir, vault_name=vault_name,vault_uid=vault_uid)

    returns, episode_end_idxes = get_episode_returns_and_term_idxes(vault)
    
    all_sorted_return_start_end = sort_concat(returns,episode_end_idxes, jnp.zeros_like(returns))
    return vault, all_sorted_return_start_end

def bin_processed_data(all_sorted_return_start_end, n_bins=500):
    # get bin edges, including final endpoint
    bin_edges = jnp.linspace(start=min(min(all_sorted_return_start_end[:,0]),0), stop = max(all_sorted_return_start_end[:,0]), num=n_bins,endpoint=True)
    print(all_sorted_return_start_end.shape[0])

    # get bin numbers
    bin_numbers = get_bin_numbers(all_sorted_return_start_end[:,0], bin_edges)
    print(bin_numbers.shape[0])

    bar_labels, bar_heights= np.unique(bin_numbers,return_counts=True)

    padded_heights = np.zeros(n_bins)
    for bar_l, bar_h in zip(bar_labels,bar_heights):
        padded_heights[int(bar_l)] = bar_h

    return bar_labels, bar_heights, padded_heights, bin_edges, bin_numbers


def get_idxes_of_similar_subsets(base_returns, comp_returns, tol=0.1):
    base_selected_idxes = []
    comp_selected_idxes = []

    comp_idx = 0

    for i, ret in enumerate(base_returns):
        # print("Run "+str(i))
        ret_dealt_with = False
        while comp_idx<len(comp_returns) and (comp_returns[comp_idx]<=(ret+tol)) and not ret_dealt_with:
            # check comp is in bracket below
            if np.abs((ret-comp_returns[comp_idx]))<tol:
                base_selected_idxes.append(i)
                comp_selected_idxes.append(comp_idx)
                ret_dealt_with = True
            comp_idx+= 1

    return base_selected_idxes, comp_selected_idxes