import jax
import jax.numpy as jnp
import numpy as np
from flashbax.vault import Vault
from og_marl.vault_utils.subsample_smaller import stitch_vault_from_sampled_episodes_
from og_marl.vault_utils.download_vault import check_directory_exists_and_not_empty
import pickle
from os.path import exists

# cumulative summing per-episode
def get_episode_returns_and_term_idxes(offline_data):

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
def sort_concat(returns,eps_ends):

    episode_start_idxes = eps_ends[:-1]+1
    episode_start_idxes = jnp.insert(episode_start_idxes,0,0).reshape(-1,1)
    sorting_idxes = jnp.lexsort(jnp.array([returns[:,0]]), axis=-1)
    # print(sorting_idxes)

    return_start_end = jnp.concatenate([returns,episode_start_idxes.reshape(-1,1),eps_ends],axis=-1)

    # return, start, end sorted by return value ascending
    sorted_return_start_end = return_start_end[sorting_idxes]
    return sorted_return_start_end

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


def subsample_similar(first_vault_info, second_vault_info, new_rel_dir, new_vault_name):

    # check that a subsampled vault by the same name does not already exist
    if check_directory_exists_and_not_empty(f"./{new_rel_dir}/{new_vault_name}"):
        print(f"Vault '{new_rel_dir}/{new_vault_name.strip('.vlt')}' already exists. To subsample from scratch, please remove the current subsampled vault from its directory.")
        return

    # store the return start end info
    return_start_end_list = []
    experience_list = []
    for vault_info in [first_vault_info, second_vault_info]:

        # unpack info
        rel_dir = vault_info['rel_dir']
        vault_name = vault_info['vault_name']
        vault_uid = vault_info['uid']

        # check that the vault to be subsampled exists
        if not check_directory_exists_and_not_empty(f"./{rel_dir}/{vault_name}"):
            print(f"Vault './{rel_dir}/{vault_name}' does not exist and cannot be subsampled.")
            return
        
        # get the vault and unpack its experience
        vlt = Vault(rel_dir=rel_dir, vault_name=vault_name,vault_uid=vault_uid)

        all_data = vlt.read()
        offline_data = all_data.experience
        del all_data

        # basic information about the vault
        returns, episode_end_idxes = get_episode_returns_and_term_idxes(offline_data)

        # no need to save the returns if they already exist! 
        if not exists(f"./{rel_dir}/{vault_name}/{vault_uid}/returns.pickle"):
            with open(f"{rel_dir}/{vault_name}/{vault_uid}/returns.pickle","wb") as f:
                pickle.dump(returns,f)

    
        # sort the episodes by return and store them
        return_start_end_list.append(sort_concat(returns,episode_end_idxes))
        experience_list.append(offline_data)

    # extract the returns specfically
    base_ret = return_start_end_list[0][:,0]
    comp_ret = return_start_end_list[1][:,0]

    # get two sets of episode indices, s.t. the episode returns are as similar as possible
    b,c = get_idxes_of_similar_subsets(base_ret,comp_ret,tol=0.01)

    # take the specified episodes
    vlt1_samples = return_start_end_list[0][b,:]
    vlt2_samples = return_start_end_list[1][c,:]

    for experience,sample_idxes,uid in zip(experience_list,[vlt1_samples,vlt2_samples],[first_vault_info['uid'], second_vault_info['uid']]):

        # use the specified episodes to write a new vault
        timesteps_written = stitch_vault_from_sampled_episodes_(
                                                experience=experience,
                                                len_start_end_sample=sample_idxes,
                                                dest_vault_name=new_vault_name,
                                                vault_uid=uid,
                                                rel_dir=new_rel_dir,
                                                )
        
        # save the number of timesteps actually written
        with open(f"{new_rel_dir}/{new_vault_name}/{uid}/timesteps.pickle","wb") as f:
            pickle.dump(timesteps_written,f)

        # save the returns list of the episodes
        with open(f"{new_rel_dir}/{new_vault_name}/{uid}/returns.pickle","wb") as f:
            pickle.dump(sample_idxes[:,0],f)

    return