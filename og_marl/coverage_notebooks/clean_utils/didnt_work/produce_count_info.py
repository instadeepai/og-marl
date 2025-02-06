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
import pickle

def get_vals_counts_infos(pairs,rewards):
    vals, indices, counts = np.unique(pairs,axis=1, return_inverse=True,return_counts=True)
    bucketed_rewards = [[] for _ in range(vals.shape[1])]
    print("Number of unique: "+str(vals.shape[1]))
    print("Most repeated pair's count: "+str(max(counts)))

    for i, idx in enumerate(indices):
        bucketed_rewards[idx].append(float(rewards[i]))
        
    return vals, indices, counts, bucketed_rewards

def get_unique_obs_actions_with_reward(offline_data, return_per_agent):
    '''
    Here, we concatenate actions to observations, states.
    API:
    offline_data["observations"]: must be of form (B,T,A,x)
    offline_data["rewards"]: must be of form (B,T,A)
    offline_data["actions"]: must be of form (B,T,A,x) [or (B,T,A) -> will be converted to (B,T,A,1)]
    offline_data['infos']["state"]: must be of form (B,T,x)
    '''

    observations = offline_data["observations"]
    rewards = offline_data['rewards'][0, :, 0] # rewards are to be bucketed and so need low dimensionality
    actions = offline_data["actions"]
    states = offline_data['infos']["state"]

    # match observation shape to action shape for concatenation. Could be B,T,A,x, with any form of x.
    # If actions have shape x > 1, flatten it.

    if len(actions.shape)==3:
        # expand the actions dimensions for easy adding
        actions = np.expand_dims(offline_data["actions"],axis=-1)
    joint_obs_pairs = np.concatenate((observations,actions),axis=-1)

    reshaped_actions = offline_data["actions"].reshape((*offline_data["actions"].shape[:2],-1))
    state_pairs = np.concatenate((states,reshaped_actions),axis=-1)

    keys = []
    unique_pairs = {}
    indices_inverse = {}
    unique_obs_act_counts = {}
    rewards_per_pair = {}

    if return_per_agent:
        for agent_id in range(actions.shape[2]):

            # add to dicts
            new_key = "agent_"+str(agent_id)
            print(new_key)
            keys.append(new_key)

            # get count, indices, vals
            unique_pairs[new_key], indices_inverse[new_key], unique_obs_act_counts[new_key], rewards_per_pair[new_key] = get_vals_counts_infos(joint_obs_pairs[:,:,agent_id,:],rewards)

    new_key = "joint"
    print(new_key)
    keys.append(new_key)
    unique_pairs[new_key], indices_inverse[new_key], unique_obs_act_counts[new_key], rewards_per_pair[new_key] = get_vals_counts_infos(joint_obs_pairs,rewards)

    new_key = 'state'
    print(new_key)
    keys.append(new_key)
    unique_pairs[new_key], indices_inverse[new_key], unique_obs_act_counts[new_key], rewards_per_pair[new_key] = get_vals_counts_infos(state_pairs,rewards)

    return unique_pairs, indices_inverse, unique_obs_act_counts, rewards_per_pair, keys


def create_count_information(rel_dir,vault_name,uid,need_to_reconstruct=False):
    vlt = Vault(rel_dir=rel_dir, vault_name=vault_name, vault_uid=uid)
    all_data = vlt.read()
    offline_data = all_data.experience
    del vlt
    del all_data

    vals, indices, counts, rewards, keys = get_unique_obs_actions_with_reward(offline_data,return_per_agent=True)

    with open(rel_dir+"/"+vault_name+"/"+uid+"/count_info.pickle","wb") as f:
        if need_to_reconstruct:
            pickle.dump((vals,indices,counts, rewards),f)            
        else:
            pickle.dump((counts, rewards),f)

    return keys