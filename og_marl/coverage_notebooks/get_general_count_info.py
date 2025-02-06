# This file provides the necessary functions for 
#   taking a Vault (by specifying its name, relative directory and uid)
#   for each transition, taking the state, per-agent observation and joint obs
#   for each transition, optionally adding to that state, obs or joint obs another part of the transition
#   for each transition, having some target that you want to bucket according to the appended state, obs or joint obs pairs
#   storing all the information you get with pickle in the uid folder

# Example: I want to know which rewards were linked to unique state-action, obs-action and joint-obs-action pairs.
# I use the main function create_count_information
# I specify that add_to_state is "actions"
# And that get_variability_of is "rewards"
# as output of get_unique_obs_actions_with_reward, you will get:
#   num_unique: number of unique state-action pairs, obs-action pairs and joint-obs-action pairs, 
#   count_vals: the set of counts of the unique pairs (i.e. if ab came up twice, cd came up three times and ef came up twice, count_vals={2,3}), 
#   count_counts: the number of times each count value appeared (i.e. if ab came up twice, cd came up three times and ef came up twice, count_counts=(2,1)), 
#   bucketed_stats: per-unique-state-action-pair, what are the rewards experienced for those? we only care about the pairs that repeat at least once, 
#   top_5_vals: what are the top 5 most-occurring pairs?,
#   top_5_counts: how often do the 5 most-frequent pairs appear?, 
#   keys: just the keys for the different types of pairs ("state","joint","agent_0" etc) The main reason why we have this is to abstract the number of agents away
# These values get stored.
# Finally, we have the processed_rewards_info
# This is basically figuring out the prob of the same state-action pair giving rise to the same reward, when the state action pair is seen for a second time
# This number is highly influenced though by the number of repeats
# For example, you may have that (s1,a1) appears twice, and its reward is the same.
# And then you have (s2,a2), which appears fifteen times - much more likely that the rewards won't be the same.


import numpy as np
from flashbax.vault import Vault
import pickle
import os

def get_prepruned_count_information(countables, linked_stats):
    """ 
    This function assumes you want to count the countables to
    - get the number of unique countables,
    - get the frequencies of the counts,
    and, where a countable appears more than once,
    - retain the values of the linked stats so that you can see if there is variance.
    """

    vals, indices, counts = np.unique(countables,axis=1, return_inverse=True,return_counts=True)

    # get top five and their respective counts
    top_five_indices = np.argpartition(counts,-5)[-5:]
    # print(top_five_indices)
    top_five_vals = vals[0,top_five_indices]
    top_five_counts = counts[top_five_indices]

    # Want power law information? This is what you need! Many transitions only appear once.
    count_vals, count_counts = np.unique(counts, return_counts=True)

    # this should be universal-check
    num_unique = vals.shape[1]

    bucketed_stats = {}

    # vals must be more than a single value, since it's cast to tuples
    for i, idx in enumerate(indices):
        if counts[idx]>1:
            try:
                bucketed_stats[idx].append(tuple(linked_stats[i])) # maybe this needs to change, but unsure
            except:
                bucketed_stats[idx] = [tuple(linked_stats[i])]
            
    return num_unique, count_vals, count_counts, bucketed_stats, top_five_vals, top_five_counts

def get_unique_obs_actions_with_reward(offline_data, return_per_agent, stat_add_to_state="", stat_get_variation='actions'):
    '''
    Here, we concatenate actions to observations, states.
    API:
    offline_data["observations"]: must be of form (B,T,A,x)
    offline_data["rewards"]: must be of form (B,T,A)
    offline_data["actions"]: must be of form (B,T,A,x) [or (B,T,A) -> will be converted to (B,T,A,1)]
    offline_data['infos']["state"]: must be of form (B,T,x)
    '''


    # Make the blocks to be counted

    # match observation shape to action shape for concatenation. Could be B,T,A,x, with any form of x.
    # If actions have shape x > 1, flatten it.
    if len(stat_add_to_state)>0:
        # stat_add_to_state = stat_add_to_state[0] # for some reason it makes it a tuple automatically
        # works for actions - try generalise in future
        if len(offline_data[stat_add_to_state].shape)==3:
            # expand the actions dimensions for easy adding
            stat_block = np.expand_dims(offline_data[stat_add_to_state],axis=-1)

        joint_obs_pairs = np.concatenate((offline_data["observations"],stat_block),axis=-1)


        stat_block_without_agent_dim = offline_data[stat_add_to_state].reshape((*offline_data[stat_add_to_state].shape[:2],-1))
        state_pairs = np.concatenate((offline_data['infos']["state"],stat_block_without_agent_dim),axis=-1)
    else:
        joint_obs_pairs = offline_data["observations"]
        state_pairs = offline_data['infos']["state"]

    # Correctly shape the things where variance is measured. Improve to be mre general soon
    if stat_get_variation=='rewards':
        stat_to_bucket = offline_data[stat_get_variation][0, :, 0][...,np.newaxis] # rewards are to be bucketed and so need low dimensionality
    elif stat_get_variation=='actions':
        stat_to_bucket = offline_data[stat_get_variation].reshape((*offline_data[stat_get_variation].shape[:2],-1))


    keys = []
    num_unique = {}
    count_vals = {}
    count_counts = {}
    bucketed_stats = {}
    top_5_vals = {}
    top_5_counts = {}

    if return_per_agent:
        for agent_id in range(offline_data['actions'].shape[2]):

            # add to dicts
            new_key = "agent_"+str(agent_id)
            keys.append(new_key)

            # get count, indices, vals
            num_unique[new_key], count_vals[new_key], count_counts[new_key], bucketed_stats[new_key], top_5_vals[new_key], top_5_counts[new_key] = get_prepruned_count_information(joint_obs_pairs[:,:,agent_id,:],stat_to_bucket)

    new_key = "joint"
    keys.append(new_key)
    num_unique[new_key], count_vals[new_key], count_counts[new_key], bucketed_stats[new_key], top_5_vals[new_key], top_5_counts[new_key] = get_prepruned_count_information(joint_obs_pairs,stat_to_bucket)

    new_key = 'state'
    keys.append(new_key)
    num_unique[new_key], count_vals[new_key], count_counts[new_key], bucketed_stats[new_key], top_5_vals[new_key], top_5_counts[new_key] = get_prepruned_count_information(state_pairs,stat_to_bucket)

    return num_unique, count_vals, count_counts, bucketed_stats, top_5_vals, top_5_counts, keys


def get_dist_(bucketed_stats,keys):
    probs = {}
    num_repeated_occurrences = {}
    for k,key in enumerate(keys):
        these_rewards = bucketed_stats[key]
        total_transitions_available = np.sum([len(bucket) for bucket in these_rewards.values()])
        prob_of_repeat = 0

        for reward_bucket in these_rewards.values():
            len_bucket = len(reward_bucket)
            prob_of_bucket = len_bucket/total_transitions_available
            _, reward_counts = np.unique(reward_bucket,return_counts=True)

            prob_2_rewards_the_same = 0

            for reward_count in reward_counts:
                prob_of_reward_given_bucket = reward_count/len_bucket
                prob_of_second_chosen_reward_same_as_first = (reward_count-1)/(len_bucket-1)

                prob_2_rewards_the_same +=  prob_of_reward_given_bucket*prob_of_second_chosen_reward_same_as_first
            
            prob_of_repeat += prob_of_bucket*prob_2_rewards_the_same
            
        probs[key] = prob_of_repeat
        num_repeated_occurrences[key] = total_transitions_available

    return probs, num_repeated_occurrences


def create_count_information(rel_dir,vault_name,uid,get_variability_of,add_to_state,store_raw_reward_info=False):
    print(f"{rel_dir}/{vault_name}/{uid}")
    vlt = Vault(rel_dir=rel_dir, vault_name=vault_name, vault_uid=uid)
    all_data = vlt.read()
    offline_data = all_data.experience
    del vlt
    del all_data

    num_unique, count_vals, count_counts, bucketed_stats, top_5_vals, top_5_counts, keys = get_unique_obs_actions_with_reward(offline_data,stat_get_variation=get_variability_of,stat_add_to_state=add_to_state,return_per_agent=True)

    base_dir_for_file = rel_dir+"/"+vault_name+"/"+uid+"/"+f"var_{get_variability_of}_wrt_{add_to_state}"
    os.makedirs(base_dir_for_file,exist_ok=True)

    with open(base_dir_for_file+"/number_unique.pickle","wb") as f:
        pickle.dump(num_unique,f)

    with open(base_dir_for_file+"/top_five.pickle","wb") as f:
        pickle.dump((top_5_vals, top_5_counts),f)

    with open(base_dir_for_file+"/count_frequencies.pickle","wb") as f:
        pickle.dump((count_vals, count_counts),f)

    if store_raw_reward_info:
        with open(base_dir_for_file+"/bucketed_rewards.pickle","wb") as f:
            pickle.dump(bucketed_stats,f)

    prob, repeated_occurrences = get_dist_(bucketed_stats,keys)

    with open(base_dir_for_file+"/processed_reward_info.pickle","wb") as f:
        pickle.dump((prob, repeated_occurrences),f)

    return keys