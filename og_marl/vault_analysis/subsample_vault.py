# Copyright 2024 InstaDeep Ltd. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import jax
import pickle
import os
import numpy as np
import flashbax as fbx
from flashbax.vault import Vault
import copy
import flashbax
from flashbax.buffers.trajectory_buffer import TrajectoryBufferState
from og_marl.vault_analysis.offline_dataset import analyse_vault
from og_marl.vault_analysis.offline_dataset import check_directory_exists_and_not_empty

from typing import Dict, List
from chex import Array

# subsample vault smaller

def get_length_start_end(
    experience: Dict[str, Array], terminal_key: str = "terminals"
) -> Array:
    
    # extract terminals
    terminal_flag = experience[terminal_key][0, :, ...].all(axis=-1)

    # list of indices of terminal entries
    term_idxes = np.argwhere(terminal_flag)

    # construct start indices using entries after terminals
    start_idxes = np.zeros_like(term_idxes)
    start_idxes[1:] = term_idxes[:-1]

    # get the length per-episode (TODO maybe redundant)
    lengths = term_idxes-start_idxes

    # concatenate for easier unpacking
    len_start_end = np.concatenate((lengths,start_idxes,term_idxes),axis=1)

    return len_start_end


def select_episodes_uniformly_up_to_n_transitions(len_start_end,n):

    # shuffle idxes of all the episodes from the vault
    shuffled_idxes = np.arange(len_start_end.shape[0])
    np.random.shuffle(shuffled_idxes)

    # grab the lengths of the shuffled episodes in order and cumulatively sum
    shuffled_lengths = len_start_end[shuffled_idxes,0]
    shuffled_cumsum_of_eps_lengths = np.cumsum(shuffled_lengths)

    # select the first x episodes until we reach the target in the cumsum
    selected_idxes = shuffled_idxes[np.where(shuffled_cumsum_of_eps_lengths<n)]
    print(shuffled_cumsum_of_eps_lengths[-1])

    # get the lengths, starts and ends of the data- shuffled subsample
    randomly_sampled_len_start_end = len_start_end[selected_idxes,:]

    return randomly_sampled_len_start_end


# given the indices of the required episodes, stitch a vault and save under a user-specified name
def stitch_vault_from_sampled_episodes_(
    experience: Dict[str, Array], len_start_end_sample, dest_vault_name: str, vault_uid: str, rel_dir: str, n: int
) -> Array:

    dest_buffer = fbx.make_trajectory_buffer(
        # Sampling parameters
        sample_batch_size=1,
        sample_sequence_length=1,
        period=1,
        # Not important in this example, as we are not adding to the buffer
        max_length_time_axis=n,
        min_length_time_axis=100,
        add_batch_size=1,
    )

    dummy_experience = jax.tree_map(lambda x: x[0, 0, ...], experience)

    dest_state = dest_buffer.init(dummy_experience)
    buffer_add = jax.jit(dest_buffer.add, donate_argnums=0)
    dest_vault = flashbax.vault.Vault(
        experience_structure=dest_state.experience,
        vault_name=dest_vault_name,
        vault_uid=vault_uid,
        rel_dir = rel_dir,
    )

    for start,end in zip(len_start_end_sample[:,1],len_start_end_sample[:,2]):
        sample_experience = jax.tree_util.tree_map(lambda x: x[:,int(start+1):int(end+1),...], experience)
        dest_state = buffer_add(dest_state, sample_experience)
        
    timesteps_written = dest_vault.write(dest_state)

    print(timesteps_written)
     
    return timesteps_written

def subsample_smaller_vault(rel_dir: str,vault_name: str, vault_uid: str, target_number_of_transitions: int = 500000):

    vlt = Vault(rel_dir=rel_dir, vault_name=vault_name+'.vlt', vault_uid=vault_uid)

    # read in data
    all_data = vlt.read()
    offline_data = all_data.experience

    # get per-episode length, start and end indexes in the vault data
    len_start_end = get_length_start_end(offline_data)

    # get a number of transitions from randomly sampled episodes within one episode length of the target num of transitions
    len_start_end_sample = select_episodes_uniformly_up_to_n_transitions(len_start_end,target_number_of_transitions)

    # access the correctly sampled episodes from the original vault and write it to a new destination vault
    new_vault_name = vault_name + "_" + str(target_number_of_transitions)+'.vlt'
    timesteps_written = stitch_vault_from_sampled_episodes_(offline_data, len_start_end_sample, new_vault_name, vault_uid, rel_dir, target_number_of_transitions)

    path = rel_dir+vault_name+".vlt/"+vault_uid+"/timesteps.pickle"
    print(path)

    with open(rel_dir+new_vault_name+"/"+vault_uid+"/timesteps.pickle","wb") as f:
        pickle.dump(timesteps_written,f)

    return new_vault_name

def find_all_datasets_in_vault(rel_dir,vault_name):
    vault_uids = sorted(
        next(os.walk(os.path.join(rel_dir, vault_name)))[1],
        reverse=True,
    )
    return vault_uids