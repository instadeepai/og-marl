import numpy as np
from flashbax.vault import Vault

def get_saco(rel_dir,vault_name,uid):
    vlt = Vault(rel_dir=rel_dir, vault_name=vault_name, vault_uid=uid)
    all_data = vlt.read()
    offline_data = all_data.experience
    del vlt
    del all_data

    states = offline_data['infos']["state"]

    num_tot = states.shape[1]

    reshaped_actions = offline_data["actions"].reshape((*offline_data["actions"].shape[:2],-1))
    state_pairs = np.concatenate((states,reshaped_actions),axis=-1)

    unique_vals = np.unique(state_pairs,axis=1)

    saco = len(unique_vals)/num_tot

    return saco