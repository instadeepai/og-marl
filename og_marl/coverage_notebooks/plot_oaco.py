import numpy as np
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import pandas as pd
import os

def plot_oaco(rel_dir,vault_name,vault_uids,random_dataset_pos=-2,norm_wrt_state=False,variability_type="var_actions_wrt_None"):
    num_unique = {}
    for uid in vault_uids:
        with open(os.path.join(rel_dir,vault_name,uid,variability_type,"number_unique.pickle"),"rb") as f:
            num_unique[uid] = pickle.load(f)

    unique_df = pd.DataFrame(num_unique)
    unique_df.transpose

    # normalise wrt random if random dataset exists
    if random_dataset_pos!=-2:
        unique_df = unique_df.div(unique_df.iloc[:,random_dataset_pos],axis=0)

    # normalise wrt random if random dataset exists
    if norm_wrt_state:
        unique_df = unique_df.div(unique_df.iloc[-1,:],axis=1)

    sns.heatmap(unique_df,annot=True,fmt=".2f",square=False)
    # plt.title(vault_name)
    plt.savefig(os.path.join(rel_dir,vault_name,f"{variability_type}_OACo_heatmap.pdf"),format='pdf',bbox_inches='tight')
    plt.show()
    return

def plot_top_five(rel_dir,vault_name,vault_uids,variability_type):
    num_unique = {}
    top_5_vals = {}
    top_5_counts = {}
    for uid in vault_uids:
        with open(os.path.join(rel_dir,vault_name,uid,variability_type,"number_unique.pickle"),"rb") as f:
            num_unique[uid] = pickle.load(f)

        with open(os.path.join(rel_dir,vault_name,uid,variability_type,"top_five.pickle"),"rb") as f:
           (top_5_vals, top_5_counts) = pickle.load(f)

    
    
    return


def plot_count_frequencies(rel_dir,vault_name,vault_uids,variability_type):
    count_vals = {}
    count_counts = {}
    for uid in vault_uids:
        with open(os.path.join(rel_dir,vault_name,uid,variability_type,"count_frequencies.pickle"),"rb") as f:
            (count_vals[uid], count_counts[uid]) = pickle.load(f)

    keys = list(count_vals[uid].keys())

    # plot per-agent
    fig, ax = plt.subplots(1,len(keys),figsize=(4*len(keys),3))

    for j, key in enumerate(keys):
        for uid in vault_uids:

            ax[j].scatter(np.log(count_vals[uid][key]),np.log(count_counts[uid][key]),label=uid)
            ax[j].set_xlabel("Counts")
            ax[j].set_ylabel("How often they occur")
            ax[j].set_title(key)

    ax[j].legend(bbox_to_anchor = (1,1))

    plt.savefig(os.path.join(rel_dir,vault_name,f"{variability_type}_per_agent_loglog.pdf"),format='pdf',bbox_inches="tight")
    plt.show()

    # plot per-dataset
    fig, ax = plt.subplots(1,len(vault_uids), figsize=(4*len(vault_uids),3))

    for key in keys:
        for i,uid in enumerate(vault_uids):

            ax[i].scatter(np.log(count_vals[uid][key]),np.log(count_counts[uid][key]),label=key,s=8)
            ax[i].set_title(uid)
    ax[i].legend(bbox_to_anchor=(1,1))

    plt.savefig(os.path.join(rel_dir,vault_name,f"{variability_type}_per_dataset_loglog.pdf"),format='pdf',bbox_inches='tight')
    plt.show()
    return


def plot_reward_variability(rel_dir,vault_name,vault_uids,variability_type):
    probs = {}
    for uid in vault_uids:
        with open(os.path.join(rel_dir,vault_name,uid,variability_type,"processed_reward_info.pickle"),"rb") as f:
            (probs[uid], _) = pickle.load(f)

    probs_df = pd.DataFrame(probs)
    probs_df.transpose

    sns.heatmap(probs_df,annot=True)
    # plt.title(vault_name)
    plt.savefig(os.path.join(rel_dir,vault_name,f"{variability_type}_reward_variability_heatmap.pdf"),format='pdf',bbox_inches='tight')
    plt.show()
    return
