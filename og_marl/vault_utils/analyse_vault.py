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

from typing import Dict, List, Tuple, Optional

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns
from chex import Array
from flashbax.vault import Vault
import numpy as np
from scipy import stats
from og_marl.vault_utils.download_vault import get_available_uids
from tabulate import tabulate


def get_structure_descriptors(
    experience: Dict[str, Array], n_head: int = 1, done_flags: tuple = ("terminals",),
) -> Tuple[Dict[str, Array], Dict[str, Array], int]:
    struct = jax.tree.map(lambda x: x.shape, experience)

    head = jax.tree.map(lambda x: x[0, :n_head, ...], experience)

    # allow for "terminals" and "truncations" to be combined into one "done"
    if len(done_flags)==1:
        terminal_flag = experience[done_flags[0]][0, :, ...].all(axis=-1) # .all is for all agents
    elif len(done_flags)==2:
        done_1 = experience[done_flags[0]][0, :, ...].all(axis=-1)
        done_2 = experience[done_flags[1]][0, :, ...].all(axis=-1)

        terminal_flag = jnp.logical_or(done_1,done_2)
    else:
        print("Too many done flags. Please revise.")
        return struct, head, 0

    num_episodes = int(jnp.sum(terminal_flag))

    return struct, head, num_episodes


def describe_structure(
    vault_name: str,
    vault_uids: Optional[List[str]] = None,
    rel_dir: str = "vaults",
    n_head: int = 0,
    done_flags: tuple = ("terminals",),
) -> Dict[str, Array]:
    # get all uids if not specified
    if vault_uids is None:
        vault_uids = get_available_uids(f"./{rel_dir}/{vault_name}")

    # get structure, number of transitions and the head of the block
    heads = {}
    structs = {}
    single_values = []
    for uid in vault_uids:
        vlt = Vault(vault_name=vault_name, rel_dir=rel_dir, vault_uid=uid)
        exp = vlt.read().experience
        n_trans = exp["actions"].shape[1]

        struct, head, n_traj = get_structure_descriptors(exp, n_head, done_flags)

        print(str(uid) + "\n-----")
        for key, val in struct.items():
            print(f"{str(key)+':': <15}{str(val): >15}")
        print("\n")

        heads[uid] = head
        structs[uid] = struct

        single_values.append([uid, n_trans, n_traj])

    print(tabulate(single_values, headers=["Uid", "Transitions", "Trajectories"]))

    return heads


def get_episode_return_descriptors(
    experience: Dict[str, Array], done_flags: tuple = ("terminals",),
) -> Tuple[float, float, float, float, Array]:
    episode_returns = calculate_returns(experience, done_flags = done_flags)

    mean = jnp.mean(episode_returns)
    stddev = jnp.std(episode_returns)
    mini = jnp.min(episode_returns)
    maxi = jnp.max(episode_returns)
    # extra
    mode = stats.mode(episode_returns)
    median = np.median(episode_returns)
    kurt = stats.kurtosis(episode_returns)
    range = maxi-mini
    quartile_1, quartile_3 = np.percentile(episode_returns,[25,75])
    interquartile_range = quartile_3 - quartile_1
    skewness = stats.skew(episode_returns)

    return mean, stddev, maxi, mini, mode, median, kurt, range, interquartile_range, skewness, episode_returns


def plot_eps_returns_violin(
    all_uid_returns: Dict[str, Array], vault_name: str, save_path: str = ""
) -> None:
    sns.set_theme(style="whitegrid")  # Set seaborn theme with a light blue background
    plt.figure(figsize=(6, 6))  # Adjust figsize as needed

    sns.kdeplot(data=list(all_uid_returns.values()), fill=True)
    plt.xlabel("Episode Return")
    plt.ylabel("Density")
    plt.legend().set_visible(False)
    plt.title(f"Density Estimation")
    # plt.xticks()
    if len(save_path) > 0:
        plt.savefig(save_path, format="pdf", bbox_inches="tight")

    plt.show()
    return


def plot_eps_returns_hist(
    all_uid_returns: Dict[str, Array],
    vault_name: str,
    n_bins: int,
    min_return: float,
    max_return: float,
    save_path: str = "",
) -> None:
    vault_uids = list(all_uid_returns.keys())

    sns.set_theme(style="whitegrid")  # Set seaborn theme with a light blue background
    fig, ax = plt.subplots(
        1,
        len(vault_uids),
        figsize=(6, 6),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    colors = sns.color_palette()

    for i, uid in enumerate(vault_uids):
        counts, bins = np.histogram(
            all_uid_returns[uid], bins=n_bins, range=(min_return, max_return)
        )
        ax[0, i].stairs(counts, bins, fill=True, color=colors[i])
        ax[0, i].set_title("Histogram")
        ax[0, i].set_xlabel("Episode return")
    ax[0, 0].set_ylabel("Frequency")
    fig.tight_layout()
    if len(save_path) > 0:
        plt.savefig(save_path, bbox_inches="tight")
    # fig.suptitle(f"Histogram of distributions of episode returns for {vault_name}")
    fig.tight_layout()
    plt.show()
    return


def describe_episode_returns(
    vault_name: str,
    vault_uids: Optional[List[str]] = None,
    rel_dir: str = "vaults",
    plot_hist: bool = True,
    save_hist: bool = False,
    plot_violin: bool = True,
    save_violin: bool = False,
    plot_saving_rel_dir: str = "vaults",
    n_bins: Optional[int] = 50,
    done_flags: tuple = ("terminals",),
) -> None:
    """Describe a vault.

    From the specified directory and for the specified uids,
    describes vaults according to their episode returns.
    The descriptors include a table of episode return mean, standard deviation, min and max.
    Additionally, the distributions of episode returns are visualised in histograms
    and violin plots. n_bins is how many bins the histogram should have.
    """
    # get all uids if not specified
    if vault_uids is None:
        vault_uids = get_available_uids(f"./{rel_dir}/{vault_name}")

    single_values = []
    all_uid_eps_returns = {}
    for uid in vault_uids:
        vlt = Vault(vault_name=vault_name, rel_dir=rel_dir, vault_uid=uid)
        exp = vlt.read().experience

        mean, stddev, max_ret, min_ret, mode, median, kurt, range, interquartile_range, skewness, episode_returns = get_episode_return_descriptors(exp, done_flags)
        all_uid_eps_returns[uid] = episode_returns

        single_values.append([uid, mean, stddev, max_ret, min_ret, mode, median, kurt, range, interquartile_range, skewness])

    print(tabulate(single_values, headers=["Uid", "Mean", "Stddev", "Max", "Min", "Mode", "Median", "Kurtosis", "Range", "Interquartile_range", "Skewness"]))

    if plot_saving_rel_dir == "vaults":
        plot_saving_rel_dir = rel_dir

    if plot_hist:
        min_of_all = min([x[4] for x in single_values])
        max_of_all = max([x[3] for x in single_values])
        if save_hist:
            plot_eps_returns_hist(
                all_uid_eps_returns,
                vault_name,
                n_bins,
                min_of_all,
                max_of_all,
                save_path=f"{plot_saving_rel_dir}/{vault_name.removesuffix('.vlt')}_histogram.pdf",
            )
        else:
            plot_eps_returns_hist(all_uid_eps_returns, vault_name, n_bins, min_of_all, max_of_all)

    if plot_violin:
        if save_violin:
            plot_eps_returns_violin(
                all_uid_eps_returns,
                vault_name,
                save_path=f"{plot_saving_rel_dir}/{vault_name.removesuffix('.vlt')}_violin_plot.pdf",
            )
        else:
            plot_eps_returns_violin(all_uid_eps_returns, vault_name)

    return


def calculate_returns(
    experience: Dict[str, Array], reward_key: str = "rewards", done_flags: tuple = ("terminals",),
) -> Array:
    """Calculate the returns in a dataset of experience.

    Args:
        experience (Dict[str, Array]): experience coming from an OG-MARL vault.
        reward_key (str, optional):
            Dictionary key of the rewards in the experience data.
            Defaults to "rewards".
        terminal_key (str, optional):
            Dictionary key of the terminals in the experience data.
            Defaults to "terminals".

    Returns:
        Array: jnp array of returns for each episode.
    """
    # Experience is of dimension of (1, T, N, *E)
    # We want all the time data, but just from one agent
    experience_one_agent = jax.tree.map(lambda x: x[0, :, 0, ...], experience)
    rewards = experience_one_agent[reward_key]
    
    if len(done_flags)==1:
        terminals = jnp.array(experience[done_flags[0]][0].all(axis=-1).squeeze(), dtype=jnp.float32) # .all is for all agents
    elif len(done_flags)==2:
        done_1 = jnp.array(experience[done_flags[0]][0].all(axis=-1).squeeze(), dtype=jnp.float32)
        done_2 = jnp.array(experience[done_flags[1]][0].all(axis=-1).squeeze(), dtype=jnp.float32)
        terminals = jnp.logical_or(done_1,done_2).astype(jnp.float32)

    def sum_rewards(terminals: Array, rewards: Array) -> Array:
        def scan_fn(carry: Array, inputs: Array) -> Array:
            terminal, reward = inputs
            new_carry = carry + reward
            new_carry = jax.lax.cond(
                terminal.all(),
                lambda _: reward,  # Reset the cumulative sum if terminal
                lambda _: new_carry,  # Continue accumulating otherwise
                operand=None,
            )
            return new_carry, new_carry

        _, cumulative_rewards = jax.lax.scan(
            scan_fn,
            jnp.zeros_like(terminals[0]),
            (
                # We shift the terminals one timestep rightwards,
                # for our cumulative sum approach to work
                jnp.pad(terminals, ((1, 0)))[:-1],
                rewards,
            ),
        )
        return cumulative_rewards[terminals == 1]

    episode_returns = sum_rewards(terminals, rewards)
    return episode_returns


def get_saco(experience: Dict[str, Array], decimals: int = 4) -> Tuple[float, Array, Array]:
    """Calculate the joint SACo in a dataset of experience.

    Args:
        experience (Dict[str, Array]): experience coming from an OG-MARL vault.

    Returns:
        float: The joint SACo value for that dataset.
        Array: numpy array containing the counts of unique pairs.
        Array: numpy array containing the counts of counts of unique pairs.
    """
    states = experience["infos"]["state"]

    num_tot = states.shape[1]

    reshaped_actions = experience["actions"].reshape((*experience["actions"].shape[:2], -1))
    reshaped_states = states.reshape((*experience["infos"]["state"].shape[:2], -1))
    state_pairs = np.concatenate((reshaped_states, reshaped_actions), axis=-1)

    unique_vals, counts = np.unique(state_pairs.round(decimals=decimals), axis=1, return_counts=True)
    count_vals, count_freq = np.unique(counts, return_counts=True)

    saco = unique_vals.shape[1] / num_tot
    # return saco, count_vals, count_freq
    return saco

def get_average_oaco(experience: Dict[str, Array]) -> Tuple[float, Array, Array]:
    """Calculate the joint SACo in a dataset of experience.

    Args:
        experience (Dict[str, Array]): experience coming from an OG-MARL vault.

    Returns:
        float: The joint SACo value for that dataset.
        Array: numpy array containing the counts of unique pairs.
        Array: numpy array containing the counts of counts of unique pairs.
    """
    states = experience["infos"]["state"]

    obs = experience["observations"]
    actions = experience["actions"]

    T,N = obs.shape[1], obs.shape[2]

    obs_act_pairs = np.concatenate((obs, actions[...,jnp.newaxis]), axis=-1)

    oaco_sum = 0
    for i in range(N):
        unique_vals, counts = np.unique(obs_act_pairs[0,:,i], axis=0, return_counts=True)
        oaco_sum += np.sum(unique_vals.shape[0] / T)

    aoaco = oaco_sum / N

    return aoaco

def plot_count_frequencies(
    all_count_vals: Dict[str, Array], all_count_freq: Dict[str, Array], save_path: str = ""
) -> None:
    """Plots the frequencies of counts of state-action pairs.

    Args:
        all_count_vals (Dict[str, Array]): for each uid (key), the counts of state-action pairs
        all_count_freq (Dict[str, Array]):
            for each uid (key), the number of times
            a state-action pair appears a specific number of times
        save_path (string): path to save the plot to. If empty, the figure is unsaved.

    Artefacts:
        plt shows a log-log plot of state-action pair count frequencies per dataset
        if save_plot is True, plt saves the figure as a pdf at location save_path

    """
    vault_uids = list(all_count_vals.keys())
    colors = sns.color_palette()

    # plot the power law showing count frequencies
    for i, uid in enumerate(vault_uids):
        plt.scatter(
            np.log(all_count_vals[uid].astype(float)),
            np.log(all_count_freq[uid].astype(float)),
            label=uid,
            color=colors[i],
        )

    plt.xlabel("Count (log base 10)")
    plt.ylabel("Frequency of count (log base 10)")
    plt.legend()

    if len(save_path) > 0:
        plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.title("Frequency of unique pair counts power law")
    plt.show()
    return


def describe_coverage(
    vault_name: str,
    vault_uids: Optional[List[str]] = None,
    rel_dir: str = "vaults",
    plot_count_freq: bool = True,
    save_plot: bool = False,
) -> None:
    """Provides coverage, structural and episode return descriptors of a Vault of datasets.

    Args:
        vault_name (string): the name of the Vault,
            not containing the .vlt suffix
        vault_uids (List[str]): a list of uids of datasets in the Vault,
            use if we only describe a subset of all datasets in the Vault
        rel_dir (string): relative  directory of the Vault
        plot_count_freq (bool):
            True when the user wants to generate a plot of state-action count frequencies
        save_plot (bool): True when the user wants to save the generated plot

    Artefacts:
        A table is printed containing for each dataset in the list of uids:
        - Joint SACo
        if plot_count_freq is True,
            plt shows a log-log plot of state-action pair count frequencies per dataset
        if save_plot is True, plt saves the figure as a pdf under the vault_name directory

    """
    # get all uids if not specified
    if vault_uids is None:
        vault_uids = get_available_uids(f"./{rel_dir}/{vault_name}")

    single_values = []
    all_uid_count_vals = {}
    all_uid_count_freq = {}
    for uid in vault_uids:
        vlt = Vault(vault_name=vault_name, rel_dir=rel_dir, vault_uid=uid)
        exp = vlt.read().experience

        saco, count_vals, count_freq = get_saco(exp)
        all_uid_count_freq[uid] = count_freq
        all_uid_count_vals[uid] = count_vals

        single_values.append([uid, saco])

    print(tabulate(single_values, headers=["Uid", "Joint SACo"]))

    if plot_count_freq:
        if save_plot:
            plot_count_frequencies(
                all_uid_count_vals,
                all_uid_count_freq,
                save_path=f"{rel_dir}/{vault_name}/count_frequency_loglog.pdf",
            )
        else:
            plot_count_frequencies(all_uid_count_vals, all_uid_count_freq)

    return


def descriptive_summary(
    vault_name: str,
    vault_uids: Optional[List[str]] = None,
    rel_dir: str = "vaults",
    plot_hist: bool = True,
    save_hist: bool = False,
    n_bins: int = 40,
    done_flags: tuple = ("terminals",),
) -> Dict[str, Array]:
    """Provides coverage, structural and episode return descriptors of a Vault of datasets.

    Args:
        vault_name (string): the name of the Vault, not containing the .vlt suffix
        vault_uids (List[str]): a list of uids of datasets in the Vault,
            use if we only describe a subset of all datasets in the Vault
        rel_dir (string): relative  directory of the Vault
        plot_hist (bool): True when the user wants to generate a histogram
        save_hist (bool): True when the user wants to save a generated histogram
        n_bins (integer): number of bins to use when generating a histogram

    Returns:
        all_returns (Dict[str, Array]): for each uid (key),
            an Array of all episode returns in that dataset

    Artefacts:
        A table is printed containing for each dataset in the list of uids:
        - Episode return:
        -- mean
        -- standard deviation
        -- min
        -- max
        - Num of trajectories
        - Num of transitions
        - Joint SACo
        if plot_hist is True, plt shows a histogram per dataset in the list of uids
        if save_hist is True, plt saves the figure as a pdf under the vault_name directory

    """
    if vault_uids is None:
        vault_uids = get_available_uids(f"./{rel_dir}/{vault_name}")

    all_returns = {}
    single_values = []
    for uid in vault_uids:
        vlt = Vault(vault_name=vault_name, rel_dir=rel_dir, vault_uid=uid)
        exp = vlt.read().experience

        saco, _, _ = get_saco(exp)
        mean, stddev, max_ret, min_ret, mode, median, kurt, range, interquartile_range, skewness, episode_returns = get_episode_return_descriptors(exp, done_flags)
        n_traj = len(episode_returns)
        n_trans = exp["actions"].shape[1]

        aoaco = get_average_oaco(exp)

        single_values.append([uid, mean, stddev, min_ret, max_ret, mode, median, kurt, range, interquartile_range, skewness, n_trans, n_traj, saco, aoaco])
        all_returns[uid] = episode_returns

    print(
        tabulate(
            single_values,
            headers=[
                "Uid",
                "Mean",
                "Stddev",
                "Min return",
                "Max return",
                "Mode", 
                "Median", 
                "Kurtosis", 
                "Range", 
                "Interquartile_range", 
                "Skewness",
                "Transitions",
                "Trajectories",
                "Joint SACo",
                "Average OACo",
            ],
        )
    )

    if plot_hist:
        min_of_all = min([x[3] for x in single_values])
        max_of_all = max([x[4] for x in single_values])
        if save_hist:
            plot_eps_returns_hist(
                all_returns,
                vault_name,
                n_bins,
                min_of_all,
                max_of_all,
                save_path=f"{rel_dir}/{vault_name}/histogram.pdf",
            )
        else:
            plot_eps_returns_hist(all_returns, vault_name, n_bins, min_of_all, max_of_all)

    return all_returns

def compare_vaults_qq_plots(vault_1_name, vault_1_uid, rel_dir="vaults/", save_plot=False):

    vlt1 = Vault(vault_name=vault_1_name, rel_dir=rel_dir, vault_uid=vault_1_uid)
    exp1 = vlt1.read().experience
    data1 = calculate_returns(exp1)

    # Create QQ plot
    sns.set_theme(style="whitegrid")  # Set seaborn theme with a light blue background
    fig, ax = plt.subplots(figsize=(6, 6))
    stats.probplot(data1, dist="norm", plot=ax)

    # Customize axis labels
    ax.set_xlabel("Theoretical Quantiles (Normal Distribution)")
    ax.set_ylabel("Empirical Quantiles (Sample Data)")
    ax.set_title("QQ Plot")
    
    if save_plot:
        plt.savefig(f"{rel_dir}/qq_plot.pdf", bbox_inches="tight")

    plt.show()



