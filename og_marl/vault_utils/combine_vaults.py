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

from git import Optional

import jax
import pickle
import flashbax as fbx
from flashbax.vault import Vault
import flashbax
from og_marl.vault_utils.download_vault import (
    check_directory_exists_and_not_empty,
    get_available_uids,
)


def get_all_vaults(
    vault_name: str,
    vault_uids: Optional[list[str]] = None,
    rel_dir: str = "vaults",
) -> list[Vault]:
    if vault_uids is None:
        vault_uids = get_available_uids(f"./{rel_dir}/{vault_name}")

    vlts = []
    for uid in vault_uids:
        vlts.append(Vault(rel_dir=rel_dir, vault_name=vault_name, vault_uid=uid))
    return vlts


def stitch_vault_from_many(vlts: list[Vault], vault_name: str, vault_uid: str, rel_dir: str) -> int:
    all_data = vlts[0].read()
    offline_data = all_data.experience

    dest_buffer = fbx.make_trajectory_buffer(
        # Sampling parameters
        sample_batch_size=1,
        sample_sequence_length=1,
        period=1,
        # Not important in this example, as we are not adding to the buffer
        max_length_time_axis=offline_data["actions"].shape[1] + 1,
        min_length_time_axis=100,
        add_batch_size=offline_data["actions"].shape[0],
    )

    dummy_experience = jax.tree_map(lambda x: x[0, 0, ...], offline_data)
    del offline_data
    del all_data

    dest_state = dest_buffer.init(dummy_experience)
    buffer_add = jax.jit(dest_buffer.add, donate_argnums=0)
    dest_vault = flashbax.vault.Vault(
        experience_structure=dest_state.experience,
        vault_name=vault_name,
        vault_uid=vault_uid,
        rel_dir=rel_dir,
    )

    tot_timesteps = 0
    for vlt in vlts:
        all_data = vlt.read()
        offline_data = all_data.experience

        dest_state = buffer_add(dest_state, offline_data)

        timesteps_written = dest_vault.write(dest_state)

        tot_timesteps += timesteps_written
        del offline_data
        del all_data

    return tot_timesteps


def combine_vaults(rel_dir: str, vault_name: str, vault_uids: Optional[list[str]] = None) -> str:
    # check that the vault to be combined exists
    if not check_directory_exists_and_not_empty(f"./{rel_dir}/{vault_name}"):
        print(f"Vault './{rel_dir}/{vault_name}' does not exist and cannot be combined.")
        return f"./{rel_dir}/{vault_name}"

    # if uids aren't specified, use all uids for subsampling
    if vault_uids is None:
        vault_uids = get_available_uids(f"./{rel_dir}/{vault_name}")

    # name of subsampled vault (at task level)
    print(vault_uids)
    uids_str = "_".join(vault_uids)
    new_vault_name = vault_name.removesuffix(".vlt") + "_combined.vlt"

    # check that a subsampled vault by the same name does not already exist
    if check_directory_exists_and_not_empty(f"./{rel_dir}/{new_vault_name}"):
        print(
            f"Vault '{rel_dir}/{new_vault_name.removesuffix('.vlt')}' already exists. To combine from scratch, please remove the current combined vault from its directory." #noqa
        )
        return new_vault_name

    vlts = get_all_vaults(rel_dir=rel_dir, vault_name=vault_name, vault_uids=vault_uids)

    timesteps = stitch_vault_from_many(vlts, new_vault_name, uids_str, rel_dir)

    with open(f"{rel_dir}/{new_vault_name}/{uids_str}/timesteps.pickle", "wb") as f:
        pickle.dump(timesteps, f)

    return new_vault_name
