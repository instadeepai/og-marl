# Copyright 2023 InstaDeep Ltd. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Implementation of IDDPG+BC in PyTorch"""
from typing import Any, Dict, Optional, Tuple

import copy
import hydra
import numpy as np
import jax
from omegaconf import DictConfig
import torch
import torch.nn.functional as F
import tree
from chex import Numeric

from og_marl.environments import get_environment, BaseEnvironment
from og_marl.loggers import BaseLogger, WandbLogger
from og_marl.vault_utils.download_vault import download_and_unzip_vault
from og_marl.replay_buffers import Experience, FlashbaxReplayBuffer
from og_marl.baselines.torch_systems.networks import DeepRNNPolicy, StateAndActionCritic
from og_marl.baselines.base import BaseOfflineSystem
from og_marl.baselines.torch_systems.utils import (
    batch_concat_agent_id_to_obs,
    concat_agent_id_to_obs,
    expand_batch_and_agent_dim_of_time_major_sequence,
    merge_batch_and_agent_dim_of_time_major_sequence,
    switch_two_leading_dims,
    unroll_rnn,
)


class IDDPGBCSystem(BaseOfflineSystem):
    """Independent DDPG + BC in PyTorch.

    NOTE: the critic conditions on states and individual agent actions.
    """

    def __init__(
        self,
        environment: BaseEnvironment,
        logger: BaseLogger,
        linear_layer_dim: int = 64,
        recurrent_layer_dim: int = 64,
        discount: float = 0.99,
        target_update_rate: float = 0.005,
        critic_learning_rate: float = 3e-4,
        policy_learning_rate: float = 3e-4,
        add_agent_id_to_obs: bool = True,
        bc_alpha: float = 2.5,  # BC
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__(environment=environment, logger=logger)

        self.device = torch.device(device)
        self.discount = discount
        self.add_agent_id_to_obs = add_agent_id_to_obs
        self.bc_alpha = bc_alpha
        self.target_update_rate = target_update_rate

        # Calculate input dimension
        obs_dim = self.environment.observation_shape[0]
        if self.add_agent_id_to_obs:
            obs_dim += len(self.environment.agents)  # Add agent ID dimension

        # Policy network
        self.policy_network = DeepRNNPolicy(
            input_dim=obs_dim,
            linear_layer_dim=linear_layer_dim,
            recurrent_layer_dim=recurrent_layer_dim,
            output_dim=self.environment.num_actions
        ).to(self.device)

        # Target policy network
        self.target_policy_network = copy.deepcopy(self.policy_network).to(self.device)

        # Critic networks (twin critics)
        state_dim = self.environment.state_shape[0]
        if self.add_agent_id_to_obs:
            state_dim += len(self.environment.agents)  # Add agent ID dimension
        self.critic_network_1 = StateAndActionCritic(
            state_dim, len(self.environment.agents), self.environment.num_actions, self.add_agent_id_to_obs
        ).to(self.device)
        self.critic_network_2 = copy.deepcopy(self.critic_network_1).to(self.device)

        # Target critic networks
        self.target_critic_network_1 = copy.deepcopy(self.critic_network_1).to(self.device)
        self.target_critic_network_2 = copy.deepcopy(self.critic_network_2).to(self.device)

        # Optimizers
        critic_params = list(self.critic_network_1.parameters()) + list(self.critic_network_2.parameters())
        self.critic_optimizer = torch.optim.Adam(critic_params, lr=critic_learning_rate)
        self.policy_optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=policy_learning_rate)

        # Reset the recurrent neural network
        self.rnn_states = {
            agent: self.policy_network.initial_state(1, self.device) 
            for agent in self.environment.agents
        }

    def reset(self) -> None:
        """Called at the start of a new episode."""
        # Reset the recurrent neural network
        self.rnn_states = {
            agent: self.policy_network.initial_state(1, self.device) 
            for agent in self.environment.agents
        }

    def select_actions(
        self,
        observations: Dict[str, np.ndarray],
        legal_actions: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, np.ndarray]:
        # Convert observations to tensors
        observations_tensor = {}
        for agent, obs in observations.items():
            observations_tensor[agent] = torch.from_numpy(obs).float().to(self.device)
        
        actions, next_rnn_states = self._select_actions(observations_tensor, self.rnn_states)
        self.rnn_states = next_rnn_states
        
        return tree.map_structure(lambda x: x[0].cpu().numpy(), actions)

    def _select_actions(
        self,
        observations: Dict[str, torch.Tensor],
        rnn_states: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        actions = {}
        next_rnn_states = {}
        
        with torch.no_grad():
            for i, agent in enumerate(self.environment.agents):
                agent_observation = observations[agent]
                if self.add_agent_id_to_obs:
                    agent_observation = concat_agent_id_to_obs(
                        agent_observation, i, len(self.environment.agents)
                    )
                agent_observation = agent_observation.unsqueeze(0)  # add batch dimension
                action, next_rnn_states[agent] = self.policy_network(
                    agent_observation, rnn_states[agent]
                )
                
                # Store agent action
                actions[agent] = action

        return actions, next_rnn_states

    def train_step(self, experience: Experience) -> Dict[str, Numeric]:
        experience = jax.tree.map(lambda x: np.array(x), experience)
        logs = self._train_step(experience)
        return {k: v.item() if torch.is_tensor(v) else v for k, v in logs.items()}

    def _train_step(self, experience: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        # Unpack the batch and convert to tensors
        observations = torch.from_numpy(experience["observations"]).float().to(self.device)  # (B,T,N,O)
        replay_actions = torch.clamp(
            torch.from_numpy(experience["actions"]).float().to(self.device), -1.0, 1.0
        )  # (B,T,N,A)
        env_states = torch.from_numpy(experience["infos"]["state"]).float().to(self.device)  # (B,T,S)
        rewards = torch.from_numpy(experience["rewards"]).float().to(self.device)  # (B,T,N)
        truncations = torch.from_numpy(experience["truncations"]).float().to(self.device)  # (B,T,N)
        terminals = torch.from_numpy(experience["terminals"]).float().to(self.device)  # (B,T,N)

        # When to reset the RNN hidden state
        resets = torch.maximum(terminals, truncations).bool()  # equivalent to logical 'or'

        # Get dims
        B, T, N = replay_actions.shape[:3]

        # Maybe add agent ids to observation
        if self.add_agent_id_to_obs:
            observations = batch_concat_agent_id_to_obs(observations)

        # Make time-major
        observations = switch_two_leading_dims(observations)
        replay_actions = switch_two_leading_dims(replay_actions)
        rewards = switch_two_leading_dims(rewards)
        terminals = switch_two_leading_dims(terminals)
        env_states = switch_two_leading_dims(env_states)
        resets = switch_two_leading_dims(resets)

        # Unroll target policy
        with torch.no_grad():
            target_actions = unroll_rnn(
                self.target_policy_network,
                merge_batch_and_agent_dim_of_time_major_sequence(observations),
                merge_batch_and_agent_dim_of_time_major_sequence(resets),
            )
            target_actions = expand_batch_and_agent_dim_of_time_major_sequence(target_actions, B, N)

            # Target critics
            target_qs_1 = self.target_critic_network_1(env_states, target_actions)
            target_qs_2 = self.target_critic_network_2(env_states, target_actions)

            # Take minimum between two target critics
            target_qs = torch.minimum(target_qs_1, target_qs_2)

            # Compute Bellman targets
            targets = rewards[:-1] + self.discount * (1 - terminals[:-1]) * target_qs[1:].squeeze(-1)

        # Zero gradients
        self.critic_optimizer.zero_grad()
        self.policy_optimizer.zero_grad()

        # Online critics
        qs_1 = self.critic_network_1(env_states, replay_actions).squeeze(-1)
        qs_2 = self.critic_network_2(env_states, replay_actions).squeeze(-1)

        # Squared TD-error
        critic_loss_1 = F.mse_loss(qs_1[:-1], targets)
        critic_loss_2 = F.mse_loss(qs_2[:-1], targets)

        # Combine critic loss
        critic_loss = (critic_loss_1 + critic_loss_2) / 2

        ###################
        ### Policy Loss ###
        ###################
        online_actions = unroll_rnn(
            self.policy_network,
            merge_batch_and_agent_dim_of_time_major_sequence(observations),
            merge_batch_and_agent_dim_of_time_major_sequence(resets),
        )
        online_actions = expand_batch_and_agent_dim_of_time_major_sequence(online_actions, B, N)

        # Unroll online policy
        policy_qs_1 = self.critic_network_1(env_states, online_actions)
        policy_qs_2 = self.critic_network_2(env_states, online_actions)
        policy_qs = torch.minimum(policy_qs_1, policy_qs_2)

        policy_loss = (
            -(self.bc_alpha / torch.mean(torch.abs(policy_qs))) * torch.mean(policy_qs)
            + 1e-3 * torch.mean(torch.pow(online_actions, 2))
            + torch.mean(torch.pow(online_actions - replay_actions, 2))
        )

        # Train critics first
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Then train policy
        self.policy_optimizer.zero_grad()
        
        # Recompute online actions for policy gradient
        online_actions = unroll_rnn(
            self.policy_network,
            merge_batch_and_agent_dim_of_time_major_sequence(observations),
            merge_batch_and_agent_dim_of_time_major_sequence(resets),
        )
        online_actions = expand_batch_and_agent_dim_of_time_major_sequence(online_actions, B, N)
        
        # Recompute policy loss
        policy_qs_1 = self.critic_network_1(env_states, online_actions)
        policy_qs_2 = self.critic_network_2(env_states, online_actions)
        policy_qs = torch.minimum(policy_qs_1, policy_qs_2)

        policy_loss = (
            -(self.bc_alpha / torch.mean(torch.abs(policy_qs))) * torch.mean(policy_qs)
            + 1e-3 * torch.mean(torch.pow(online_actions, 2))
            + torch.mean(torch.pow(online_actions - replay_actions, 2))
        )
        
        policy_loss.backward()
        self.policy_optimizer.step()

        # Soft target update
        tau = self.target_update_rate
        with torch.no_grad():
            # Update target critics
            for target_param, param in zip(self.target_critic_network_1.parameters(), self.critic_network_1.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
            for target_param, param in zip(self.target_critic_network_2.parameters(), self.critic_network_2.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
            
            # Update target policy
            for target_param, param in zip(self.target_policy_network.parameters(), self.policy_network.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

        return {
            "mean_dataset_q_values": torch.mean((qs_1 + qs_2) / 2),
            "critic_loss": critic_loss,
            "policy_loss": policy_loss,
            "mean_chosen_q_values": torch.mean((policy_qs_1 + policy_qs_2) / 2),
        }


@hydra.main(version_base=None, config_path="configs", config_name="iddpg_bc")
def run_experiment(cfg: DictConfig) -> None:
    print(cfg)

    jax.config.update('jax_platform_name', 'cpu')

    env = get_environment(cfg["task"]["source"], cfg["task"]["env"], cfg["task"]["scenario"], seed=cfg["seed"])

    buffer = FlashbaxReplayBuffer(
        sequence_length=cfg["replay"]["sequence_length"],
        sample_period=cfg["replay"]["sample_period"],
        seed=cfg["seed"],
    )

    download_and_unzip_vault(cfg["task"]["source"], cfg["task"]["env"], cfg["task"]["scenario"])

    buffer.populate_from_vault(cfg["task"]["source"], cfg["task"]["env"], cfg["task"]["scenario"], str(cfg["task"]["dataset"]))

    wandb_config = {
        "system": cfg["system_name"],
        "seed": cfg["seed"],
        "training_steps": cfg["training_steps"],
        **cfg["task"],
        **cfg["replay"],
        **cfg["system"],
    }
    logger = WandbLogger(project=cfg["wandb_project"], config=wandb_config)

    system = IDDPGBCSystem(env, logger, **cfg["system"])

    torch.manual_seed(cfg["seed"])

    system.train(buffer, training_steps=int(cfg["training_steps"]))


if __name__ == "__main__":
    run_experiment()
