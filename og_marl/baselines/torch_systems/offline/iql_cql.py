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
"""Implementation of IQL+CQL in PyTorch"""
from typing import Any, Dict, Tuple

import copy
import numpy as np
import jax
import torch
import torch.nn.functional as F
import tree
import hydra
from omegaconf import DictConfig
from chex import Numeric

from og_marl.environments import get_environment, BaseEnvironment
from og_marl.loggers import BaseLogger
from og_marl.replay_buffers import Experience, FlashbaxReplayBuffer
from og_marl.baselines.base import BaseOfflineSystem
from og_marl.loggers import WandbLogger
from og_marl.vault_utils.download_vault import download_and_unzip_vault
from og_marl.baselines.torch_systems.networks import DeepRNN
from og_marl.baselines.torch_systems.utils import (
    batch_concat_agent_id_to_obs,
    concat_agent_id_to_obs,
    expand_batch_and_agent_dim_of_time_major_sequence,
    gather,
    merge_batch_and_agent_dim_of_time_major_sequence,
    switch_two_leading_dims,
    unroll_rnn,
)

class IQLCQLSystem(BaseOfflineSystem):
    """IQL+CQL System in PyTorch.

    Independent Q-Learners with Conservative Q-Learning for stable offline training.
    """

    def __init__(
        self,
        environment: BaseEnvironment,
        logger: BaseLogger,
        cql_weight: float = 2.0,
        linear_layer_dim: int = 64,
        recurrent_layer_dim: int = 64,
        discount: float = 0.99,
        target_update_period: int = 200,
        learning_rate: float = 3e-4,
        add_agent_id_to_obs: bool = True,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__(environment, logger)

        self.device = torch.device(device)
        self.discount = discount
        self.add_agent_id_to_obs = add_agent_id_to_obs

        # Calculate input dimension
        obs_dim = self.environment.observation_shape[0]
        if self.add_agent_id_to_obs:
            obs_dim += len(self.environment.agents)  # Add agent ID dimension

        # Q-network
        self.q_network = DeepRNN(
            input_dim=obs_dim,
            linear_layer_dim=linear_layer_dim,
            recurrent_layer_dim=recurrent_layer_dim,
            output_dim=self.environment.num_actions
        ).to(self.device)

        # Target Q-network
        self.target_q_network = copy.deepcopy(self.q_network).to(self.device)
        self.target_update_period = target_update_period

        # Optimizer
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # Recurrent neural network hidden states for evaluation
        self.rnn_states = {
            agent: self.q_network.initial_state(1, self.device) 
            for agent in self.environment.agents
        }

        # CQL
        self.cql_weight = cql_weight

    def reset(self) -> None:
        """Called at the start of a new episode during evaluation."""
        self.rnn_states = {
            agent: self.q_network.initial_state(1, self.device) 
            for agent in self.environment.agents
        }

    def select_actions(
        self,
        observations: Dict[str, np.ndarray],
        legal_actions: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        # Convert to tensors
        observations = tree.map_structure(
            lambda x: torch.from_numpy(x).float().to(self.device), observations
        )
        legal_actions = tree.map_structure(
            lambda x: torch.from_numpy(x).bool().to(self.device), legal_actions
        )
        
        actions, next_rnn_states = self._select_actions(
            observations, legal_actions, self.rnn_states
        )
        self.rnn_states = next_rnn_states
        
        return tree.map_structure(lambda x: x.cpu().numpy(), actions)

    def _select_actions(
        self,
        observations: Dict[str, torch.Tensor],
        legal_actions: Dict[str, torch.Tensor],
        rnn_states: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        actions = {}
        next_rnn_states = {}
        
        with torch.no_grad():
            for i, agent in enumerate(self.agents):
                agent_observation = observations[agent]
                if self.add_agent_id_to_obs:
                    agent_observation = concat_agent_id_to_obs(
                        agent_observation, i, len(self.agents)
                    )
                
                agent_observation = agent_observation.unsqueeze(0)  # add batch dimension
                q_values, next_rnn_states[agent] = self.q_network(
                    agent_observation, rnn_states[agent]
                )

                agent_legal_actions = legal_actions[agent]
                masked_q_values = torch.where(
                    agent_legal_actions,
                    q_values[0],
                    torch.tensor(-99999999.0, device=self.device),
                )
                greedy_action = torch.argmax(masked_q_values)
                actions[agent] = greedy_action

        return actions, next_rnn_states

    def train_step(self, experience: Experience) -> Dict[str, Numeric]:
        experience = jax.tree.map(lambda x: np.array(x), experience)
        logs = self._train_step(experience)
        return {k: v.item() if torch.is_tensor(v) else v for k, v in logs.items()}

    def _train_step(self, experience: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        # Unpack the experience and convert to tensors
        observations = torch.from_numpy(experience["observations"]).float().to(self.device)  # (B,T,N,O)
        actions = torch.from_numpy(experience["actions"]).long().to(self.device)  # (B,T,N)
        rewards = torch.from_numpy(experience["rewards"]).float().to(self.device)  # (B,T,N)
        truncations = torch.from_numpy(experience["truncations"]).float().to(self.device)  # (B,T)
        terminals = torch.from_numpy(experience["terminals"]).float().to(self.device)  # (B,T)
        legal_actions = torch.from_numpy(experience["infos"]["legals"]).bool().to(self.device)  # (B,T,N,A)

        # When to reset the RNN hidden state
        resets = torch.maximum(terminals, truncations).bool()  # equivalent to logical 'or'

        # Get dims
        B, T, N, A = legal_actions.shape

        # Maybe add agent ids to observation
        if self.add_agent_id_to_obs:
            observations = batch_concat_agent_id_to_obs(observations)

        # Make time-major
        observations = switch_two_leading_dims(observations)
        resets = switch_two_leading_dims(resets)

        # Merge batch_dim and agent_dim
        observations = merge_batch_and_agent_dim_of_time_major_sequence(observations)
        resets = merge_batch_and_agent_dim_of_time_major_sequence(resets)

        # Unroll target network
        with torch.no_grad():
            target_qs_out = unroll_rnn(self.target_q_network, observations, resets)

        # Expand batch and agent_dim
        target_qs_out = expand_batch_and_agent_dim_of_time_major_sequence(target_qs_out, B, N)

        # Make batch-major again
        target_qs_out = switch_two_leading_dims(target_qs_out)

        # Zero gradients
        self.optimizer.zero_grad()

        # Unroll online network
        qs_out = unroll_rnn(self.q_network, observations, resets)

        # Expand batch and agent_dim
        qs_out = expand_batch_and_agent_dim_of_time_major_sequence(qs_out, B, N)

        # Make batch-major again
        qs_out = switch_two_leading_dims(qs_out)

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qs = gather(qs_out, actions, dim=3)

        # Max over target Q-Values/ Double q learning
        qs_out_selector = torch.where(legal_actions, qs_out, torch.tensor(-9999999.0, device=self.device))
        cur_max_actions = torch.argmax(qs_out_selector, dim=3)
        target_max_qs = gather(target_qs_out, cur_max_actions, dim=-1)

        # Compute targets
        targets = (
            rewards[:, :-1] + (1 - terminals[:, :-1]) * self.discount * target_max_qs[:, 1:]
        )
        targets = targets.detach()

        # TD-Error Loss
        td_loss = F.mse_loss(chosen_action_qs[:, :-1], targets)

        #############
        #### CQL ####
        #############

        cql_loss = torch.mean(
            torch.logsumexp(qs_out, dim=-1, keepdim=True)[:, :-1]
        ) - torch.mean(chosen_action_qs[:, :-1])

        #############
        #### end ####
        #############

        # Total loss
        loss = td_loss + self.cql_weight * cql_loss

        # Backward pass
        loss.backward()
        self.optimizer.step()

        # Maybe update target network
        if self.training_step_ctr % self.target_update_period == 0:
            self.target_q_network.load_state_dict(self.q_network.state_dict())

        return {
            "loss": loss,
            "cql_loss": cql_loss,
            "td_loss": td_loss,
            "mean_q_values": torch.mean(qs_out),
            "mean_chosen_q_values": torch.mean(chosen_action_qs),
        }


@hydra.main(version_base=None, config_path="configs", config_name="iql_cql")
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

    system = IQLCQLSystem(env, logger, **cfg["system"])

    torch.manual_seed(cfg["seed"])

    system.train(buffer, training_steps=int(cfg["training_steps"]))


if __name__ == "__main__":
    run_experiment()
