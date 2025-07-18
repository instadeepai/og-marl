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
from typing import Tuple
import torch
import torch.nn.functional as F
import torch.nn as nn

class DeepRNN(nn.Module):
    """PyTorch equivalent of Sonnet's DeepRNN with GRU."""
    
    def __init__(self, input_dim: int, linear_layer_dim: int, recurrent_layer_dim: int, output_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, linear_layer_dim)
        self.gru = nn.GRU(linear_layer_dim, recurrent_layer_dim, batch_first=False)
        self.linear2 = nn.Linear(recurrent_layer_dim, output_dim)
        self.recurrent_layer_dim = recurrent_layer_dim
        
    def forward(self, x: torch.Tensor, hidden_state: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.linear1(x))
        
        if hidden_state is None:
            hidden_state = self.initial_state(x.shape[0] if len(x.shape) > 1 else 1, x.device)
            
        # GRU expects (seq_len, batch, input_size) if batch_first=False
        if len(x.shape) == 2:  # (batch, features)
            x = x.unsqueeze(0)  # (1, batch, features)
            
        gru_out, new_hidden = self.gru(x, hidden_state)
        output = F.relu(gru_out)
        output = self.linear2(output)
        
        if output.shape[0] == 1:  # Remove seq dimension if it was added
            output = output.squeeze(0)
            
        return output, new_hidden
    
    def initial_state(self, batch_size: int, device: torch.device = None) -> torch.Tensor:
        if device is None:
            device = next(self.parameters()).device
        return torch.zeros(1, batch_size, self.recurrent_layer_dim, device=device)


class DeepRNNPolicy(nn.Module):
    """PyTorch equivalent of Sonnet's DeepRNN with tanh activation for continuous actions."""
    
    def __init__(self, input_dim: int, linear_layer_dim: int, recurrent_layer_dim: int, output_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, linear_layer_dim)
        self.gru = nn.GRU(linear_layer_dim, recurrent_layer_dim, batch_first=False)
        self.linear2 = nn.Linear(recurrent_layer_dim, output_dim)
        self.recurrent_layer_dim = recurrent_layer_dim
        
    def forward(self, x: torch.Tensor, hidden_state: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.linear1(x))
        
        if hidden_state is None:
            hidden_state = self.initial_state(x.shape[0] if len(x.shape) > 1 else 1, x.device)
            
        # GRU expects (seq_len, batch, input_size) if batch_first=False
        if len(x.shape) == 2:  # (batch, features)
            x = x.unsqueeze(0)  # (1, batch, features)
            
        gru_out, new_hidden = self.gru(x, hidden_state)
        output = F.relu(gru_out)
        output = torch.tanh(self.linear2(output))  # Use tanh for continuous actions
        
        if output.shape[0] == 1:  # Remove seq dimension if it was added
            output = output.squeeze(0)
            
        return output, new_hidden
    
    def initial_state(self, batch_size: int, device: torch.device = None) -> torch.Tensor:
        if device is None:
            device = next(self.parameters()).device
        return torch.zeros(1, batch_size, self.recurrent_layer_dim, device=device)


class StateAndActionCritic(nn.Module):
    """PyTorch equivalent of TensorFlow StateAndActionCritic."""
    
    def __init__(self, state_dim: int, num_agents: int, num_actions: int, add_agent_id: bool = True, hidden_dim: int = 128):
        super().__init__()
        self.N = num_agents
        self.A = num_actions
        self.add_agent_id = add_agent_id
        
        input_dim = state_dim + num_actions
        self._critic_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, states: torch.Tensor, agent_actions: torch.Tensor) -> torch.Tensor:
        """Forward pass of critic network.
        
        Args:
            states: [T,B,S] state tensor
            agent_actions: [T,B,N,A] agent actions tensor
        """
        from .utils import batch_concat_agent_id_to_obs
        
        T, B = states.shape[:2]
        
        # Repeat states for each agent [T,B,S] -> [T,B,N,S]
        states_repeated = states.unsqueeze(2).repeat(1, 1, self.N, 1)
        
        # Concatenate states and agent actions
        critic_input = torch.cat([states_repeated, agent_actions], dim=-1)
        
        # Add agent IDs if required
        if self.add_agent_id:
            critic_input = batch_concat_agent_id_to_obs(critic_input)
        
        q_values = self._critic_network(critic_input)
        
        return q_values
