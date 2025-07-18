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
import torch
import torch.nn.functional as F
import torch.nn as nn

def gather(values: torch.Tensor, indices: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """PyTorch equivalent of TF gather operation."""
    # Ensure indices have the same number of dimensions as values for the selected dimension
    if dim == -1:
        dim = len(values.shape) - 1
    
    # Create the proper shape for indices
    indices_shape = list(values.shape)
    indices_shape[dim] = 1
    indices_expanded = indices.unsqueeze(dim)
    
    # Expand indices to match all dimensions except the gather dimension
    for i in range(len(indices_shape)):
        if i != dim and indices_expanded.shape[i] == 1 and indices_shape[i] > 1:
            expand_dims = [-1] * len(indices_shape)
            expand_dims[i] = indices_shape[i]
            indices_expanded = indices_expanded.expand(*expand_dims)
    
    gathered = torch.gather(values, dim, indices_expanded)
    return gathered.squeeze(dim)


def switch_two_leading_dims(x: torch.Tensor) -> torch.Tensor:
    """Switch first two dimensions of tensor."""
    return x.transpose(0, 1)


def merge_batch_and_agent_dim_of_time_major_sequence(x: torch.Tensor) -> torch.Tensor:
    """Merge batch and agent dimensions for time-major sequence."""
    T, B, N = x.shape[:3]
    trailing_dims = x.shape[3:]
    return x.reshape(T, B * N, *trailing_dims)


def expand_batch_and_agent_dim_of_time_major_sequence(x: torch.Tensor, B: int, N: int) -> torch.Tensor:
    """Expand merged batch and agent dimensions back."""
    T, BN = x.shape[:2]
    assert BN == B * N
    trailing_dims = x.shape[2:]
    return x.reshape(T, B, N, *trailing_dims)


def concat_agent_id_to_obs(obs: torch.Tensor, agent_id: int, N: int) -> torch.Tensor:
    """Add agent ID to observation."""
    is_vector_obs = len(obs.shape) == 1
    
    if is_vector_obs:
        agent_id_tensor = F.one_hot(torch.tensor(agent_id, device=obs.device), num_classes=N).float()
    else:
        h, w = obs.shape[:2]
        agent_id_tensor = torch.full((h, w, 1), (agent_id / N) + 1 / (2 * N), dtype=torch.float32, device=obs.device)
    
    if not is_vector_obs and len(obs.shape) == 2:  # if no channel dim
        obs = obs.unsqueeze(-1)
    
    return torch.cat([agent_id_tensor, obs], dim=-1)


def batch_concat_agent_id_to_obs(obs: torch.Tensor) -> torch.Tensor:
    """Add agent IDs to batched observations."""
    B, T, N = obs.shape[:3]
    is_vector_obs = len(obs.shape) == 4
    device = obs.device
    
    agent_ids = []
    for i in range(N):
        if is_vector_obs:
            agent_id = F.one_hot(torch.tensor(i, device=device), num_classes=N).float()
        else:
            h, w = obs.shape[3:5]
            agent_id = torch.full((h, w, 1), (i / N) + 1 / (2 * N), dtype=torch.float32, device=device)
        agent_ids.append(agent_id)
    
    agent_ids = torch.stack(agent_ids, dim=0).to(device)
    # Repeat along time and batch dims
    agent_ids = agent_ids.unsqueeze(0).repeat(T, 1, *[1]*(len(agent_ids.shape)-1))  # (T, N, ...)
    agent_ids = agent_ids.unsqueeze(0).repeat(B, *[1]*(len(agent_ids.shape)))  # (B, T, N, ...)
    
    if not is_vector_obs and len(obs.shape) == 5:  # if no channel dim
        obs = obs.unsqueeze(-1)
    
    return torch.cat([agent_ids, obs], dim=-1)


def unroll_rnn(rnn_network: nn.Module, inputs: torch.Tensor, resets: torch.Tensor) -> torch.Tensor:
    """Unroll RNN with reset handling."""
    T, B = inputs.shape[:2]
    device = inputs.device
    
    outputs = []
    hidden_state = rnn_network.initial_state(B, device)
    
    for i in range(T):
        output, hidden_state = rnn_network(inputs[i], hidden_state)
        outputs.append(output)
        
        # Reset hidden state where needed
        reset_mask = resets[i].unsqueeze(0).unsqueeze(-1)  # (1, B, 1)
        initial_state = rnn_network.initial_state(B, device)
        hidden_state = torch.where(reset_mask, initial_state, hidden_state)
    
    return torch.stack(outputs, dim=0)