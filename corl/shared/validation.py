from typing import List, Tuple, Dict, Optional
import numpy as np
import torch
from torch import nn 
from tqdm import tqdm

from corl.shared.utils import get_saved_dataset

def discounted_return(
        reward_array: np.ndarray,
        discount_ratio: float
) -> np.ndarray :
    cumsum = np.zeros_like(reward_array)
    cumsum[-1] = reward_array[-1]
    for t in reversed(range(reward_array.shape[0] - 1)):
        cumsum[t] = reward_array[t] + discount_ratio * cumsum[t + 1]
    return cumsum

def MC_bias(
        trajectories: List[ Dict [str, np.ndarray] ], # 'rewards', 'observations', 'rewards', 'next_observations'
        critics: List[nn.Module],
        discount_ratio: float,
        device: str
):  
    bias_result = []
    for trajectory in (trajectories):
        states = to_tensor(trajectory['observations'], device)
        actions = to_tensor(trajectory['actions'], device)
        q_values = torch.min(torch.cat([critic(states, actions) for critic in critics], dim=-1), dim=-1).values
        returns = to_tensor(discounted_return(trajectory['rewards'], discount_ratio), device)
        bias = (q_values - returns).detach().cpu().numpy()
        bias_result.append(bias)
    return bias_result


def Q_gap(
        trajectories: List[Dict[str, np.ndarray]], # 'rewards', 'observations', 'rewards', 'next_observations'
        critics: List[nn.Module],
        actor: nn.Module,
        device: str
):  
    q_gap_list = []
    for trajectory in (trajectories):
        states = to_tensor(trajectory['observations'], device)
        actions = to_tensor(trajectory['actions'], device)
        q_values = torch.min(torch.cat([critic(states, actions) for critic in critics], dim=-1), dim=-1).values
        acted_q_values = torch.min(torch.cat([critic(states, actor.get_action_array(states)).reshape(-1, 1) for critic in critics], dim=-1), dim=-1).values
        q_gap = (q_values - acted_q_values).detach().cpu().numpy()
        q_gap_list.append(q_gap)
    return q_gap_list

def valid_bellman_error():
    pass

def to_tensor(array, device):
    return torch.tensor(array).to(device)

def validate(data, critics, actor, discount, device):
    q_gap = np.concatenate(Q_gap(data, critics, actor, device))
    mc_bias = np.concatenate(MC_bias(data, critics, discount, device))
    return q_gap, mc_bias