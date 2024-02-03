# Utilities for diffusion.
from typing import Optional, List, Union

import d4rl
import gym
import numpy as np
import torch
from torch import nn

from hydra.utils import instantiate

import pyrootutils

path = pyrootutils.find_root(search_from=__file__, indicator=".aug-project-root")
pyrootutils.set_root(path = path,
                     project_root_env_var = True,
                     dotenv = True,
                     
                     pythonpath = True)

from src.diffusion.elucidated_diffusion import ElucidatedDiffusion
from src.data.norm import BaseNormalizer


# Make transition dataset from data.
def make_inputs(
        env: gym.Env,
        modelled_terminals: bool = False,
) -> np.ndarray:
    dataset = d4rl.qlearning_dataset(env)
    obs = dataset['observations']
    actions = dataset['actions']
    next_obs = dataset['next_observations']
    rewards = dataset['rewards']
    inputs = np.concatenate([obs, actions, rewards[:, None], next_obs], axis=1)
    if modelled_terminals:
        terminals = dataset['terminals'].astype(np.float32)
        inputs = np.concatenate([inputs, terminals[:, None]], axis=1)
    return inputs


# Convert diffusion samples back to (s, a, r, s') format.
# split diffusion trajectory
# split diffusion transition
def split_diffusion_transition(
        samples: Union[np.ndarray, torch.Tensor],
        env: gym.Env,
        modelled_terminals: bool = False,
        terminal_threshold: Optional[float] = None,
):
    # Compute dimensions from env
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    # Split samples into (s, a, r, s') format
    obs = samples[:, :obs_dim]
    actions = samples[:, obs_dim:obs_dim + action_dim]
    rewards = samples[:, obs_dim + action_dim]
    next_obs = samples[:, obs_dim + action_dim + 1: obs_dim + action_dim + 1 + obs_dim]
    if modelled_terminals:
        terminals = samples[:, -1]
        if terminal_threshold is not None:
            if isinstance(terminals, torch.Tensor):
                terminals = (terminals > terminal_threshold).float()
            else:
                terminals = (terminals > terminal_threshold).astype(np.float32)
        return obs, actions, rewards, next_obs, terminals
    else:
        return obs, actions, rewards, next_obs


def split_diffusion_trajectory(
        samples: Union[np.ndarray, torch.Tensor],
        env: gym.Env,
        modalities : List[str] = ["observations", "actions", "rewards"],
):
    # Compute dimensions from env
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    # Split samples into (s, a, r, s') format
    obs = samples[:, :-1, :obs_dim]
    actions = samples[:, :-1, obs_dim:obs_dim + act_dim]
    next_obs = samples[:, 1:, :obs_dim]
    b, t, d = obs.shape

    if "rewards" in modalities:
        rewards = samples[:, :-1, obs_dim + act_dim]
        rewards = rewards.reshape(b, t, 1)
        
        return obs, actions, rewards, next_obs

    return obs, actions, next_obs


def construct_diffusion_model(
        data_shape: dict,
        normalizer: BaseNormalizer,
        denoising_network,
        edm_config,
        disable_terminal_norm: bool = False,
        skip_dims: List[int] = [],
        cond_dim: Optional[int] = 0,
) -> ElucidatedDiffusion:
    D = 0
    for k, v in data_shape.items():
        T, d = v
        D += d
    model = instantiate(denoising_network,
                        horizon=T,
                        d_in=D, 
                        cond_dim=cond_dim)

    #if disable_terminal_norm:
    #    terminal_dim = D - 1
    #    if terminal_dim not in skip_dims:
    #        skip_dims.append(terminal_dim)

    if skip_dims:
        print(f"Skipping normalization for dimensions {skip_dims}.")

    return ElucidatedDiffusion(
        net=model,
        normalizer=normalizer,
        data_shape=data_shape,
        **edm_config
    )
