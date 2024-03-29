import os
import random
from typing import Any, Dict, List, Optional, Tuple, Union, Dict

import wandb
import pickle
import uuid
import numpy as np
import torch
import torch.nn as nn
import gym
import copy


def wandb_init(config: dict) -> None:
    wandb.init(
        config=config,
        project=config["project"],
        group=config["group"],
        name=config["name"],
        id=str(uuid.uuid4()),
    )
    wandb.config = {
        "environment": config['env'],
        "seed": config['seed']
    }

    wandb.run.save()

def set_seed(
        seed: int, env: Optional[gym.Env] = None, deterministic_torch: bool = False
):
    if env is not None:
        env.seed(seed)
        env.action_space.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)

def wrap_env(
    env: gym.Env,
    state_mean: Union[np.ndarray, float] = 0.0,
    state_std: Union[np.ndarray, float] = 1.0,
    reward_scale: float = 1.0,
) -> gym.Env:
    def normalize_state(state):
        return (state - state_mean) / state_std

    def scale_reward(reward):
        return reward_scale * reward

    env = gym.wrappers.TransformObservation(env, normalize_state)
    if reward_scale != 1.0:
        env = gym.wrappers.TransformReward(env, scale_reward)
    return env

def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)


def compute_mean_std(states: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std


def normalize_states(states: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (states - mean) / std


@torch.no_grad()
def eval_actor(
        env: gym.Env, actor: nn.Module, device: str, n_episodes: int, seed: int
) -> np.ndarray:
    env.seed(seed)
    actor.eval()
    episode_rewards = []
    for _ in range(n_episodes):
        state, done = env.reset(), False
        episode_reward = 0.0
        while not done:
            action = actor.act(state, device)
            state, reward, done, _ = env.step(action)
            episode_reward += reward
        episode_rewards.append(episode_reward)

    actor.train()
    return np.asarray(episode_rewards)


@torch.no_grad()
def eval_actor_return_evalbuffer(
        env: gym.Env, actor: nn.Module, device: str, n_episodes: int, seed: int
) -> np.ndarray:
    env.seed(seed)
    actor.eval()
    episode_rewards = []
    evaluation_buffer = []
    
    for _ in range(n_episodes):
        state, done = env.reset(), False
        episode_reward = 0.0
        while not done:
            temp = {}
            action = actor.act(state, device)
            temp['observations'] = state.reshape(1,-1).astype(np.float32)
            state, reward, done, _ = env.step(action)
            episode_reward += reward
            temp['next_observations'] = state.reshape(1,-1).astype(np.float32)
            temp['actions'] = action.reshape(1,-1).astype(np.float32)
            temp['rewards'] = reward.reshape(1,).astype(np.float32)
            temp['done'] = done
            evaluation_buffer.append(temp)
            
        episode_rewards.append(episode_reward)

    actor.train()
    return np.asarray(episode_rewards), np.array(evaluation_buffer)

# Training and evaluation logic
@torch.no_grad()
def eval_rollout(
    model, 
    env: gym.Env,
    target_return: float,
    device: str = "cpu",
) -> Tuple[float, float]:
    states = torch.zeros(
        1, model.episode_len + 1, model.state_dim, dtype=torch.float, device=device
    )
    actions = torch.zeros(
        1, model.episode_len, model.action_dim, dtype=torch.float, device=device
    )
    returns = torch.zeros(1, model.episode_len + 1, dtype=torch.float, device=device)
    time_steps = torch.arange(model.episode_len, dtype=torch.long, device=device)
    time_steps = time_steps.view(1, -1)

    states[:, 0] = torch.as_tensor(env.reset(), device=device)
    returns[:, 0] = torch.as_tensor(target_return, device=device)

    # cannot step higher than model episode len, as timestep embeddings will crash
    episode_return, episode_len = 0.0, 0.0
    for step in range(model.episode_len):
        # first select history up to step, then select last seq_len states,
        # step + 1 as : operator is not inclusive, last action is dummy with zeros
        # (as model will predict last, actual last values are not important)
        predicted_actions = model(  # fix this noqa!!!
            states[:, : step + 1][:, -model.seq_len :],
            actions[:, : step + 1][:, -model.seq_len :],
            returns[:, : step + 1][:, -model.seq_len :],
            time_steps[:, : step + 1][:, -model.seq_len :],
        )
        predicted_action = predicted_actions[0, -1].cpu().numpy()
        next_state, reward, done, info = env.step(predicted_action)
        # at step t, we predict a_t, get s_{t + 1}, r_{t + 1}
        actions[:, step] = torch.as_tensor(predicted_action)
        states[:, step + 1] = torch.as_tensor(next_state)
        returns[:, step + 1] = torch.as_tensor(returns[:, step] - reward)

        episode_return += reward
        episode_len += 1

        if done:
            break

    return episode_return, episode_len

def merge_dictionary(list_of_Dict: List[Dict]) -> Dict:
    merged_data = {}

    for d in list_of_Dict:
        for k, v in d.items():
            if k not in merged_data.keys():
                merged_data[k] = [v]
            else:
                merged_data[k].append(v)

    for k, v in merged_data.items():
        merged_data[k] = np.concatenate(merged_data[k])

    return merged_data

def get_saved_dataset(env: str) -> Dict:
    with open(f'./data/{env}.pkl','rb') as f:
        data = pickle.load(f)
    return merge_dictionary(data)


def get_GTA_dataset(env: str, step: Optional[int]=None) -> Dict:
    data = np.load(f'data/generated_data/{env}.npz', allow_pickle=True) # data는 Dict의 list로 되어 있다.
    config_dict = data['config'].item()
    data = data['data'].squeeze()
    metadata = {}
    try:
        metadata['diffusion_horizon'] = config_dict['construct_diffusion_model']['denoising_network']['horizon']
    except:
        metadata['diffusion_horizon'] = 1
    metadata['diffusion_backbone'] = config_dict['construct_diffusion_model']['denoising_network']['_target_'].split('.')[-1]
    metadata['conditioned'] = True if config_dict['construct_diffusion_model']['denoising_network']['cond_dim'] != 0 else False
    metadata['guidance_target_multiple'] = config_dict['SimpleDiffusionGenerator']['guidance_rewardscale']
    metadata['noise_level'] = config_dict['SimpleDiffusionGenerator']['noise_level']
    return merge_dictionary([*data]), metadata

def get_dataset(config):
    if config.GDA is None or config.GDA == 'None':
        dataset = get_saved_dataset(config.env)
        return dataset, {}
    elif 'GTA' in config.GDA:
        generated_dataset, metadata = get_GTA_dataset(config.env, config.step)
    else:
        raise RuntimeError("GDA must be one of the 'GTA' or 'None'.")
    dataset = merge_dictionary([get_saved_dataset(config.env), generated_dataset])
    return dataset, metadata

def get_trajectory_dataset(config):
    metadata = {}
    env = config.env
    if config.GDA is None or 'None' in config.GDA:
        with open(f'./data/{env}.pkl','rb') as f:
            dataset = np.load(f, allow_pickle=True)
            return dataset, metadata
    else:
        data = np.load(f'data/generated_data/{env}.npz', allow_pickle=True) # data는 Dict의 list로 되어 있다.
        config_dict = data['config'].item()
        data = data['data'].squeeze()
    
        with open(f'./data/{env}.pkl','rb') as f:
            dataset = np.load(f, allow_pickle=True) # data는 Dict의 list로 되어 있다.
            data = (data.tolist() +(dataset))
        try:
            metadata['diffusion_horizon'] = config_dict['construct_diffusion_model']['denoising_network']['horizon']
        except:
            metadata['diffusion_horizon'] = 1
        metadata['diffusion_backbone'] = config_dict['construct_diffusion_model']['denoising_network']['_target_'].split('.')[-1]
        metadata['conditioned'] = True if config_dict['construct_diffusion_model']['denoising_network']['cond_dim'] != 0 else False
        metadata['guidance_target_multiple'] = config_dict['SimpleDiffusionGenerator']['guidance_rewardscale']
        metadata['noise_level'] = config_dict['SimpleDiffusionGenerator']['noise_level']
        return data, metadata
