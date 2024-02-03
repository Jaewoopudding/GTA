# source: https://github.com/gwthomas/IQL-PyTorch
# https://arxiv.org/pdf/2110.06169.pdf
import copy
import os
import datetime
import random
import uuid
import pickle
from dataclasses import dataclass, field
from pathlib import Path
import argparse
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import d4rl
import gym
import numpy as np
import pyrallis
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.distributions import MultivariateNormal
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics import MeanAbsolutePercentageError
import pyrootutils

path = pyrootutils.find_root(search_from = __file__, indicator=".aug-project-root")
pyrootutils.set_root(path = path,
                     project_root_env_var = True,
                     dotenv = True,
                     pythonpath = True)

from corl.shared.buffer import ReplayBuffer, RewardNormalizer, StateNormalizer, DiffusionConfig
from corl.shared.logger import Logger
from corl.shared.utils  import wandb_init, set_seed, wrap_env, soft_update, compute_mean_std, normalize_states, eval_actor, get_saved_dataset, get_generated_dataset, merge_dictionary

from corl.shared.s4rl import S4RLAugmentation


TensorBatch = List[torch.Tensor]
os.environ["WANDB_MODE"] = "online"

EXP_ADV_MAX = 100.0
LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0



@dataclass
class TrainConfig:
    # Experiment
    device: str = "cuda:3"
    s4rl_augmentation_type: str = 'identical'
    iteration: int = 2
    env: str = "halfcheetah-medium-v2"  # OpenAI gym environment name
    seed: int = 0  # Sets Gym, PyTorch and Numpy seeds
    step: int = 1000000 # Generated Data Augmentation 모델 학습 step 수

    eval_freq: int = int(5e2)  # How often (time steps) we evaluate
    max_timesteps: int = int(1e5)  # Max time steps to run environment
    checkpoints_path: str = "./reward_model"  # Save path
    save_checkpoints: bool = True  # Save model checkpoints
    log_every: int = 100
    load_model: str = ""  # Model load file name, "" doesn't load

    buffer_size: int = 20_000_000  # Replay buffer size
    batch_size: int = 256  # Batch size for all networks
    normalize: bool = True  # Normalize states
    # Wandb logging
    project: str = "augmentation"
    group: str = "reward_model"
    name: str = "reward_model"
    # Diffusion config
    
    # Network size
    network_width: int = 128
    network_depth: int = 2
    
    

    def __post_init__(self):
        self.name = f"{self.name}-{self.env}-{self.s4rl_augmentation_type}-{str(uuid.uuid4())[:4]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)




def asymmetric_l2_loss(u: torch.Tensor, tau: float) -> torch.Tensor:
    return torch.mean(torch.abs(tau - (u < 0).float()) * u ** 2)


class Squeeze(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(dim=self.dim)




class MLP(nn.Module):
    def __init__(
            self,
            dims,
            activation_fn: Callable[[], nn.Module] = nn.ReLU,
            output_activation_fn: Callable[[], nn.Module] = None,
            squeeze_output: bool = False,
    ):
        super().__init__()
        n_dims = len(dims)
        if n_dims < 2:
            raise ValueError("MLP requires at least two dims (input and output)")

        layers = []
        for i in range(n_dims - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(activation_fn())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        if output_activation_fn is not None:
            layers.append(output_activation_fn())
        if squeeze_output:
            if dims[-1] != 1:
                raise ValueError("Last dim must be 1 when squeezing")
            layers.append(Squeeze(-1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)



class RewardModel():
    def __init__(
            self,
            state_dim : int,
            action_dim : int,
            hidden_dim : int,
            optimizer : str = "adam",
            dim_mults : tuple = (1, 2, 4, 1),
            device : str = "cuda",
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device

        input_dim = state_dim*2 + action_dim
        dims = [input_dim, *map(lambda m: hidden_dim * m, dim_mults), 1]

        self.model = MLP(dims).to(device)
        if optimizer == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4)
        elif optimizer == "adamw":
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=3e-4)

        self.total_it = 0
        self.mape_score = MeanAbsolutePercentageError().to(device)

    def train(self, batch):
        log_dict = {}
        self.total_it += 1

        state, action, reward, next_state, done = batch

        data = torch.cat([state, action, next_state], dim=-1)
        
        loss = F.mse_loss(self.model(data), reward)
        mape = self.mape_score(self.model(data), reward)

        log_dict["train_loss"] = loss.item()
        log_dict["train_mape"] = mape.item()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return log_dict

    @torch.no_grad()
    def eval(self, batch):
        log_dict = {}

        state, action, reward, next_state, done = batch

        data = torch.cat([state, action, next_state], dim=-1)
        
        loss = F.mse_loss(self.model(data), reward)
        mape = self.mape_score(self.model(data), reward)

        log_dict["val_loss"] = loss.item()
        log_dict["val_mape"] = mape.item()
        
        return log_dict

    def state_dict(self):
        return self.model.state_dict()
    
    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)


    @torch.no_grad()
    def inference(self, state, action, next_state):
        data = torch.cat([state, action, next_state], dim=-1)
        return self.model(data)

        

    
@pyrallis.wrap()
def train(config : TrainConfig):
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default=config.env)
    args = parser.parse_args()
    print(args.env)
    env = gym.make(args.env)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    ##### result folder #####
    
    config.checkpoints_path = os.path.join("./reward_model/", config.env + "_checkpoints/")

    if not os.path.exists(config.checkpoints_path):
        os.makedirs(config.checkpoints_path)


    ##### LOADING DATASET #####
    with open(f'./data/{args.env}.pkl', 'rb') as f:
        dataset = pickle.load(f)
    train_dataset = merge_dictionary(dataset[:int(len(dataset)*0.9)])
    val_dataset = merge_dictionary(dataset[int(len(dataset)*0.9):])

    if config.normalize:
        state_mean, state_std = compute_mean_std(train_dataset["observations"], eps=1e-3)
    else:
        state_mean, state_std = 0, 1

    augmentation = S4RLAugmentation(type=config.s4rl_augmentation_type)
    replay_buffer = ReplayBuffer(
        state_dim=state_dim,
        action_dim=action_dim,
        buffer_size=config.buffer_size,
        augmentation=augmentation,
        state_normalizer=StateNormalizer(state_mean, state_std)
    )
    replay_buffer.load_d4rl_dataset(train_dataset)

    val_replay_buffer = ReplayBuffer(
        state_dim=state_dim,
        action_dim=action_dim,
        buffer_size=config.buffer_size,
        augmentation=augmentation,
        state_normalizer=StateNormalizer(state_mean, state_std)
    )
    val_replay_buffer.load_d4rl_dataset(val_dataset)

    
    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)
        logger = Logger(config.checkpoints_path, seed=config.seed)
    else:
        logger = Logger('./tmp', seed=config.seed)

    seed = config.seed
    set_seed(seed, env)

    trainer = RewardModel(state_dim, 
                          action_dim, 
                          config.network_width, 
                          optimizer="adam",
                          dim_mults=(1, 2, 4, 1), 
                          device=config.device)

    
    wandb_init(vars(config))


    for t in range(int(config.max_timesteps)):
        batch = replay_buffer.sample(config.batch_size, iteration=config.iteration) 
        batch = [b.to(config.device) for b in batch]
        log_dict = trainer.train(batch)

        if t % config.log_every == 0:
            wandb.log(log_dict, step=trainer.total_it)
            logger.log({'step': trainer.total_it, **log_dict}, mode='train')

        if t % config.eval_freq == 0 or t == config.max_timesteps - 1:
            with torch.no_grad():
                batch = val_replay_buffer.sample(config.batch_size, iteration=config.iteration)
                batch = [b.to(config.device) for b in batch]
                log_dict = trainer.eval(batch)
            if config.checkpoints_path is not None and config.save_checkpoints and t % 30000 == 0:
                torch.save(
                    trainer.state_dict(),
                    os.path.join(config.checkpoints_path, f"checkpoint_{t}.pt"),
                )
            wandb.log(log_dict, step=trainer.total_it)
            logger.log({'step': trainer.total_it, **log_dict}, mode='eval')

    torch.save(trainer.state_dict(), os.path.join(config.checkpoints_path, f"checkpoint_{t}.pt"))



if __name__ == "__main__":
    train()
