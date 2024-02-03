import os
import copy
import numpy as np
import torch
from torch import nn
from torch.distributions import Bernoulli
import torch.nn.functional as F
import einops
from einops.layers.torch import Rearrange
from einops import rearrange
import pdb
import wandb
import tqdm
from copy import deepcopy

import corl.diffusers.utils as utils
from utils.arrays import batch_to_device, to_np, to_device, apply_dict, to_torch
from utils.timer import Timer
from src.dynamics.reward import RewardModel

from corl.diffusers.helpers import (
    cosine_beta_schedule,
    extract,
    apply_conditioning,
    Losses,
    SinusoidalPosEmb,
    Downsample1d,
    Upsample1d,
    Conv1dBlock,
)

from corl.diffusers.sequence import SequenceDataset
from corl.diffusers.normalization import CDFNormalizer
from corl.shared.utils  import wandb_init, set_seed, wrap_env, soft_update, compute_mean_std, normalize_states, eval_actor, get_saved_dataset, get_generated_dataset, merge_dictionary
from corl.diffusers.decision_diffuser import GaussianInvDynDiffusion, TemporalUnet, Trainer, Config, cycle
from corl.diffusers.normalization import Normalizer

import gym
from tqdm import trange

class Config:
    # misc
    seed = 100
    device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
    bucket = '/home/jaewoo/practices/diffusers/Augmentation-For-OfflineRL/diffuser_models'
    dataset = 'hopper-medium-expert-v2'

    ## model
    model = 'models.TemporalUnet'
    diffusion = 'models.GaussianInvDynDiffusion'
    horizon = 20
    n_diffusion_steps = 100
    action_weight = 10
    loss_weights = None
    loss_discount = 1
    predict_epsilon = True
    dim_mults = (1, 4, 8)
    returns_condition = True
    calc_energy=False
    dim=128
    condition_dropout=0.25
    condition_guidance_w = 1.2
    test_ret = 0.9
    renderer = 'utils.MuJoCoRenderer'

    ## dataset
    loader = 'datasets.SequenceDataset'
    normalizer = 'CDFNormalizer'
    preprocess_fns = []
    clip_denoised = True
    use_padding = True
    include_returns = True
    discount = 0.99
    max_path_length = 1000
    hidden_dim = 256
    ar_inv = False
    train_only_inv = False
    termination_penalty = -100
    returns_scale = 1000.0 # Determined using rewards from the dataset

    ## training
    n_steps_per_epoch = 10000
    loss_type = 'l2'
    n_train_steps = int(2e6 / 4) ### batch size resizing
    batch_size = 128
    learning_rate = 2e-4
    gradient_accumulate_every = 2
    ema_decay = 0.995
    log_freq = 1000
    save_freq = 10000
    sample_freq = 10000
    n_saves = 5
    save_parallel = False
    n_reference = 8
    save_checkpoints = False

    ## 
    project: str = "diffusers"
    group: str = "Decision Diffuser:D4RL"
    name: str = "Decision Diffuser"

def main():
    config = Config()

    # wandb_init({
    #     'project': config.project,
    #     'group': config.group,
    #     'name': config.name,
    #     "s4rl_augmentation_type": 'None',
    #     'env': config.dataset,
    #     'seed': 0,
    # })

    dataset = SequenceDataset(
        env=config.dataset,
        horizon=config.horizon,
        normalizer=CDFNormalizer,
        preprocess_fns=config.preprocess_fns,
        use_padding=config.use_padding,
        max_path_length=config.max_path_length,
        include_returns=config.include_returns,
        returns_scale=config.returns_scale,
        discount=config.discount,
        termination_penalty=config.termination_penalty
    )   

    print("===============<dataset has been set>===============")

    observation_dim = dataset.observation_dim
    action_dim = dataset.action_dim

    model = TemporalUnet(
        horizon=config.horizon,
        transition_dim=observation_dim,
        cond_dim=observation_dim,
        dim_mults=config.dim_mults,
        returns_condition=config.returns_condition,
        dim=config.dim,
        condition_dropout=config.condition_dropout,
        calc_energy=config.calc_energy,
    ).to(config.device)

    print("===============<denoiser network has been set>===============")

    diffusion = GaussianInvDynDiffusion(
        model=model,
        horizon=config.horizon,
        observation_dim=observation_dim,
        action_dim=action_dim,
        n_timesteps=config.n_diffusion_steps,
        loss_type=config.loss_type,
        clip_denoised=config.clip_denoised,
        predict_epsilon=config.predict_epsilon,
        hidden_dim=config.hidden_dim,
        ar_inv=config.ar_inv,
        train_only_inv=config.train_only_inv,
        action_weight=config.action_weight,
        loss_weights=config.loss_weights,
        loss_discount=config.loss_discount,
        returns_condition=config.returns_condition,
        condition_guidance_w=config.condition_guidance_w
    ).to(config.device)

    print("===============<diffusion has been set>===============")

    trainer = Trainer(
            diffusion_model=diffusion,
            dataset=dataset,
            train_batch_size=config.batch_size,
            train_lr=config.learning_rate,
            gradient_accumulate_every=config.gradient_accumulate_every,
            ema_decay=config.ema_decay,
            sample_freq=config.sample_freq,
            save_freq=config.save_freq,
            log_freq=config.log_freq,
            label_freq=int(config.n_train_steps // config.n_saves),
            save_parallel=config.save_parallel,
            bucket=config.bucket,
            n_reference=config.n_reference,
            train_device=config.device,
            save_checkpoints=config.save_checkpoints,
        )
    
    
    reward_model = RewardModel(observation_dim, 
                          action_dim, 
                          hidden_dim=128, 
                          optimizer="adam",
                          dim_mults=(1, 2, 4, 1), 
                          device=config.device)
    
    reward_model.load_state_dict(torch.load(f'reward_model/{config.dataset}.pt'))

    state_dict = torch.load(os.path.join(f"diffuser_models/{config.dataset}.pt"))
    trainer.model.load_state_dict(state_dict['model'])
    trainer.ema_model.load_state_dict(state_dict['ema'])
    trainer.ema_model.reward_model = reward_model
    print("===============<trained diffuser / reward model has been loaded>===============")

    dataloader = cycle(torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, num_workers=0, shuffle=True, pin_memory=True))

    train_dataset = merge_dictionary(dataset[:int(len(dataset)*0.9)])
    val_dataset = merge_dictionary(dataset[int(len(dataset)*0.9):])

    if config.normalize:
        state_mean, state_std = compute_mean_std(train_dataset["observations"], eps=1e-3)

    MULTIPLE = 5
    GENERATION_TARGET_AMOUNT = 5_000_000
    iteration_steps = int(GENERATION_TARGET_AMOUNT / MULTIPLE / config.batch_size / config.horizon)

    print("Generating Multiple:", MULTIPLE)
    print("Generating Target Amount:", GENERATION_TARGET_AMOUNT)

    augmented_states = []
    augmented_actions = []
    augmented_rewards = []
    augmented_next_states = []

    returns = to_device(Config.test_ret * torch.ones(config.batch_size * MULTIPLE, 1), config.device)

    for i in tqdm.tqdm(range(iteration_steps)):
        batch = next(dataloader)
        batch = batch_to_device(batch, device=config.device)
        states, actions = trainer.ema_model.back_and_forth(batch.trajectories, noising_ratio=0.5, multiple=5, returns=returns, verbose=False)
        states = states.detach().cpu().detach().numpy()
        actions = actions.detach().cpu().detach().numpy()
        states = dataset.normalizer.unnormalize(states, 'observations')
        actions = dataset.normalizer.unnormalize(actions, 'actions')
        
        reward_model_input = torch.tensor(np.concatenate([(states.reshape(-1, observation_dim) - dataset.observation_mean) / dataset.observation_std, actions.reshape(-1, action_dim)], axis=-1)).to(config.device)
        rewards = reward_model(reward_model_input).detach().cpu().numpy()

        augmented_states.append(states.reshape((config.batch_size * MULTIPLE), config.horizon - 1, -1)[:, :-1, :])
        augmented_actions.append(actions.reshape((config.batch_size * MULTIPLE), config.horizon - 1, -1)[:, :-1, :])
        augmented_rewards.append(rewards.reshape((config.batch_size * MULTIPLE), config.horizon - 1, 1)[:, :-1, :])
        augmented_next_states.append(states.reshape((config.batch_size * MULTIPLE), config.horizon - 1, -1)[:, 1:, :])

    states = np.concatenate(augmented_states)
    actions = np.concatenate(augmented_actions)
    rewards = np.concatenate(augmented_rewards)
    next_states = np.concatenate(augmented_next_states)

    transitions = np.array([{'observations':state, 'actions':action, 'rewards':reward, 'next_observations':next_state} for state, action, reward, next_state in zip(states, actions, rewards, next_states)])
    np.save(os.path.join("data/generated_data", config.dataset + "_debug1"), transitions)


# dict_keys(['observations', 'actions', 'next_observations', 'rewards'])
    


    



if __name__ == '__main__':
    main()

    