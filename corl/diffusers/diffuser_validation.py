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
# from ml_logger import logger

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
from corl.diffusers.decision_diffuser import GaussianInvDynDiffusion, TemporalUnet, Trainer, Config
from corl.diffusers.normalization import Normalizer

import gym
from tqdm import trange

class Config:
    # misc
    seed = 100
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
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
    returns_scale = 400.0 # Determined using rewards from the dataset

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

    wandb_init({
        'project': config.project,
        'group': config.group,
        'name': config.name,
        "s4rl_augmentation_type": 'None',
        'env': config.dataset,
        'seed': 0,
    })

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
    

    state_dict = torch.load("/home/jaewoo/practices/diffusers/Augmentation-For-OfflineRL/data/generated_data/hopper-medium-expert-v2_debug1_with_reward.npy")

    trainer.step = state_dict['step']
    trainer.model.load_state_dict(state_dict['model'])
    trainer.ema_model.load_state_dict(state_dict['ema'])

    print("##################### Model Loading has been completed #####################")

    NUM_EVAL = 40

    env_list = [gym.make(config.dataset) for _ in range(NUM_EVAL)]
    dones = [0 for _ in range(NUM_EVAL)]
    episode_rewards = [0 for _ in range(NUM_EVAL)]

    assert trainer.ema_model.condition_guidance_w == config.condition_guidance_w

    returns = to_device(Config.test_ret * torch.ones(NUM_EVAL, 1), config.device)

    t = 0
    obs_list = [env.reset()[None] for env in env_list]
    obs = np.concatenate(obs_list, axis=0)
    recorded_obs = [deepcopy(obs[:, None])]
        
    for _ in trange(1000, desc=f"episode running..", leave=False):
        if sum(dones) ==  NUM_EVAL:
            break

        obs = dataset.normalizer.normalize(obs, 'observations')
        conditions = {0: to_torch(obs, device=config.device)}
        samples = trainer.ema_model.conditional_sample(conditions, returns=returns)
        obs_comb = torch.cat([samples[:, 0, :], samples[:, 1, :]], dim=-1)
        obs_comb = obs_comb.reshape(-1, 2*observation_dim)
        action = trainer.ema_model.inv_model(obs_comb)

        samples = to_np(samples)
        action = to_np(action)

        action = dataset.normalizer.unnormalize(action, 'actions')

        obs_list = []
        for i in range(NUM_EVAL):
            this_obs, this_reward, this_done, _ = env_list[i].step(action[i])
            obs_list.append(this_obs[None])
            if this_done:
                if dones[i] == 1:
                    pass
                else:
                    dones[i] = 1
                    episode_rewards[i] += this_reward
            else:
                if dones[i] == 1:
                    pass
                else:
                    episode_rewards[i] += this_reward

        obs = np.concatenate(obs_list, axis=0)
        recorded_obs.append(deepcopy(obs[:, None]))
        t += 1

    recorded_obs = np.concatenate(recorded_obs, axis=1)
    episode_rewards = np.array(episode_rewards)

    wandb.log(
                {
                    f"eval/return_mean": np.mean(episode_rewards),
                    f"eval/return_std": np.std(episode_rewards),
                }
            )




if __name__ == '__main__':
    main()