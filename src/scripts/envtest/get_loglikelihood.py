# Train diffusion model on D4RL transitions.
import os
import argparse
from datetime import datetime
import random
from tqdm import tqdm
import pickle
from typing import Optional, Union, List
import math

import gym
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import wandb

from accelerate import PartialState

from hydra import initialize, compose
from hydra.utils import instantiate
from omegaconf import OmegaConf

import pyrootutils

path = pyrootutils.find_root(search_from=__file__, indicator=".aug-project-root")
pyrootutils.set_root(path = path,
                     project_root_env_var = True,
                     dotenv = True,
                     pythonpath = True)


from src.diffusion.elucidated_diffusion import Trainer, define_rewardweighting_sampler, define_terminal_sampler
from src.data.norm import MinMaxNormalizer, normalizer_factory
from src.diffusion.utils import make_inputs,  split_diffusion_trajectory, construct_diffusion_model
from corl.shared.utils  import get_saved_dataset, merge_dictionary
from corl.shared.buffer import DiffusionDataset, DiffusionTrajectoryDataset
from src.dynamics.reward import RewardModel, TrainConfig
from corl.shared.utils  import compute_mean_std, normalize_states, merge_dictionary
from src.diffusion.train_diffuser import SimpleDiffusionGenerator



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='halfcheetah-medium-v2')
    parser.add_argument('--config_path', type=str, default='../../../configs')
    parser.add_argument('--config_name', type=str, default='smallmixer_denoiser_v4.yaml')
    parser.add_argument('--test_dataset', type=str, default='results/halfcheetah-medium-v2/smallmixer_denoiser_v4/2024-01-17/18:21/5M-1_3x-smallmixer-50-sar-temp2_0.npz')
    #
    parser.add_argument('--results_folder', type=str, default='./results')
    parser.add_argument('--use_gpu', action='store_true', default=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_samples', action='store_true', default=True)
    parser.add_argument('--load_checkpoint', action='store_true')
    parser.add_argument('--ckpt_path', type=str, default="results/halfcheetah-medium-v2/smallmixer_denoiser_v4/2023-12-23/model-1000000.pt")
    args = parser.parse_args()
        
    with initialize(version_base=None, config_path=args.config_path):
        cfg = compose(config_name=args.config_name)

    # Set seed.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.use_gpu:
        torch.cuda.manual_seed(args.seed)

    # Create the environment and dataset.
    env = gym.make(args.dataset)
    action_dim = env.action_space.shape[0]
    obs_dim = env.observation_space.shape[0]
    reward_dim = 1

    data_shape = {
        "observations": (cfg.Dataset.seq_len+1, obs_dim),
        "actions": (cfg.Dataset.seq_len+1, action_dim),
        "rewards": (cfg.Dataset.seq_len+1, reward_dim),
    }

    with open(f'./data/{args.dataset}.pkl','rb') as f:
        dataset = pickle.load(f) 

    test = np.load(args.test_dataset, allow_pickle=True)
    test_data = test["data"]
    test_config = test["config"].item()
    
    data = merge_dictionary(dataset)
    inputs = []
    for k in cfg.Dataset.modalities : 
        if k == "rewards":
            data[k] = data[k].reshape(-1,1)
        inputs.append(data[k])
    inputs = np.concatenate(inputs, axis=-1)
    inputs = torch.from_numpy(inputs).float()
    D = inputs.shape[-1]

    normalizer = normalizer_factory(cfg.Dataset.normalizer_type, inputs, skip_dims=[])

    
    dataset = DiffusionTrajectoryDataset(
        test_data,
        args.dataset,
        seq_len = cfg.Dataset.seq_len,
        discounted_return = cfg.Dataset.discounted_return,
        gamma = cfg.Dataset.gamma,
        restore_rewards=cfg.Dataset.restore_rewards,
        penalty = 0,  
    )

    now = datetime.now()
    date_ = now.strftime("%Y-%m-%d")
    time_ = now.strftime("%H:%M")
    model_nm = args.config_name.split('.')[0]
    
    fname = args.dataset+"/"+model_nm+"/"+date_+"/"+time_
    
    resfolder = os.path.join(args.results_folder, fname)
    if not os.path.exists(resfolder):
        os.makedirs(resfolder)
    
    resfolder = args.test_dataset[:-len(args.test_dataset.split('/')[-1])]
    
    
    # with open(os.path.join(resfolder, "config.yaml"), "w") as f:
    #     OmegaConf.save(cfg, f)

    # Create the diffusion model and trainer.
    diffusion = construct_diffusion_model(data_shape=data_shape,
                                          normalizer=normalizer,
                                          denoising_network=cfg.construct_diffusion_model.denoising_network,
                                          edm_config=cfg.ElucidatedDiffusion,
                                          disable_terminal_norm = cfg.construct_diffusion_model.disable_terminal_norm,
                                          cond_dim=cfg.construct_diffusion_model.denoising_network.cond_dim,
                                          )
    
    trainer = Trainer(
        diffusion_model=diffusion,
        dataset=dataset,
        dataset_nm = args.dataset,
        results_folder=resfolder,
        train_batch_size=cfg.Trainer.train_batch_size,
        train_lr=cfg.Trainer.train_lr,
        lr_scheduler=cfg.Trainer.lr_scheduler,
        weight_decay=cfg.Trainer.weight_decay,
        train_num_steps=cfg.Trainer.train_num_steps,
        save_and_sample_every=cfg.Trainer.save_and_sample_every,
        cond_dim=cfg.construct_diffusion_model.denoising_network.cond_dim,
        modalities = cfg.Dataset.modalities,
        reweighted_training = cfg.Trainer.reweighted_training,
        reward_scale = cfg.Dataset.reward_scale,
        discounted_return = cfg.Dataset.discounted_return
    )


    distributed_state = PartialState()
    if trainer.accelerator.is_main_process:
        trainer.ema.to(distributed_state.device)
    # Load the last checkpoint.
    #trainer.load(milestone=trainer.train_num_steps)
    trainer.load(ckpt_path=args.ckpt_path)

    
    # Generate samples and save them.
    if trainer.accelerator.is_main_process:
        batch_size = 1500

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        generator = SimpleDiffusionGenerator(
                    env=env,
                    ema_model=trainer.ema.ema_model,
                    modalities=cfg.Dataset.modalities,
                    sample_batch_size=batch_size,
                )
        diffusion = trainer.ema.ema_model
        lls = []
        for d in tqdm(loader):
            states, actions, rewards, next_states, returns, time_steps, terminals = d
            
            samples, cond, terminals, cond_state = generator.prepare_sampling_data(states,
                                                                                   actions,
                                                                                   rewards,
                                                                                   returns,
                                                                                   next_states,
                                                                                   terminals,
                                                                                   device=distributed_state.device,
                                                                                   reward_scale=cfg.Dataset.reward_scale,
                                                                                   guidance_rewardscale=cfg.SimpleDiffusionGenerator.guidance_rewardscale,
                                                                                   fixed_rewardscale=cfg.SimpleDiffusionGenerator.fixed_rewardscale,
                                                                                   cond_state = False,
                                                                                   max_conditioning_return = dataset.episode_rewards.max(),
                                                                                   discounted_return = cfg.Dataset.discounted_return,
                                                                                   )

            ll = diffusion.log_likelihood(samples, cond=cond)

            ll = ll[0].cpu().numpy()
            lls.append(ll)

            if len(lls)==3:
                temp = np.concatenate(lls, axis=0)

                save_file_name = args.test_dataset.split('/')[-1].split('.')[0] + "_loglikelihood_temp.npz"
                savepath = os.path.join(resfolder, save_file_name)
                np.savez(savepath, 
                        data = temp,
                        config = test_config)
        


        lls = np.concatenate(lls, axis=0)

        save_file_name = args.test_dataset.split('/')[-1].split('.')[0] + "_loglikelihood.npz"
        savepath = os.path.join(resfolder, save_file_name)
        np.savez(savepath, 
                data = lls,
                config = test_config)
        


            