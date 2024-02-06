# Train diffusion model on D4RL transitions.
import os
import argparse
from datetime import datetime
import random
import tqdm
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


from src.diffusion.elucidated_diffusion import Trainer, define_rewardweighting_sampler
from src.data.norm import MinMaxNormalizer, normalizer_factory
from src.diffusion.utils import split_diffusion_trajectory, construct_diffusion_model
from corl.shared.utils  import merge_dictionary
from corl.shared.buffer import DiffusionDataset, DiffusionTrajectoryDataset
from corl.shared.utils  import merge_dictionary




class SimpleDiffusionGenerator:
    def __init__(
            self,
            env: gym.Env,
            ema_model,
            num_sample_steps: int = 128,
            sample_batch_size: int = 1000,
            modalities : List[str] = ["observations", "actions", "rewards"]
    ):
        self.env = env
        self.diffusion = ema_model
        self.diffusion.eval()
        # Clamp samples if normalizer is MinMaxNormalizer
        self.clamp_samples = isinstance(self.diffusion.normalizer, MinMaxNormalizer)
        print(f'Clamping samples: {self.clamp_samples}')
        self.num_sample_steps = num_sample_steps
        self.sample_batch_size = sample_batch_size
        self.modalities = modalities
        print(f'Sampling using: {self.num_sample_steps} steps, {self.sample_batch_size} batch size.')

    
    def prepare_sampling_data(
            self,
            states,
            actions,
            rewards,
            returns,
            next_states,
            terminals,
            device,
            reward_scale : float = 1.0,
            guidance_rewardscale : Optional[float] = None,
            fixed_rewardscale : Optional[float] = None,
            cond_state : bool = False,
            max_conditioning_return : Optional[float] = None,
            discounted_return : bool = False,
    ):
        if discounted_return:
            conditioning_value = returns[:,0,:].cpu().numpy()
        else:
            conditioning_value = rewards.squeeze().cpu().numpy().sum(axis=-1)
        cond = np.copy(conditioning_value)
        if guidance_rewardscale is not None: 
            conditioning_value *= guidance_rewardscale
            cond[conditioning_value>0] = cond[conditioning_value>0] * guidance_rewardscale
            reverse_guidancescale = -guidance_rewardscale+2
            cond[conditioning_value<0] = cond[conditioning_value<0] * reverse_guidancescale
        cond = cond.reshape(-1, 1)


        B, T, D = states.shape
        rewards = rewards.reshape(B, T, 1)

        data = []
        for mod in self.modalities:
            if mod == 'observations':
                data.append(states)
            elif mod == 'actions':
                data.append(actions)
            elif mod == 'rewards':
                data.append(rewards) 
        last_state = next_states[:,-1, None,:]
        last_action = torch.zeros_like(actions[:,-1, None,:])
        last_reward = torch.zeros_like(rewards[:,-1, None,:])
        last_transition = torch.cat([last_state, last_action, last_reward], dim=-1).to(device)
            
        data = torch.cat(data, dim=-1).to(device)
        data = torch.cat([data, last_transition], dim=1)

        if cond_state:
            cond_state = {}
            is_terminate = terminals.sum(dim=-1)
            for batch_idx, term in enumerate(is_terminate):
                if term == 1:
                    cond_state[batch_idx] = data[batch_idx,-2:,:] # condintion on last state, action, reward, state
                    cond[batch_idx] = conditioning_value[batch_idx] 

        else:
            cond_state = None


        cond = torch.from_numpy(cond).to(device)
        cond *= reward_scale

        return data, cond, terminals, cond_state

    def sample_back_and_forth(
            self,
            data_loader,
            num_samples: int,
            noise_level : float = 0.5, 
            temperature : float = 1.0,
            reward_scale : float = 1.0,
            guidance_rewardscale : Optional[float] = None,
            fixed_rewardscale : Optional[float] = None,
            device : str = "cuda",
            state_conditioning : bool = False,
            max_conditioning_return : Optional[float] = None,
            discounted_return : bool = False,
            retain_original : bool = False
    ) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        # assert num_samples % self.sample_batch_size == 0, 'num_samples must be a multiple of sample_batch_size'
        num_batches = num_samples // self.sample_batch_size
        if num_samples % self.sample_batch_size != 0:
            num_batches += 1
        
        generated_samples = []
        loader_iterator = iter(data_loader)
        for i in range(num_batches):
            try:
                states, actions, rewards, next_states, returns, time_steps, terminals, rtg= next(loader_iterator)
            except StopIteration:
                loader_iterator = iter(data_loader)
                states, actions, rewards, next_states, returns, time_steps, terminals, rtg = next(loader_iterator)
            state_dim = states.shape[-1]
            samples, cond, terminals, cond_state = self.prepare_sampling_data(states,
                                                                              actions,
                                                                              rewards,
                                                                              returns,
                                                                              next_states,
                                                                              terminals,
                                                                              device,
                                                                              reward_scale,
                                                                              guidance_rewardscale,
                                                                              fixed_rewardscale,
                                                                              state_conditioning,
                                                                              max_conditioning_return,
                                                                              discounted_return,
                                                                              )
            if not state_conditioning:
                cond_state = None

            print(f'Generating split {i + 1} of {num_batches}')
            sampled_outputs = self.diffusion.sample_back_and_forth(
                samples=samples,
                num_sample_steps=self.num_sample_steps,
                clamp=self.clamp_samples,
                cond = cond,
                cond_state = cond_state,
                noise_level = noise_level,
                temperature = temperature,
                state_dim = state_dim
            )
            sampled_outputs = sampled_outputs.cpu().numpy()


            if "rewards" in self.modalities:
                gen_state, gen_action, gen_reward, gen_next_obs = split_diffusion_trajectory(
                                                                        samples = sampled_outputs, 
                                                                        env = self.env,
                                                                        modalities = self.modalities,)
                
                for b in range(gen_state.shape[0]):
                    if retain_original:
                        temp = {
                        "observations": gen_state[b,:,:],
                        "actions": gen_action[b,:,:],
                        "next_observations": gen_next_obs[b,:,:],
                        "rewards": gen_reward[b,:,:].squeeze(),
                        "returns": returns[b,:,:].squeeze().cpu().numpy(),
                        "terminals": terminals[b,:].cpu().numpy(),
                        "timesteps": time_steps[b,:].cpu().numpy(),
                        "original_observations": states[b,:,:].cpu().numpy(),
                        "original_actions": actions[b,:,:].cpu().numpy(),
                        "original_next_observations": next_states[b,:,:].cpu().numpy(),
                        "original_rewards": rewards[b,:,:].squeeze().cpu().numpy(),
                        "RTG" : rtg[b,:].squeeze().cpu().numpy(),
                        }
                    else:
                        temp = {
                            "observations": gen_state[b,:,:],
                            "actions": gen_action[b,:,:],
                            "next_observations": gen_next_obs[b,:,:],
                            "rewards": gen_reward[b,:,:].squeeze(),
                            "returns": returns[b,:,:].squeeze().cpu().numpy(),
                            "terminals": terminals[b,:].cpu().numpy(),
                            "timesteps": time_steps[b,:].cpu().numpy(),
                            "RTG" : rtg[b,:].squeeze().cpu().numpy(),
                        }
                    generated_samples.append(temp)
            else : 
                obs, actions, next_obs =  split_diffusion_trajectory(
                                                        samples = sampled_outputs, 
                                                        env = self.env,
                                                        modalities = self.modalities,)
            

                for b in range(self.sample_batch_size):
                    temp = {
                        "observations": obs[b,:,:],
                        "actions": actions[b,:,:],
                        "next_observations": next_obs[b,:,:],
                    }
                    generated_samples.append(temp)


        return generated_samples





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='halfcheetah-medium-v2')
    parser.add_argument('--config_path', type=str, default='../../configs')
    parser.add_argument('--config_name', type=str, default='gta.yaml')
    parser.add_argument('--wandb_project', type=str, default="gta")
    parser.add_argument('--wandb_group', type=str, default="gta")
    #
    parser.add_argument('--results_folder', type=str, default='./results')
    parser.add_argument('--use_gpu', action='store_true', default=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_samples', action='store_true', default=True)
    parser.add_argument('--load_checkpoint', action='store_true')
    parser.add_argument('--ckpt_path', type=str, default=None)
    
    parser.add_argument('--back_and_forth', action='store_true')
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

    if "penalty" in cfg.Dataset:
        penalty = cfg.Dataset.penalty
        print("terminal penalty : ",penalty)
    else:
        penalty = None
        print("terminal penalty : 100")

    if "episode" in cfg.Dataset:
        print("load with Diffusion Dataset")
        if cfg.Dataset.episode:
            dataset = DiffusionDataset(
                dataset,
                args.dataset,
                seq_len = cfg.Dataset.seq_len,
                discounted_return = cfg.Dataset.discounted_return,
                gamma = cfg.Dataset.gamma,
                penalty = penalty,  
            )
    else:
        print("load with Diffusion Trajectory Dataset")
        dataset = DiffusionTrajectoryDataset(
            dataset,
            args.dataset,
            seq_len = cfg.Dataset.seq_len,
            discounted_return = cfg.Dataset.discounted_return,
            gamma = cfg.Dataset.gamma,
            restore_rewards=cfg.Dataset.restore_rewards,
            penalty = penalty,  
        )


    now = datetime.now()
    date_ = now.strftime("%Y-%m-%d")
    time_ = now.strftime("%H:%M")
    model_nm = args.config_name.split('.')[0]
    
    fname = args.dataset+"/"+model_nm+"/"+date_+"/"+time_
    
    resfolder = os.path.join(args.results_folder, fname)
    if not os.path.exists(resfolder):
        os.makedirs(resfolder)
    
    
    with open(os.path.join(resfolder, "config.yaml"), "w") as f:
        OmegaConf.save(cfg, f)

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

    if not args.load_checkpoint:
        # Initialize logging.
        wandb.init(
            project=args.wandb_project,
            config=args,
            group=args.wandb_group,
            name=resfolder.split('/')[-1],
        )
        # Train model.
        trainer.train()
    else:
        distributed_state = PartialState()
        if trainer.accelerator.is_main_process:
            trainer.ema.to(distributed_state.device)
        # Load the last checkpoint.
        trainer.load(ckpt_path=args.ckpt_path)

    if args.load_checkpoint:
        # Generate samples and save them.
        if trainer.accelerator.is_main_process:
            if args.save_samples:
                sample_batch_size = min(len(dataset),cfg.SimpleDiffusionGenerator.sample_batch_size)
                generator = SimpleDiffusionGenerator(
                    env=env,
                    ema_model=trainer.ema.ema_model,
                    modalities=cfg.Dataset.modalities,
                    sample_batch_size=sample_batch_size,
                )
                if args.back_and_forth:
                    
                    if cfg.SimpleDiffusionGenerator.weighted_sampling:
                        print(f"sampling with weighted sampling {cfg.SimpleDiffusionGenerator.weight_param}")
                        sample_loader = DataLoader(dataset, 
                                                batch_size=sample_batch_size,
                                                sampler=define_rewardweighting_sampler(dataset, args.dataset, cfg.Dataset.reward_scale, cfg.SimpleDiffusionGenerator.weight_param))                
                    else:
                        sample_loader = DataLoader(dataset, shuffle=True, batch_size=sample_batch_size)                
                    # Generate samples via back-and-forth
                        
                    num_transitions = cfg.SimpleDiffusionGenerator.save_num_transitions
                    # samllest num_transition which is bigger than cfg.SimpleDiffusionGenerator.save_num_transitions
                    # and divisible by both cfg.Dataset.seq_len and sample_batch_size

                    lcm = cfg.Dataset.seq_len * sample_batch_size
                    num_transitions = ((num_transitions + lcm - 1) // lcm) * lcm
                    num_samples = num_transitions // cfg.Dataset.seq_len
                    

                    generated_samples = generator.sample_back_and_forth(
                        data_loader=sample_loader,
                        num_samples=num_samples,
                        noise_level = cfg.SimpleDiffusionGenerator.noise_level,
                        temperature=cfg.SimpleDiffusionGenerator.temperature,
                        device=distributed_state.device,
                        guidance_rewardscale=cfg.SimpleDiffusionGenerator.amplify_returnscale,
                        reward_scale=cfg.Dataset.reward_scale,
                        state_conditioning = cfg.SimpleDiffusionGenerator.state_conditioning,
                        max_conditioning_return = dataset.trajectory_returns.max(),
                        discounted_return = cfg.Dataset.discounted_return,
                        retain_original=True
                    )


                save_file_name = f"gta_smaples.npz"
                
                gen_sample = np.array(generated_samples)
                np.random.shuffle(gen_sample)
                gen_sample = gen_sample[:(cfg.SimpleDiffusionGenerator.save_num_transitions//cfg.Dataset.seq_len)+1]
                savepath = os.path.join(f'data/generated_data/{args.dataset}', save_file_name)
                np.savez(savepath, 
                        data = gen_sample,
                        config = dict(cfg))
            

    else:
        print("diffusion training is done")

            