import gym
from gym.vector import AsyncVectorEnv
#import gymnasium as gym
import numpy as np
import torch
import argparse
from tqdm import tqdm
import pyrootutils
import wandb
import yaml
from typing import List
import os
import time

path = pyrootutils.find_root(search_from=__file__, indicator=".aug-project-root")
pyrootutils.set_root(path = path,
                     project_root_env_var = True,
                     dotenv = True,
                     pythonpath = True)


import src.envs as envs


# from src.data.norm import MinMaxNormalizer, normalizer_factory
# from src.diffusion.utils import make_inputs,  split_diffusion_transition, construct_diffusion_model
# from corl.shared.utils  import get_saved_dataset
# from corl.shared.buffer import DiffusionDataset
#  # to register extended environments



def transfer_flatten_to_sim(state, env):

    sim = env.unwrapped.sim

    idx_time = 0
    idx_qpos = idx_time + 1
    idx_qvel = idx_qpos + sim.model.nq

    qpos = state[idx_qpos:idx_qpos + sim.model.nq]
    qvel = state[idx_qvel:idx_qvel + sim.model.nv]
    
    return qpos, qvel



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='halfcheetah-medium-v2')
    parser.add_argument('--data_path', type=str, default="/home/sujin/project/Augmentation-For-OfflineRL/results/halfcheetah-medium-v2/smallmixer_denoiser_v4_reweighting_tune/2024-01-28/23:06/0M-1_3x-smallmixer-50-sar-temp2_0_10.npz")
    parser.add_argument('--batch_num', type=int, default=6)
    parser.add_argument('--batch_id', type=int, default=0)
    parser.add_argument('--test_partial', action='store_true', default=False)
    parser.add_argument('--percentage', type=float, default=1.0)
    args = parser.parse_args()
    
    # wandb.init(
    #     project = args.wandb_project,
    #     entity = args.wandb_entity,
    #     config = config,
    #     name = data_name
    # )
    
    env = args.dataset.split('-')[0]
    if env == 'hopper':
        env_nm = "HopperExt-v2"
    elif env == 'halfcheetah':
        env_nm = "HalfCheetahExt-v2"
    elif env == 'walker2d':
        env_nm = "Walker2dExt-v2"

    original = np.load(f"data/{args.dataset}.npy", allow_pickle=True)
    obs = []
    rewards = []
    for epi in original:
        obs.append(epi["observations"])
        rewards.append(epi["rewards"])
    obs = np.concatenate(obs, axis=0)
    rewards = np.concatenate(rewards, axis=0)

    state_mean = obs.mean(axis=0)
    state_std = obs.std(axis=0)+1e-3

    reward_mean = rewards.mean(axis=0)
    reward_std = rewards.std(axis=0)+1e-3

    data_name = args.data_path.split('/')[-1]
    dir_path = args.data_path[:-len(data_name)]
    samples = np.load(args.data_path, allow_pickle=True)


    dynamic_mse = []
    reward_mse = []
    real_rewards = []
    gen_rewards = []

    if "npz" in data_name:
        data = samples["data"]
        config = samples["config"].item()
    else:
        data = samples
        config = {}

    if args.test_partial:
        # np.random.seed(1)
        # np.random.shuffle(data)
        # data = data[:len(data)//5]
        if args.percentage > 0:
            crit = len(data)//5
            reward = [d["rewards"].sum() for d in data]
            idx = np.argsort(reward)[::-1]
            len_ = int(len(data)*args.percentage)
            data = data[idx[:len_]]

            if len(data) > crit:
                np.random.seed(1)
                np.random.shuffle(data)
                data = data[:crit]
        else:
            print("test randomly")
            np.random.seed(1)
            np.random.shuffle(data)
            data = data[:len(data)//10]

            


    boundaries = np.linspace(0, len(data), args.batch_num+1)
    ranges = []
    for i in range(args.batch_num):
        ranges.append(np.arange(boundaries[i], boundaries[i+1], dtype=int))
    epis = ranges[args.batch_id]

    print(f"batch {args.batch_id} has {len(epis)} episodes")
    for epi in tqdm(epis):
        for timestep in range(data[epi]["observations"].shape[0]):
            env = gym.make(env_nm)
            state = data[epi]["observations"][timestep]
            action = data[epi]["actions"][timestep]
            next_state = data[epi]["next_observations"][timestep]
            env.reset(state = state)
            real_obs, real_reward, done, _ = env.step(action)
            real_obs = (real_obs-state_mean)/state_std
            real_reward = (real_reward-reward_mean)/reward_std
            next_state = (next_state-state_mean)/state_std
            rew = data[epi]["rewards"][timestep]
            rew = (rew-reward_mean)/reward_std

            mse = np.square(real_obs-next_state).mean()
            rewardmse = np.square(real_reward-rew).mean()
            
            real_reward = real_reward*reward_std+reward_mean
            reward_mse.append(rewardmse)
            dynamic_mse.append(mse)
            real_rewards.append(real_reward)
            gen_rewards.append(data[epi]["rewards"][timestep])

            env.close()
    
    dynamic_mse = np.array(dynamic_mse)
    reward_mse = np.array(reward_mse)
    real_rewards = np.array(real_rewards)
    gen_rewards = np.array(gen_rewards)
    

    data_nm = data_name.split('.')[0]
    resfolder = f"./statistics/{args.dataset}/{data_nm}_{args.percentage}"
    if not os.path.exists(resfolder):
        os.makedirs(resfolder)

    np.savez(f"{resfolder}/batch{args.batch_id}.npz", 
             dynamic_mse=dynamic_mse, 
             reward_mse=reward_mse, 
             real_rewards=real_rewards,
             gen_rewards=gen_rewards,
             config = config)

