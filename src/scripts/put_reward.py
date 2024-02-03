import gym
import os
from typing import List, Tuple, Union
import numpy as np
import torch
import argparse
from tqdm import tqdm

import pyrootutils

path = pyrootutils.find_root(search_from = __file__, indicator=".aug-project-root")
pyrootutils.set_root(path = path,
                     project_root_env_var = True,
                     dotenv = True,
                     pythonpath = True)

from src.dynamics.reward import RewardModel, TrainConfig


import src.environments as environments # to register extended environments

from corl.shared.buffer import ReplayBuffer, RewardNormalizer, StateNormalizer, DiffusionConfig
from corl.shared.logger import Logger
from corl.shared.utils  import wandb_init, set_seed, wrap_env, soft_update, compute_mean_std, normalize_states, eval_actor, get_saved_dataset, get_generated_dataset, merge_dictionary

from corl.shared.s4rl import S4RLAugmentation




TensorBatch = List[torch.Tensor]
os.environ["WANDB_MODE"] = "online"

EXP_ADV_MAX = 100.0
LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='hopper-medium-expert-v2')
    parser.add_argument('--data_path', type=str, default="/home/sujin/project/Augmentation-For-OfflineRL/results/hopper-medium-expert-v2/")
    parser.add_argument('--data_name', type=str, default="hopper-medium-expert-v2_debug1.npy")
    parser.add_argument('--ckpt_name', type=str, default="checkpoint_99999.pt")
    args = parser.parse_args()
    
    config = TrainConfig
    
    env = args.dataset.split('-')[0]
    if env == 'hopper':
        env_nm = "HopperExt-v2"
    elif env == 'halfcheetah':
        env_nm = "HalfCheetahExt-v2"
    elif env == 'walker2d':
        env_nm = "Walker2dExt-v2"

    env = gym.make(env_nm)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    trainer = RewardModel(state_dim, 
                          action_dim, 
                          config.network_width, 
                          optimizer="adam",
                          dim_mults=(1, 2, 4, 1), 
                          device=config.device)
    
    print("load model")
    ckpt_path = f"./reward_model/{args.dataset}_checkpoints/{args.ckpt_name}"
    ckpt = torch.load(ckpt_path)
    trainer.load_state_dict(ckpt)

    data_nm =  args.data_path + args.data_name
    data = np.load(data_nm, allow_pickle=True)
    dataset = merge_dictionary(data)
    
    if config.normalize:
        state_mean, state_std = compute_mean_std(dataset["observations"], eps=1e-3)
    else:
        state_mean, state_std = 0, 1


    for epi in tqdm(range(0, len(data))):
        state = data[epi]["observations"] # (31, 17)
        action = data[epi]["actions"] # (31, 6)
        next_state = data[epi]["next_observations"] # (31, 17)

        state = normalize_states(state, state_mean, state_std)
        next_state = normalize_states(next_state, state_mean, state_std)

        state = torch.tensor(state, dtype=torch.float, device=config.device)
        action = torch.tensor(action, dtype=torch.float, device=config.device)
        next_state = torch.tensor(next_state, dtype=torch.float, device=config.device)


        reward = trainer.inference(state, action, next_state)
        data[epi]["rewards"] = reward.reshape(-1,).cpu().numpy()



    mses = []
    reward_mses = []
    for epi in tqdm(range(0, len(data),500)):
        for timestep in range(data[epi]["observations"].shape[0]):
            env = gym.make(env_nm)
            state = data[epi]["observations"][timestep]
            action = data[epi]["actions"][timestep]
            next_state = data[epi]["next_observations"][timestep]
            env.reset(state = state)
            real_obs, real_reward, done, _ = env.step(action)
            mse = np.square(real_obs-next_state).mean()
            reward_mse = np.square(real_reward-data[epi]["rewards"][timestep]).mean()
            mses.append(mse)
            reward_mses.append(reward_mse)

        
    mses = np.array(mses)
    mses = np.sort(mses)

    print("dynamic_mse")
    print("mean, std, min, max")
    print(mses.mean())
    print(mses.std())
    print(mses.min())
    print(mses.max())

    print("dynamic_mse")
    l = len(mses)
    print("q1, q2, q3")
    print(mses[int(l*0.25)])
    print(mses[int(l*0.5)])
    print(mses[int(l*0.75)])

    reward_mses = np.array(reward_mses)
    reward_mses = np.sort(reward_mses)

    print("reward_mse")
    print("mean, std, min, max")
    print(reward_mses.mean())
    print(reward_mses.std())
    print(reward_mses.min())
    print(reward_mses.max())

    print("reward_mse")
    l = len(reward_mses)
    print("q1, q2, q3")
    print(reward_mses[int(l*0.25)])
    print(reward_mses[int(l*0.5)])
    print(reward_mses[int(l*0.75)])
    nm = args.data_name.replace(".npy", "")
    np.save(f"{args.data_path}/{nm}_with_reward.npy", data)





