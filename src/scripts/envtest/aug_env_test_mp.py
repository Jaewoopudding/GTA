import gym
from stable_baselines3.common.vec_env import SubprocVecEnv
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
import ray
import time

import warnings
warnings.filterwarnings("ignore")


path = pyrootutils.find_root(search_from=__file__, indicator=".aug-project-root")
pyrootutils.set_root(path = path,
                     project_root_env_var = True,
                     dotenv = True,
                     pythonpath = True)


import src.envs as environments


# from src.data.norm import MinMaxNormalizer, normalizer_factory
# from src.diffusion.utils import make_inputs,  split_diffusion_transition, construct_diffusion_model
# from corl.shared.utils  import get_saved_dataset
# from corl.shared.buffer import DiffusionDataset
 # to register extended environments



def transfer_flatten_to_sim(state, env):

    sim = env.unwrapped.sim

    idx_time = 0
    idx_qpos = idx_time + 1
    idx_qvel = idx_qpos + sim.model.nq

    qpos = state[idx_qpos:idx_qpos + sim.model.nq]
    qvel = state[idx_qvel:idx_qvel + sim.model.nv]
    
    return qpos, qvel


# class VectorizedEnv:
#     def __init__(self,
#                  envs : List[gym.Env],
#                  ):
#         self.envs = envs
    
#     @ray.remote
#     def reset(self, states):
#         self.envs = [env.reset(state=state) for env, state in zip(self.envs, states)]
#         return self.envs
    
#     @ray.remote
#     def step(self, actions):
#         next_observations = []
#         rewards = []
#         dones = []

#         for env, action in zip(self.envs, actions):
#             next_obs, reward, done, _ = env.step(action)
#             next_observations.append(next_obs)
#             rewards.append(reward)
#             dones.append(done)
        
#         return np.array(next_observations), np.array(rewards), np.array(dones)


#@ray.remote(num_cpus=36)
def reset_and_step(env, state, action):
    env.reset(state=state)
    next_obs, reward, done, _ = env.step(action)
    env.close()
    return next_obs, reward

@ray.remote(num_cpus=24)
def env_test(env_nm, data):
    reward_mse = []
    dynamic_mse = []
    real_rewards = []
    iteration = len(data)-1
    import src.envs as environments
    for epi in tqdm(range(iteration)):
        for timestep in range(data[epi]["observations"].shape[0]):
            env = gym.make(env_nm)
            state = data[epi]["observations"][timestep]
            action = data[epi]["actions"][timestep]
            next_state = data[epi]["next_observations"][timestep]
            env.reset(state = state)
            real_obs, real_reward, done, _ = env.step(action)

            mse = np.square(real_obs-next_state).mean()
            rewardmse = np.square(real_reward-data[epi]["rewards"][timestep]).mean()

            reward_mse.append(rewardmse)
            dynamic_mse.append(mse)
            real_rewards.append(real_reward)

            env.close()
    return reward_mse, dynamic_mse, real_rewards


@ray.remote(num_cpus=32)
def env_test2(env_nm, data):
    reward_mse = []
    dynamic_mse = []
    real_rewards = []
    iteration = len(data)-1
    import src.envs as environments
    for epi in tqdm(range(iteration)):
        envs = [gym.make(env_nm) for _ in range(len(data[epi]["observations"]))]
        
        states = data[epi]["observations"]
        actions = data[epi]["actions"]
        real_next = data[epi]["next_observations"]

        envs = [reset_and_step(env = env, state = state, action = action) for state, action, env in zip(states, actions, envs)]
        
        next_state = np.array([e[0] for e in envs])
        rewards = np.fromiter((e[1] for e in envs), dtype=float)


        dynamicmse = np.square(real_next-next_state).mean()
        rewardmse = np.square(rewards-data[epi]["rewards"]).mean()


        reward_mse.append(rewardmse)
        dynamic_mse.append(dynamicmse)
        real_rewards.append(rewards)
    
    return reward_mse, dynamic_mse, real_rewards


def make_and_reset(env_nm, state):
    env = gym.make(env_nm)
    #env.reset(state=state)
    return env


def batchify(data, batch_num):
    states = data["observations"]
    actions = data["actions"]
    next_state = data["next_observations"]

    state_batch = []
    action_batch = []
    next_state_batch = []

    data_len = len(data)
    batch_size = data_len//batch_num
    for i in range(batch_num):
        if i == batch_num-1:
            state_batch.append(states[i*batch_size:])
            action_batch.append(actions[i*batch_size:])
            next_state_batch.append(next_state[i*batch_size:])

        else:
            state_batch.append(states[i*batch_size:(i+1)*batch_size])
            action_batch.append(actions[i*batch_size:(i+1)*batch_size])
            next_state_batch.append(next_state[i*batch_size:(i+1)*batch_size])

    return state_batch, action_batch, next_state_batch



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='halfcheetah-medium-v2')
    parser.add_argument('--data_path', type=str, default="/home/sujin/project/Augmentation-For-OfflineRL/results/halfcheetah-medium-v2/temporalattention_denoiser/2023-11-22/16:50/")
    parser.add_argument('--data_name', type=str, default="5M-1_1x-temporalattention-50-sar.npy")
    parser.add_argument('--id', type=int, default=None)
    args = parser.parse_args()
    
    # with open(f"{args.data_path}/config.yaml", "r") as f:
    #     config = yaml.load(f, Loader=yaml.FullLoader)
    
    # wandb.init(
    #     project = args.wandb_project,
    #     entity = args.wandb_entity,
    #     config = config,
    #     name = args.data_name
    # )
    #ray.init()
    
    env = args.dataset.split('-')[0]
    if env == 'hopper':
        env_nm = "HopperExt-v2"
    elif env == 'halfcheetah':
        env_nm = "HalfCheetah-v2"
    elif env == 'walker2d':
        env_nm = "Walker2dExt-v2"


    data_nm =  args.data_path + args.data_name
    data = np.load(data_nm, allow_pickle=True)


    dynamic_mse = []
    reward_mse = []
    real_rewards = []

    iteration = args.num_of_sample if args.num_of_sample is not None else len(data)-1

    batch_num = 1

    for epi in tqdm(range(iteration)):
        
        states, actions, next_state = batchify(data[epi],batch_num)
        states = data[epi]["observations"]
        actions = data[epi]["actions"]
        next_state = data[epi]["next_observations"]

        #for n in range(batch_num):
        envs = AsyncVectorEnv([lambda : make_and_reset(env_nm=env_nm, state = s) for s in states])
        envs.reset()
        obs, rewards, done, info = envs.step(actions)
        envs.close()

        dynamicmse = np.square(obs-next_state).mean()
        rewardmse = np.square(rewards-data[epi]["rewards"]).mean()


        reward_mse.append(rewardmse)
        dynamic_mse.append(dynamicmse)
        real_rewards.append(rewards)


    print("dynamic_mse")
    print("mean, std, min, max")
    dynamic_mse_mean = dynamic_mse.mean()
    dynamic_mse_std = dynamic_mse.std()
    dynamic_mse_min = dynamic_mse.min()
    dynamic_mse_max = dynamic_mse.max() 
    print(dynamic_mse_mean)
    print(dynamic_mse_std)
    print(dynamic_mse_min)
    print(dynamic_mse_max)

    print("dynamic_mse")
    l = len(dynamic_mse)
    print("q1, q2, q3")
    dynamic_mse_q1 = dynamic_mse[int(l*0.25)]
    dynamic_mse_q2 = dynamic_mse[int(l*0.5)]
    dynamic_mse_q3 = dynamic_mse[int(l*0.75)]
    print(dynamic_mse_q1)
    print(dynamic_mse_q2)
    print(dynamic_mse_q3)

    reward_mse = np.array(reward_mse)
    reward_mse = np.sort(reward_mse)

    reward_mse_mean =reward_mse.mean()
    reward_mse_std =reward_mse.std()
    reward_mse_min =reward_mse.min()
    reward_mse_max =reward_mse.max()

    print("reward_mse")
    print("mean, std, min, max")
    print(reward_mse_mean)
    print(reward_mse_std)
    print(reward_mse_min)
    print(reward_mse_max)    

    print("reward_mse")
    l = len(reward_mse)
    print("q1, q2, q3")

    reward_mse_q1 = reward_mse[int(l*0.25)]
    reward_mse_q2 = reward_mse[int(l*0.5)]
    reward_mse_q3 = reward_mse[int(l*0.75)]

    print(reward_mse_q1)
    print(reward_mse_q2)
    print(reward_mse_q3)


    real_rewards = np.array(real_rewards)
    real_rewards = np.sort(real_rewards)

    print("real_rewards")
    print("mean, std, min, max")

    oracle_reward_mean = real_rewards.mean()
    oracle_reward_std = real_rewards.std()
    oracle_reward_min = real_rewards.min()
    oracle_reward_max = real_rewards.max()

    print(oracle_reward_mean)
    print(oracle_reward_std)
    print(oracle_reward_min)
    print(oracle_reward_max)

    print("real_rewards")
    l = len(real_rewards)
    print("q1, q2, q3")

    oracle_reward_q1 = real_rewards[int(l*0.25)]
    oracle_reward_q2 = real_rewards[int(l*0.5)]
    oracle_reward_q3 = real_rewards[int(l*0.75)]

    print(oracle_reward_q1)
    print(oracle_reward_q2)
    print(oracle_reward_q3)


    stats = {
        args.data_name : {
            "dynamic_mse_mean" : dynamic_mse_mean,
            "dynamic_mse_std"  : dynamic_mse_std,
            "dynamic_mse_min"  : dynamic_mse_min,
            "dynamic_mse_max"  : dynamic_mse_max,
            "dynamic_mse_q1"   : dynamic_mse_q1 ,
            "dynamic_mse_q2"   : dynamic_mse_q2 ,
            "dynamic_mse_q3"   : dynamic_mse_q3 ,            
            
            "reward_mse_mean" : reward_mse_mean,
            "reward_mse_std"  : reward_mse_std,
            "reward_mse_min"  : reward_mse_min,
            "reward_mse_max"  : reward_mse_max,
            "reward_mse_q1"   : reward_mse_q1 ,
            "reward_mse_q2"   : reward_mse_q2 ,
            "reward_mse_q3"   : reward_mse_q3 ,

            "oracle_reward_mean" : oracle_reward_mean,
            "oracle_reward_std"  : oracle_reward_std,
            "oracle_reward_min"  : oracle_reward_min,
            "oracle_reward_max"  : oracle_reward_max,
            "oracle_reward_q1"   : oracle_reward_q1 ,
            "oracle_reward_q2"   : oracle_reward_q2 ,
            "oracle_reward_q3"   : oracle_reward_q3 ,
        }
    }

    import pandas as pd
    df = pd.DataFrame.from_dict(stats, orient='index')
    try:
        data = pd.read_csv(f"./statistics/data_statistics.csv", index_col=0)
        df = pd.concat([data, df])
    except:
        pass    
    df.to_csv("./statistics/data_statistics.csv")
    
    print(df)
    print("=========<STATISTIC ANALYSIS IS FINISHED>=========")
    
            
            
