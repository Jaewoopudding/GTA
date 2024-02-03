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
import pickle
import time
import os
import plotly.graph_objects as go


path = pyrootutils.find_root(search_from=__file__, indicator=".aug-project-root")
pyrootutils.set_root(path = path,
                     project_root_env_var = True,
                     dotenv = True,
                     pythonpath = True)


from corl.shared.buffer import DiffusionDataset
import src.envs as envs


def plot_rewards(train_data, generated_reward, real_reward, path):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=train_data,
                               name = "train_data",
                               histnorm='probability density'))
    fig.add_trace(go.Histogram(x=generated_reward,
                               name = "generated_reward",
                               histnorm='probability density'))
    fig.add_trace(go.Histogram(x=real_reward,
                               name = "real_reward",
                               histnorm='probability density'))
    # Overlay both histograms
    fig.update_layout(barmode='overlay')
    # Reduce opacity to see both histograms
    fig.update_traces(opacity=0.75)

    data_nm = path.split('/')[-2]


    fig.update_layout(
        title_text=data_nm, # title of plot
        xaxis_title_text='Trajectory Rewards', # xaxis label
        yaxis_title_text='probability', # yaxis label
    )
    fig.show()
    fig.write_image(path)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='halfcheetah-medium-v2')
    parser.add_argument('--data_path', type=str, default="/home/orl/jaewoo/s4rl/s4rl_halfcheetah-medium-v2.npy")
    parser.add_argument('--percentage', type=float, default=0.1)
    args = parser.parse_args()

    data_nm = args.data_path.split('/')[-1].split('.')[0]
    resfolder = f"./statistics/{args.dataset}/{data_nm}_{args.percentage}"
    datas = os.listdir(resfolder)
    data_nm = f"{data_nm}_{args.percentage}"



    dynamic_mse = []
    reward_mse = []
    real_rewards = []
    gen_rewards = []
    for d in datas:
        if d.endswith(".npz"):
            data = np.load(f"{resfolder}/{d}", allow_pickle=True)
            dynamic_mse.append(data["dynamic_mse"])
            reward_mse.append(data["reward_mse"])
            real_rewards.append(data["real_rewards"])
            gen_rewards.append(data["gen_rewards"])

    cfg = data["config"].item()
    if "Dataset" in cfg.keys():
        seq_len = cfg["Dataset"]["seq_len"]
    else:
        seq_len = 100
            
    with open(f'./data/{args.dataset}.pkl','rb') as f:
        dataset = pickle.load(f) 
    dataset = DiffusionDataset(
        dataset,
        dataset_nm=args.dataset,
        seq_len = seq_len,
        discounted_return=True,
        gamma=0.99,
        restore_rewards=True,
    )
    train_data = dataset.trajectory_rewards

    dynamic_mse = np.concatenate(dynamic_mse)
    reward_mse = np.concatenate(reward_mse)
    real_rewards = np.concatenate(real_rewards)
    gen_rewards = np.concatenate(gen_rewards)

    real = real_rewards.reshape(-1, seq_len).sum(axis=1)
    gen = gen_rewards.reshape(-1, seq_len).sum(axis=1)


    print("=========<START PLOTTING>=========")
    plot_rewards(train_data, gen, real, f"{resfolder}/rewards_dist.png")

    print("=========<START STATISTIC ANALYSIS>=========")
    dynamic_mse = np.sort(dynamic_mse)

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
        data_nm : {
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

            "dataset" : args.dataset,
        }
    }

    import pandas as pd
    df = pd.DataFrame.from_dict(stats, orient='index')
    try:
        data = pd.read_csv(f"./statistics/data_statistics_{args.dataset}.csv", index_col=0)
        df = pd.concat([data, df])
    except:
        pass    
    df.to_csv(f"./statistics/data_statistics_{args.dataset}.csv")
    
    print(df)
    print("=========<STATISTIC ANALYSIS IS FINISHED>=========")
    