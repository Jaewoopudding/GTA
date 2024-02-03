import numpy as np
import torch
import argparse
from tqdm import tqdm
import pyrootutils
from typing import List
import os

path = pyrootutils.find_root(search_from=__file__, indicator=".aug-project-root")
pyrootutils.set_root(path = path,
                     project_root_env_var = True,
                     dotenv = True,
                     pythonpath = True)


import src.envs as envs
from corl.shared.buffer import DiffusionTrajectoryDataset



def calc_mmd(x, y, 
             kernel = "rbf", 
             device = "cuda:0"):
    """Emprical maximum mean discrepancy. The lower the result
       the more evidence that distributions are the same.

    Args:
        x: first sample, distribution P
        y: second sample, distribution Q

    borrowed from https://www.kaggle.com/code/onurtunali/maximum-mean-discrepancy/notebook
    """
    x, y = torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx, ry = (xx.diag().unsqueeze(0).expand_as(xx)), (yy.diag().unsqueeze(0).expand_as(yy))
    dxx, dyy, dxy = rx.t() + rx - 2. * xx, ry.t() + ry - 2. * yy, rx.t() + ry - 2. * zz
    XX, YY, XY = (torch.zeros(xx.shape).to(device), torch.zeros(xx.shape).to(device), torch.zeros(xx.shape).to(device))
    if kernel == "rbf":
        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5*dxx/a)
            YY += torch.exp(-0.5*dyy/a)
            XY += torch.exp(-0.5*dxy/a)

    elif kernel == "multiscale":
        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a**2 * (a**2 + dxx)**-1
            YY += a**2 * (a**2 + dyy)**-1
            XY += a**2 * (a**2 + dxy)**-1
        
    return torch.mean(XX + YY - 2. * XY)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='halfcheetah-medium-v2')
    parser.add_argument('--data_path', type=str, default="/home/orl/ablations-halfcheetah-medium-v2/envtest")
    parser.add_argument('--test_data_nm', type=str, default="5M-1_3x-smallmixer-25-sar-temp2_0.npz")
    parser.add_argument('--test_partial', action='store_true', default=True)
    parser.add_argument('--percentage', type=float, default=0.1)
    parser.add_argument('--kernel', type=str, default="rbf")
    parser.add_argument('--device', type=str, default="cuda:7")
    args = parser.parse_args()
    device = args.device
    kernel = args.kernel    
    
    env = args.dataset.split('-')[0]
    if env == 'hopper':
        env_nm = "HopperExt-v2"
    elif env == 'halfcheetah':
        env_nm = "HalfCheetahExt-v2"
    elif env == 'walker2d':
        env_nm = "Walker2dExt-v2"


    generated_dataset = np.load(f"{args.data_path}/{args.test_data_nm}", allow_pickle=True)

    generated_data = generated_dataset["data"]
    generated_config = generated_dataset["config"].item()

    if args.test_partial:
        # np.random.seed(1)
        # np.random.shuffle(data)
        # data = data[:len(data)//5]
        reward = [d["rewards"].sum() for d in generated_data]
        idx = np.argsort(reward)[::-1]
        len_ = int(len(generated_data)*args.percentage)
        generated_data = generated_data[idx[:len_]]

    # boundaries = np.linspace(0, len(generated_data), args.batch_num+1)
    # ranges = []
    # for i in range(args.batch_num):
    #     ranges.append(np.arange(boundaries[i], boundaries[i+1], dtype=int))

    original_data = np.load(f"data/{args.dataset}.npy", allow_pickle=True)
    dataset = DiffusionTrajectoryDataset(
        dataset = original_data,
        dataset_nm = args.dataset,
        seq_len = 31,
        discounted_return = True,
        gamma = 0.99,
        penalty = 100,
    )
    subtrajectories = dataset.dataset
    org_samples = []
    gen_samples = []
    for sub in subtrajectories:
        states, actions, rewards, next_states, returns, time_steps, terminals, rtg = sub
        sample = np.concatenate([states, actions, rewards], axis=-1)
        org_samples.append(sample)
    gen_samples = []
        
    for gen in generated_data:
        sample = np.concatenate([gen["observations"], gen["actions"], gen["rewards"].reshape(-1,1)], axis=-1)
        gen_samples.append(sample)
    
    
    org_samples = np.array(org_samples)
    gen_samples = np.array(gen_samples)
    print(org_samples.shape, gen_samples.shape)

    B, T, D = org_samples.shape
    org_samples = org_samples.reshape(-1, T*D)
    gen_samples = gen_samples.reshape(-1, T*D)
    
    print(org_samples.shape, gen_samples.shape)

    
    # dataset = DiffusionTrajectoryDataset(
    #     dataset = original_data,
    #     dataset_nm = args.dataset,
    #     seq_len = 31,
    #     discounted_return = True,
    #     gamma = 0.99,
    #     penalty = 100,
    # )
    # subtrajectories = dataset.dataset
    # org_samples = []
    # gen_samples = []
    # for sub in subtrajectories:
    #     states, actions, rewards, next_states, returns, time_steps, terminals, rtg = sub
    #     sample = np.concatenate([states, actions, rewards], axis=-1)
    #     org_samples.append(sample)
        
    # for gen in generated_data:
    #     sample = np.concatenate([gen["observations"], gen["actions"], gen["rewards"].reshape(-1,1)], axis=-1)
    #     gen_samples.append(sample)

    # org_samples = np.array(org_samples)
    # gen_samples = np.array(gen_samples)

    # B, T, D = org_samples.shape
    # org_samples = org_samples.reshape(-1, T*D)
    # # gen_samples = gen_samples.reshape(-1, T*D)
    # import pdb; pdb.set_trace()
    print("====================Computing MMD====================")
    num_epoch = 100
    np.random.seed(1)
    mmd_scores = []
    for _ in tqdm(range(50)):
        np.random.shuffle(org_samples)
        t_org = org_samples[:10000]
        np.random.shuffle(gen_samples)
        t_gen = gen_samples[:10000]
        mmd = calc_mmd(t_org, t_gen, kernel, device)
        mmd_scores.append(mmd.cpu().numpy())

    data_info = args.data_path.split('/')[-2]
    resfolder = f"./statistics/quality_metrics/{args.dataset}/{args.test_data_nm}_{args.percentage}"
    if not os.path.exists(resfolder):
        os.makedirs(resfolder)
    
    mmd_scores = np.array(mmd_scores)
    print("mmd_scores")
    print("mean, std, min, max")
    mmd_scores_mean = mmd_scores.mean()
    mmd_scores_std = mmd_scores.std()
    mmd_scores_min = mmd_scores.min()
    mmd_scores_max = mmd_scores.max()

    data_id = f"{args.dataset}_{args.test_data_nm}_{args.percentage}_trajectory"
    stats = {
        data_id: {
            "mean" : mmd_scores_mean,
            "std" : mmd_scores_std,
            "min" : mmd_scores_min,
            "max" : mmd_scores_max,
        }
    }

    import pandas as pd
    df = pd.DataFrame.from_dict(stats, orient='index')
    try:
        data = pd.read_csv(f"./statistics/quality_metrics/MMDs.csv", index_col=0)
        df = pd.concat([data, df])
    except:
        pass    
    df.to_csv("./statistics/quality_metrics/MMDs.csv")
    

    

