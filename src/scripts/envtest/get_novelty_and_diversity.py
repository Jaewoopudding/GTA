import numpy as np
import tqdm
import argparse
import scipy.spatial as ss
import torch
import pandas as pd
import time
import os

from corl.shared.utils import merge_dictionary

COMPARISON_TARGET = ['observations', 'actions', 'rewards', 'next_observations']
COMPARISON_TARGET_ORIGINAL = ['original_observations', 'original_actions', 'original_rewards', 'original_next_observations']

def percentage_trajectory_cutting(data, ratio):
    rewards_sum_array = [d['rewards'].sum() for d in data]
    sort_array = np.argsort(rewards_sum_array)
    cut_target = int(sort_array.shape[0] * ratio)
    idx = sort_array[-cut_target:]
    return data[idx]

def percentage_transition_cutting(data, ratio):
    rewards_sum_array = data['rewards']
    sort_array = np.argsort(rewards_sum_array)
    cut_target = int(sort_array.shape[0] * ratio)
    idx = sort_array[-cut_target:]
    return_data = {}
    for k, v in data.items():
        return_data[k] = v[idx]
    return [return_data]

def get_list_from_dict(dictionary):
    transition_count = dictionary['observations'].shape[0]
    return np.hstack([dictionary[key].reshape(transition_count, -1) for key in dictionary if key in COMPARISON_TARGET])

def get_original_list_from_dict(dictionary):
    transition_count = dictionary['observations'].shape[0]
    return np.hstack([dictionary[key].reshape(transition_count, -1) for key in dictionary if key in COMPARISON_TARGET_ORIGINAL])

def normalize(target, mean, std):
    return (target - mean) / std

def unnormalize(target, mean, std):
    return (target * std) + mean

def get_novelty_and_diversity(datapath, device, num_of_samples, args):
    generated_dataset = np.load(datapath, allow_pickle=True)
    original_dataset = torch.tensor(get_list_from_dict(merge_dictionary(np.load(f"data/{args.dataset}.npy", allow_pickle=True)))).to(device)
    mean = original_dataset.mean(dim=0).to(device)
    std = original_dataset.std(dim=0).to(device)
    # original_dataset = normalize(original_dataset, mean, std)

    try:
        generation_config = generated_dataset['config']
        generated_data = percentage_trajectory_cutting(generated_dataset['data'], ratio=args.ratio)
        generated_data = merge_dictionary(generated_data)
        if COMPARISON_TARGET_ORIGINAL[0] in generated_dataset['data'][0].keys():
            original_data = torch.tensor(get_original_list_from_dict(generated_data)).to(device)
        generated_data = torch.tensor(get_list_from_dict(generated_data)).to(device)
        # generated_data = normalize(generated_data, mean, std)

        if COMPARISON_TARGET_ORIGINAL[0] in generated_dataset['data'][0].keys():
            distance = torch.nn.functional.mse_loss(generated_data, original_data).detach().cpu().numpy()
        else:
            distance = 0

            print("data type: GTA")
    except:
        print("data type: SER")
        generation_config = {}
        data = {}
        for key in generated_dataset.files:
            data[key] = generated_dataset[key]
        generated_data = percentage_transition_cutting(data, ratio=args.ratio)
        generated_data = merge_dictionary(generated_data)
        generated_data = torch.tensor(get_list_from_dict(generated_data)).to(device)
        # generated_data = normalize(generated_data, mean, std)
        distance = 0

    gen_idx = np.random.choice(generated_data.shape[0], size=num_of_samples)

    
    original_idx = np.random.choice(original_dataset.shape[0], size=num_of_samples)

    dist_matrix = torch.cdist(original_dataset[original_idx], generated_data[gen_idx])
    novelty = dist_matrix.min(axis=0).values.mean().detach().cpu().numpy()
    diversity = torch.pdist(generated_data[gen_idx]).mean().detach().cpu().numpy()
    if True:
        print(torch.pdist(original_dataset[original_idx]).mean().detach().cpu().numpy())
        print(torch.pdist(generated_data[gen_idx]).mean().detach().cpu().numpy())
    return novelty, diversity, distance, generation_config

def main():
    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='halfcheetah-medium-v2')
    parser.add_argument('--datapath', type=str, default="data/div_nov_targets/5M-1_0x-smallmixer-10-sar-temp0_0.npz")
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--num_of_samples', type=int, default=100_000)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--ratio', type=float, default=1.0)
    args = parser.parse_args()

    if args.seed is None:
        seed = np.random.choice(10000000, 1)
    else:
        seed = args.seed
    np.random.seed(seed)
    
    num_of_samples = args.num_of_samples
    device = args.device

    # file_list = []
    # for (root, directories, files) in os.walk(args.data_path):
    #     for f in files:
    #         if '.npz' in f:
    #             file_list.append(f)

    novelty, diversity, distance, generation_config = get_novelty_and_diversity(args.datapath, device, num_of_samples, args)

    # noise_level = generation_config.item()['SimpleDiffusionGenerator']['noise_level']
    # temperature = generation_config.item()['SimpleDiffusionGenerator']['temperature']
    # guidance_rewardscale = generation_config.item()['SimpleDiffusionGenerator']['guidance_rewardscale']
    experiment_name = '-'.join([args.datapath.split('/')[-1]])

    novelty_and_diversity = {
        'novelty': novelty,
        'diversity': diversity,
        'distance': distance,
        'ratio': args.ratio,
        'samples': num_of_samples,
        'seed': seed,
        'elapsed_time': round(time.time() - start, 3)
    }

    if generation_config != {}:
        novelty_and_diversity['guidance_rewardscale'] = generation_config.item()['SimpleDiffusionGenerator']['guidance_rewardscale']
        novelty_and_diversity['noise_level'] = generation_config.item()['SimpleDiffusionGenerator']['noise_level']

    df = pd.DataFrame(novelty_and_diversity, index=[experiment_name])

    try:
        data = pd.read_csv(f"./statistics/novelty_diversity_and_distance.csv", index_col=0)
        df = pd.concat([data, df])
    except:
        print("create new csv file")    
    df.to_csv("./statistics/novelty_diversity_and_distance.csv")
    return novelty, diversity

if __name__=='__main__':
    main()