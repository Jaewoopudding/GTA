import numpy as np
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Any, DefaultDict
import random
from collections import defaultdict
from tqdm.auto import tqdm, trange  # noqa
import argparse
import pickle


def merge_dictionary(list_of_Dict: List[Dict]) -> Dict:
    merged_data = {}

    for d in list_of_Dict:
        for k, v in d.items():
            if k not in merged_data.keys():
                merged_data[k] = [v]
            else:
                merged_data[k].append(v)

    for k, v in merged_data.items():
        merged_data[k] = np.concatenate(merged_data[k])

    return merged_data

def get_saved_dataset(env: str) -> Dict:
    with open(f'./data/{env}.pkl','rb') as f:
        data = pickle.load(f) 
    return merge_dictionary(data)



class DiffusionDataset(Dataset):
    def __init__(
            self, 
            dataset: Dict, 
            seq_len: int = 10, 
            reward_scale: float = 1.0,
        ):
        self.seq_len = seq_len
        self.dataset, info = self._load_d4rl_trajectories(dataset, gamma=1.0)
        self.reward_scale = reward_scale
        print("The Number of Trajectories: ",self.dataset.__len__())


    def __getitem__(self, index):
        traj = self.dataset[index]
        traj_len = traj["observations"].shape[0]
        start_idx = random.randint(0, traj_len - self.seq_len)

        states = traj["observations"][start_idx : start_idx + self.seq_len]
        actions = traj["actions"][start_idx : start_idx + self.seq_len]
        rewards = traj["rewards"][start_idx : start_idx + self.seq_len]
        returns = traj["returns"][start_idx : start_idx + self.seq_len]
        time_steps = np.arange(start_idx, start_idx + self.seq_len)

        returns = returns * self.reward_scale

        return states, actions, rewards, returns, time_steps



    def _load_d4rl_trajectories(
        self, dataset: Dict, gamma: float = 1.0
    ) -> Tuple[List[DefaultDict[str, np.ndarray]], Dict[str, Any]]:
        traj, traj_len = [], []

        data_ = defaultdict(list)
        for i in trange(dataset["rewards"].shape[0], desc="Processing trajectories"):
            data_["observations"].append(dataset["observations"][i])
            data_["actions"].append(dataset["actions"][i])
            data_["rewards"].append(dataset["rewards"][i])

            try:
                timeout_flag = np.linalg.norm(
                        dataset["observations"][i + 1] - dataset["next_observations"][i]
                    ) > 1e-6 \
                    or dataset["terminals"][i] == 1.0
            except:
                timeout_flag = True

            if timeout_flag:
                if len(data_["observations"]) > self.seq_len:
                    episode_data = {k: np.array(v, dtype=np.float32) for k, v in data_.items()}
                    # return-to-go if gamma=1.0, just discounted returns else
                    episode_data["returns"] = self._discounted_cumsum(
                        episode_data["rewards"], gamma=gamma
                    )

                    traj.append(episode_data)
                    traj_len.append(episode_data["actions"].shape[0])
                    # reset trajectory buffer
                    data_ = defaultdict(list)
                else:
                    data_ = defaultdict(list)

        # needed for normalization, weighted sampling, other stats can be added also
        info = {
            "obs_mean": dataset["observations"].mean(0, keepdims=True),
            "obs_std": dataset["observations"].std(0, keepdims=True) + 1e-6,
            "traj_lens": np.array(traj_len),
        }
        return traj, info

    def _discounted_cumsum(self, x: np.ndarray, gamma: float) -> np.ndarray:
        cumsum = np.zeros_like(x)
        cumsum[-1] = x[-1]
        for t in reversed(range(x.shape[0] - 1)):
            cumsum[t] = x[t] + gamma * cumsum[t + 1]
        return cumsum
    
    def __len__(self):
        return len(self.dataset)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="halfcheeta-medium-v2")
    parser.add_argument("--seq_len", type=int, default=10)
    parser.add_argument("--reward_scale", type=float, default=1.0)
    parser.add_argument("--save_dir", type=str, default="./data")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    data = get_saved_dataset(args.dataset)
    dataset = DiffusionDataset(data, args.seq_len, args.reward_scale)

    trajectories = dataset.dataset



    traj = trajectories[index]



    traj_len = traj["observations"].shape[0]
    start_idx = random.randint(0, traj_len - args.seq_len)

    states = traj["observations"][start_idx : start_idx + args.seq_len]
    actions = traj["actions"][start_idx : start_idx + args.seq_len]
    rewards = traj["rewards"][start_idx : start_idx + args.seq_len]
    returns = traj["returns"][start_idx : start_idx + args.seq_len]
    time_steps = np.arange(start_idx, start_idx + args.seq_len)

    returns = returns * args.reward_scale



    
