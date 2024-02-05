# Shared functions for the CORL algorithms.
from typing import Dict, List, Optional, Tuple, DefaultDict, Any
import random

import numpy as np
import torch
from tqdm.auto import tqdm # noqa
from torch.utils.data import Dataset
import torch

TensorBatch = List[torch.Tensor]


def return_reward_range(dataset, max_episode_steps):
    returns, lengths = [], []
    ep_ret, ep_len = 0.0, 0
    for r, d in zip(dataset["rewards"], dataset["terminals"]):
        ep_ret += float(r)
        ep_len += 1
        if d or ep_len == max_episode_steps:
            returns.append(ep_ret)
            lengths.append(ep_len)
            ep_ret, ep_len = 0.0, 0
    lengths.append(ep_len)  # but still keep track of number of steps
    assert sum(lengths) == len(dataset["rewards"])
    return min(returns), max(returns)


class RewardNormalizer:
    def __init__(self, dataset, env_name, max_episode_steps=1000):
        self.env_name = env_name
        self.scale = 1.
        self.shift = 0.
        if any(s in env_name for s in ("halfcheetah", "hopper", "walker2d")):
            min_ret, max_ret = return_reward_range(dataset, max_episode_steps)
            self.scale = max_episode_steps / (max_ret - min_ret)
        elif "antmaze" in env_name:
            self.shift = -1.

    def __call__(self, reward):
        return (reward + self.shift) * self.scale


class StateNormalizer:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def to_torch(self, device: str):
        self.mean = torch.tensor(self.mean, device=device)
        self.std = torch.tensor(self.std, device=device)

    def __call__(self, state):
        return (state - self.mean) / self.std
    
    def unnormalize(self, state):
        return (state * self.std) + self.mean


class ReplayBufferBase:
    def __init__(
            self,
            device: str = "cpu",
            reward_normalizer: Optional[RewardNormalizer] = None,
            state_normalizer: Optional[StateNormalizer] = None,
    ):
        self.reward_normalizer = reward_normalizer
        self.state_normalizer = state_normalizer
        if self.state_normalizer is not None:
            self.state_normalizer.to_torch(device)
        self._device = device

    # Un-normalized samples.
    def _sample(self, batch_size: int, **kwargs) -> TensorBatch:
        raise NotImplementedError

    def sample(self, batch_size: int, **kwargs) -> TensorBatch:
        states, actions, rewards, next_states, dones = self._sample(batch_size, **kwargs)
        if self.reward_normalizer is not None:
            rewards = self.reward_normalizer(rewards)
        if self.state_normalizer is not None:
            states = self.state_normalizer(states)
            next_states = self.state_normalizer(next_states)
        return [states, actions, rewards, next_states, dones]

class ReplayBuffer(ReplayBufferBase):
    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            buffer_size: int,
            device: str = "cpu",
            reward_normalizer: Optional[RewardNormalizer] = None,
            state_normalizer: Optional[StateNormalizer] = None,
    ):
        super().__init__(
            device, reward_normalizer, state_normalizer
        )
        self._buffer_size = buffer_size
        self._pointer = 0 # we currently use self._pointer as self._size
        self._ptr = 0 # pointer for MOPO fake_sampler

        self._states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._actions = torch.zeros(
            (buffer_size, action_dim), dtype=torch.float32, device=device
        )
        self._rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._next_states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)

    @property
    def empty(self):
        return self._pointer == 0

    @property
    def full(self):
        return self._pointer == self._buffer_size

    def __len__(self):
        return self._pointer

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    # Loads data in d4rl format, i.e. from Dict[str, np.array].
    def load_d4rl_dataset(self, data: Dict[str, np.ndarray]):
        if not self.empty:
            raise ValueError("Trying to load data into non-empty replay buffer")
        n_transitions = data["observations"].shape[0]
        if n_transitions > self._buffer_size:
            raise ValueError(
                "Replay buffer is smaller than the dataset you are trying to load!"
            )
        self._states[:n_transitions] = self._to_tensor(data["observations"])
        self._actions[:n_transitions] = self._to_tensor(data["actions"])
        if data["rewards"].ndim == 1:
            self._rewards[:n_transitions] = self._to_tensor(data["rewards"][..., None])
        else:
            self._rewards[:n_transitions] = self._to_tensor(data["rewards"])
        self._next_states[:n_transitions] = self._to_tensor(data["next_observations"])
        if 'terminals' in data.keys():
            self._dones[:n_transitions] = self._to_tensor(data["terminals"][..., None])
        else:
            self._dones[:n_transitions] = torch.zeros_like(self._rewards[:n_transitions]).to(torch.bool)
        self._pointer = n_transitions

        print(f"Dataset size: {n_transitions}")

    def _sample(self, batch_size: int, **kwargs) -> TensorBatch:
        indices = np.random.randint(0, self._pointer, size=batch_size)
        states = self._states[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        next_states = self._next_states[indices]
        dones = self._dones[indices]
        return [states, actions, rewards, next_states, dones]

    def add_transition_batch(self, batch: TensorBatch):
        states, actions, rewards, next_states, dones = batch
        batch_size = states.shape[0]

        # If the buffer is full, do nothing.
        if self.full:
            return
        if self._pointer + batch_size > self._buffer_size:
            # Trim the samples to fit the buffer size.
            states = states[: self._buffer_size - self._pointer]
            actions = actions[: self._buffer_size - self._pointer]
            rewards = rewards[: self._buffer_size - self._pointer]
            next_states = next_states[: self._buffer_size - self._pointer]
            dones = dones[: self._buffer_size - self._pointer]
            batch_size = states.shape[0]

        self._states[self._pointer: self._pointer + batch_size] = states
        self._actions[self._pointer: self._pointer + batch_size] = actions
        self._rewards[self._pointer: self._pointer + batch_size] = rewards
        self._next_states[self._pointer: self._pointer + batch_size] = next_states
        self._dones[self._pointer: self._pointer + batch_size] = dones
        self._pointer += batch_size
        
    def add_batch(self, obss: np.ndarray, next_obss: np.ndarray, actions: np.ndarray, rewards: np.ndarray, terminals: np.ndarray) -> None:
        batch_size = len(obss)
        indexes = np.arange(self._ptr, self._ptr + batch_size) % self._buffer_size

        self._states[indexes] = self._to_tensor(obss.copy())
        self._next_states[indexes] = self._to_tensor(next_obss.copy())
        self._actions[indexes] = self._to_tensor(actions.copy())
        self._rewards[indexes] = self._to_tensor(rewards.copy())
        self._dones[indexes] = self._to_tensor(terminals.copy())

        self._ptr = (self._ptr + batch_size) % self._buffer_size
        self._pointer = min(self._pointer + batch_size, self._buffer_size)


class DiffusionDataset(Dataset):
    def __init__(
            self, 
            dataset: Dict, 
            dataset_nm : str,
            seq_len: int = 10, 
            discounted_return: bool = False,
            gamma: float = 0.99,
            restore_rewards: Optional[bool] = False,
            penalty: Optional[int] = 100,
        ):
        self.seq_len = seq_len
        self.discounted_return = discounted_return
        self.dataset_nm = dataset_nm
        self.restore_rewards = restore_rewards
        penalty=100 if penalty is None else penalty
        self.penalty = penalty
        self.dataset, info = self._load_d4rl_trajectories(dataset, gamma=gamma)
        self.trajectory_rewards = info["trajectory_rewards"]
        self.episode_rewards = info["episode_rewards"]
        self.trajectory_returns = info["trajectory_returns"]
        print("The Number of Trajectories: ",self.dataset.__len__())


    def __getitem__(self, index):
        traj = self.dataset[index]
        traj_len = traj["observations"].shape[0]
        start_idx = random.randint(0, traj_len - self.seq_len)

        states = traj["observations"][start_idx : start_idx + self.seq_len]
        actions = traj["actions"][start_idx : start_idx + self.seq_len]
        rewards = traj["rewards"][start_idx : start_idx + self.seq_len]
        returns = traj["returns"][start_idx : start_idx + self.seq_len]
        terminals = traj["terminals"][start_idx : start_idx + self.seq_len]
        time_steps = np.arange(start_idx, start_idx + self.seq_len)

        next_states = traj["next_observations"][start_idx : start_idx + self.seq_len]
        

        rtg = traj["RTG"][start_idx : start_idx + self.seq_len]
        

        return states, actions, rewards, next_states, returns, time_steps, terminals, rtg


    def _load_d4rl_trajectories(
        self, dataset: np.array, gamma: float = 1.0
    ) -> Tuple[List[DefaultDict[str, np.ndarray]], Dict[str, Any]]:
        traj, traj_len = [], []
        episode_rewards = []
        for episode_data in tqdm(dataset) :
            if len(episode_data["observations"]) < self.seq_len:
                continue
            
            episode_data["RTG"] = self._discounted_cumsum(
                episode_data["rewards"], gamma=1.0
            )

            if self.discounted_return and ("maze" not in self.dataset_nm):
                if episode_data["terminals"].any():
                    assert (episode_data["terminals"][-1]) and (not episode_data["terminals"][:-1].any())
                    episode_data["rewards"][-1] -= self.penalty

            episode_data["returns"] = self._discounted_cumsum(
                episode_data["rewards"], gamma=gamma
            )
            if episode_data["returns"].ndim == 1:
                episode_data["returns"] = episode_data["returns"][..., None]
                episode_data["rewards"] = episode_data["rewards"][..., None]
            
            if self.discounted_return and ("maze" not in self.dataset_nm) and (self.restore_rewards):
                if episode_data["terminals"].any():
                    assert (episode_data["terminals"][-1]) and (not episode_data["terminals"][:-1].any())
                    episode_data["rewards"][-1] += self.penalty
            
            traj.append(episode_data)
            traj_len.append(episode_data["actions"].shape[0])
            episode_rewards.append(episode_data["returns"][0])

        trajectory_rewards = []
        trajectory_returns = []
        for tr, tlen in zip(traj, traj_len):
            for traj_idx in range(tlen - self.seq_len + 1):
                start_idx = traj_idx
                if self.discounted_return and ("maze" not in self.dataset_nm):
                    if tr["terminals"][start_idx : start_idx + self.seq_len].any():
                        trajectory_reward = tr["rewards"][start_idx : start_idx + self.seq_len].sum() - self.penalty
                    else:
                        trajectory_reward = tr["rewards"][start_idx : start_idx + self.seq_len].sum()
                else: 
                    trajectory_reward = tr["rewards"][start_idx : start_idx + self.seq_len].sum()   
                trajectory_rewards.append(trajectory_reward)
                trajectory_returns.append(tr["returns"][start_idx])


        info = {
            "trajectory_rewards" : np.array(trajectory_rewards),
            "episode_rewards" : np.array(episode_rewards).squeeze(),
            "trajectory_returns" : np.array(trajectory_returns)
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


class DiffusionTrajectoryDataset(DiffusionDataset):
    def __init__(
            self, 
            dataset: Dict, 
            dataset_nm : str,
            seq_len: int = 10, 
            discounted_return: bool = False,
            gamma : float = 0.99,
            restore_rewards: Optional[bool] = False,
            penalty: Optional[int] = 100,
        ):
        self.seq_len = seq_len
        self.discounted_return = discounted_return
        self.dataset_nm = dataset_nm
        self.restore_rewards = restore_rewards
        penalty=100 if penalty is None else penalty
        self.penalty = penalty
        self.dataset, info = self._load_d4rl_trajectories(dataset, gamma=gamma)
        self.trajectory_rewards = info["trajectory_rewards"]
        self.episode_rewards = info["episode_rewards"]
        self.trajectory_returns = info["trajectory_returns"]
        self.terminal_idx = info["terminal_idx"]
        self.terminal_rate = info["terminal_rate"]
        print("The Number of Trajectories: ",self.dataset.__len__())

    def __getitem__(self, index):
        return self.dataset[index]

    def _load_d4rl_trajectories(
        self, dataset: Dict, gamma: float = 1.0
    ) -> Tuple[List[DefaultDict[str, np.ndarray]], Dict[str, Any]]:
        traj, traj_len = [], []
        episode_rewards = []
        terminal_idx = []
        terminal_cnt = 0
        non_terminal_cnt = 0
        for episode_data in tqdm(dataset) :
            if len(episode_data["observations"]) < self.seq_len:
                continue
            
            episode_data["RTG"] = self._discounted_cumsum(
                episode_data["rewards"], gamma=1.0
            )

            non_terminal_cnt += len(episode_data["observations"])
            if episode_data["terminals"].any():
                terminal_cnt+=1
                non_terminal_cnt-=1
            if self.discounted_return and ("maze" not in self.dataset_nm):
                if episode_data["terminals"].any():
                    assert (episode_data["terminals"][-1]) and (not episode_data["terminals"][:-1].any())
                    episode_data["rewards"][-1] -= self.penalty

            episode_data["returns"] = self._discounted_cumsum(
                episode_data["rewards"], gamma=gamma
            )
            if episode_data["returns"].ndim == 1:
                episode_data["returns"] = episode_data["returns"][..., None]
                episode_data["rewards"] = episode_data["rewards"][..., None]
            
            if self.discounted_return and ("maze" not in self.dataset_nm) and (self.restore_rewards):
                if episode_data["terminals"].any():
                    assert (episode_data["terminals"][-1]) and (not episode_data["terminals"][:-1].any())
                    episode_data["rewards"][-1] += self.penalty

            traj.append(episode_data)
            traj_len.append(episode_data["actions"].shape[0])
            episode_rewards.append(episode_data["returns"][0])

        subtrajectories = []
        trajectory_rewards = []
        trajectory_returns = []
        for tr, tlen in zip(traj, traj_len):
            for traj_idx in range(tlen - self.seq_len + 1):
                start_idx = traj_idx
                states = tr["observations"][start_idx : start_idx + self.seq_len]
                actions = tr["actions"][start_idx : start_idx + self.seq_len]
                rewards = tr["rewards"][start_idx : start_idx + self.seq_len]
                returns = tr["returns"][start_idx : start_idx + self.seq_len]
                terminals = tr["terminals"][start_idx : start_idx + self.seq_len]
                time_steps = np.arange(start_idx, start_idx + self.seq_len)
                next_states = tr["next_observations"][start_idx : start_idx + self.seq_len]
                

                rtg = tr["RTG"][start_idx : start_idx + self.seq_len]

                if self.discounted_return:
                    if tr["terminals"][start_idx : start_idx + self.seq_len].any() and ("maze" not in self.dataset_nm):
                        trajectory_reward = tr["rewards"][start_idx : start_idx + self.seq_len].sum() - 100
                    else:
                        trajectory_reward = tr["rewards"][start_idx : start_idx + self.seq_len].sum()
                else: 
                    trajectory_reward = tr["rewards"][start_idx : start_idx + self.seq_len].sum() 

                if tr["terminals"][start_idx : start_idx + self.seq_len].any():
                    terminal_idx.append(1)
                else:
                    terminal_idx.append(0)  
                trajectory_rewards.append(trajectory_reward)
                trajectory_returns.append(tr["returns"][start_idx])
                
                subtrajectories.append([states, actions, rewards, next_states, returns, time_steps, terminals, rtg])

        info = {
            "trajectory_rewards" : np.array(trajectory_rewards),
            "episode_rewards" : np.array(episode_rewards).squeeze(),
            "trajectory_returns" : np.array(trajectory_returns),
            "terminal_idx" : np.array(terminal_idx),
            "terminal_rate" : terminal_cnt / (non_terminal_cnt+terminal_cnt),
        }

        return subtrajectories, info


def prepare_replay_buffer(
        state_dim: int,
        action_dim: int,
        buffer_size: int,
        dataset: Dict[str, np.ndarray],
        device: str = "cpu",
        reward_normalizer: Optional[RewardNormalizer] = None,
        state_normalizer: Optional[StateNormalizer] = None,
):
    buffer_args = {
        'reward_normalizer': reward_normalizer,
        'state_normalizer': state_normalizer,
        'device': device,
    }


    print('=============<Loading D4RL dataset.>=============')
    replay_buffer = ReplayBuffer(
        state_dim=state_dim,
        action_dim=action_dim,
        buffer_size=buffer_size,
        **buffer_args,
    )

    replay_buffer.load_d4rl_dataset(dataset)

    return replay_buffer
