# Shared functions for the CORL algorithms.
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, DefaultDict, Any
from functools import partial
from collections import defaultdict
import random

import gym
import numpy as np
import torch
from tqdm.auto import tqdm, trange  # noqa
from torch.utils.data import IterableDataset, Dataset
import torch

from src.data.norm import MinMaxNormalizer
from src.diffusion.utils import make_inputs, construct_diffusion_model, split_diffusion_transition, split_diffusion_trajectory
from corl.shared.utils import merge_dictionary
from corl.shared.s4rl import S4RLAugmentation

TensorBatch = List[torch.Tensor]


@dataclass
class DiffusionConfig:
    path: Optional[str] = None  # Path to model checkpoints or .npz file with diffusion samples
    num_steps: int = 128  # Number of diffusion steps
    sample_limit: int = -1  # If not -1, limit the number of diffusion samples to this number


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
            augmentation: Optional[S4RLAugmentation] = None
    ):
        self.reward_normalizer = reward_normalizer
        self.state_normalizer = state_normalizer
        if self.state_normalizer is not None:
            self.state_normalizer.to_torch(device)
        self.augmentation = augmentation
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
            
        if self.augmentation:
            try:
                q_function = kwargs["q_function"]
            except:
                q_function = None

            cat_func = partial(torch.cat, dim=0)
            iteration = kwargs['iteration']
            augmented_batch= []
            for _ in range(iteration):
                augmented_states, augmented_next_states = self.augmentation(states, actions, next_states, q_function=q_function)
                augmented_batch.append([augmented_states, actions, rewards, augmented_next_states, dones])
            return list(map(cat_func, list(zip(*augmented_batch))))
        else:
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
            augmentation: Optional[S4RLAugmentation] = None
    ):
        super().__init__(
            device, reward_normalizer, state_normalizer, augmentation
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
        # TODO: Make this part compatible regardless of reward dimension (or make reward dimension consistent)
        if data["rewards"].ndim == 1:
            self._rewards[:n_transitions] = self._to_tensor(data["rewards"][..., None])
        else:
            self._rewards[:n_transitions] = self._to_tensor(data["rewards"])
        self._next_states[:n_transitions] = self._to_tensor(data["next_observations"])
        if 'terminals' in data.keys():
            self._dones[:n_transitions] = self._to_tensor(data["terminals"][..., None])
            # self._dones[:n_transitions] = torch.zeros_like(self._rewards[:n_transitions]).to(torch.bool)
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
    
    # For MOPO
    def sample_all(self) -> Dict[str, np.ndarray]:
        return {
            "observations": self._states[:self._pointer].cpu().numpy(),
            "actions": self._actions[:self._pointer].cpu().numpy(),
            "next_observations": self._next_states[:self._pointer].cpu().numpy(),
            "terminals": self._dones[:self._pointer].cpu().numpy(),
            "rewards": self._rewards[:self._pointer].cpu().numpy(),
        }
        
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


class DiffusionGenerator(ReplayBufferBase):
    def __init__(
            self,
            env_name: str,
            diffusion_path: str,
            use_ema: bool = True,
            num_steps: int = 32,
            batch_parallelism: int = 100,  # How many batches to generate each diffusion sample.
            device: str = "cpu",
            max_samples: int = -1,
            reward_normalizer: Optional[RewardNormalizer] = None,
            state_normalizer: Optional[StateNormalizer] = None,
    ):
        super().__init__(
            device, reward_normalizer, state_normalizer,
        )
        # Create the environment
        self.env = gym.make(env_name)
        inputs = make_inputs(self.env)
        inputs = torch.from_numpy(inputs).float()
        self.diffusion = construct_diffusion_model(inputs=inputs).to(device)

        data = torch.load(diffusion_path)
        if use_ema:
            ema_dict = data['ema']
            ema_dict = {k: v for k, v in ema_dict.items() if k.startswith('ema_model')}
            ema_dict = {k.replace('ema_model.', ''): v for k, v in ema_dict.items()}
            self.diffusion.load_state_dict(ema_dict)
        else:
            self.diffusion.load_state_dict(data['model'])
        self.diffusion.eval()
        # Clamp samples if normalizer is MinMaxNormalizer
        self.clamp_samples = isinstance(self.diffusion.normalizer, MinMaxNormalizer)
        self.num_steps = num_steps

        # Batching of diffusion samples
        self.batch_parallelism = batch_parallelism
        self.cache = []
        self.cache_pointer = 0

        # If max samples is not -1, then we will limit to that many unique samples.
        if max_samples != -1:
            print(f"Limiting to {max_samples} samples.")
            self.replay_buffer = ReplayBuffer(
                state_dim=self.env.observation_space.shape[0],
                action_dim=self.env.action_space.shape[0],
                buffer_size=max_samples,
                device=device,
                reward_normalizer=reward_normalizer,
                state_normalizer=state_normalizer,
            )
        else:
            self.replay_buffer = None

    def _sample_from_diffusion(self, batch_size: int, **kwargs) -> TensorBatch:
        sampled_outputs = self.diffusion.sample(
            batch_size=batch_size,
            num_sample_steps=self.num_steps,
            clamp=self.clamp_samples,
            **kwargs,
        )
        x =  split_diffusion_transition(sampled_outputs, self.env)

        # Use the ground-truth done function if the diffusion model doesn't model it.
        if len(x) == 4:
            observations, actions, rewards, next_observations = x
            terminals = torch.zeros_like(next_observations[..., 0]).float()
        else:
            observations, actions, rewards, next_observations, terminals = x

        if self.replay_buffer is not None:
            self.replay_buffer.add_transition_batch(
                [observations, actions, rewards[..., None], next_observations, terminals[..., None]])
            print(f'Samples collected: {self.replay_buffer._pointer}.')
        return [observations, actions, rewards, next_observations, terminals]

    def _sample(self, batch_size: int, **kwargs) -> TensorBatch:
        # If max samples reached, sample from replay buffer.
        if self.replay_buffer is not None and self.replay_buffer.full:
            return self.replay_buffer._sample(batch_size)

        # Otherwise, sample from diffusion.
        if self.batch_parallelism == 1:
            return self._sample_from_diffusion(batch_size, **kwargs)
        else:
            diffusion_sample_size = batch_size * self.batch_parallelism
            if len(self.cache) == 0 or self.cache_pointer == diffusion_sample_size:
                self.cache = self._sample_from_diffusion(diffusion_sample_size, **kwargs)
                self.cache_pointer = 0
            batch = [x[self.cache_pointer: self.cache_pointer + batch_size] for x in self.cache]
            self.cache_pointer += batch_size
            return batch


class SequenceDataset(Dataset):
    def __init__(
            self, 
            dataset: Dict, 
            seq_len: int = 10, 
            reward_scale: float = 1.0,
            augmentation: str = None,
            guidance_target_multiple: float = 1.,
            **kwargs
        ):
        self.reward_scale = reward_scale
        self.seq_len = seq_len
        self.guidance_target_multiple = guidance_target_multiple
        self.dataset, info = self._load_d4rl_trajectories(dataset, gamma=1.0)
        if augmentation:
            self.augmentation = S4RLAugmentation(augmentation, **kwargs)
            self.augmentation.trajectory_flag()
        else:
            self.augmentation = None
    
        self.state_mean = info["obs_mean"]
        self.state_std = info["obs_std"]
        # https://github.com/kzl/decision-transformer/blob/e2d82e68f330c00f763507b3b01d774740bee53f/gym/experiment.py#L116 # noqa
        self.sample_prob = info["traj_lens"] / info["traj_lens"].sum()
        print("The Number of Trajectories: ",self.dataset.__len__())


    def __getitem__(self, index): 
        traj = self.dataset[index]
        ## TODO ##
        start_idx = random.randint(0, traj["rewards"].shape[0] - 1) # SEQ LEN PADDING
        # https://github.com/kzl/decision-transformer/blob/e2d82e68f330c00f763507b3b01d774740bee53f/gym/experiment.py#L128 # noqa
        states = traj["observations"] #[start_idx : start_idx + self.seq_len]
        actions = traj["actions"] #[start_idx : start_idx + self.seq_len]
        returns = traj["returns"] #[start_idx : start_idx + self.seq_len]
        if self._gen_flag:
            if 'timesteps' in traj.keys():
                time_steps = traj["timesteps"]#[start_idx : start_idx + self.seq_len]
            else:
                time_steps = np.arange(start_idx, start_idx + self.seq_len)
        else:
            time_steps = np.arange(start_idx, start_idx + self.seq_len)
        if self.augmentation:
            states = self.augmentation(states)
        states = (states - self.state_mean) / self.state_std
        
        returns = returns * self.reward_scale
        # pad up to seq_len if needed
        mask = np.hstack(
            [np.ones(states.shape[0]), np.zeros(self.seq_len - states.shape[0])]
        )

        ## BUG OCCURED HERE 12.13 ##
        if states.shape[0] < self.seq_len:
            states = self._pad_along_axis(states, pad_to=self.seq_len)
            actions = self._pad_along_axis(actions, pad_to=self.seq_len)
            returns = self._pad_along_axis(returns, pad_to=self.seq_len)
            time_steps = self._pad_along_axis(time_steps, pad_to=self.seq_len)
        return self._to_tensor(states), self._to_tensor(actions) , self._to_tensor(returns) , self._to_tensor(time_steps) , self._to_tensor(mask)
    
    def _to_tensor(self, array):
        return torch.tensor(array, dtype=torch.float32)

    def _attach_rtg(self, reward_array, gamma):
        return [self._discounted_cumsum(reward_array, gamma=gamma)]
        
    def _attach_generated_rtg(self, reward_array, rtg_init, gamma):
        temp = np.zeros_like(reward_array)
        temp[1:] = reward_array[:-1]
        rewards_cumsum = np.cumsum(temp)
        return [np.ones_like(reward_array) * rtg_init * self.guidance_target_multiple - rewards_cumsum]

    def _load_d4rl_trajectories(
        self, dataset: Dict, gamma: float = 1.0
    ) -> Tuple[List[DefaultDict[str, np.ndarray]], Dict[str, Any]]:
        traj, traj_len = [], []
        subtrajectories = []
        data_ = defaultdict(list)
        if "returns" in dataset[0].keys(): # 생성된 데이터에만 returns 키가 존재한다. 반드시 데이터를 합칠 때, gen data가 먼저 와야 한다. 
            self._gen_flag = True
        else:
            self._gen_flag = False

        for i in trange(dataset.__len__(), desc='Processing Trajectories'):
            data_["observations"].append(dataset[i]["observations"])
            data_["actions"].append(dataset[i]["actions"])
            data_["rewards"].append(dataset[i]["rewards"])
            if self._gen_flag:
                if 'returns' in dataset[i].keys():
                    data_['timesteps'].append(dataset[i]['timesteps'])
                    data_['returns'] = self._attach_generated_rtg(dataset[i]["rewards"], dataset[i]['RTG'][0] ,gamma)
                else:
                    data_['returns'] = self._attach_rtg(dataset[i]["rewards"], gamma)

            if not self._gen_flag:
                data_['returns'] = self._attach_rtg(dataset[i]["rewards"], gamma)

            episode_data = {k: np.array(v[0], dtype=np.float32) for k, v in data_.items()}
            traj.append(episode_data)
            traj_len.append(episode_data["actions"].shape[0])

            # reset trajectory buffer
            data_ = defaultdict(list)


            tr = episode_data
            tlen = episode_data["actions"].shape[0]
            for traj_idx in range(tlen - self.seq_len + 1):
                subtrajectory = {}
                start_idx = traj_idx
                subtrajectory['observations'] = tr["observations"][start_idx : start_idx + self.seq_len]
                subtrajectory['actions'] = tr["actions"][start_idx : start_idx + self.seq_len]
                subtrajectory['rewards'] = tr["rewards"][start_idx : start_idx + self.seq_len]
                subtrajectory['returns'] = tr["returns"][start_idx : start_idx + self.seq_len]

                if 'timesteps' in tr.keys():
                    subtrajectory['timesteps'] = tr['timesteps'][start_idx : start_idx + self.seq_len]

                subtrajectories.append(subtrajectory)

    
        merged_dataset= merge_dictionary(dataset)
        info = {
            "obs_mean": merged_dataset["observations"].mean(0, keepdims=True),
            "obs_std": merged_dataset["observations"].std(0, keepdims=True) + 1e-6,
            "traj_lens": np.array(traj_len),
        }
        return subtrajectories, info

    def _pad_along_axis(
        self, arr: np.ndarray, pad_to: int, axis: int = 0, fill_value: float = 0.0
    ) -> np.ndarray:
        pad_size = pad_to - arr.shape[axis]
        if pad_size <= 0:
            return arr

        npad = [(0, 0)] * arr.ndim
        npad[axis] = (0, pad_size)
        return np.pad(arr, pad_width=npad, mode="constant", constant_values=fill_value)


    def _discounted_cumsum(self, x: np.ndarray, gamma: float) -> np.ndarray:
        cumsum = np.zeros_like(x)
        cumsum[-1] = x[-1]
        for t in reversed(range(x.shape[0] - 1)):
            cumsum[t] = x[t] + gamma * cumsum[t + 1]
        return cumsum
    
    def __len__(self):
        return len(self.dataset)


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





class DiffusionIterableDataset(SequenceDataset):
    def __init__(
            self, 
            dataset: Dict, 
            seq_len: int = 10, 
            reward_scale: float = 1.0,
        ):
        self.seq_len = seq_len
        self.dataset, info = self._load_d4rl_trajectories(dataset, gamma=1.0)
        self.reward_scale = reward_scale
        self.sample_prob = info["traj_lens"] / info["traj_lens"].sum()
        print("The Number of Trajectories: ",self.dataset.__len__())


    def __prepare_sample(self, traj_idx, start_idx):
        traj = self.dataset[traj_idx]
        # https://github.com/kzl/decision-transformer/blob/e2d82e68f330c00f763507b3b01d774740bee53f/gym/experiment.py#L128 # noqa
        states = traj["observations"][start_idx : start_idx + self.seq_len]
        actions = traj["actions"][start_idx : start_idx + self.seq_len]
        rewards = traj["rewards"][start_idx : start_idx + self.seq_len]
        returns = traj["returns"][start_idx : start_idx + self.seq_len]
        time_steps = np.arange(start_idx, start_idx + self.seq_len)
        
        returns = returns * self.reward_scale

        return states, actions, rewards, returns, time_steps

    def __iter__(self):
        while True:
            traj_idx = np.random.choice(len(self.dataset), p=self.sample_prob)
            start_idx = random.randint(0, self.dataset[traj_idx]["rewards"].shape[0] - self.seq_len)
            yield self.__prepare_sample(traj_idx, start_idx)


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




def prepare_replay_buffer(
        state_dim: int,
        action_dim: int,
        buffer_size: int,
        dataset: Dict[str, np.ndarray],
        env_name: str,
        diffusion_config: DiffusionConfig,
        device: str = "cpu",
        s4rl_augmentation_type: str = None,
        std_scale=1e-2, 
        uniform_scale=1e-2, 
        adv_scale=1e-2, 
        reward_normalizer: Optional[RewardNormalizer] = None,
        state_normalizer: Optional[StateNormalizer] = None,
):
    buffer_args = {
        'reward_normalizer': reward_normalizer,
        'state_normalizer': state_normalizer,
        'device': device,
    }

    augmentation = S4RLAugmentation(
            type=s4rl_augmentation_type,
            std_scale = std_scale,
            uniform_scale = uniform_scale,
            adv_scale = adv_scale
    )

    if diffusion_config.path is None:
        print('=============<Loading standard D4RL dataset.>=============')
        replay_buffer = ReplayBuffer(
            state_dim=state_dim,
            action_dim=action_dim,
            buffer_size=buffer_size,
            augmentation=augmentation,
            **buffer_args,
        )
        replay_buffer.load_d4rl_dataset(dataset)
    elif diffusion_config.path.endswith(".npz"):
        print('=============<Loading diffusion dataset.>=============')
        diffusion_dataset = np.load(diffusion_config.path)
        diffusion_dataset = {key: diffusion_dataset[key] for key in diffusion_dataset.files}

        if diffusion_config.sample_limit != -1:
            # Limit the number of samples
            for key in diffusion_dataset.keys():
                diffusion_dataset[key] = diffusion_dataset[key][:diffusion_config.sample_limit]
            print('Limited diffusion dataset to {} samples'.format(diffusion_config.sample_limit))

        replay_buffer = ReplayBuffer(
            state_dim=state_dim,
            action_dim=action_dim,
            buffer_size=diffusion_dataset['rewards'].shape[0],
            **buffer_args,
        )
        replay_buffer.load_d4rl_dataset(diffusion_dataset)
    elif diffusion_config.path.endswith(".pt"):
        print('Loading diffusion model.')
        # Load gin config from the same directory.
        gin_path = os.path.join(os.path.dirname(diffusion_config.path), 'config.gin')
        gin.parse_config_file(gin_path, skip_unknown=True)

        replay_buffer = DiffusionGenerator(
            env_name=env_name,
            diffusion_path=diffusion_config.path,
            use_ema=True,
            num_steps=diffusion_config.num_steps,
            max_samples=diffusion_config.sample_limit,
            **buffer_args,
        )
    else:
        raise ValueError("Unknown diffusion_path format")

    return replay_buffer
