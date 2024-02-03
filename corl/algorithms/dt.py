import os
import random
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, DefaultDict, Dict, List, Optional, Tuple, Union

import d4rl  # noqa
import gym
import numpy as np
import pyrallis
import torch
import torch.nn as nn
import wandb
from torch.nn import functional as F
from torch.utils.data import DataLoader, IterableDataset, Dataset
from tqdm.auto import tqdm, trange  # noqa
from torch.utils.data import WeightedRandomSampler
import pyrootutils

path = pyrootutils.find_root(search_from = __file__, indicator=".aug-project-root")
pyrootutils.set_root(path = path,
                     project_root_env_var = True,
                     dotenv = True,
                     pythonpath = True)

from corl.shared.utils  import wandb_init, get_trajectory_dataset, merge_dictionary

@dataclass
class TrainConfig:
    # wandb params
    device: str = "cuda:2"
    s4rl_augmentation_type: str = 'identical'
    std_scale: float = 0.0003
    uniform_scale: float = 0.0003
    adv_scale: float = 0.0001
    iteration: int = 2
    env: str = "maze2d-medium-v1"  # OpenAI gym environment name
    seed: int = 0  # Sets Gym, PyTorch and Numpy seeds
    eval_seed: int = 42
    GDA: str = 'GTA'  # "gda only" 'gda with original' None
    step: int = 100000 # Generated Data Augmentation 모델 학습 step 수
    data_mixture_type: str = 'mixed'
    GDA_id: str = None

    # wandb params
    project: str = 'DT'
    group: str = "DT-D4RL"
    name: str = "DT"
    diffusion_horizon: int = 31
    diffusion_backbone: str = 'mixer' # 'mixer', 'temporal'

    conditioned: bool = False
    data_volume: int = 5e6
    generation_type: str = 's' # 's,a' 's,a,r'
    guidance_temperature: float = 1.2
    guidance_target_multiple: float = 1.1

    # model params
    embedding_dim: int = 128
    num_layers: int = 3
    num_heads: int = 1
    seq_len: int = 20
    episode_len: int = 1000
    attention_dropout: float = 0.1
    residual_dropout: float = 0.1
    embedding_dropout: float = 0.1
    max_action: float = 1.0
    # training params
    learning_rate: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 1e-4
    clip_grad: Optional[float] = 0.25
    batch_size: int = 1024
    max_timesteps: int = 100_000
    warmup_steps: int = 10_000
    reward_scale: float = 0.001
    num_workers: int = 4
    # evaluation params
    target_returns: Tuple[float, ...] = (12000.0, 6000.0)
    eval_episodes: int = 100
    eval_every: int = 10_000
    # general params
    checkpoints_path: Optional[str] = 'checkpoints'
    deterministic_torch: bool = False
    datapath: str = None

    def __post_init__(self):
        self.name = f"{self.name}-{self.env}-{self.s4rl_augmentation_type}-{str(uuid.uuid4())[:4]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)
        if self.s4rl_augmentation_type == 'identical':
            self.iteration = 1
        if self.GDA is None:
            self.diffusion_horizon = None
            self.diffusion_backbone = None
            self.conditioned = None
            self.data_volume = None
            self.generation_type = None
            self.guidance_temperature = None
            self.guidance_target_multiple = None
        if (self.datapath is not None) and (self.datapath != 'None'):
            self.GDA = os.path.splitext(os.path.basename(self.datapath))[0]
        if self.GDA_id is not None:
            self.GDA = self.GDA + f'_{self.GDA_id}'
        if self.data_mixture_type is not None:
            self.GDA = self.GDA + f'_{self.data_mixture_type}'
        if self.env.split('-')[0] == 'halfcheetah':
            self.target_returns: Tuple[float, ...] = (12000.0,)
        elif self.env.split('-')[0] == 'walker2d':
            self.target_returns: Tuple[float, ...] = (5000.0,)
        elif self.env.split('-')[0] == 'hopper':
            self.target_returns: Tuple[float, ...] = (3600.0,)
        elif self.env.split('-')[0] == "maze2d":
            self.target_returns: Tuple[float, ...] = (280.0, )
        else:
            raise ValueError(f"env: {self.env}")


# general utils
def set_seed(
    seed: int, env: Optional[gym.Env] = None, deterministic_torch: bool = False
):
    if env is not None:
        env.seed(seed)
        env.action_space.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)


def wrap_env(
    env: gym.Env,
    state_mean: Union[np.ndarray, float] = 0.0,
    state_std: Union[np.ndarray, float] = 1.0,
    reward_scale: float = 1.0,
) -> gym.Env:
    def normalize_state(state):
        return (state - state_mean) / state_std

    def scale_reward(reward):
        return reward_scale * reward

    env = gym.wrappers.TransformObservation(env, normalize_state)
    if reward_scale != 1.0:
        env = gym.wrappers.TransformReward(env, scale_reward)
    return env


# some utils functionalities specific for Decision Transformer
def pad_along_axis(
    arr: np.ndarray, pad_to: int, axis: int = 0, fill_value: float = 0.0
) -> np.ndarray:
    pad_size = pad_to - arr.shape[axis]
    if pad_size <= 0:
        return arr

    npad = [(0, 0)] * arr.ndim
    npad[axis] = (0, pad_size)
    return np.pad(arr, pad_width=npad, mode="constant", constant_values=fill_value)


def discounted_cumsum(x: np.ndarray, gamma: float) -> np.ndarray:
    cumsum = np.zeros_like(x)
    cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        cumsum[t] = x[t] + gamma * cumsum[t + 1]
    return cumsum

def attach_generated_rtg(reward_array, rtg_init, guidance_target_multiple, env_type: str = 'locomotion'):
    temp = np.zeros_like(reward_array)
    temp[1:] = reward_array[:-1]
    rewards_cumsum = np.cumsum(temp)
    multiplier = guidance_target_multiple if env_type == 'locomotion' else 1
    return np.ones_like(reward_array) * rtg_init * multiplier - rewards_cumsum



def load_d4rl_trajectories(
    env: Dict, gamma: float = 1.0, augmentation: float = None
) -> Tuple[List[DefaultDict[str, np.ndarray]], Dict[str, Any]]:
    dataset, metadata = get_trajectory_dataset(env)
    traj, traj_len = [], []
    gen_count = 0
    original_count = 0

    if 'maze' in env.env:
        env_type = 'maze'
    else:
        env_type = 'locomotion'

    for subtrajectory in dataset:
        if 'RTG' in subtrajectory.keys():
            traj.append({
                'observations': subtrajectory['observations'],
                'actions': subtrajectory['actions'],
                'rewards': subtrajectory['rewards'],
                'returns': attach_generated_rtg(subtrajectory['rewards'], subtrajectory['RTG'][0], metadata['guidance_target_multiple'], env_type),
                'timesteps': subtrajectory['timesteps']
            })
            traj_len.append(subtrajectory['actions'].shape[0])
            gen_count += 1
        else:
            traj.append({
                'observations': subtrajectory['observations'],
                'actions': subtrajectory['actions'],
                'rewards': subtrajectory['rewards'],
                'returns': discounted_cumsum(subtrajectory['rewards'], gamma=gamma)
            })
            traj_len.append(subtrajectory['actions'].shape[0])
            original_count += 1

    s4rl_string='No S4RL augmentation' if augmentation is None else f"S4RL augmentation scale: {augmentation}"

    print(f"================<Dataset information>================\nEnvironment: {env.env}\nGenerated trajectories: {gen_count}, original trajectories: {original_count}\n{s4rl_string}\n=====================================================")

    # data_ = defaultdict(list)
    # for i in trange(dataset["rewards"].shape[0], desc="Processing trajectories"):
    #     data_["observations"].append(dataset["observations"][i])
    #     data_["actions"].append(dataset["actions"][i])
    #     data_["rewards"].append(dataset["rewards"][i])

    #     if dataset["terminals"][i] or dataset["timeouts"][i]:
    #         episode_data = {k: np.array(v, dtype=np.float32) for k, v in data_.items()}
    #         # return-to-go if gamma=1.0, just discounted returns else
    #         episode_data["returns"] = discounted_cumsum(
    #             episode_data["rewards"], gamma=gamma
    #         )
    #         traj.append(episode_data)
    #         traj_len.append(episode_data["actions"].shape[0])
    #         # reset trajectory buffer
    #         data_ = defaultdict(list)

    # needed for normalization, weighted sampling, other stats can be added also
            
    merged_dataset = merge_dictionary(dataset)
    info = {
        "obs_mean": merged_dataset["observations"].mean(0, keepdims=True),
        "obs_std": merged_dataset["observations"].std(0, keepdims=True) + 1e-6,
        "traj_lens": np.array(traj_len),
    }
    return traj, info


class SequenceDataset(Dataset):
    def __init__(self, config: Dict, seq_len: int = 10, reward_scale: float = 1.0, augmentation_scale: float = None):
        self.dataset, info = load_d4rl_trajectories(config, gamma=1.0, augmentation=augmentation_scale)
        self.reward_scale = reward_scale
        self.seq_len = seq_len
        self.augmentation_scale = augmentation_scale

        self.state_mean = info["obs_mean"]
        self.state_std = info["obs_std"]
        # https://github.com/kzl/decision-transformer/blob/e2d82e68f330c00f763507b3b01d774740bee53f/gym/experiment.py#L116 # noqa
        self.sample_prob = info["traj_lens"] / info["traj_lens"].sum()

    def __prepare_sample(self, traj_idx, start_idx):
        traj = self.dataset[traj_idx]
        # https://github.com/kzl/decision-transformer/blob/e2d82e68f330c00f763507b3b01d774740bee53f/gym/experiment.py#L128 # noqa
        states = traj["observations"][start_idx : start_idx + self.seq_len]
        actions = traj["actions"][start_idx : start_idx + self.seq_len]
        returns = traj["returns"][start_idx : start_idx + self.seq_len]
        if 'timesteps' in traj.keys():
            time_steps = np.arange(traj['timesteps'][0], traj['timesteps'][0] + self.seq_len)
        else:
            time_steps = np.arange(start_idx, start_idx + self.seq_len)

        states = (states - self.state_mean) / self.state_std
        if self.augmentation_scale:
            states = states + np.random.randn(states.shape[0], states.shape[1]) * self.augmentation_scale
        returns = returns * self.reward_scale
        # pad up to seq_len if needed
        mask = np.hstack(
            [np.ones(states.shape[0]), np.zeros(self.seq_len - states.shape[0])]
        )
        if states.shape[0] < self.seq_len:
            states = pad_along_axis(states, pad_to=self.seq_len)
            actions = pad_along_axis(actions, pad_to=self.seq_len)
            returns = pad_along_axis(returns, pad_to=self.seq_len)

        return states, actions, returns, time_steps, mask

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        start_idx = random.randint(0, self.dataset[index]["rewards"].shape[0] - 1)
        return self.__prepare_sample(index, start_idx)


# Decision Transformer implementation
class TransformerBlock(nn.Module):
    def __init__(
        self,
        seq_len: int,
        embedding_dim: int,
        num_heads: int,
        attention_dropout: float,
        residual_dropout: float,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.drop = nn.Dropout(residual_dropout)

        self.attention = nn.MultiheadAttention(
            embedding_dim, num_heads, attention_dropout, batch_first=True
        )
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.GELU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
            nn.Dropout(residual_dropout),
        )
        # True value indicates that the corresponding position is not allowed to attend
        self.register_buffer(
            "causal_mask", ~torch.tril(torch.ones(seq_len, seq_len)).to(bool)
        )
        self.seq_len = seq_len

    # [batch_size, seq_len, emb_dim] -> [batch_size, seq_len, emb_dim]
    def forward(
        self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        causal_mask = self.causal_mask[: x.shape[1], : x.shape[1]]

        norm_x = self.norm1(x)
        attention_out = self.attention(
            query=norm_x,
            key=norm_x,
            value=norm_x,
            attn_mask=causal_mask,
            key_padding_mask=padding_mask,
            need_weights=False,
        )[0]
        # by default pytorch attention does not use dropout
        # after final attention weights projection, while minGPT does:
        # https://github.com/karpathy/minGPT/blob/7218bcfa527c65f164de791099de715b81a95106/mingpt/model.py#L70 # noqa
        x = x + self.drop(attention_out)
        x = x + self.mlp(self.norm2(x))
        return x


class DecisionTransformer(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        seq_len: int = 10,
        episode_len: int = 1000,
        embedding_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 8,
        attention_dropout: float = 0.0,
        residual_dropout: float = 0.0,
        embedding_dropout: float = 0.0,
        max_action: float = 1.0,
    ):
        super().__init__()
        self.emb_drop = nn.Dropout(embedding_dropout)
        self.emb_norm = nn.LayerNorm(embedding_dim)

        self.out_norm = nn.LayerNorm(embedding_dim)
        # additional seq_len embeddings for padding timesteps
        self.timestep_emb = nn.Embedding(episode_len + seq_len, embedding_dim)
        self.state_emb = nn.Linear(state_dim, embedding_dim)
        self.action_emb = nn.Linear(action_dim, embedding_dim)
        self.return_emb = nn.Linear(1, embedding_dim)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    seq_len=3 * seq_len,
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    attention_dropout=attention_dropout,
                    residual_dropout=residual_dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.action_head = nn.Sequential(nn.Linear(embedding_dim, action_dim), nn.Tanh())
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.episode_len = episode_len
        self.max_action = max_action

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(
        self,
        states: torch.Tensor,  # [batch_size, seq_len, state_dim]
        actions: torch.Tensor,  # [batch_size, seq_len, action_dim]
        returns_to_go: torch.Tensor,  # [batch_size, seq_len]
        time_steps: torch.Tensor,  # [batch_size, seq_len]
        padding_mask: Optional[torch.Tensor] = None,  # [batch_size, seq_len]
    ) -> torch.FloatTensor:
        batch_size, seq_len = states.shape[0], states.shape[1]
        # [batch_size, seq_len, emb_dim]
        time_emb = self.timestep_emb(time_steps)
        state_emb = self.state_emb(states.to(torch.float32)) + time_emb
        act_emb = self.action_emb(actions) + time_emb
        returns_emb = self.return_emb(returns_to_go.unsqueeze(-1)) + time_emb

        # [batch_size, seq_len * 3, emb_dim], (r_0, s_0, a_0, r_1, s_1, a_1, ...)
        sequence = (
            torch.stack([returns_emb, state_emb, act_emb], dim=1)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, 3 * seq_len, self.embedding_dim)
        )
        if padding_mask is not None:
            # [batch_size, seq_len * 3], stack mask identically to fit the sequence
            padding_mask = (
                torch.stack([padding_mask, padding_mask, padding_mask], dim=1)
                .permute(0, 2, 1)
                .reshape(batch_size, 3 * seq_len)
            )
        # LayerNorm and Dropout (!!!) as in original implementation,
        # while minGPT & huggingface uses only embedding dropout
        out = self.emb_norm(sequence)
        out = self.emb_drop(out)

        for block in self.blocks:
            out = block(out, padding_mask=padding_mask)

        out = self.out_norm(out)
        # [batch_size, seq_len, action_dim]
        # predict actions only from state embeddings
        out = self.action_head(out[:, 1::3]) * self.max_action
        return out


# Training and evaluation logic
@torch.no_grad()
def eval_rollout(
    model: DecisionTransformer,
    env: gym.Env,
    target_return: float,
    device: str = "cpu",
) -> Tuple[float, float]:
    states = torch.zeros(
        1, model.episode_len + 1, model.state_dim, dtype=torch.float, device=device
    )
    actions = torch.zeros(
        1, model.episode_len, model.action_dim, dtype=torch.float, device=device
    )
    returns = torch.zeros(1, model.episode_len + 1, dtype=torch.float, device=device)
    time_steps = torch.arange(model.episode_len, dtype=torch.long, device=device)
    time_steps = time_steps.view(1, -1)

    states[:, 0] = torch.as_tensor(env.reset(), device=device)
    returns[:, 0] = torch.as_tensor(target_return, device=device)

    # cannot step higher than model episode len, as timestep embeddings will crash
    episode_return, episode_len = 0.0, 0.0
    for step in range(model.episode_len):
        # first select history up to step, then select last seq_len states,
        # step + 1 as : operator is not inclusive, last action is dummy with zeros
        # (as model will predict last, actual last values are not important)
        predicted_actions = model(  # fix this noqa!!!
            states[:, : step + 1][:, -model.seq_len :],
            actions[:, : step + 1][:, -model.seq_len :],
            returns[:, : step + 1][:, -model.seq_len :],
            time_steps[:, : step + 1][:, -model.seq_len :],
        )
        predicted_action = predicted_actions[0, -1].cpu().numpy()
        next_state, reward, done, info = env.step(predicted_action)
        # at step t, we predict a_t, get s_{t + 1}, r_{t + 1}
        actions[:, step] = torch.as_tensor(predicted_action)
        states[:, step + 1] = torch.as_tensor(next_state)
        returns[:, step + 1] = torch.as_tensor(returns[:, step] - reward)

        episode_return += reward
        episode_len += 1

        if done:
            break

    return episode_return, episode_len

def cycle(dl):
    while True:
        for data in dl:
            yield data

@pyrallis.wrap()
def train(config: TrainConfig):
    set_seed(config.seed, deterministic_torch=config.deterministic_torch)
    # init wandb session for logging
    config.project = config.env
    wandb_init(asdict(config))

    # data & dataloader setup
    s4rl = None if config.s4rl_augmentation_type=='identical' else config.std_scale
    dataset = SequenceDataset(
        config, seq_len=config.seq_len, reward_scale=config.reward_scale, augmentation_scale=s4rl
    )
    sampler = WeightedRandomSampler(dataset.sample_prob, len(dataset.sample_prob))

    trainloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        pin_memory=True,
        num_workers=config.num_workers,
        persistent_workers=True,
        sampler=sampler
    )
    # evaluation environment with state & reward preprocessing (as in dataset above)
    eval_env = wrap_env(
        env=gym.make(config.env),
        state_mean=dataset.state_mean,
        state_std=dataset.state_std,
        reward_scale=config.reward_scale,
    )
    # model & optimizer & scheduler setup
    config.state_dim = eval_env.observation_space.shape[0]
    config.action_dim = eval_env.action_space.shape[0]
    model = DecisionTransformer(
        state_dim=config.state_dim,
        action_dim=config.action_dim,
        embedding_dim=config.embedding_dim,
        seq_len=config.seq_len,
        episode_len=config.episode_len,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        attention_dropout=config.attention_dropout,
        residual_dropout=config.residual_dropout,
        embedding_dropout=config.embedding_dropout,
        max_action=config.max_action,
    ).to(config.device)

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=config.betas,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optim,
        lambda steps: min((steps + 1) / config.warmup_steps, 1),
    )
    # save config to the checkpoint
    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    trainloader_iter = cycle(trainloader)
    for step in trange(config.step, desc="Training"):
        batch = next(trainloader_iter)
        states, actions, returns, time_steps, mask = [b.to(config.device) for b in batch]
        # True value indicates that the corresponding key value will be ignored
        padding_mask = ~mask.to(torch.bool)

        predicted_actions = model(
            states=states,
            actions=actions,
            returns_to_go=returns,
            time_steps=time_steps,
            padding_mask=padding_mask,
        )
        loss = F.mse_loss(predicted_actions, actions.detach(), reduction="none")
        # [batch_size, seq_len, action_dim] * [batch_size, seq_len, 1]
        loss = (loss * mask.unsqueeze(-1)).mean()

        optim.zero_grad()
        loss.backward()
        if config.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)
        optim.step()
        scheduler.step()

        wandb.log(
            {
                "train_loss": loss.item(),
                "learning_rate": scheduler.get_last_lr()[0],
            },
            step=step,
        )

        # validation in the env for the actual online performance
        if step % config.eval_every == 0 or step == config.step - 1:
            model.eval()
            for target_return in config.target_returns:
                eval_env.seed(config.eval_seed)
                eval_returns = []
                for _ in trange(config.eval_episodes, desc="Evaluation", leave=False):
                    eval_return, eval_len = eval_rollout(
                        model=model,
                        env=eval_env,
                        target_return=target_return * config.reward_scale,
                        device=config.device,
                    )
                    # unscale for logging & correct normalized score computation
                    eval_returns.append(eval_return / config.reward_scale)

                normalized_scores = (
                    eval_env.get_normalized_score(np.array(eval_returns)) * 100
                )
                wandb.log(
                    {
                        f"eval/{target_return}_return_mean": np.mean(eval_returns),
                        f"eval/{target_return}_return_std": np.std(eval_returns),
                        f"eval/{target_return}_normalized_score_mean": np.mean(
                            normalized_scores
                        ),
                        f"eval/{target_return}_normalized_score_std": np.std(
                            normalized_scores
                        ),
                    },
                    step=step,
                )
                wandb.log({
                "result/d4rl_normalized_score": np.max(np.mean(normalized_scores))
                })
            model.train()

    if config.checkpoints_path is not None:
        checkpoint = {
            "model_state": model.state_dict(),
            "state_mean": dataset.state_mean,
            "state_std": dataset.state_std,
        }
        torch.save(checkpoint, os.path.join(config.checkpoints_path, "dt_checkpoint.pt"))


if __name__ == "__main__":
    train()