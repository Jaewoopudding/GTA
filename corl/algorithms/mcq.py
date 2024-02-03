# source: https://github.com/yihaosun1124/OfflineRL-Kit
import os
import random
import uuid
import time
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import yaml

import d4rl
import gym
import numpy as np
from tqdm import tqdm
from collections import deque
import pyrallis
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.distributions import Normal, TanhTransform, TransformedDistribution
import pyrootutils

path = pyrootutils.find_root(search_from = __file__, indicator=".aug-project-root")
pyrootutils.set_root(path = path,
                     project_root_env_var = True,
                     dotenv = True,
                     pythonpath = True)

from corl.shared.buffer import prepare_replay_buffer, RewardNormalizer, StateNormalizer, DiffusionConfig
from corl.shared.logger import Logger
from corl.shared.utils  import wandb_init, set_seed, wrap_env, soft_update, compute_mean_std, normalize_states, eval_actor, get_saved_dataset, get_generated_dataset, merge_dictionary, get_dataset
from corl.shared.policy import BasePolicy, SACPolicy
from corl.shared.validation import validate

TensorBatch = List[torch.Tensor]
os.environ["WANDB_MODE"] = "online"


@dataclass
class TrainConfig:
    # Experiment
    device: str = "cuda:6"
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    s4rl_augmentation_type: str = 'identical'
    std_scale: float = 0.0003
    uniform_scale: float = 0.0003
    adv_scale: float = 0.0001
    iteration: int = 2
    env: str = "hopper-medium-v2"   # OpenAI gym environment name
    seed: int = 0  # Sets Gym, PyTorch and Numpy seeds
    GDA: str = None  # "gda only" 'gda with original' None
    step: int = 1000000 # Generated Data Augmentation 모델 학습 step 수
    data_mixture_type: str = 'mixed'
    GDA_id: str = None

    # Wandb logging. 자동화 가능한 것은 자동화 해야 함. 데이터 생성할 때, 메타데이터를 첨부하도록 해야 함. 
    # project: str = env # 연동 안됨. 다시 정의해야 함 
    group: str = "MCQ-D4RL"
    name: str = "MCQ"
    project: str = env
    diffusion_horizon: int = 31
    diffusion_backbone: str = 'mixer' # 'mixer', 'temporal'
    
    conditioned: bool = False
    data_volume: int = 5e6
    generation_type: str = 's' # 's,a' 's,a,r'
    guidance_temperature: float = 1.2
    guidance_target_multiple: float = 2

    eval_freq: int = int(5e3)  # How often (time steps) we evaluate
    val_freq: int = int(1e6) # Measuring Q Overestimation
    n_episodes: int = 10  # How many episodes run during evaluation
    max_timesteps: int = int(1e6)  # Max time steps to run environment
    checkpoints_path: Optional[str] = "checkpoints"
    save_checkpoints: bool = True  # Save model checkpoints
    log_every: int = 1000
    load_model: str = ""  # Model load file name, "" doesn't load
    
    ######################################### Changed Part #######################
    # MCQ
    buffer_size: int = 12_000_000  # Replay buffer size - TODO: never used maybe removed
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 0.005
    auto_alpha: bool = True
    target_entropy: Optional[int] = None
    alpha_lr: float = 3e-4
    lmbda: float = 0.9
    num_sampled_actions: int = 10
    behavior_policy_lr: float = 1e-3
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    network_width: int = 400
    network_depth: int = 2
    normalize: bool = True  # Normalize states
    normalize_reward: bool = False  # Normalize reward
    ##############################################################################
    
    datapath: str = None
    # Diffusion config

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
    
    
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Union[List[int], Tuple[int]],
        output_dim: Optional[int] = None,
        activation: nn.Module = nn.ReLU,
        dropout_rate: Optional[float] = None
    ) -> None:
        super().__init__()
        hidden_dims = [input_dim] + list(hidden_dims)
        model = []
        for in_dim, out_dim in zip(hidden_dims[:-1], hidden_dims[1:]):
            model += [nn.Linear(in_dim, out_dim), activation()]
            if dropout_rate is not None:
                model += [nn.Dropout(p=dropout_rate)]

        self.output_dim = hidden_dims[-1]
        if output_dim is not None:
            model += [nn.Linear(hidden_dims[-1], output_dim)]
            self.output_dim = output_dim
        self.model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    

class ActorProb(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        dist_net: nn.Module,
        device: str = "cpu"
    ) -> None:
        super().__init__()

        self.device = torch.device(device)
        self.backbone = backbone.to(device)
        self.dist_net = dist_net.to(device)

    def forward(self, obs: Union[np.ndarray, torch.Tensor]) -> torch.distributions.Normal:
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        logits = self.backbone(obs)
        dist = self.dist_net(logits)
        return dist
    
    @torch.no_grad()
    def act(self, obs: Union[np.ndarray, torch.Tensor], device: str = "cpu") -> np.ndarray:
        dist = self(obs)
        action, _ = dist.mode()
        return action.cpu().detach().numpy().flatten()
    
    @torch.no_grad()
    def get_action_array(self, state: torch.tensor) -> torch.tensor:
        dist = self(state)
        return dist.mode()[0]
        


class Critic(nn.Module):
    def __init__(self, backbone: nn.Module, device: str = "cpu") -> None:
        super().__init__()

        self.device = torch.device(device)
        self.backbone = backbone.to(device)
        latent_dim = getattr(backbone, "output_dim")
        self.last = nn.Linear(latent_dim, 1).to(device)

    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        actions: Optional[Union[np.ndarray, torch.Tensor]] = None
    ) -> torch.Tensor:
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        if actions is not None:
            actions = torch.as_tensor(actions, device=self.device, dtype=torch.float32).flatten(1)
            obs = torch.cat([obs, actions], dim=1)
        logits = self.backbone(obs)
        values = self.last(logits)
        return values


# Vanilla Variational Auto-Encoder 
class VAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        latent_dim: int,
        max_action: Union[int, float],
        device: str = "cpu"
    ) -> None:
        super(VAE, self).__init__()
        self.e1 = nn.Linear(input_dim + output_dim, hidden_dim)
        self.e2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, latent_dim)
        self.log_std = nn.Linear(hidden_dim, latent_dim)

        self.d1 = nn.Linear(input_dim + latent_dim, hidden_dim)
        self.d2 = nn.Linear(hidden_dim, hidden_dim)
        self.d3 = nn.Linear(hidden_dim, output_dim)

        self.max_action = max_action
        self.latent_dim = latent_dim
        self.device = torch.device(device)

        self.to(device=self.device)


    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = F.relu(self.e1(torch.cat([obs, action], 1)))
        z = F.relu(self.e2(z))

        mean = self.mean(z)
        # Clamped for numerical stability 
        log_std = self.log_std(z).clamp(-4, 15)
        std = torch.exp(log_std)
        z = mean + std * torch.randn_like(std)

        u = self.decode(obs, z)

        return u, mean, std

    def decode(self, obs: torch.Tensor, z: Optional[torch.Tensor] = None) -> torch.Tensor:
        # When sampling from the VAE, the latent vector is clipped to [-0.5, 0.5]
        if z is None:
            z = torch.randn((obs.shape[0], self.latent_dim)).to(self.device).clamp(-0.5,0.5)

        a = F.relu(self.d1(torch.cat([obs, z], 1)))
        a = F.relu(self.d2(a))
        return self.max_action * torch.tanh(self.d3(a))
    
    
class NormalWrapper(Normal):
    def log_prob(self, actions):
        return super().log_prob(actions).sum(-1, keepdim=True)

    def entropy(self):
        return super().entropy().sum(-1)

    def mode(self):
        return self.mean


class TanhNormalWrapper(Normal):
    def log_prob(self, action, raw_action=None):
        if raw_action is None:
            raw_action = self.arctanh(action)
        log_prob = super().log_prob(raw_action).sum(-1, keepdim=True)
        eps = 1e-6
        log_prob = log_prob - torch.log((1 - action.pow(2)) + eps).sum(-1, keepdim=True)
        return log_prob

    def mode(self):
        raw_action = self.mean
        action = torch.tanh(self.mean)
        return action, raw_action

    def arctanh(self, x):
        one_plus_x = (1 + x).clamp(min=1e-6)
        one_minus_x = (1 - x).clamp(min=1e-6)
        return 0.5 * torch.log(one_plus_x / one_minus_x)

    def rsample(self):
        raw_action = super().rsample()
        action = torch.tanh(raw_action)
        return action, raw_action
    
    
class DiagGaussian(nn.Module):
    def __init__(
        self,
        latent_dim,
        output_dim,
        unbounded=False,
        conditioned_sigma=False,
        max_mu=1.0,
        sigma_min=-5.0,
        sigma_max=2.0
    ):
        super().__init__()
        self.mu = nn.Linear(latent_dim, output_dim)
        self._c_sigma = conditioned_sigma
        if conditioned_sigma:
            self.sigma = nn.Linear(latent_dim, output_dim)
        else:
            self.sigma_param = nn.Parameter(torch.zeros(output_dim, 1))
        self._unbounded = unbounded
        self._max = max_mu
        self._sigma_min = sigma_min
        self._sigma_max = sigma_max

    def forward(self, logits):
        mu = self.mu(logits)
        if not self._unbounded:
            mu = self._max * torch.tanh(mu)
        if self._c_sigma:
            sigma = torch.clamp(self.sigma(logits), min=self._sigma_min, max=self._sigma_max).exp()
        else:
            shape = [1] * len(mu.shape)
            shape[1] = -1
            sigma = (self.sigma_param.view(shape) + torch.zeros_like(mu)).exp()
        return NormalWrapper(mu, sigma)


class TanhDiagGaussian(DiagGaussian):
    def __init__(
        self,
        latent_dim,
        output_dim,
        unbounded=False,
        conditioned_sigma=False,
        max_mu=1.0,
        sigma_min=-5.0,
        sigma_max=2.0
    ):
        super().__init__(
            latent_dim=latent_dim,
            output_dim=output_dim,
            unbounded=unbounded,
            conditioned_sigma=conditioned_sigma,
            max_mu=max_mu,
            sigma_min=sigma_min,
            sigma_max=sigma_max
        )

    def forward(self, logits):
        mu = self.mu(logits)
        if not self._unbounded:
            mu = self._max * torch.tanh(mu)
        if self._c_sigma:
            sigma = torch.clamp(self.sigma(logits), min=self._sigma_min, max=self._sigma_max).exp()
        else:
            shape = [1] * len(mu.shape)
            shape[1] = -1
            sigma = (self.sigma_param.view(shape) + torch.zeros_like(mu)).exp()
        return TanhNormalWrapper(mu, sigma)
    
    
class MCQPolicy(SACPolicy):
    """
    Mildly Conservative Q-Learning <Ref: https://arxiv.org/abs/2206.04745>
    """

    def __init__(
        self,
        actor: nn.Module,
        critic1: nn.Module,
        critic2: nn.Module,
        behavior_policy: nn.Module,
        actor_optim: torch.optim.Optimizer,
        critic1_optim: torch.optim.Optimizer,
        critic2_optim: torch.optim.Optimizer,
        behavior_policy_optim: torch.optim.Optimizer,
        tau: float = 0.005,
        gamma: float  = 0.99,
        alpha: Union[float, Tuple[float, torch.Tensor, torch.optim.Optimizer]] = 0.2,
        lmbda: float = 0.7,
        num_sampled_actions: int = 10
    ) -> None:
        super().__init__(
            actor,
            critic1,
            critic2,
            actor_optim,
            critic1_optim,
            critic2_optim,
            tau=tau,
            gamma=gamma,
            alpha=alpha
        )

        self.behavior_policy = behavior_policy
        self.behavior_policy_optim = behavior_policy_optim
        self._lmbda = lmbda
        self._num_sampled_actions = num_sampled_actions

    def learn(self, batch: TensorBatch) -> Dict[str, float]:
        # Original
        # obss, actions, next_obss, rewards, terminals = batch["observations"], batch["actions"], \
        #     batch["next_observations"], batch["rewards"], batch["terminals"]
        
        # Revised
        obss, actions, rewards, next_obss, terminals = batch
        
        # update behavior policy
        recon, mean, std = self.behavior_policy(obss, actions)
        recon_loss = F.mse_loss(recon, actions)
        KL_loss	= -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
        vae_loss = recon_loss + KL_loss

        self.behavior_policy_optim.zero_grad()
        vae_loss.backward()
        self.behavior_policy_optim.step()

        # update critic
        with torch.no_grad():
            next_actions, next_log_probs = self.actforward(next_obss)
            next_q = torch.min(
                self.critic1_old(next_obss, next_actions), self.critic2_old(next_obss, next_actions)
            ) - self._alpha * next_log_probs
            target_q_for_in_actions = rewards + self._gamma * (1 - terminals) * next_q
        q1_in, q2_in = self.critic1(obss, actions), self.critic2(obss, actions)
        critic1_loss_for_in_actions = ((q1_in - target_q_for_in_actions).pow(2)).mean()
        critic2_loss_for_in_actions = ((q2_in - target_q_for_in_actions).pow(2)).mean()

        s_in = torch.cat([obss, next_obss], dim=0)
        with torch.no_grad():
            s_in_repeat = torch.repeat_interleave(s_in, self._num_sampled_actions, 0)
            sampled_actions = self.behavior_policy.decode(s_in_repeat)
            target_q1_for_ood_actions = self.critic1_old(s_in_repeat, sampled_actions).reshape(s_in.shape[0], -1).max(1)[0].reshape(-1, 1)
            target_q2_for_ood_actions = self.critic2_old(s_in_repeat, sampled_actions).reshape(s_in.shape[0], -1).max(1)[0].reshape(-1, 1)
            target_q_for_ood_actions = torch.min(target_q1_for_ood_actions, target_q2_for_ood_actions)
            ood_actions, _ = self.actforward(s_in)
        
        q1_ood, q2_ood = self.critic1(s_in, ood_actions), self.critic2(s_in, ood_actions)
        critic1_loss_for_ood_actions = ((q1_ood - target_q_for_ood_actions).pow(2)).mean()
        critic2_loss_for_ood_actions = ((q2_ood - target_q_for_ood_actions).pow(2)).mean()

        critic1_loss = self._lmbda * critic1_loss_for_in_actions + (1 - self._lmbda) * critic1_loss_for_ood_actions
        self.critic1_optim.zero_grad()
        critic1_loss.backward()
        self.critic1_optim.step()

        critic2_loss = self._lmbda * critic2_loss_for_in_actions + (1 - self._lmbda) * critic2_loss_for_ood_actions
        self.critic2_optim.zero_grad()
        critic2_loss.backward()
        self.critic2_optim.step()

        # update actor
        a, log_probs = self.actforward(obss)
        q1a, q2a = self.critic1(obss, a), self.critic2(obss, a)

        actor_loss = - torch.min(q1a, q2a).mean() + self._alpha * log_probs.mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        if self._is_auto_alpha:
            log_probs = log_probs.detach() + self._target_entropy
            alpha_loss = -(self._log_alpha * log_probs).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self._alpha = torch.clamp(self._log_alpha.detach().exp(), 0.0, 1.0)

        self._sync_weight()

        result = {
            "loss/actor": actor_loss.item(),
            "loss/critic1": critic1_loss.item(),
            "loss/critic2": critic2_loss.item(),
            "loss/behavior_policy": vae_loss.item()
        }

        if self._is_auto_alpha:
            result["loss/alpha"] = alpha_loss.item()
            result["alpha"] = self._alpha.item()

        return result
    
    
class MFPolicyTrainer:
    def __init__(
        self,
        policy: BasePolicy,
        # eval_env: gym.Env,
        # buffer: ReplayBuffer,
        # logger: Logger,
        # epoch: int = 1000,
        # step_per_epoch: int = 1000,
        # batch_size: int = 256,
        # eval_episodes: int = 10,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    ) -> None:
        self.policy = policy
        # self.eval_env = eval_env
        # buffer에서 sample하는 부분이 밖에 존재하기 때문에 삭제
        # self.buffer = buffer
        # self.logger = logger

        # self._epoch = epoch
        # self._step_per_epoch = step_per_epoch
        # self._batch_size = batch_size
        # self._eval_episodes = eval_episodes
        self.lr_scheduler = lr_scheduler
        
        self.total_it = 0

    # def train(self) -> Dict[str, float]:
    #     start_time = time.time()

    #     num_timesteps = 0
    #     last_10_performance = deque(maxlen=10)
    #     # train loop
    #     for e in range(1, self._epoch + 1):

    #         self.policy.train()

    #         pbar = tqdm(range(self._step_per_epoch), desc=f"Epoch #{e}/{self._epoch}")
    #         for it in pbar:
    #             batch = self.buffer.sample(self._batch_size)
    #             loss = self.policy.learn(batch)
    #             pbar.set_postfix(**loss)

    #             for k, v in loss.items():
    #                 self.logger.logkv_mean(k, v)
                
    #             num_timesteps += 1

    #         if self.lr_scheduler is not None:
    #             self.lr_scheduler.step()
            
    #         # evaluate current policy
    #         eval_info = self._evaluate()
    #         ep_reward_mean, ep_reward_std = np.mean(eval_info["eval/episode_reward"]), np.std(eval_info["eval/episode_reward"])
    #         ep_length_mean, ep_length_std = np.mean(eval_info["eval/episode_length"]), np.std(eval_info["eval/episode_length"])
    #         norm_ep_rew_mean = self.eval_env.get_normalized_score(ep_reward_mean) * 100
    #         norm_ep_rew_std = self.eval_env.get_normalized_score(ep_reward_std) * 100
    #         last_10_performance.append(norm_ep_rew_mean)
    #         self.logger.logkv("eval/normalized_episode_reward", norm_ep_rew_mean)
    #         self.logger.logkv("eval/normalized_episode_reward_std", norm_ep_rew_std)
    #         self.logger.logkv("eval/episode_length", ep_length_mean)
    #         self.logger.logkv("eval/episode_length_std", ep_length_std)
    #         self.logger.set_timestep(num_timesteps)
    #         self.logger.dumpkvs()
        
    #         # save checkpoint
    #         torch.save(self.policy.state_dict(), os.path.join(self.logger.checkpoint_dir, "policy.pth"))

    #     self.logger.log("total time: {:.2f}s".format(time.time() - start_time))
    #     torch.save(self.policy.state_dict(), os.path.join(self.logger.model_dir, "policy.pth"))
    #     self.logger.close()

    #     return {"last_10_performance": np.mean(last_10_performance)}
    
    # Original Code는 train()을 한 번만 호출. 기존 evaluation 코드를 최대한 바꾸지 않기 위해 이 부분을 고쳤음
    def train(self, batch: TensorBatch) -> Dict[str, float]:
        log_dict = {}
        self.total_it += 1

        loss = self.policy.learn(batch)
        for k, v in loss.items():
            log_dict[k] = v
            
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log_dict
    
    # def _evaluate(self) -> Dict[str, List[float]]:
    #     self.policy.eval()
    #     obs = self.eval_env.reset()
    #     eval_ep_info_buffer = []
    #     num_episodes = 0
    #     episode_reward, episode_length = 0, 0

    #     while num_episodes < self._eval_episodes:
    #         action = self.policy.select_action(obs.reshape(1,-1), deterministic=True)
    #         next_obs, reward, terminal, _ = self.eval_env.step(action.flatten())
    #         episode_reward += reward
    #         episode_length += 1

    #         obs = next_obs

    #         if terminal:
    #             eval_ep_info_buffer.append(
    #                 {"episode_reward": episode_reward, "episode_length": episode_length}
    #             )
    #             num_episodes +=1
    #             episode_reward, episode_length = 0, 0
    #             obs = self.eval_env.reset()
        
    #     return {
    #         "eval/episode_reward": [ep_info["episode_reward"] for ep_info in eval_ep_info_buffer],
    #         "eval/episode_length": [ep_info["episode_length"] for ep_info in eval_ep_info_buffer]
    #     }

    def state_dict(self) -> Dict[str, Any]:
        return {
            "policy": self.policy,
            "total_it": self.total_it
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.policy.load_state_dict(state_dict["actor_lr_schedule"])
        self.total_it = state_dict["total_it"]

    

@pyrallis.wrap()
def train(config: TrainConfig):
    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
    config.project = config.env
    env = gym.make(config.env)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    obs_space = env.observation_space.shape
    action_dim = np.prod(env.action_space.shape)
    hidden_dims = [config.network_width for _ in range(config.network_depth)]
    
    ##### LOADING DATASET #####
    dataset, metadata = get_dataset(config)
    trajectory_data = np.load(f'./data/{config.env}.pkl', allow_pickle=True)
    for k, v in metadata.items():
        setattr(config, k, v)

    if config.normalize:
        state_mean, state_std = compute_mean_std(dataset["observations"], eps=1e-3)
    else:
        state_mean, state_std = 0, 1
    env = wrap_env(env, state_mean=state_mean, state_std=state_std)
    replay_buffer = prepare_replay_buffer(
        state_dim=state_dim,
        action_dim=action_dim,
        buffer_size=config.buffer_size,
        dataset=dataset,
        env_name=config.env,
        device=config.device,
        s4rl_augmentation_type=config.s4rl_augmentation_type,
        std_scale=config.std_scale, 
        uniform_scale=config.uniform_scale, 
        adv_scale=config.adv_scale, 
        reward_normalizer=RewardNormalizer(dataset, config.env) if config.normalize_reward else None,
        state_normalizer=StateNormalizer(state_mean, state_std),
        diffusion_config=config.diffusion,
    )

    config.max_action = float(env.action_space.high[0])


    # Set seeds
    seed = config.seed
    set_seed(seed, env)
    
    # create policy model
    actor_backbone = MLP(input_dim=np.prod(obs_space), hidden_dims=hidden_dims, dropout_rate=0.1)
    critic1_backbone = MLP(input_dim=np.prod(obs_space) + action_dim, hidden_dims=hidden_dims)
    critic2_backbone = MLP(input_dim=np.prod(obs_space) + action_dim, hidden_dims=hidden_dims)
    dist = TanhDiagGaussian(
        latent_dim=getattr(actor_backbone, "output_dim"),
        output_dim=action_dim,
        unbounded=True,
        conditioned_sigma=True
    )
    actor = ActorProb(actor_backbone, dist, config.device)
    critic1 = Critic(critic1_backbone, config.device)
    critic2 = Critic(critic2_backbone, config.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=config.actor_lr)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=config.critic_lr)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=config.critic_lr)

    if config.auto_alpha:
        target_entropy = config.target_entropy if config.target_entropy \
            else -env.action_space.shape[0]

        config.target_entropy = target_entropy

        log_alpha = torch.zeros(1, requires_grad=True, device=config.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=config.alpha_lr)
        alpha = (target_entropy, log_alpha, alpha_optim)
    else:
        alpha = config.alpha

    behavior_policy = VAE(
        input_dim=np.prod(obs_space),
        output_dim=action_dim,
        hidden_dim=750,
        latent_dim=action_dim*2,
        max_action=config.max_action,
        device=config.device
    )
    behavior_policy_optim = torch.optim.Adam(behavior_policy.parameters(), lr=config.behavior_policy_lr)

    # create policy
    policy = MCQPolicy(
        actor,
        critic1,
        critic2,
        behavior_policy,
        actor_optim,
        critic1_optim,
        critic2_optim,
        behavior_policy_optim,
        tau=config.tau,
        gamma=config.gamma,
        alpha=alpha,
        lmbda=config.lmbda,
        num_sampled_actions=config.num_sampled_actions
    )
    
    print("---------------------------------------")
    print(f"Training MCQ, Env: {config.env}, Seed: {seed}")
    print("---------------------------------------")
    
    # create policy trainer
    trainer = MFPolicyTrainer(
        policy=policy,
        # eval_env=env,
        # buffer=replay_buffer,
        # logger=logger,
        # epoch=config.epoch,
        # step_per_epoch=config.step_per_epoch,
        # batch_size=config.batch_size,
        # eval_episodes=config.eval_episodes
    )
    
    if config.load_model != "":
        policy_file = Path(config.load_model)
        trainer.load_state_dict(torch.load(policy_file))
        actor = trainer.actor

    wandb_init(vars(config))
    
    evaluations = []
    for t in range(int(config.max_timesteps)):
        # 내가 이해하기로는, q_function이 argument로 들어가야 하는 이유는
        # S4RL augmentation의 adversarial attack augmentation을 위해서 q_function이 필요하기 떄문인 것 같은데,
        # 어차피 사용 안 하니까 None으로 넣어줘도 되는 건가?
        batch = replay_buffer.sample(config.batch_size, q_function=critic1, iteration=config.iteration )
        batch = [b.to(config.device) for b in batch]
        log_dict = trainer.train(batch)

        if t % config.log_every == 0:
            wandb.log(log_dict, step=trainer.total_it)

        # Evaluate episode
        if t % config.eval_freq == 0 or t == config.max_timesteps - 1:
            print(f"Time steps: {t + 1}")
            eval_scores = eval_actor(
                env,
                actor,
                device=config.device,
                n_episodes=config.n_episodes,
                seed=config.seed,
            )
            eval_score = eval_scores.mean()
            normalized_eval_score = env.get_normalized_score(eval_score) * 100.0
            evaluations.append(normalized_eval_score)
            print("---------------------------------------")
            print(
                f"Evaluation over {config.n_episodes} episodes: "
                f"{eval_score:.3f} , D4RL score: {normalized_eval_score:.3f}"
            )
            print("---------------------------------------")
            log_dict = {"d4rl_normalized_score": normalized_eval_score,
                        "result/d4rl_normalized_score": normalized_eval_score}
            wandb.log(log_dict, step=trainer.total_it)
            
        if (config.val_freq != 0) and (t % config.val_freq == 0 or t == config.max_timesteps - 1) :
            q_gap, bias = validate(trajectory_data, [critic1, critic2], actor, config.gamma, config.device)
            print("---------------------------------------")
            print(
                f"Q gap: {q_gap.mean():.3f}, MC bias: {bias.mean():.3f} "
            )
            print("---------------------------------------")
            wandb.log({
                'Q gap mean': q_gap.mean(),
                'Q gap std': q_gap.std(),
                'MC bias mean': bias.mean(),
                'MC bias std': bias.std()
                },
                step=trainer.total_it
            )

    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)
        np.save(os.path.join(config.checkpoints_path, f"evaluation_results.npy") ,np.array(evaluations))
    if config.checkpoints_path is not None and config.save_checkpoints:
        torch.save(
            trainer.state_dict(),
            os.path.join(config.checkpoints_path, f"checkpoint_{t}.pt"),
        )
    return evaluations
   
if __name__ == "__main__":
    train() 
    