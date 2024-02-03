# source: https://github.com/yihaosun1124/OfflineRL-Kit
import os
import random
import uuid
import time
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import d4rl
import gym
import numpy as np
from tqdm import tqdm
from collections import deque, defaultdict
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

from corl.shared.buffer import prepare_replay_buffer, RewardNormalizer, StateNormalizer, DiffusionConfig, ReplayBuffer
from corl.shared.utils  import wandb_init, set_seed, wrap_env, soft_update, compute_mean_std, normalize_states, eval_actor, get_saved_dataset, get_generated_dataset, merge_dictionary, get_dataset
from corl.shared.policy import BasePolicy, SACPolicy, CQLPolicy
from corl.shared.dynamics import BaseDynamics, EnsembleDynamics, StandardScaler
from corl.shared.ensemble import EnsembleDynamicsModel
from corl.shared.termination_fns import get_termination_fn

TensorBatch = List[torch.Tensor]
os.environ["WANDB_MODE"] = "online"


@dataclass
class TrainConfig:
    # Experiment
    device: str = "cuda:0"
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    s4rl_augmentation_type: str = 'identical'
    std_scale: float = 0.0003
    uniform_scale: float = 0.0003
    adv_scale: float = 0.0001
    iteration: int = 2
    env: str = "halfcheetah-medium-v2"   # OpenAI gym environment name
    seed: int = 0  # Sets Gym, PyTorch and Numpy seeds
    GDA: str = 'None' # "5M-1_25x-smallmixer-50-sar"  # "gda only" 'gda with original' None
    step: int = 1000000 # Generated Data Augmentation 모델 학습 step 수
    data_mixture_type: str = 'mixed'
    GDA_id: str = None

    # Wandb logging. 자동화 가능한 것은 자동화 해야 함. 데이터 생성할 때, 메타데이터를 첨부하도록 해야 함. 
    # project: str = env # 연동 안됨. 다시 정의해야 함
    group: str = "COMBO-D4RL"
    name: str = "COMBO"
    diffusion_horizon: int = 31
    diffusion_backbone: str = 'mixer' # 'mixer', 'temporal'
    
    conditioned: bool = False
    data_volume: int = 5e6
    generation_type: str = 's' # 's,a' 's,a,r'
    guidance_temperature: float = 1.2
    guidance_target_multiple: float = 2

    eval_freq: int = int(5e3)  # How often (time steps) we evaluate
    n_episodes: int = 10  # How many episodes run during evaluation
    max_timesteps: int = int(1e6)  # Max time steps to run environment
    checkpoints_path: Optional[str] = "checkpoints"
    save_checkpoints: bool = True  # Save model checkpoints
    log_every: int = 1000
    load_model: str = ""  # Model load file name, "" doesn't load
    
    ######################################### Changed Part #######################
    # COMBO 
    # suggested hypers

    #     halfcheetah-medium-v2: rollout-length=5, cql-weight=0.5
    #     hopper-medium-v2: rollout-length=5, cql-weight=5.0
    #     walker2d-medium-v2: rollout-length=1, cql-weight=5.0
    #     halfcheetah-medium-replay-v2: rollout-length=5, cql-weight=0.5
    #     hopper-medium-replay-v2: rollout-length=5, cql-weight=0.5
    #     walker2d-medium-replay-v2: rollout-length=1, cql-weight=0.5
    #     halfcheetah-medium-expert-v2: rollout-length=5, cql-weight=5.0
    #     hopper-medium-expert-v2: rollout-length=5, cql-weight=5.0
    #     walker2d-medium-expert-v2: rollout-length=1, cql-weight=5.0


    buffer_size: int = 6_000_000
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 0.005
    alpha: float = 0.2
    auto_alpha: bool = True
    target_entropy: Optional[int] = None
    alpha_lr: float = 1e-4
    dynamics_lr: float = 1e-3

    cql_weight: float = 0.5
    temperature: float = 1.
    max_q_backup: bool = False
    deterministic_backup: bool = True
    with_lagrange: bool = False
    lagrange_threshold: float = 10.
    cql_alpha_lr: float = 3e-4
    num_repeat_actions: int = 10
    uniform_rollout: bool = False
    rho_s: str = 'mix'

    # List argument cannot be passed
    hidden_dim1: int = 256
    hidden_dim2: int = 256
    hidden_dim3: int = 256

    dynamics_hidden_dim1: int = 200
    dynamics_hidden_dim2: int = 200
    dynamics_hidden_dim3: int = 200
    dynamics_hidden_dim4: int = 200

    dynamics_weight_decay1: float = 2.5e-5
    dynamics_weight_decay2: float = 5e-5
    dynamics_weight_decay3: float = 7.5e-5
    dynamics_weight_decay4: float = 7.5e-5
    dynamics_weight_decay5: float = 1e-4

    max_epochs: int = 100
    max_epochs_since_update: int = 5
    n_ensemble: int = 7
    n_elites: int = 5
    rollout_freq: int = 1000
    rollout_batch_size: int = 50000
    rollout_length: int = 5
    penalty_coef: float = 2.5
    model_retain_epochs: int = 5
    real_ratio: float = 0.05
    load_dynamics_path: Optional[str] = None
    
    actor_lr: float = 1e-4
    critic_lr: float = 3e-4
    network_width: int = 256
    network_depth: int = 2
    normalize: bool = True  # Normalize states
    normalize_reward: bool = False  # Normalize reward
    ##############################################################################

    datapath: str = None #'data/generated_data/halfcheetah-medium-v2/smallmixer/5M-1_25x-smallmixer-50-sar.npz'

    
    # Diffusion config

    def __post_init__(self):
        self.name = f"{self.name}-{self.env}-{self.s4rl_augmentation_type}-{str(uuid.uuid4())[:4]}"
        self.hidden_dims = [self.hidden_dim1, self.hidden_dim2, self.hidden_dim3]
        self.dynamics_hidden_dims = [self.dynamics_hidden_dim1, self.dynamics_hidden_dim2, self.dynamics_hidden_dim3, self.dynamics_hidden_dim4]
        self.dynamics_weight_decays = [self.dynamics_weight_decay1, self.dynamics_weight_decay2, self.dynamics_weight_decay3, self.dynamics_weight_decay4, self.dynamics_weight_decay5]
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


class COMBOPolicy(CQLPolicy):
    """
    Conservative Offline Model-Based Policy Optimization <Ref: https://arxiv.org/abs/2102.08363>
    """

    def __init__(
        self,
        dynamics: BaseDynamics,
        actor: nn.Module,
        critic1: nn.Module,
        critic2: nn.Module,
        actor_optim: torch.optim.Optimizer,
        critic1_optim: torch.optim.Optimizer,
        critic2_optim: torch.optim.Optimizer,
        action_space: gym.spaces.Space,
        tau: float = 0.005,
        gamma: float  = 0.99,
        alpha: Union[float, Tuple[float, torch.Tensor, torch.optim.Optimizer]] = 0.2,
        cql_weight: float = 1.0,
        temperature: float = 1.0,
        max_q_backup: bool = False,
        deterministic_backup: bool = True,
        with_lagrange: bool = True,
        lagrange_threshold: float = 10.0,
        cql_alpha_lr: float = 1e-4,
        num_repeart_actions:int = 10,
        uniform_rollout: bool = False,
        rho_s: str = "mix"
    ) -> None:
        super().__init__(
            actor,
            critic1,
            critic2,
            actor_optim,
            critic1_optim,
            critic2_optim,
            action_space,
            tau=tau,
            gamma=gamma,
            alpha=alpha,
            cql_weight=cql_weight,
            temperature=temperature,
            max_q_backup=max_q_backup,
            deterministic_backup=deterministic_backup,
            with_lagrange=with_lagrange,
            lagrange_threshold=lagrange_threshold,
            cql_alpha_lr=cql_alpha_lr,
            num_repeart_actions=num_repeart_actions
        )

        self.dynamics = dynamics
        self._uniform_rollout = uniform_rollout
        self._rho_s = rho_s

    def rollout(
        self,
        init_obss: np.ndarray,
        rollout_length: int
    ) -> Tuple[Dict[str, np.ndarray], Dict]:

        num_transitions = 0
        rewards_arr = np.array([])
        rollout_transitions = defaultdict(list)

        # rollout
        observations = init_obss
        for _ in range(rollout_length):
            if self._uniform_rollout:
                actions = np.random.uniform(
                    self.action_space.low[0],
                    self.action_space.high[0],
                    size=(len(observations), self.action_space.shape[0])
                )
            else:
                actions = self.select_action(observations)
            next_observations, rewards, terminals, info = self.dynamics.step(observations, actions)
            rollout_transitions["obss"].append(observations)
            rollout_transitions["next_obss"].append(next_observations)
            rollout_transitions["actions"].append(actions)
            rollout_transitions["rewards"].append(rewards)
            rollout_transitions["terminals"].append(terminals)

            num_transitions += len(observations)
            rewards_arr = np.append(rewards_arr, rewards.flatten())

            nonterm_mask = (~terminals).flatten()
            if nonterm_mask.sum() == 0:
                break

            observations = next_observations[nonterm_mask]
        
        for k, v in rollout_transitions.items():
            rollout_transitions[k] = np.concatenate(v, axis=0)

        return rollout_transitions, \
            {"num_transitions": num_transitions, "reward_mean": rewards_arr.mean()}
    

    def learn(self, batch: Dict) -> Dict[str, float]:
        real_batch, fake_batch = batch["real"], batch["fake"]
        mix_batch = {
            "observations": torch.cat([real_batch[0], fake_batch[0]], 0),
            "actions": torch.cat([real_batch[1], fake_batch[1]], 0),
            "rewards": torch.cat([real_batch[2], fake_batch[2]], 0),
            "next_observations": torch.cat([real_batch[3], fake_batch[3]], 0),
            "terminals": torch.cat([real_batch[4], fake_batch[4]], 0),
        }
    
        obss, actions, next_obss, rewards, terminals = mix_batch["observations"], mix_batch["actions"], \
            mix_batch["next_observations"], mix_batch["rewards"], mix_batch["terminals"]
        batch_size = obss.shape[0]
        
        # update actor
        a, log_probs = self.actforward(obss)
        q1a, q2a = self.critic1(obss, a), self.critic2(obss, a)
        actor_loss = (self._alpha * log_probs - torch.min(q1a, q2a)).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        if self._is_auto_alpha:
            log_probs = log_probs.detach() + self._target_entropy
            alpha_loss = -(self._log_alpha * log_probs).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self._alpha = self._log_alpha.detach().exp()
        
        # compute td error
        if self._max_q_backup:
            with torch.no_grad():
                tmp_next_obss = next_obss.unsqueeze(1) \
                    .repeat(1, self._num_repeat_actions, 1) \
                    .view(batch_size * self._num_repeat_actions, next_obss.shape[-1])
                tmp_next_actions, _ = self.actforward(tmp_next_obss)
                tmp_next_q1 = self.critic1_old(tmp_next_obss, tmp_next_actions) \
                    .view(batch_size, self._num_repeat_actions, 1) \
                    .max(1)[0].view(-1, 1)
                tmp_next_q2 = self.critic2_old(tmp_next_obss, tmp_next_actions) \
                    .view(batch_size, self._num_repeat_actions, 1) \
                    .max(1)[0].view(-1, 1)
                next_q = torch.min(tmp_next_q1, tmp_next_q2)
        else:
            with torch.no_grad():
                next_actions, next_log_probs = self.actforward(next_obss)
                next_q = torch.min(
                    self.critic1_old(next_obss, next_actions),
                    self.critic2_old(next_obss, next_actions)
                )
                if not self._deterministic_backup:
                    next_q -= self._alpha * next_log_probs

        target_q = rewards + self._gamma * (1 - terminals) * next_q
        q1, q2 = self.critic1(obss, actions), self.critic2(obss, actions)
        critic1_loss = ((q1 - target_q).pow(2)).mean()
        critic2_loss = ((q2 - target_q).pow(2)).mean()

        # compute conservative loss
        if self._rho_s == "model":
            obss, actions, next_obss = fake_batch["observations"], \
                fake_batch["actions"], fake_batch["next_observations"]
            
        batch_size = len(obss)
        random_actions = torch.FloatTensor(
            batch_size * self._num_repeat_actions, actions.shape[-1]
        ).uniform_(self.action_space.low[0], self.action_space.high[0]).to(self.actor.device)
        # tmp_obss & tmp_next_obss: (batch_size * num_repeat, obs_dim)
        tmp_obss = obss.unsqueeze(1) \
            .repeat(1, self._num_repeat_actions, 1) \
            .view(batch_size * self._num_repeat_actions, obss.shape[-1])
        tmp_next_obss = next_obss.unsqueeze(1) \
            .repeat(1, self._num_repeat_actions, 1) \
            .view(batch_size * self._num_repeat_actions, obss.shape[-1])
        
        obs_pi_value1, obs_pi_value2 = self.calc_pi_values(tmp_obss, tmp_obss)
        next_obs_pi_value1, next_obs_pi_value2 = self.calc_pi_values(tmp_next_obss, tmp_obss)
        random_value1, random_value2 = self.calc_random_values(tmp_obss, random_actions)

        for value in [
            obs_pi_value1, obs_pi_value2, next_obs_pi_value1, next_obs_pi_value2,
            random_value1, random_value2
        ]:
            value.reshape(batch_size, self._num_repeat_actions, 1)
        
        # cat_q shape: (batch_size, 3 * num_repeat, 1)
        cat_q1 = torch.cat([obs_pi_value1, next_obs_pi_value1, random_value1], 1)
        cat_q2 = torch.cat([obs_pi_value2, next_obs_pi_value2, random_value2], 1)
        # Samples from the original dataset
        real_obss, real_actions = real_batch[0], real_batch[1]
        q1, q2 = self.critic1(real_obss, real_actions), self.critic2(real_obss, real_actions)

        conservative_loss1 = \
            torch.logsumexp(cat_q1 / self._temperature, dim=1).mean() * self._cql_weight * self._temperature - \
            q1.mean() * self._cql_weight
        conservative_loss2 = \
            torch.logsumexp(cat_q2 / self._temperature, dim=1).mean() * self._cql_weight * self._temperature - \
            q2.mean() * self._cql_weight
        
        if self._with_lagrange:
            cql_alpha = torch.clamp(self.cql_log_alpha.exp(), 0.0, 1e6)
            conservative_loss1 = cql_alpha * (conservative_loss1 - self._lagrange_threshold)
            conservative_loss2 = cql_alpha * (conservative_loss2 - self._lagrange_threshold)

            self.cql_alpha_optim.zero_grad()
            cql_alpha_loss = -(conservative_loss1 + conservative_loss2) * 0.5
            cql_alpha_loss.backward(retain_graph=True)
            self.cql_alpha_optim.step()
        
        critic1_loss = critic1_loss + conservative_loss1
        critic2_loss = critic2_loss + conservative_loss2

        # update critic
        self.critic1_optim.zero_grad()
        critic1_loss.backward(retain_graph=True)
        self.critic1_optim.step()

        self.critic2_optim.zero_grad()
        critic2_loss.backward()
        self.critic2_optim.step()

        self._sync_weight()

        result =  {
            "loss/actor": actor_loss.item(),
            "loss/critic1": critic1_loss.item(),
            "loss/critic2": critic2_loss.item()
        }

        if self._is_auto_alpha:
            result["loss/alpha"] = alpha_loss.item()
            result["alpha"] = self._alpha.item()
        if self._with_lagrange:
            result["loss/cql_alpha"] = cql_alpha_loss.item()
            result["cql_alpha"] = cql_alpha.item()
        
        return result

    
class MBPolicyTrainer:
    def __init__(
        self,
        policy: BasePolicy,
        eval_env: gym.Env,
        real_buffer: ReplayBuffer,
        fake_buffer: ReplayBuffer,
        rollout_setting: Tuple[int, int, int],
        # epoch: int = 1000,
        # step_per_epoch: int = 1000,
        batch_size: int = 256,
        real_ratio: float = 0.05,
        # eval_episodes: int = 10,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        dynamics_update_freq: int = 0
    ) -> None:
        self.policy = policy
        self.eval_env = eval_env
        self.real_buffer = real_buffer
        self.fake_buffer = fake_buffer

        self._rollout_freq, self._rollout_batch_size, \
            self._rollout_length = rollout_setting
        self._dynamics_update_freq = dynamics_update_freq
        self._batch_size = batch_size
        self._real_ratio = real_ratio
        self.lr_scheduler = lr_scheduler
        
        self.total_it = 0
        
    def train(self) -> Dict[str, float]:
        self.policy.train()
        log_dict = {}
        
        if self.total_it % self._rollout_freq == 0:
            init_obss = self.real_buffer._sample(self._rollout_batch_size)[0].cpu().numpy()
            rollout_transitions, rollout_info = self.policy.rollout(init_obss, self._rollout_length)
            self.fake_buffer.add_batch(**rollout_transitions)
            
            for k, v in rollout_info.items():
                log_dict["rollout_info/"+k] = v
                # print(f"rollout_info/{k}: {v:.2f}")
                
        real_sample_size = int(self._batch_size * self._real_ratio)
        fake_sample_size = self._batch_size - real_sample_size
        real_batch = self.real_buffer._sample(batch_size=real_sample_size)
        fake_batch = self.fake_buffer._sample(batch_size=fake_sample_size)
        batch = {"real": real_batch, "fake": fake_batch}
        loss = self.policy.learn(batch)
        
        for k, v in loss.items():
            log_dict[k] = v
            
        # update the dynamics if necessary
        if 0 < self._dynamics_update_freq and (self.total_it+1)%self._dynamics_update_freq == 0:
            dynamics_update_info = self.policy.update_dynamics(self.real_buffer)
        
        self.total_it += 1

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log_dict


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
    
    config.obs_shape = env.observation_space.shape
    config.action_dim = int(np.prod(env.action_space.shape))
    config.hidden_dims = [config.network_width for _ in range(config.network_depth)]
    
    ##### LOADING DATASET #####
    dataset, metadata = get_dataset(config)
    for k, v in metadata.items():
        setattr(config, k, v)

    if config.normalize:
        state_mean, state_std = compute_mean_std(dataset["observations"], eps=1e-3)
    else:
        state_mean, state_std = 0, 1
    # env = wrap_env(env, state_mean=state_mean, state_std=state_std)
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
    
    fake_buffer = ReplayBuffer(
        state_dim=state_dim,
        action_dim=action_dim,
        buffer_size=config.rollout_batch_size*config.rollout_length*config.model_retain_epochs,
        device=config.device,
        reward_normalizer=RewardNormalizer(dataset, config.env) if config.normalize_reward else None,
        state_normalizer=StateNormalizer(state_mean, state_std),
    )

    config.max_action = float(env.action_space.high[0])

    # Set seeds
    seed = config.seed
    set_seed(seed, env)
    
    # create policy model
    actor_backbone = MLP(input_dim=np.prod(config.obs_shape), hidden_dims=config.hidden_dims)
    critic1_backbone = MLP(input_dim=np.prod(config.obs_shape) + config.action_dim, hidden_dims=config.hidden_dims)
    critic2_backbone = MLP(input_dim=np.prod(config.obs_shape) + config.action_dim, hidden_dims=config.hidden_dims)
    dist = TanhDiagGaussian(
        latent_dim=getattr(actor_backbone, "output_dim"),
        output_dim=config.action_dim,
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
            else int(-np.prod(env.action_space.shape))
        config.target_entropy = target_entropy
        log_alpha = torch.zeros(1, requires_grad=True, device=config.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=config.alpha_lr)
        alpha = (target_entropy, log_alpha, alpha_optim)
    else:
        alpha = config.alpha

    # create dynamics
    load_dynamics_model = True if config.load_dynamics_path else False
    
    dynamics_model = EnsembleDynamicsModel(
        obs_dim=np.prod(config.obs_shape),
        action_dim=config.action_dim,
        hidden_dims=config.dynamics_hidden_dims,
        num_ensemble=config.n_ensemble,
        num_elites=config.n_elites,
        weight_decays=config.dynamics_weight_decays,
        device=config.device
    )
    dynamics_optim = torch.optim.Adam(
        dynamics_model.parameters(),
        lr=config.dynamics_lr
    )
    scaler = StandardScaler()
    termination_fn = get_termination_fn(task=config.env)
    dynamics = EnsembleDynamics(
        dynamics_model,
        dynamics_optim,
        scaler,
        termination_fn,
        penalty_coef=config.penalty_coef,
        checkpoint_path=config.checkpoints_path,
    )

    if config.load_dynamics_path:
        dynamics.load(config.load_dynamics_path)
        
    # create policy
    policy = COMBOPolicy(
        dynamics,
        actor,
        critic1,
        critic2,
        actor_optim,
        critic1_optim,
        critic2_optim,
        action_space=env.action_space,
        tau=config.tau,
        gamma=config.gamma,
        alpha=alpha,
        # TODO
        cql_weight=config.cql_weight,
        temperature=config.temperature,
        max_q_backup=config.max_q_backup,
        deterministic_backup=config.deterministic_backup,
        with_lagrange=config.with_lagrange,
        lagrange_threshold=config.lagrange_threshold,
        cql_alpha_lr=config.cql_alpha_lr,
        num_repeart_actions=config.num_repeat_actions,
        uniform_rollout=config.uniform_rollout,
        rho_s=config.rho_s
    )
    
    print("---------------------------------------")
    print(f"Training COMBO, Env: {config.env}, Seed: {seed}")
    print("---------------------------------------")
    
    # create policy trainer
    trainer = MBPolicyTrainer(
        policy=policy,
        eval_env=env,
        real_buffer=replay_buffer,
        fake_buffer=fake_buffer,
        rollout_setting=(config.rollout_freq, config.rollout_batch_size, config.rollout_length),
        batch_size=config.batch_size,
        real_ratio=config.real_ratio,
    )
    
    wandb_init(vars(config))
    
    # train
    if not load_dynamics_model:
        dynamics.train(
            data=replay_buffer.sample_all(), 
            max_epochs=config.max_epochs,
            max_epochs_since_update=config.max_epochs_since_update,
        )
    
    evaluations = []
    for t in range(int(config.max_timesteps)):
        # 내가 이해하기로는, q_function이 argument로 들어가야 하는 이유는
        # S4RL augmentation의 adversarial attack augmentation을 위해서 q_function이 필요하기 떄문인 것 같은데,
        # 어차피 사용 안 하니까 None으로 넣어줘도 되는 건가?
        log_dict = trainer.train()
        evaluations = []

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
            if config.checkpoints_path is not None and config.save_checkpoints:
                torch.save(
                    trainer.state_dict(),
                    os.path.join(config.checkpoints_path, f"checkpoint_{t}.pt"),
                )
            log_dict = {"d4rl_normalized_score": normalized_eval_score}
            wandb.log(log_dict, step=trainer.total_it)
            evaluations.append(eval_score)
    # config.evaluations = evaluations

    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    if config.checkpoints_path is not None and config.save_checkpoints:
        torch.save(
            trainer.state_dict(),
            os.path.join(config.checkpoints_path, f"checkpoint_{t}.pt"),
        )
    return evaluations


if __name__ == "__main__":
    train() 
