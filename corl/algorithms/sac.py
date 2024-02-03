# source: https://github.com/sfujim/TD3_BC
# https://arxiv.org/pdf/2106.06860.pdf
import copy
import os
import random
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import d4rl
import gym
import numpy as np
import pyrallis
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import pyrootutils

path = pyrootutils.find_root(search_from = __file__, indicator=".aug-project-root")
pyrootutils.set_root(path = path,
                     project_root_env_var = True,
                     dotenv = True,
                     pythonpath = True)

from corl.shared.buffer import prepare_replay_buffer, RewardNormalizer, StateNormalizer, DiffusionConfig
from corl.shared.utils  import wandb_init, set_seed, wrap_env, soft_update, compute_mean_std, normalize_states, eval_actor, get_saved_dataset, get_generated_dataset, merge_dictionary, get_dataset

TensorBatch = List[torch.Tensor]
os.environ["WANDB_MODE"] = "online"

## TODO config q_lr, policy_lr, autotune, target_network_freq
@dataclass
class TrainConfig:
    # Experiment
    device: str = "cuda:7"
    s4rl_augmentation_type: str = 'identical'
    std_scale: float = 0.0003
    uniform_scale: float = 0.0003
    adv_scale: float = 0.0001
    iteration: int = 2
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    env: str = "halfcheetah-medium-v2"  # OpenAI gym environment name
    seed: int = 5  # Sets Gym, PyTorch and Numpy seeds
    GDA: str = None  # "gda only" 'gda with original' None
    data_mixture_type: str = 'mixed'
    GDA_id: str = None
    
    # Wandb logging
    project: str = env
    group: str = "SAC-D4RL"
    name: str = "SAC"
    diffusion_horizon: int = 31
    diffusion_backbone: str = 'mixer' # 'mixer', 'temporal'
    
    conditioned: bool = False
    data_volume: int = 5e6
    generation_type: str = 's' # 's,a' 's,a,r'
    guidance_temperature: float = 1.2
    guidance_target_multiple: float = 2

    step: int = 1000000 # Generated Data Augmentation 모델 학습 step 수

    eval_freq: int = int(5e3)  # How often (time steps) we evaluate
    n_episodes: int = 10  # How many episodes run during evaluation
    max_timesteps: int = int(1e6)  # Max time steps to run environment
    checkpoints_path: Optional[str] = "checkpoints"
    save_checkpoints: bool = True  # Save model checkpoints
    log_every: int = 1000
    load_model: str = ""  # Model load file name, "" doesn't load
    buffer_size: int = 2_000_000  # Replay buffer size
    batch_size: int = 256  # Batch size for all networks
    discount: float = 0.99  # Discount factor

    normalize: bool = True  # Normalize states
    normalize_reward: bool = False  # Normalize reward

    # SAC config
    tau: float = 0.005  # Target network update rate
    policy_freq: int = 2  # Frequency of delayed actor updates
    q_lr: float = 1e-3
    policy_lr: float = 3e-4
    autotune: bool = True
    target_network_freq: int = 1

    # Diffusion config
    # Network size
    network_width: int = 256
    network_depth: int = 2
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

class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.observation_space.shape).prod() + np.prod(env.action_space.shape), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(env.action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu") -> np.ndarray:
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        return self.get_action(state)[0].cpu().data.numpy().flatten()



class SAC:  # noqa
    def __init__(
            self,
            actor: nn.Module,
            actor_optimizer: torch.optim.Optimizer,
            qf_1: nn.Module,
            qf_2: nn.Module,
            q_optimizer: torch.optim.Optimizer,
            qf_1_target: nn.Module,
            qf_2_target: nn.Module,
            discount: float = 0.99, # 사용됨
            tau: float = 0.005,
            policy_freq: int = 2,
            alpha: float = 2.5, # 사용됨
            log_alpha: float = None,
            target_entropy: float = None,
            autotune: bool = True,
            a_optimizer: torch.optim.Optimizer = None,
            target_network_freq: int = 1,
            device: str = "cpu",
    ):
        


        self.actor = actor
        self.actor_target = copy.deepcopy(actor)
        self.actor_optimizer = actor_optimizer
        self.qf1 = qf_1
        self.qf2 = qf_2
        self.q_optimizer = q_optimizer
        self.qf1_target = qf_1_target
        self.qf2_target = qf_2_target

        self.discount = discount
        self.tau = tau
        self.policy_freq = policy_freq

        self.alpha = alpha
        self.log_alpha = log_alpha
        self.target_entropy = target_entropy
        self.autotune = autotune
        self.a_optimizer = a_optimizer
        self.target_network_freq = target_network_freq 

        self.total_it = 0
        self.device = device

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        log_dict = {}
        self.total_it += 1

        state, action, reward, next_state, done = batch
        not_done = 1 - done

        ## state augmentation이 두번 일어나야 되는걸

        with torch.no_grad():
            # Select action according to actor and add clipped noise
            next_state_actions, next_state_log_pi, _ = self.actor.get_action(next_state)
            qf1_next_target = self.qf1_target(next_state, next_state_actions)
            qf2_next_target = self.qf2_target(next_state, next_state_actions)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward.flatten() + (1 - done.flatten()) * self.discount * (min_qf_next_target).view(-1)

        qf1_a_values = self.qf1(state, action).view(-1)
        qf2_a_values = self.qf2(state, action).view(-1)
        qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
        qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        self.q_optimizer.zero_grad()
        qf_loss.backward()
        self.q_optimizer.step()

        # Delayed actor updates
        if self.total_it % self.policy_freq == 0:
            # Compute actor loss
            for _ in range(self.policy_freq):
                pi, log_pi, _ = self.actor.get_action(state)
                qf1_pi = self.qf1(state, pi)
                qf2_pi = self.qf2(state, pi)
                min_qf_pi = torch.min(qf1_pi, qf2_pi)
                actor_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                if self.autotune:
                    with torch.no_grad():
                        _, log_pi, _ = self.actor.get_action(state)
                    alpha_loss = (-self.log_alpha.exp() * (log_pi + self.target_entropy)).mean()
                    self.a_optimizer.zero_grad()
                    alpha_loss.backward()
                    self.a_optimizer.step()
                    self.alpha = self.log_alpha.exp().item()

        if self.total_it % self.target_network_freq == 0:
            soft_update(self.qf1_target, self.qf1, self.tau)
            soft_update(self.qf2_target, self.qf2, self.tau)
            soft_update(self.actor_target, self.actor, self.tau)

        return log_dict

    def state_dict(self) -> Dict[str, Any]:
        return {
            "qf1": self.qf1.state_dict(),
            "qf2": self.qf2.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
            "actor": self.actor.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "total_it": self.total_it,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.qf1.load_state_dict(state_dict["qf1"])
        self.qf2.load_state_dict(state_dict["qf2"])
        self.q_optimizer.load_state_dict(state_dict["q_optimizer"])

        self.actor.load_state_dict(state_dict['actor'])
        self.actor_optmizer.load_state_dict(state_dict['actor_optimizer'])

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

    ##### LOADING DATASET #####
    dataset, metadata = get_dataset(config)
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

    # Set seeds
    seed = config.seed
    set_seed(seed, env)

    actor = Actor(env).to(config.device)
    qf1 = SoftQNetwork(env).to(config.device)
    qf2 = SoftQNetwork(env).to(config.device)
    qf1_target = SoftQNetwork(env).to(config.device)
    qf2_target = SoftQNetwork(env).to(config.device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = torch.optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=config.q_lr)
    actor_optimizer = torch.optim.Adam(list(actor.parameters()), lr=config.policy_lr)

    if config.autotune:
        target_entropy = -torch.prod(torch.Tensor(env.action_space.shape).to(config.device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=config.device)
        alpha = log_alpha.exp().item()
        a_optimizer = torch.optim.Adam([log_alpha], lr=config.q_lr)
    else:
        alpha = config.alpha



        pass
    kwargs = {
        "actor": actor,
        "actor_optimizer": actor_optimizer,
        "qf_1": qf1,
        "qf_2": qf2,
        "q_optimizer": q_optimizer,
        "qf_1_target": qf1_target,
        "qf_2_target": qf2_target,
        "discount": config.discount,
        "tau": config.tau,

        "policy_freq": config.policy_freq,
        "alpha": alpha,
        "log_alpha": log_alpha,
        "target_entropy": target_entropy,
        "autotune": config.autotune,
        "a_optimizer": a_optimizer,
        "target_network_freq": config.target_network_freq,
        "device": config.device,
    }

    print("---------------------------------------")
    print(f"Training SAC (Online), Env: {config.env}, Seed: {seed}")
    print("---------------------------------------")

    # Initialize actor
    trainer = SAC(**kwargs)

    if config.load_model != "":
        policy_file = Path(config.load_model)
        trainer.load_state_dict(torch.load(policy_file))
        actor = trainer.actor

    wandb_init(vars(config))

    evaluations = []
    for t in range(int(config.max_timesteps)):
        batch = replay_buffer.sample(config.batch_size, q_function=qf1, iteration=config.iteration )
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
    
    config.evaluations = evaluations
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
