"""
Main diffusion code.
Code was adapted from https://github.com/lucidrains/denoising-diffusion-pytorch
"""
import math
import pathlib
from multiprocessing import cpu_count
from typing import Optional, Sequence, Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from accelerate import Accelerator
from einops import reduce
from ema_pytorch import EMA
# from redq.algos.core import ReplayBuffer
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchdiffeq import odeint
from tqdm import tqdm


import pyrootutils

path = pyrootutils.find_root(search_from=__file__, indicator=".aug-project-root")
pyrootutils.set_root(path = path,
                     project_root_env_var = True,
                     dotenv = True,
                     pythonpath = True)


from src.data.norm import BaseNormalizer
from src.online.utils import make_inputs_from_replay_buffer


# helpers
def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def cycle(dl):
    while True:
        for data in dl:
            yield data

def define_rewardweighting_sampler(dataset,     
                                    dataset_nm,
                                    reward_scale,
                                    weight_param):                 
    '''
    weight_param: work as weight for sparse reward setting, bin num for dense reward setting
    '''

    print('Reweighted training on high reward samples')

    episode_rewards = dataset.episode_rewards
    trajectory_returns = dataset.trajectory_returns
    trajectory_returns *= reward_scale

    dense_reward = ("maze" in dataset_nm)
    ## 여기서 분기 : dense reward vs sparse reward
    
    if dense_reward:
        # episode wise ; need to set cfg.Dataset.episode = True
        data_len = len(episode_rewards)
        fail = len(episode_rewards[episode_rewards == 0])
        success = len(episode_rewards[episode_rewards > 0])

        episode_rewards[episode_rewards == 0] = 1
        episode_rewards[episode_rewards > 0] = weight_param

        sampler = WeightedRandomSampler(torch.DoubleTensor(episode_rewards), len(episode_rewards))
        
    
    else: 

        hist, bin_edges = np.histogram(trajectory_returns, bins=weight_param)
        hist = hist / np.sum(hist)

        softmin_prob_unnorm = np.exp(bin_edges[1:] / 5.0)
        softmin_prob = softmin_prob_unnorm / np.sum(softmin_prob_unnorm)

        provable_dist = softmin_prob * (hist / (hist + 1e-3))
        provable_dist = provable_dist / (np.sum(provable_dist) + 1e-7)

        bin_indices = np.digitize(trajectory_returns, bin_edges[1:])
        hist_prob = hist[np.minimum(bin_indices, weight_param-1)]

        weights = provable_dist[np.minimum(bin_indices, weight_param-1)] / (hist_prob + 1e-7)
        weights = np.clip(weights, a_min=0.0, a_max=5.0)
        weights = weights.squeeze()

        # Select samples proportional to weights
        sampler = WeightedRandomSampler(torch.DoubleTensor(weights), len(trajectory_returns))
        
    return sampler


def define_terminal_sampler(dataset,
                            dataset_nm,
                            terminal = True):                 

    terminal_idx = dataset.terminal_idx

    assert "maze" not in dataset_nm, "this sampler is not for maze envs"

    if terminal:
        sampler = WeightedRandomSampler(torch.DoubleTensor(terminal_idx), len(terminal_idx))
    else:
        sampler = WeightedRandomSampler(torch.DoubleTensor(1-terminal_idx), len(1-terminal_idx))
    return sampler



# tensor helpers
def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


def apply_conditioning(x, cond_state, state_dim):
    for idx, transition in cond_state.items():
        x[idx, -2:, :state_dim] = transition[:,:state_dim].clone()
    return x


# main class
class ElucidatedDiffusion(nn.Module):
    def __init__(
            self,
            net,
            normalizer: BaseNormalizer,
            data_shape: dict,  # shape of the input and output
            num_sample_steps: int = 32,  # number of sampling steps
            sigma_min: float = 0.002,  # min noise level
            sigma_max: float = 80,  # max noise level
            sigma_data: float = 1.0,  # standard deviation of data distribution
            rho: float = 7,  # controls the sampling schedule
            P_mean: float = -1.2,  # mean of log-normal distribution from which noise is drawn for training
            P_std: float = 1.2,  # standard deviation of log-normal distribution from which noise is drawn for training
            S_churn: float = 80,  # parameters for stochastic sampling - depends on dataset, Table 5 in paper
            S_tmin: float = 0.05,
            S_tmax: float = 50,
            S_noise: float = 1.003,
    ):
        super().__init__()
        assert net.random_or_learned_sinusoidal_cond
        self.net = net
        self.normalizer = normalizer

        # input dimensions
        self.data_shape = data_shape
        self.action_dim = data_shape['actions'][1]
        self.obs_dim = data_shape['observations'][1]
        D = 0
        for k, v in data_shape.items():
            T, d = v
            D += d
        self.event_shape = [T, D]

        # parameters
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.rho = rho
        self.P_mean = P_mean
        self.P_std = P_std
        self.num_sample_steps = num_sample_steps  # otherwise known as N in the paper
        self.S_churn = S_churn
        self.S_tmin = S_tmin
        self.S_tmax = S_tmax
        self.S_noise = S_noise

    @property
    def device(self):
        return next(self.net.parameters()).device

    # derived preconditioning params - Table 1
    def c_skip(self, sigma):
        return (self.sigma_data ** 2) / (sigma ** 2 + self.sigma_data ** 2)

    def c_out(self, sigma):
        return sigma * self.sigma_data * (self.sigma_data ** 2 + sigma ** 2) ** -0.5

    def c_in(self, sigma):
        return 1 * (sigma ** 2 + self.sigma_data ** 2) ** -0.5

    def c_noise(self, sigma):
        return log(sigma) * 0.25

    # preconditioned network output, equation (7) in the paper
    def preconditioned_network_forward(self, noised_inputs, sigma, clamp=False, cond=None):
        batch, device = noised_inputs.shape[0], noised_inputs.device

        if isinstance(sigma, float):
            #sigma = torch.full((batch,), sigma, device=device)
            sigma = torch.full((batch,), sigma, device=device)

        T,D = self.event_shape
        sigma = sigma.reshape(-1, 1)
        padded_sigma = sigma.reshape(-1, 1, 1).repeat(1, T, D) #1104


        net_out = self.net(
            self.c_in(padded_sigma) * noised_inputs,
            self.c_noise(sigma),
            cond=cond,
        )

        out = self.c_skip(padded_sigma) * noised_inputs + self.c_out(padded_sigma) * net_out

        if clamp:
            out = out.clamp(-1., 1.)

        return out

    # sample schedule, equation (5) in the paper
    def sample_schedule(self, num_sample_steps=None):
        num_sample_steps = default(num_sample_steps, self.num_sample_steps)

        N = num_sample_steps
        inv_rho = 1 / self.rho

        steps = torch.arange(num_sample_steps, device=self.device, dtype=torch.float32)
        sigmas = (self.sigma_max ** inv_rho + steps / (N - 1) * (
                self.sigma_min ** inv_rho - self.sigma_max ** inv_rho)) ** self.rho

        sigmas = F.pad(sigmas, (0, 1), value=0.)  # last step is sigma value of 0.
        return sigmas


    def sample_schedule_with_custom_timestep(self, num_sample_steps=None):

        N = num_sample_steps
        inv_rho = 1 / self.rho

        steps = torch.arange(num_sample_steps, device=self.device, dtype=torch.float32)
        sigmas = (self.sigma_max ** inv_rho + steps / (N - 1) * (
                self.sigma_min ** inv_rho - self.sigma_max ** inv_rho)) ** self.rho

        sigmas = F.pad(sigmas, (0, 1), value=0.)  # last step is sigma value of 0.
        return sigmas



    @torch.no_grad()
    def sample(
            self,
            batch_size: int = 16,
            num_sample_steps: Optional[int] = None,
            clamp: bool = True,
            cond=None,
            disable_tqdm: bool = False,
    ):
        num_sample_steps = default(num_sample_steps, self.num_sample_steps)
        shape = (batch_size, *self.event_shape)

        # get the schedule, which is returned as (sigma, gamma) tuple, and pair up with the next sigma and gamma
        sigmas = self.sample_schedule(num_sample_steps)
        gammas = torch.where(
            (sigmas >= self.S_tmin) & (sigmas <= self.S_tmax),
            min(self.S_churn / num_sample_steps, math.sqrt(2) - 1),
            0.
        )

        sigmas_and_gammas = list(zip(sigmas[:-1], sigmas[1:], gammas[:-1]))

        # inputs are noise at the beginning
        # start from   high variance randn 
        init_sigma = sigmas[0]
        inputs = init_sigma * torch.randn(shape, device=self.device)

        # gradually denoise
        for sigma, sigma_next, gamma in tqdm(sigmas_and_gammas, desc='sampling time step', mininterval=1,
                                             disable=disable_tqdm):
            sigma, sigma_next, gamma = map(lambda t: t.item(), (sigma, sigma_next, gamma))

            eps = self.S_noise * torch.randn(shape, device=self.device)  # stochastic sampling

            sigma_hat = sigma + gamma * sigma
            inputs_hat = inputs + math.sqrt(sigma_hat ** 2 - sigma ** 2) * eps

            denoised_over_sigma = self.score_fn(inputs_hat, sigma_hat, clamp=clamp, cond=cond)
            inputs_next = inputs_hat + (sigma_next - sigma_hat) * denoised_over_sigma

            # second order correction, if not the last timestep
            if sigma_next != 0:
                denoised_prime_over_sigma = self.score_fn(inputs_next, sigma_next, clamp=clamp, cond=cond)
                
                
                inputs_next = inputs_hat + 0.5 * (sigma_next - sigma_hat) * (
                        denoised_over_sigma + denoised_prime_over_sigma)
            

            inputs = inputs_next

        if clamp:
            inputs = inputs.clamp(-1., 1.)
        return self.normalizer.unnormalize(inputs)
    
    @torch.no_grad()
    def sample_back_and_forth(
            self,
            samples: torch.Tensor,
            num_sample_steps: Optional[int] = None,
            clamp: bool = False,
            cond=None,
            cond_state : torch.Tensor = None,
            disable_tqdm: bool = False,
            noise_level : float = 0.5,
            temperature : float = 1.0,
            state_dim : Optional[int] = None
    ):
        samples = samples.to(device=self.device)
        samples = self.normalizer.normalize(samples)
        
        if cond_state is not None:
            for k in cond_state.keys():
                cond_state[k] = cond_state[k].to(device=self.device)
                cond_state[k] = self.normalizer.normalize(cond_state[k])

        num_sample_steps = default(num_sample_steps, self.num_sample_steps)
        T, D = self.event_shape
        shape = samples.shape

        # sigmas = self.noise_distribution(shape[0])
        # padded_sigmas = sigmas.reshape(-1, 1, 1).repeat(1, T, D)

        # noise = torch.randn_like(samples)
        # noised_inputs = samples + padded_sigmas * noise  # alphas are 1. in the paper

        ''' ~1114
        # get the schedule, which is returned as (sigma, gamma) tuple, and pair up with the next sigma and gamma
        sigmas = self.sample_schedule(num_sample_steps)
        gammas = torch.where(
            (sigmas >= self.S_tmin) & (sigmas <= self.S_tmax),
            min(self.S_churn / num_sample_steps, math.sqrt(2) - 1),
            0.
        )

        sigmas_and_gammas = list(zip(sigmas[:-1], sigmas[1:], gammas[:-1]))

        # inputs are noise at the beginning
        # init_sigma = sigmas[0]
        # inputs = init_sigma * torch.randn(shape, device=self.device)
        inputs = samples + sigmas[num_sample_steps//2] * torch.randn(shape, device=self.device)
        '''


        # inputs are now perturbed samples
        # How to perturb K steps in ElucidatedDiffusion
        # code from https://github.com/NVlabs/edm/blob/62072d2612c7da05165d6233d13d17d71f213fee/generate.py#L66
        # schedule = "linear" : sigma(t) = t
        # scaling = "none" : s(t) = 1
        
        # Define time steps in terms of noise level.
        sigmas = self.sample_schedule(num_sample_steps)
        gammas = torch.where(
            (sigmas >= self.S_tmin) & (sigmas <= self.S_tmax),
            min(self.S_churn / num_sample_steps, math.sqrt(2) - 1),
            0.
        )

        
        sigmas_and_gammas = list(zip(sigmas[:-1], sigmas[1:], gammas[:-1]))

        noise_ts = int(num_sample_steps*(1-noise_level))


        if noise_level == 1.0:
            inputs = torch.randn(shape, device=self.device) * sigmas[0]
        else :
            inputs = samples + sigmas[noise_ts] * torch.randn(shape, device=self.device)
        
        # gradually denoise
        for sigma, sigma_next, gamma in tqdm(sigmas_and_gammas[noise_ts:], desc='sampling time step', mininterval=1,
                                             disable=disable_tqdm):
            sigma, sigma_next, gamma = map(lambda t: t.item(), (sigma, sigma_next, gamma))

            eps = self.S_noise * torch.randn(shape, device=self.device)  # stochastic sampling

            sigma_hat = sigma + gamma * sigma
            inputs_hat = inputs + math.sqrt(sigma_hat ** 2 - sigma ** 2) * eps

            denoised_over_sigma = self.score_fn(inputs_hat, sigma_hat, clamp=clamp, cond=cond)
            uncond_denoised_over_sigma = self.score_fn(inputs_hat, sigma_hat, clamp=clamp, cond=None)
            # wo cond
            denoised_over_sigma = uncond_denoised_over_sigma + temperature * (denoised_over_sigma - uncond_denoised_over_sigma)
            
            inputs_next = inputs_hat + (sigma_next - sigma_hat) * denoised_over_sigma

            

            # second order correction, if not the last timestep
            if sigma_next != 0:
                if temperature==0:
                    denoised_prime_over_sigma = self.score_fn(inputs_next, sigma_next, clamp=clamp, cond=None)
                else:
                    denoised_prime_over_sigma = self.score_fn(inputs_next, sigma_next, clamp=clamp, cond=cond)
                
                # uncond_denoised_prime_over_sigma = self.score_fn(inputs_next, sigma_next, clamp=clamp, cond=None)
                # denoised_prime_over_sigma = uncond_denoised_prime_over_sigma + temperature * (denoised_prime_over_sigma - uncond_denoised_prime_over_sigma)
            
                inputs_next = inputs_hat + 0.5 * (sigma_next - sigma_hat) * (
                        denoised_over_sigma + denoised_prime_over_sigma)
                
            # apply condition
            if cond_state is not None:
                inputs_next = apply_conditioning(inputs_next, cond_state, state_dim)


            inputs = inputs_next

        if clamp:
            inputs = inputs.clamp(-1., 1.)
        return self.normalizer.unnormalize(inputs)

    # This is known as 'denoised_over_sigma' in the lucidrains repo.
    def score_fn(
            self,
            x,
            sigma,
            clamp: bool = False,
            cond=None,
    ):
        denoised = self.preconditioned_network_forward(x, sigma, clamp=clamp, cond=cond)
        denoised_over_sigma = (x - denoised) / sigma

        return denoised_over_sigma

    # Adapted from https://github.com/crowsonkb/k-diffusion/blob/master/k_diffusion/sampling.py
    @torch.no_grad()
    def log_likelihood(self, x, atol=1e-4, rtol=1e-4, clamp=False, normalize=True, cond=None):
        # Input to the ODE solver must be in normalized space.
        if normalize:
            x = self.normalizer.normalize(x)
        v = torch.randint_like(x, 2) * 2 - 1
        s_in = x.new_ones([x.shape[0]])
        fevals = 0

        def ode_fn(sigma, x):
            nonlocal fevals
            with torch.enable_grad():
                x = x[0].detach().requires_grad_()
                sigma = sigma * s_in
                padded_sigma = sigma.view(x.shape[0], *([1] * len(self.event_shape)))
                denoised = self.preconditioned_network_forward(x, sigma, clamp=clamp, cond=cond)
                denoised_over_sigma = (x - denoised) / padded_sigma
                fevals += 1
                grad = torch.autograd.grad((denoised_over_sigma * v).sum(), x)[0]
                d_ll = (v * grad).flatten(1).sum(1)
            return denoised_over_sigma.detach(), d_ll
        
        x_min = x, x.new_zeros([x.shape[0]])
        t = x.new_tensor([self.sigma_min, self.sigma_max])
        sol = odeint(ode_fn, x_min, t, atol=atol, rtol=rtol, method='dopri5')
        latent, delta_ll = sol[0][-1], sol[1][-1]
        ll_prior = torch.distributions.Normal(0, self.sigma_max).log_prob(latent).flatten(1).sum(1)

        return ll_prior + delta_ll, {'fevals': fevals}

    # training
    def loss_weight(self, sigma):
        return (sigma ** 2 + self.sigma_data ** 2) * (sigma * self.sigma_data) ** -2

    def noise_distribution(self, batch_size):
        return (self.P_mean + self.P_std * torch.randn((batch_size,), device=self.device)).exp()

    def forward(self, inputs, cond=None):
        inputs = self.normalizer.normalize(inputs)

        batch_size, *event_shape = inputs.shape
        assert event_shape == self.event_shape, f'mismatch of event shape, ' \
                                                f'expected {self.event_shape}, got {event_shape}'

        T, D = self.event_shape
        sigmas = self.noise_distribution(batch_size)
        padded_sigmas = sigmas.reshape(batch_size, 1, 1).repeat(1, T, D)

        noise = torch.randn_like(inputs)
        noised_inputs = inputs + padded_sigmas * noise  # alphas are 1. in the paper
        denoised = self.preconditioned_network_forward(noised_inputs, sigmas, cond=cond)

        mask = torch.ones_like(inputs).to(self.device)
        mask[:, -1, self.obs_dim:] = 0.       

        losses = F.mse_loss(denoised, inputs, reduction='none')
        losses = losses * mask
        losses = reduce(losses, 'b ... -> b', 'mean')
        losses = losses * self.loss_weight(sigmas)
            
        return losses.mean()


class Trainer(object):
    def __init__(
            self,
            diffusion_model,
            dataset: Optional[torch.utils.data.Dataset] = None,
            dataset_nm: Optional[torch.utils.data.Dataset] = None,
            train_batch_size: int = 64,
            gradient_accumulate_every: int = 1,
            train_lr: float = 1e-4,
            lr_scheduler: Optional[str] = None,
            train_num_steps: int = 100000,
            ema_update_every: int = 10,
            ema_decay: float = 0.995,
            adam_betas: Tuple[float, float] = (0.9, 0.99),
            save_and_sample_every: int = 10000,
            weight_decay: float = 0.,
            results_folder: str = './results',
            amp: bool = False,
            fp16: bool = False,
            split_batches: bool = True,
            cond_dim: Optional[int] = 0,
            modalities: List[str] = ['observations', 'actions'],
            reweighted_training: bool = False,
            reward_scale : float = 1.0,
            discounted_return : bool = True,
    ):
        super().__init__()
        self.accelerator = Accelerator(
            split_batches=split_batches,
            mixed_precision='fp16' if fp16 else 'no'
        )
        self.accelerator.native_amp = amp
        self.model = diffusion_model
        self.cond_dim = cond_dim
        self.modalities = modalities
        self.reward_scale = reward_scale
        self.discounted_return = discounted_return
        

        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f'Number of trainable parameters: {num_params}.')

        self.save_and_sample_every = save_and_sample_every
        self.train_num_steps = train_num_steps
        self.gradient_accumulate_every = gradient_accumulate_every

            

        if dataset is not None:
            
            if reweighted_training:
                self.batch_size = train_batch_size

                sampler = define_rewardweighting_sampler(dataset = dataset, 
                                         dataset_nm = dataset_nm,
                                         reward_scale = self.reward_scale)
                dl = DataLoader(dataset, 
                                batch_size=self.batch_size, 
                                sampler=sampler, 
                                pin_memory=True, 
                                num_workers=0)
                dl = self.accelerator.prepare(dl)
                self.dl = cycle(dl)

            else :
                # If dataset size is less than 800K use the small batch size
                #if len(dataset) < int(8e5):
                self.batch_size = train_batch_size
                #else:
                #    self.batch_size = train_batch_size
                print(f'Using batch size: {self.batch_size}')
                # dataset and dataloader
                dl = DataLoader(dataset, 
                                batch_size=self.batch_size,
                                shuffle=True,
                                pin_memory=True, 
                                num_workers=0)
                dl = self.accelerator.prepare(dl)
                self.dl = cycle(dl)

        else:
            # No dataloader, train batch by batch
            self.batch_size = train_batch_size
            self.dl = None

        # optimizer, make sure that the bias and layer-norm weights are not decayed
        no_decay = ['bias', 'LayerNorm.weight', 'norm.weight', '.g']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': weight_decay,
            },
            {
                'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
            },
        ]
        self.opt = torch.optim.AdamW(optimizer_grouped_parameters, lr=train_lr, betas=adam_betas)

        # for logging results in a folder periodically
        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta=ema_decay, update_every=ema_update_every)
            self.results_folder = pathlib.Path(results_folder)
            self.results_folder.mkdir(exist_ok=True)

        # step counter state
        self.step = 0

        if lr_scheduler == 'linear':
            print('using LambdaLR scheduler')
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.opt,
                lambda step: max(0, 1 - step / train_num_steps)
            )
        elif lr_scheduler == 'cosine':
            print('using CosineAnnealingLR scheduler')
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.opt,
                T_max=train_num_steps
            )
        elif lr_scheduler == 'cosinewarmup':
            print('using CosineAnnealingWarmRestarts scheduler')
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.opt,
                T_0=train_num_steps // 10,
            )
        else:
            self.lr_scheduler = None

        # prepare model, dataloader, optimizer with accelerator
        self.model, self.opt, self.lr_scheduler = self.accelerator.prepare(self.model, self.opt, self.lr_scheduler)

        # .module is needed when using DDP
        try : 
            self.model.module.normalizer.to(self.accelerator.device)
        except :
            self.model.normalizer.to(self.accelerator.device)    
        
        if self.accelerator.is_main_process:
            self.ema.ema_model.normalizer.to(self.accelerator.device)

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone: Optional[str] = None, ckpt_path: Optional[str] = None):
        accelerator = self.accelerator
        device = accelerator.device

        if ckpt_path is not None:
            data = torch.load(ckpt_path, map_location=device)
        else:    
            data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data['ema'])

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    # Train for the full number of steps.
    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        with tqdm(initial=self.step, total=self.train_num_steps, disable=not accelerator.is_main_process) as pbar:
            while self.step < self.train_num_steps:
                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    states, actions, rewards, next_states, returns, time_steps, terminals, rtg = next(self.dl)
                    if self.discounted_return:
                        cond = returns[:,0,:].to(device)
                    else:
                        trajectory_reward = rewards.squeeze().sum(dim=-1)
                        cond = trajectory_reward.reshape(-1, 1).to(device)
                    cond *= self.reward_scale
                    
                    
                    data = []
                    for mod in self.modalities:
                        if mod == 'observations':
                            data.append(states)
                        elif mod == 'actions':
                            data.append(actions)
                        elif mod == 'rewards':
                            data.append(rewards) 
                    last_state = next_states[:,-1, None,:]
                    last_action = torch.zeros_like(actions[:,-1, None,:])
                    last_reward = torch.zeros_like(rewards[:,-1, None,:])
                    last_transition = torch.cat([last_state, last_action, last_reward], dim=-1)        
                        
                    data = torch.cat(data, dim=-1).to(device)
                    data = torch.cat([data, last_transition], dim=1)


                    with self.accelerator.autocast():
                        if self.cond_dim > 0 :
                            loss = self.model(data, cond=cond)
                        else :
                            loss = self.model(data)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    self.accelerator.backward(loss)

                accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                pbar.set_description(f'loss: {total_loss:.4f}')
                if accelerator.is_local_main_process:
                    wandb.log({
                        'step': self.step,
                        'loss': total_loss,
                        'lr': self.opt.param_groups[0]['lr']
                    })

                accelerator.wait_for_everyone()

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.to(device)
                    self.ema.update()

                    if self.step != 0 and self.step % self.save_and_sample_every == 0:
                        self.save(self.step)

                pbar.update(1)

                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

        accelerator.print('training complete')

    # Allow user to pass in external data.
    def train_on_batch(
            self,
            data: torch.Tensor,
            use_wandb=True,
            splits=1,  # number of splits to split the batch into
            **kwargs,
    ):
        accelerator = self.accelerator
        device = accelerator.device
        data = data.to(device)

        total_loss = 0.
        if splits == 1:
            with self.accelerator.autocast():
                loss = self.model(data, **kwargs)
                total_loss += loss.item()
            self.accelerator.backward(loss)
        else:
            assert splits > 1 and data.shape[0] % splits == 0
            split_data = torch.split(data, data.shape[0] // splits)

            for idx, d in enumerate(split_data):
                with self.accelerator.autocast():
                    # Split condition as well
                    new_kwargs = {}
                    for k, v in kwargs.items():
                        if isinstance(v, torch.Tensor):
                            new_kwargs[k] = torch.split(v, v.shape[0] // splits)[idx]
                        else:
                            new_kwargs[k] = v

                    loss = self.model(d, **new_kwargs)
                    loss = loss / splits
                    total_loss += loss.item()
                self.accelerator.backward(loss)

        accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
        if use_wandb:
            wandb.log({
                'step': self.step,
                'loss': total_loss,
                'lr': self.opt.param_groups[0]['lr'],
            })

        accelerator.wait_for_everyone()

        self.opt.step()
        self.opt.zero_grad()

        accelerator.wait_for_everyone()

        self.step += 1
        if accelerator.is_main_process:
            self.ema.to(device)
            self.ema.update()

            if self.step != 0 and self.step % self.save_and_sample_every == 0:
                self.save(self.step)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return total_loss


class REDQTrainer(Trainer):
    def __init__(
            self,
            diffusion_model,
            train_batch_size: int = 16,
            gradient_accumulate_every: int = 1,
            train_lr: float = 1e-4,
            lr_scheduler: Optional[str] = None,
            train_num_steps: int = 100000,
            ema_update_every: int = 10,
            ema_decay: float = 0.995,
            adam_betas: Tuple[float, float] = (0.9, 0.99),
            save_and_sample_every: int = 10000,
            weight_decay: float = 0.,
            results_folder: str = './results',
            amp: bool = False,
            fp16: bool = False,
            split_batches: bool = True,
            model_terminals: bool = False,
    ):
        super().__init__(
            diffusion_model,
            dataset=None,
            train_batch_size=train_batch_size,
            gradient_accumulate_every=gradient_accumulate_every,
            train_lr=train_lr,
            lr_scheduler=lr_scheduler,
            train_num_steps=train_num_steps,
            ema_update_every=ema_update_every,
            ema_decay=ema_decay,
            adam_betas=adam_betas,
            save_and_sample_every=save_and_sample_every,
            weight_decay=weight_decay,
            results_folder=results_folder,
            amp=amp,
            fp16=fp16,
            split_batches=split_batches,
        )

        self.model_terminals = model_terminals

    def train_from_redq_buffer(self, buffer , num_steps: Optional[int] = None):
        num_steps = num_steps or self.train_num_steps
        for j in range(num_steps):
            b = buffer.sample_batch(self.batch_size)
            obs = b['obs1']
            next_obs = b['obs2']
            actions = b['acts']
            rewards = b['rews'][:, None]
            done = b['done'][:, None]
            data = [obs, actions, rewards, next_obs]
            if self.model_terminals:
                data.append(done)
            data = np.concatenate(data, axis=1)
            data = torch.from_numpy(data).float()
            loss = self.train_on_batch(data, use_wandb=False)
            if j % 1000 == 0:
                print(f'[{j}/{num_steps}] loss: {loss:.4f}')

    def update_normalizer(self, buffer, device=None):
        data = make_inputs_from_replay_buffer(buffer, self.model_terminals)
        data = torch.from_numpy(data).float()
        self.model.normalizer.reset(data)
        self.ema.ema_model.normalizer.reset(data)
        if device:
            self.model.normalizer.to(device)
            self.ema.ema_model.normalizer.to(device)
