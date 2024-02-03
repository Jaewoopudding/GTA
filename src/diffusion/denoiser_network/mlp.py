# Denoiser networks for diffusion.


from typing import Optional

import torch
import torch.nn as nn
import torch.optim
from torch.nn import functional as F
from torch.distributions import Bernoulli

import pyrootutils

path = pyrootutils.find_root(search_from=__file__, indicator=".aug-project-root")
pyrootutils.set_root(path = path,
                     project_root_env_var = True,
                     dotenv = True,
                     pythonpath = True)

from src.diffusion.denoiser_network.helpers import SinusoidalPosEmb, RandomOrLearnedSinusoidalPosEmb

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d



# Residual MLP of the form x_{L+1} = MLP(LN(x_L)) + x_L
class ResidualBlock(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, activation: str = "relu", layer_norm: bool = True):
        super().__init__()
        self.linear = nn.Linear(dim_in, dim_out, bias=True)
        if layer_norm:
            self.ln = nn.LayerNorm(dim_in)
        else:
            self.ln = torch.nn.Identity()
        self.activation = getattr(F, activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.linear(self.activation(self.ln(x)))


class ResidualMLP(nn.Module):
    def __init__(
            self,
            input_dim: int,
            width: int,
            depth: int,
            output_dim: int,
            activation: str = "relu",
            layer_norm: bool = False,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, width),
            *[ResidualBlock(width, width, activation, layer_norm) for _ in range(depth)],
            nn.LayerNorm(width) if layer_norm else torch.nn.Identity(),
        )

        self.activation = getattr(F, activation)
        self.final_linear = nn.Linear(width, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.final_linear(self.activation(self.network(x)))


class ResidualMLPDenoiser(nn.Module):
    def __init__(
            self,
            d_in: int,
            dim_t: int = 128,
            mlp_width: int = 1024,
            num_layers: int = 6,
            learned_sinusoidal_cond: bool = False,
            random_fourier_features: bool = True,
            learned_sinusoidal_dim: int = 16,
            activation: str = "relu",
            layer_norm: bool = True,
            cond_dim: Optional[int] = 0,
            condition_dropout : float = 0.25,
            use_dropout : bool = True,
            force_dropout : bool = False,
    ):
        super().__init__()

        self.use_dropout = use_dropout
        self.force_dropout = force_dropout
        self.cond_dim = cond_dim

        self.proj = nn.Sequential(
            nn.Linear(d_in, 2*dim_t),
            nn.SiLU(),
            nn.Linear(2*dim_t, dim_t)
        )
        # time embeddings
        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features
        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim_t)
            fourier_dim = dim_t

        if cond_dim > 0:
            self.cond_mlp = nn.Sequential(
                #SinusoidalPosEmb(cond_dim),
                nn.Linear(cond_dim, 2*dim_t),
                nn.SiLU(),
                nn.Linear(2*dim_t, dim_t)
            )
            self.mask_dist = Bernoulli(probs=1-condition_dropout)
            dim_t = 2*dim_t 
        
        
        self.residual_mlp = ResidualMLP(
            input_dim=dim_t,
            width=mlp_width,
            depth=num_layers,
            output_dim=d_in,
            activation=activation,
            layer_norm=layer_norm,
        )

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, dim_t)
        )



    def forward(
            self,
            x: torch.Tensor,
            timesteps: torch.Tensor,
            cond=None,
    ) -> torch.Tensor:
        '''
        x : (batch, horizon, transition)
        timesteps : (batch, 1)
        '''

        # Reshape timesteps to (batch, horizon, 1)
        timesteps = timesteps.reshape(-1, 1, 1).repeat(1, x.shape[1], 1)
        # Reshape cond to (batch, horizon, 1)
        cond = cond.reshape(-1, 1, 1).repeat(1, x.shape[1], 1)

        time_embed = self.time_mlp(timesteps)

        x = self.proj(x)

        if self.cond_dim > 0:
            cond = self.cond_mlp(cond)
            if self.use_dropout or self.force_dropout:
                mask = self.mask_dist.sample(cond.shape).to(cond.device)
                cond = cond * mask
            if self.force_dropout:
                cond = cond * 0
            x = torch.cat([x, cond], dim=-1)

        x += time_embed

        return self.residual_mlp(x)


