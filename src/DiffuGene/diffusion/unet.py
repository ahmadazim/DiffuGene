from diffusers import UNet2DConditionModel, DDPMScheduler
import torch
import torch.nn as nn
import math
import numpy as np
import random

class LatentUNET2D(nn.Module):
    """
    2D UNet for 512x512x4 latent embeddings using Diffusers UNet2DConditionModel.
    Input/output: (B,4,512,512)
    """
    def __init__(
        self,
        input_channels: int = 4,
        output_channels: int = 4,
        cond_dim: int = 10,
        time_steps: int = 1000,
        layers_per_block: int = 2,
        base_channels: int = 128,
    ):
        super().__init__()

        # initial channel expansion 4 -> base_channels
        self.base_channels = base_channels
        self.input_proj = nn.Conv2d(input_channels, base_channels, kernel_size=3, padding=1)

        # embed covariates into the UNet's cross‑attention space
        self.cond_emb = nn.Sequential(
            nn.Linear(cond_dim, base_channels),
            nn.SiLU(),
            nn.Linear(base_channels, base_channels),
        )

        # learnable null condition embedding for classifier‑free guidance
        self.null_cond_emb = nn.Parameter(torch.randn(1, base_channels))

        # 6 spatial stages with multipliers [1,1,2,3,4,4] on base=base_channels
        self.unet = UNet2DConditionModel(
            sample_size=(512, 512),
            in_channels=base_channels,
            out_channels=base_channels,
            layers_per_block=layers_per_block,
            block_out_channels=[base_channels, base_channels, 2*base_channels, 3*base_channels, 4*base_channels, 4*base_channels],
            down_block_types=[
                "DownBlock2D",     # 512→256, 128→128 (e.g.)
                "DownBlock2D",     # 256→128, 128→128
                "DownBlock2D",     # 128→64,  128→256
                "DownBlock2D",     # 64→32,    256→384 
                "AttnDownBlock2D", # 32→16,    384→512 (self‑attention)
                "AttnDownBlock2D", # 16→8,     512→512 (self‑attention)
            ],
            up_block_types=[
                "AttnUpBlock2D",   # 8→16,     512→512 (self‑attention)
                "AttnUpBlock2D",   # 16→32,    512→384 (self‑attention)
                "UpBlock2D",       # 32→64,    384→256
                "UpBlock2D",       # 64→128,   256→128
                "UpBlock2D",       # 128→256,  128→128
                "UpBlock2D",       # 256→512,  128→128
            ],
            mid_block_type="UNetMidBlock2DCrossAttn",
            attention_head_dim=64,
            cross_attention_dim=128,
        )

        # final channel contraction 128 -> 4
        self.output_proj = nn.Conv2d(base_channels, output_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor):
        """
        x: (B,4,512,512) latent map
        t: (B,)         timestep
        c: (B,cond_dim) covariates
        """
        x = self.input_proj(x)               # (B,base_channels,512,512)
        cond_emb = self.cond_emb(c)          # (B,base_channels)
        cond_emb = cond_emb.unsqueeze(1)     # (B,1,base_channels) – add sequence dimension
        x = self.unet(x, t, encoder_hidden_states=cond_emb).sample
        x = self.output_proj(x)              # (B,4,512,512)
        return x


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def noise_pred_loss(eps_pred: torch.Tensor,
                    eps_true: torch.Tensor,
                    t: torch.Tensor,
                    scheduler: DDPMScheduler,
                    simplified_loss: bool = True) -> torch.Tensor:
    # fetch β_t and cumulative ᾱ_t, broadcast to match (B,C,…)
    beta_t    = scheduler.betas[t].view(-1, *([1] * (eps_true.dim()-1)))
    alpha_bar = scheduler.alphas_cumprod[t].view(-1, *([1] * (eps_true.dim()-1)))
    # instantaneous α_t = 1 - β_t, and σ_t^2 = β_t
    alpha_inst = 1 - beta_t
    sigma2_t   = beta_t

    # DDPM loss weight: β_t^2 / (σ_t^2 * α_t * (1 - ᾱ_t))
    if simplified_loss: 
        weight = 1.0
    else:
        weight = beta_t.pow(2) / (sigma2_t * alpha_inst * (1 - alpha_bar))
    return (weight * (eps_pred - eps_true).pow(2)).mean()

def v_pred_loss(v_pred: torch.Tensor,
                x0: torch.Tensor,
                eps: torch.Tensor,
                t: torch.Tensor,
                scheduler: DDPMScheduler,
                gamma: float = 10.0) -> torch.Tensor:
    """
    MSE on v = sqrt(alpha_bar)*eps - sqrt(1-alpha_bar)*x0, with light Min-SNR weight:
        w(t) = SNR / (SNR + gamma),  SNR = alpha_bar / (1-alpha_bar)
    higher gamma keeps weighting close to 1
    """
    a_bar = scheduler.alphas_cumprod[t].view(-1, *([1] * (x0.dim()-1)))
    a_s   = torch.sqrt(a_bar)
    sig   = torch.sqrt(1.0 - a_bar)
    v_tgt = a_s * eps - sig * x0
    snr   = a_bar / (1.0 - a_bar + 1e-12)
    w     = (snr / (snr + gamma)).view(-1, *([1] * (x0.dim()-1)))
    return (w * (v_pred - v_tgt).pow(2)).mean()