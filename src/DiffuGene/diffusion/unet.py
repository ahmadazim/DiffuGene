from diffusers import UNet2DConditionModel, DDPMScheduler
import torch
import torch.nn as nn
import math
import numpy as np
import random

class LatentUNET2D(nn.Module):
    """
    2D UNet for 64x64x64 latent embeddings using Diffusers' UNet2DConditionModel.
    Input/output: (B,64,64,64)
    """
    def __init__(
        self,
        input_channels: int = 64,
        output_channels: int = 64,
        cond_dim: int = 10,
        time_steps: int = 1000,
        layers_per_block: int = 2,
    ):
        super().__init__()

        # initial channel expansion 64 -> 256
        self.input_proj = nn.Conv2d(input_channels, 256, kernel_size=3, padding=1)
        
        # embed covariates into the UNet's cross-attention space
        self.cond_emb = nn.Sequential(
            nn.Linear(cond_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
        )
        
        # Learnable null condition embedding for classifier-free guidance
        self.null_cond_emb = nn.Parameter(torch.randn(1, 256))

        self.unet = UNet2DConditionModel(
            sample_size=(64, 64),
            in_channels=256,
            out_channels=256,
            layers_per_block=layers_per_block,
            # 6 spatial stages with multipliers [1, 1, 2, 3, 4, 4] on base=256
            block_out_channels=[256, 256, 512, 768, 1024, 1024],
            down_block_types=[
                "DownBlock2D",     # 64×64 → 32×32,   256→256
                "DownBlock2D",     # 32×32 → 16×16,   256→256
                "DownBlock2D",     # 16×16 →  8× 8,   256→512
                "AttnDownBlock2D", #  8× 8 →  4× 4,   512→768 (self-attn)
                "DownBlock2D",     #  4× 4 →  2× 2,   768→1024
                "AttnDownBlock2D", #  2× 2 →  1× 1,  1024→1024 (self-attn)
            ],
            up_block_types=[
                "AttnUpBlock2D",   #  1× 1 →  2× 2,  1024→1024 (self-attn)
                "UpBlock2D",       #  2× 2 →  4× 4, 1024→768
                "AttnUpBlock2D",   #  4× 4 →  8× 8,  768→512 (self-attn)
                "UpBlock2D",       #  8× 8 → 16×16, 512→256
                "UpBlock2D",       # 16×16 → 32×32, 256→256
                "UpBlock2D",       # 32×32 → 64×64, 256→256
            ],
            mid_block_type="UNetMidBlock2DCrossAttn",
            attention_head_dim=64,
            cross_attention_dim=256,
        )
        # final channel contraction 256 -> 16
        self.output_proj = nn.Conv2d(256, output_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor):
        """
        x: (B,64,64,64) latent map  
        t: (B,)         timestep  
        c: (B,cond_dim) covariates  
        """
        x = self.input_proj(x)                # (B,256,64,64)
        cond_emb = self.cond_emb(c)           # (B,256)
        cond_emb = cond_emb.unsqueeze(1)      # (B,1,256) - add sequence dimension for cross-attention
        x = self.unet(x, t, encoder_hidden_states=cond_emb).sample
        x = self.output_proj(x)               # (B,64,64,64)
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