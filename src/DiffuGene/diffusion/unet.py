from diffusers import UNet2DConditionModel, DDPMScheduler
import torch
import torch.nn as nn
import math
import numpy as np
import random

class LatentUNET2D(nn.Module):
    """
    2D UNet for 128x128x64 latent embeddings using Diffusers UNet2DConditionModel.
    Input/output: (B,64,128,128)
    """
    def __init__(
        self,
        input_channels: int = 64,
        output_channels: int = 64,
        cond_dim: int = 10,
        time_steps: int = 1000,
        layers_per_block: int = 2,
        base_channels: int = 256,
    ):
        super().__init__()

        # initial channel expansion 64 -> base_channels
        self.base_channels = base_channels
        self.input_proj = nn.Conv2d(input_channels, base_channels, kernel_size=3, padding=1)

        # embed covariates into the UNet's cross‑attention space
        self.cross_attention_dim = 128
        self.cond_emb = nn.Sequential(
            nn.Linear(cond_dim, base_channels),
            nn.SiLU(),
            nn.Linear(base_channels, self.cross_attention_dim),
        )

        # learnable null condition embedding for classifier‑free guidance
        self.null_cond_emb = nn.Parameter(torch.randn(1, self.cross_attention_dim))

        # 6 spatial stages with multipliers [1,1,2,3,4,4] on base=base_channels
        self.unet = UNet2DConditionModel(
            sample_size=(128, 128),
            in_channels=base_channels,
            out_channels=base_channels,
            layers_per_block=layers_per_block,
            # 6 spatial stages with multipliers [1, 1, 2, 3, 4, 4] on base=256
            block_out_channels=[base_channels, base_channels, 2*base_channels, 3*base_channels, 4*base_channels, 4*base_channels],
            down_block_types=[
                "DownBlock2D",     # 64×64 → 32×32,   256→256
                "DownBlock2D",     # 32×32 → 16×16,   256→256
                "DownBlock2D",     # 16×16 →  8× 8,   256→512
                "AttnDownBlock2D", #  8× 8 →  4× 4,   512→768 (self-attn)
                "DownBlock2D",     #  4× 4 →  2× 2,   768→1024
                "AttnDownBlock2D", #  2× 2 →  1× 1,   1024→1024 (self-attn)
            ],
            up_block_types=[
                "AttnUpBlock2D",   #  1× 1 →  2× 2,  1024→1024 (self-attn)
                "UpBlock2D",       #  2× 2 →  4× 4,  1024→768
                "AttnUpBlock2D",   #  4× 4 →  8× 8,  768→512 (self-attn)
                "UpBlock2D",       #  8× 8 → 16×16,  512→256
                "UpBlock2D",       # 16×16 → 32×32,  256→256
                "UpBlock2D",       # 32×32 → 64×64,  256→256
            ],
            mid_block_type="UNetMidBlock2DCrossAttn",
            attention_head_dim=64,
            cross_attention_dim=self.cross_attention_dim,
        )

        # final channel contraction base_channels -> output_channels
        self.output_proj = nn.Conv2d(base_channels, output_channels, kernel_size=3, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor,
        cfg_drop_prob: float = None,
        return_pair: bool = False,
    ):
        """
        x: (B,C,H,W) latent map
        t: (B,)         timestep
        c: (B,cond_dim) covariates
        cfg_drop_prob: optional classifier-free guidance drop probability. When
            provided, the forward builds a 2× batch (uncond + cond) and returns
            the two outputs for loss computation.
        return_pair: when True with cfg_drop_prob, return (out_uncond, out_cond).
                     Otherwise return the conditional output only.
        """
        if c is None:
            raise ValueError("Conditional UNet forward expects covariates `c`.")

        if cfg_drop_prob is None:
            # Standard conditional forward (single branch)
            x = self.input_proj(x)               # (B,base_channels,H,W)
            cond_emb = self.cond_emb(c)          # (B,base_channels)
            cond_emb = cond_emb.unsqueeze(1)     # (B,1,base_channels) – add sequence dimension
            x = self.unet(x, t, encoder_hidden_states=cond_emb).sample
            x = self.output_proj(x)              # (B,C,H,W)
            return x

        # Classifier-free guidance training: build dropped + full conditional batches
        B = x.size(0)
        cond_emb = self.cond_emb(c).unsqueeze(1)  # (B,1,hidden)
        hidden_dim = cond_emb.size(-1)
        null_emb = self.null_cond_emb.unsqueeze(0).expand(B, 1, hidden_dim)
        if null_emb.dtype != cond_emb.dtype:
            null_emb = null_emb.to(dtype=cond_emb.dtype)

        # Randomly drop some examples
        mask = (torch.rand(B, device=x.device) < cfg_drop_prob)
        dropped_emb = cond_emb.clone()
        dropped_emb[mask] = null_emb[mask]

        # Double the batch for uncond/cond passes
        x_in = torch.cat([x, x], dim=0)
        t_in = torch.cat([t, t], dim=0)
        emb_in = torch.cat([dropped_emb, cond_emb], dim=0)

        h = self.input_proj(x_in)
        out = self.unet(h, t_in, encoder_hidden_states=emb_in).sample
        out = self.output_proj(out)
        out_uncond, out_cond = out.chunk(2, dim=0)

        if return_pair:
            return out_uncond, out_cond
        return out_cond


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
                scheduler: DDPMScheduler) -> torch.Tensor:
    """
    Standard v-prediction MSE:
        v = sqrt(alpha_bar) * eps − sqrt(1 − alpha_bar) * x0
    """
    a_bar = scheduler.alphas_cumprod[t].view(-1, *([1] * (x0.dim()-1)))
    sqrt_ab = torch.sqrt(a_bar)
    sqrt_one_minus_ab = torch.sqrt(1.0 - a_bar)
    v_target = sqrt_ab * eps - sqrt_one_minus_ab * x0
    return (v_pred - v_target).pow(2).mean()