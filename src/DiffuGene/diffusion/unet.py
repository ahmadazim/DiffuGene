from diffusers import UNet2DModel, DDPMScheduler
import torch
import torch.nn as nn
import math
import numpy as np
import random

class SinusoidalEmbeddings(nn.Module):
    def __init__(self, time_steps:int, embed_dim: int):
        super().__init__()
        position = torch.arange(time_steps).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim))
        embeddings = torch.zeros(time_steps, embed_dim, requires_grad=False)
        embeddings[:, 0::2] = torch.sin(position * div)
        embeddings[:, 1::2] = torch.cos(position * div)
        # self.embeddings = embeddings
        self.register_buffer('embeddings', embeddings)

    def forward(self, x, t):
        return self.embeddings[t]#.to(x.device)


class LatentUNET2D(nn.Module):
    """
    2D UNet for 16x16x16 latent embeddings using Diffusers' UNet2DModel.
    Input/output: (B,16,16,16)
    """
    def __init__(
        self,
        input_channels: int = 16,
        output_channels: int = 16,
        time_steps: int = 1000,
        layers_per_block: int = 2,
    ):
        super().__init__()
        # sinusoidal time embeddings
        self.time_embed = SinusoidalEmbeddings(time_steps, embed_dim=512)

        # UNet2D backbone
        # self.unet = UNet2DModel(
        #     sample_size=(16, 16),
        #     in_channels=input_channels,
        #     out_channels=output_channels,
        #     layers_per_block=layers_per_block,
        #     block_out_channels=[256, 256, 512, 512],
        #     down_block_types   = ["DownBlock2D",
        #                           "AttnDownBlock2D", 
        #                           "AttnDownBlock2D", 
        #                           "DownBlock2D"],
        #     up_block_types     = ["UpBlock2D",
        #                           "AttnUpBlock2D",
        #                           "AttnUpBlock2D",
        #                           "UpBlock2D"],
        #     attention_head_dim = 64
        # )
        
        # initial channel expansion 16 -> 128
        self.input_proj = nn.Conv2d(input_channels, 128, kernel_size=1)

        self.unet = UNet2DModel(
            sample_size=(16, 16),
            in_channels=128,
            out_channels=128,
            layers_per_block=layers_per_block,
            # 4 spatial stages with channel multipliers 1,2,3,4
            block_out_channels=[128, 256, 384, 512],
            down_block_types=[
                 "DownBlock2D",         # spatial:16×16→8×8,   channels:128→128
                 "AttnDownBlock2D",     # spatial:8×8→4×4,     channels:128→256 (self-attn)
                 "DownBlock2D",         # spatial:4×4→2×2,     channels:256→384
                 "AttnDownBlock2D",     # spatial:2×2→1×1,     channels:384→512 (self-attn)
             ],
             up_block_types=[
                 "AttnUpBlock2D",
                 "UpBlock2D",
                 "AttnUpBlock2D",
                 "UpBlock2D",
             ],
             mid_block_type="UNetMidBlock2DSelfAttn",
             attention_head_dim=64,
        )
        # final channel contraction 128 -> 16
        self.output_proj = nn.Conv2d(128, output_channels, kernel_size=1)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        x: (B,16,16,16)
        t: (B,)
        """
        # add time embeddings if needed by custom layers
        _ = self.time_embed(x, t)
        # # Diffusers UNet2DModel expects (B,C,H,W)
        # return self.unet(x, t).sample
        # expand channels, run through UNet, then project back
        x = self.input_proj(x)
        x = self.unet(x, t).sample
        x = self.output_proj(x)
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
