"""
Genomic Latent Diffusion Comparison
===================================

This module implements three generative pipelines for discrete genotype data:

1. **UNet latent diffusion** - uses a pre-trained VAE to map genotypes into
   2D latent feature maps (e.g., 16x16x64).  A convolutional U-Net
   backbone is trained to predict the velocity of the noising process for
   diffusion models.  DDIM sampling is used at inference time.

2. **Diffusion Transformer (DiT)** - learns a 1D autoencoder to compress
   genotype sequences directly into a sequence of latent tokens.  A
   Transformer model replaces the convolutional U-Net in the diffusion
   pipeline.  This design is motivated by recent work showing that
   replacing U-Nets with Transformers operating on latent patches can
   improve generative performance.

3. **Scalable Interpolant Transformer (SiT)** - builds upon DiT by
   training the Transformer under a flow-matching objective.  The
   interpolant framework connects data and noise through flexible
   continuous-time processes; learning the velocity field directly
   generalizes diffusion models and often yields better generative
   performance.  The velocity loss integrates over
   time and encourages the model to estimate the drift of the probability
   flow ODE.

This code illustrates how to set up these pipelines in PyTorch using
standard libraries such as diffusers.  The implementation is meant to
be instructional and is not optimized for production use.  It includes
placeholder functions where domain-specific metrics (e.g. minor allele
frequency and linkage disequilibrium penalties) would be implemented.
"""

import math
import copy
import glob
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any
import os
import h5py
from tqdm import tqdm
import numpy as np
import pandas as pd
import bisect
from typing import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from diffusers import UNet2DModel, DDPMScheduler, DDIMScheduler

class LatentDataset(Dataset):
    """Simple dataset wrapper for pre-computed latents.

    Args:
        latent_path: path to a tensor file containing latents of shape
          (N, C, H, W) saved via torch.save.
    """

    def __init__(self, latent_path: str) -> None:
        super().__init__()
        self.latents = torch.load(latent_path)
        if self.latents.dim() != 4:
            raise ValueError(
                f"Latents must be a 4-D tensor of shape (N, C, H, W). Received {self.latents.shape}."
            )

    def __len__(self) -> int:
        return self.latents.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.latents[idx]


class GenotypeDataset(Dataset):
    """Dataset wrapper for one-hot encoded genotype sequences.

    Each item is an integer tensor of shape (L,) encoding dosages
    (0, 1 or 2).  Additional covariate information can be returned if
    required for conditional generation.

    Args:
        sequences: tensor of shape (N, L) containing genotype calls.
        covariates: optional tensor of shape (N, C_cov) with external
          conditioning variables.
    """

    def __init__(self, sequences: torch.Tensor, covariates: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        if sequences.dim() != 2:
            raise ValueError("Genotype sequences must be 2-D (NxL).")
        self.sequences = sequences.long()
        self.covariates = covariates

    def __len__(self) -> int:
        return self.sequences.shape[0]

    def __getitem__(self, idx: int):
        x = self.sequences[idx]
        if self.covariates is None:
            return x
        else:
            return x, self.covariates[idx]


def positional_encoding(length: int, dim: int) -> torch.Tensor:
    """Generates a 1D sinusoidal positional encoding.

    Args:
        length: number of positions (tokens).
        dim: embedding dimension (must be even).

    Returns:
        A tensor of shape (length, dim) containing positional encodings.
    """
    if dim % 2 != 0:
        raise ValueError("Positional encoding dimension must be even.")
    pe = torch.zeros(length, dim)
    position = torch.arange(0, length, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding followed by a linear projection.

    This module converts continuous (or discrete) timesteps into a learned
    vector representation that can be added to token embeddings.  The
    design mirrors the Fourier time embedding used in diffusion models.
    """

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.linear = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Computes a positional/fourier embedding for the given timesteps.

        Args:
            t: tensor of shape (batch,) with values in [0, 1].

        Returns:
            A tensor of shape (batch, 1, embedding_dim) ready to add to
            token embeddings.
        """
        # Convert to shape (batch, embedding_dim)
        half_dim = self.embedding_dim // 2
        # Create frequencies
        freqs = torch.exp(
            -torch.arange(0, half_dim, dtype=torch.float32, device=t.device) * (math.log(10000.0) / half_dim)
        )
        # Expand t
        t = t.unsqueeze(1)  # (B, 1)
        sinusoid = torch.cat([torch.sin(t * freqs), torch.cos(t * freqs)], dim=1)
        return self.linear(sinusoid).unsqueeze(1)


class ConditionEmbedding(nn.Module):
    """Simple conditioning embedding projecting covariates to model dimension."""

    def __init__(self, input_dim: int, embed_dim: int) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x).unsqueeze(1)  # (B, 1, embed_dim)


class TransformerDiffusionModel(nn.Module):
    """A simple Transformer-based denoiser operating on token sequences.

    This model acts as the backbone for both DiT and SiT experiments.  It
    consists of a stack of Transformer encoder layers that process the
    sequence of latent tokens.  Conditioning on time and covariates is
    achieved by adding their embeddings to the token embeddings before
    passing them through the encoder.

    Args:
        token_dim: dimensionality of each token.
        num_layers: number of Transformer encoder layers.
        num_heads: number of attention heads per layer.
        mlp_ratio: width expansion factor in the feed-forward network.
    """

    def __init__(self, token_dim: int, num_layers: int = 8, num_heads: int = 8, mlp_ratio: int = 4) -> None:
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=num_heads,
            dim_feedforward=token_dim * mlp_ratio,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        # Final linear projection to predict velocity/noise with same dimension
        self.out_proj = nn.Linear(token_dim, token_dim)

    def forward(
        self,
        tokens: torch.Tensor,
        time_emb: torch.Tensor,
        cond_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute model output for given tokens and conditioning.

        Args:
            tokens: input sequence of shape (B, N, D).
            time_emb: time embedding of shape (B, 1, D).
            cond_emb: optional covariate embedding of shape (B, 1, D).

        Returns:
            Predicted noise/velocity of shape (B, N, D).
        """
        h = tokens + time_emb
        if cond_emb is not None:
            h = h + cond_emb
        h = self.encoder(h)
        return self.out_proj(h)


def find_best_ck(x: int, max_c: int = 5, *, min_k: Optional[int] = None) -> Tuple[int, int]:
    """Pick (c,k) so c*2^k ~= x (optionally enforce k >= min_k)."""
    best_c, best_k = 1, 0
    best_err = abs(1.0 - x)
    for c in range(1, max_c + 1):
        k = 0
        while True:
            val = c * (1 << k)
            if abs(val - x) < best_err:
                best_err = abs(val - x)
                best_c, best_k = c, k
            if val > 10 * x:
                break
            k += 1
    # Apply lower bound on k
    if min_k is not None and best_k < int(min_k):
        best_k = int(min_k)
        c_est = int(round(float(x) / float(1 << best_k)))
        best_c = min(max(1, c_est), max_c)
    return best_c, best_k


def pixel_unshuffle_1d(x: torch.Tensor, factor: int = 2) -> torch.Tensor:
    """1D pixel-unshuffle: (B,C,L) -> (B,C*factor,L//factor). factor must divide L."""
    B, C, L = x.shape
    if L % factor != 0:
        raise ValueError(f"Sequence length {L} is not divisible by factor {factor}.")
    new_L = L // factor
    x = x.view(B, C, new_L, factor)
    x = x.permute(0, 1, 3, 2).contiguous()  # (B, C, factor, new_L)
    x = x.view(B, C * factor, new_L)
    return x


def pixel_shuffle_1d(x: torch.Tensor, factor: int = 2) -> torch.Tensor:
    """Inverse of pixel_unshuffle_1d: (B,C,L) -> (B,C//factor,L*factor). C must be divisible by factor."""
    B, C, L = x.shape
    if C % factor != 0:
        raise ValueError(f"Number of channels {C} is not divisible by factor {factor}.")
    new_C = C // factor
    x = x.view(B, new_C, factor, L)
    x = x.permute(0, 1, 3, 2).contiguous()  # (B, new_C, L, factor)
    x = x.view(B, new_C, L * factor)
    return x


class PixelUnshuffleDownSample1D(nn.Module):
    """Shortcut: 1D pixel_unshuffle without averaging. Requires out_channels == in_channels * factor."""

    def __init__(self, in_channels: int, out_channels: int, factor: int = 2) -> None:
        super().__init__()
        assert out_channels == in_channels * factor, (
            f"1D non-averaging shortcut requires out_channels={in_channels*factor}, got {out_channels}"
        )
        self.factor = int(factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return pixel_unshuffle_1d(x, self.factor)  # (B, inC*factor, L//factor)


class ResidualDownsample1D(nn.Module):
    """Main path: Conv(stride=2). Shortcut: pixel_unshuffle (no averaging in 1d)."""

    def __init__(self, in_channels: int, out_channels: int, factor: int = 2) -> None:
        super().__init__()
        self.factor = int(factor)
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=3, stride=self.factor, padding=1, bias=False
        )
        self.shortcut = PixelUnshuffleDownSample1D(in_channels, out_channels, factor=self.factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv1(x) + self.shortcut(x)


class PixelShuffleUpSample1D(nn.Module):
    """Shortcut: 1D pixel_shuffle without replication. Requires in_channels == out_channels * factor."""

    def __init__(self, in_channels: int, out_channels: int, factor: int = 2) -> None:
        super().__init__()
        assert in_channels == out_channels * factor, (
            f"1D non-replicating shortcut requires in_channels={out_channels*factor}, got {in_channels}"
        )
        self.factor = int(factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return pixel_shuffle_1d(x, self.factor)  # (B, inC//factor, L*factor)


class ResidualUpsample1D(nn.Module):
    """Main path: Deconv(stride=2). Shortcut: pixel_shuffle (no replication in 1d)."""

    def __init__(self, in_channels: int, out_channels: int, factor: int = 2) -> None:
        super().__init__()
        self.factor = int(factor)
        self.deconv = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size=self.factor, stride=self.factor, bias=False
        )
        self.shortcut = PixelShuffleUpSample1D(in_channels, out_channels, factor=self.factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.deconv(x) + self.shortcut(x)


class GenotypeAutoencoder(nn.Module):
    """One-dimensional autoencoder for genotype sequences.

    This implementation compresses a sequence of length L with integer
    entries (0/1/2) into a latent sequence of fixed length latent_length
    and token dimension latent_dim.  It mirrors the deterministic
    pixel-shuffle architecture described in the question but does not
    reshape into a 2D lattice.  A variational formulation can easily be
    obtained by adding learnable mean and log-variance heads and a
    sampling step.

    Args:
        input_length: number of SNPs (L).
        latent_length: number of tokens in the compressed representation.
        latent_dim: dimension of each token.
        embed_dim: dimension of the initial per-SNP embedding.
    """

    def __init__(
        self,
        input_length: int,
        latent_length: int = 64,
        latent_dim: int = 256,
        embed_dim: int = 8,
        max_c: int = 5,
        enable_structured_latent: bool = True,
    ) -> None:
        super().__init__()
        self.input_length = int(input_length)
        self.latent_length = int(latent_length)
        self.latent_dim = int(latent_dim)
        self.embed_dim = int(embed_dim)
        self.enable_structured_latent = bool(enable_structured_latent)
        self._last_masked_channels: int = int(self.latent_dim)
        self._last_active_fraction: float = 1.0
        self._last_latent_mask: Optional[torch.Tensor] = None

        # ---- bookkeeping: choose pad/truncate length target_len = c * 2^k ----
        if self.latent_length <= 0 or (self.latent_length & (self.latent_length - 1)) != 0:
            raise ValueError(f"latent_length must be a positive power of two. Got {self.latent_length}")
        self.target_exp = int(math.log2(self.latent_length))

        # Choose (c,k) with k >= target_exp so num_down >= 0
        c, k = find_best_ck(self.input_length, max_c=int(max_c), min_k=self.target_exp)
        self.c = int(c)
        self.L1 = int(k)  # so 2^L1 is the post-stride length
        self.target_len = int(self.c * (1 << self.L1))  # pad/truncate length

        self.pad_1d = self.target_len > self.input_length
        self.truncate_1d = self.target_len < self.input_length

        # Downsample depth to reach latent_length: 2^L1 -> 2^target_exp
        self.num_down = int(self.L1 - self.target_exp)
        if self.num_down < 0:
            raise ValueError(f"num_down < 0 (L1={self.L1}, target_exp={self.target_exp}).")

        # Channels: after num_down steps, channels double each time
        # start_channels * 2^num_down = latent_dim
        if self.latent_dim % (1 << self.num_down) != 0:
            raise ValueError(
                f"latent_dim must be divisible by 2^num_down. latent_dim={self.latent_dim}, num_down={self.num_down}"
            )
        self.start_channels = int(self.latent_dim // (1 << self.num_down))

        # ---- modules ----
        # Input embedding: (onehot+dosage)=4 -> embed_dim
        self.input_embed = nn.Conv1d(4, self.embed_dim, kernel_size=1, bias=False)

        # Stride-c reduction: (B,embed_dim,target_len) -> (B,start_channels,2^L1)
        self.conv_reduce = nn.Conv1d(
            self.embed_dim, self.start_channels, kernel_size=self.c, stride=self.c, bias=False
        )

        # Residual pixel-unshuffle downsampling blocks: halve length, double channels
        down_blocks: List[ResidualDownsample1D] = []
        in_ch = self.start_channels
        for _ in range(self.num_down):
            out_ch = in_ch * 2
            down_blocks.append(ResidualDownsample1D(in_ch, out_ch, factor=2))
            in_ch = out_ch
        if in_ch != self.latent_dim:
            raise RuntimeError(f"Sanity check failed: final encoder channels {in_ch} != latent_dim {self.latent_dim}")
        self.down_blocks = nn.ModuleList(down_blocks)

        # Mirror upsampling blocks: double length, halve channels
        up_blocks: List[ResidualUpsample1D] = []
        in_ch = self.latent_dim
        for _ in range(self.num_down):
            out_ch = in_ch // 2
            up_blocks.append(ResidualUpsample1D(in_ch, out_ch, factor=2))
            in_ch = out_ch
        if in_ch != self.start_channels:
            raise RuntimeError(
                f"Sanity check failed: final decoder channels {in_ch} != start_channels {self.start_channels}"
            )
        self.up_blocks = nn.ModuleList(up_blocks)

        # Invert stride-c reduction: (B,start_channels,2^L1) -> (B,embed_dim,target_len)
        self.deconv_expand = nn.ConvTranspose1d(
            self.start_channels, self.embed_dim, kernel_size=self.c, stride=self.c, bias=False
        )

        # Final projection to logits over three states
        self.output_proj = nn.Conv1d(self.embed_dim, 3, kernel_size=1, bias=True)

    def _sample_latent_channel_mask(self, device: torch.device) -> Tuple[torch.Tensor, int]:
        """
        Sample a channel-wise mask over latent token channels.

        Returns:
            mask: shape (1, 1, latent_dim), values in {0, 1}
            c_prime: number of active channels kept (front channels kept)
        """
        C = int(self.latent_dim)
        if (not self.enable_structured_latent) or C <= 0 or C < 16:
            mask = torch.ones(1, 1, C, device=device)
            return mask, C

        # Match ae.py strategy: c' in {16, 20, 24, ..., C}
        c_values = list(range(16, C + 1, 4))
        if len(c_values) == 0:
            mask = torch.ones(1, 1, C, device=device)
            return mask, C

        idx = torch.randint(len(c_values), (1,), device=device).item()
        c_prime = int(c_values[idx])
        mask = torch.zeros(1, 1, C, device=device)
        mask[:, :, :c_prime] = 1.0
        return mask, c_prime

    def decode_with_latent_mask(self, z: torch.Tensor, *, enable_masking: bool = True) -> torch.Tensor:
        """
        Decode with channel-wise latent masking.
        During eval/inference, or if masking disabled, uses all channels.
        """
        if (not self.training) or (not enable_masking):
            self._last_latent_mask = None
            self._last_active_fraction = 1.0
            self._last_masked_channels = int(self.latent_dim)
            return self.decode(z)

        mask, c_prime = self._sample_latent_channel_mask(z.device)
        self._last_latent_mask = mask
        self._last_masked_channels = int(c_prime)
        self._last_active_fraction = float(c_prime) / float(self.latent_dim)
        return self.decode(z * mask)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encodes integer genotypes into a latent token sequence.

        Args:
            x: integer tensor of shape (B, L) with values 0/1/2.

        Returns:
            Latent tokens z of shape (B, latent_length, latent_dim).
        """
        B, L = x.shape
        # Build 4-channel input: one-hot + dosage
        x_onehot = F.one_hot(x, num_classes=3).float()  # (B, L, 3)
        x_dosage = x.float().unsqueeze(-1)  # (B, L, 1)
        x4 = torch.cat([x_onehot, x_dosage], dim=-1).permute(0, 2, 1).contiguous()  # (B, 4, L)
        h = self.input_embed(x4)  # (B, embed_dim, L)

        # Pad or truncate to target_len = c * 2^L1
        if self.pad_1d:
            pad_len = self.target_len - L
            h = F.pad(h, (0, pad_len))  # right-pad
        elif self.truncate_1d:
            h = h[..., : self.target_len]  # right-truncate

        # Stride-c reduction: target_len -> 2^L1
        h = self.conv_reduce(h)  # (B, start_channels, 2^L1)

        # Residual downsample blocks: 2^L1 -> latent_length
        for down in self.down_blocks:
            h = down(h)

        # h is (B, latent_dim, latent_length) by construction
        return h.permute(0, 2, 1).contiguous()  # (B, latent_length, latent_dim)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decodes latent tokens back to genotype logits.

        Args:
            z: latent tensor of shape (B, latent_length, latent_dim).

        Returns:
            Logits over three genotype states of shape (B, L, 3).
        """
        h = z.permute(0, 2, 1).contiguous()  # (B, latent_dim, latent_length)

        # Residual upsample blocks: latent_length -> 2^L1
        for up in self.up_blocks:
            h = up(h)

        # Invert stride-c: 2^L1 -> target_len
        h = self.deconv_expand(h)  # (B, embed_dim, target_len)

        # Crop or pad back to exact input_length
        if self.pad_1d:
            h = h[..., : self.input_length]
        elif self.truncate_1d:
            pad_len = self.input_length - self.target_len
            if pad_len > 0:
                h = F.pad(h, (0, pad_len))

        logits = self.output_proj(h)  # (B, 3, input_length)
        return logits.permute(0, 2, 1).contiguous()  # (B, input_length, 3)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encodes and decodes genotypes.

        Returns both the decoded logits and the latent representation.
        """
        self._last_latent_mask = None
        self._last_active_fraction = 1.0
        self._last_masked_channels = int(self.latent_dim)
        z = self.encode(x)
        logits = self.decode(z)
        return logits, z


def reconstruction_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Cross-entropy loss between predicted logits and discrete genotypes."""
    B, L, C = logits.shape
    loss = F.cross_entropy(logits.view(B * L, C), target.view(B * L), reduction="mean")
    return loss


# Placeholder functions for domain-specific regularizers.  In a full
# implementation these would compute the minor allele frequency (MAF)
# penalty, linkage disequilibrium (LD) penalty and other smoothness terms.
def maf_loss_fn(decoded: torch.Tensor, true_data: torch.Tensor) -> torch.Tensor:
    """Computes a minor allele frequency penalty.

    Args:
        decoded: decoded logits of shape (B, L, 3).
        true_data: integer genotypes (B, L).

    Returns:
        Scalar tensor representing the MAF loss.
    """
    # Convert logits to probabilities
    probs = F.softmax(decoded, dim=-1)
    # Compute allele frequencies per site (mean dosage / 2)
    dosage_pred = probs[..., 1] + 2 * probs[..., 2]
    dosage_true = true_data.float()
    maf_pred = dosage_pred.mean(dim=0) / 2.0
    maf_true = dosage_true.mean(dim=0) / 2.0
    return F.mse_loss(maf_pred, maf_true)


def ld_loss_fn(decoded: torch.Tensor, true_data: torch.Tensor, window: int = 100) -> torch.Tensor:
    """Computes a local linkage disequilibrium penalty.

    Correlation structures of nearby SNPs in sliding windows are matched.

    Args:
        decoded: decoded logits (B, L, 3).
        true_data: integer genotypes (B, L).
        window: window size for computing LD.

    Returns:
        Scalar tensor representing the LD loss.
    """
    probs = F.softmax(decoded, dim=-1)
    dosage_pred = probs[..., 1] + 2 * probs[..., 2]
    dosage_true = true_data.float()
    L = dosage_true.shape[1]
    # Compute LD (correlation) in local windows
    loss = 0.0
    count = 0
    for start in range(0, L, window // 2):
        end = min(start + window, L)
        if end - start < 2:
            continue
        pred_block = dosage_pred[:, start:end]
        true_block = dosage_true[:, start:end]
        # Normalize
        pred_block = (pred_block - pred_block.mean(dim=0)) / (pred_block.std(dim=0) + 1e-5)
        true_block = (true_block - true_block.mean(dim=0)) / (true_block.std(dim=0) + 1e-5)
        # Compute covariance matrices and correlation difference
        cov_pred = pred_block.T @ pred_block / (pred_block.shape[0] - 1)
        cov_true = true_block.T @ true_block / (true_block.shape[0] - 1)
        loss += F.mse_loss(cov_pred, cov_true)
        count += 1
    if count == 0:
        return torch.tensor(0.0, device=decoded.device)
    return loss / count


def total_variation_loss(z: torch.Tensor) -> torch.Tensor:
    """1D total variation penalty on latent sequences."""
    return torch.mean(torch.abs(z[:, 1:] - z[:, :-1]))


def _build_ema_model(
    model: nn.Module,
    ema_shadow: Optional[Dict[str, torch.Tensor]],
    prefix: str = "",
) -> Optional[nn.Module]:
    """Create a detached copy of model with EMA values loaded."""
    if ema_shadow is None:
        return None
    ema_model = copy.deepcopy(model)
    state = ema_model.state_dict()
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        key = f"{prefix}{name}" if prefix else name
        if key in ema_shadow:
            state[name] = ema_shadow[key].detach().clone()
    ema_model.load_state_dict(state, strict=False)
    return ema_model


def train_unet_latent_diffusion(
    latent_path: str,
    num_epochs: int = 10,
    batch_size: int = 32,
    device: str = "cuda",
    ema_decay: float = 0.9999,
    use_ema: bool = True,
) -> Tuple[UNet2DModel, Optional[UNet2DModel]]:
    """Trains a UNet on pre-computed latents using a velocity parameterization.

    Args:
        latent_path: path to torch.saved latents of shape (N, C, H, W).
        num_epochs: number of training epochs.
        batch_size: mini-batch size.
        device: computation device.
        ema_decay: decay factor for EMA in (0, 1).
        use_ema: whether to track and apply EMA weights.
    """
    dataset = LatentDataset(latent_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    # Determine latent shape
    C, H, W = dataset.latents.shape[1:]
    # Define UNet model; prediction_type="v_prediction" for velocity parameterization
    model = UNet2DModel(
        sample_size=H,
        in_channels=C,
        out_channels=C,
        layers_per_block=2,
        block_out_channels=(64, 128, 256, 512),
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    ).to(device)
    # Scheduler with cosine betas; velocity prediction
    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="v_prediction",
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    ema_shadow: Optional[Dict[str, torch.Tensor]] = None
    if use_ema:
        if not (0.0 < ema_decay < 1.0):
            raise ValueError(f"ema_decay must be in (0, 1). Got {ema_decay}")
        ema_shadow = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    for epoch in range(num_epochs):
        for latents in dataloader:
            latents = latents.to(device)
            bsz = latents.shape[0]
            # Sample diffusion timesteps
            t = torch.randint(0, scheduler.num_train_timesteps, (bsz,), device=device)
            noise = torch.randn_like(latents)
            # Create noisy latent using scheduler
            noisy_latents = scheduler.add_noise(latents, noise, t)
            # Target velocity computed from noise and latent
            v_target = scheduler.get_velocity(latents, noise, t)
            # Predict velocity
            v_pred = model(noisy_latents, t).sample
            loss = F.mse_loss(v_pred, v_target)
            loss.backward()
            optimizer.step()
            if use_ema and ema_shadow is not None:
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        if param.requires_grad:
                            ema_shadow[name].mul_(ema_decay).add_(param.detach(), alpha=(1.0 - ema_decay))
            optimizer.zero_grad()
        # Optionally monitor GPU memory usage
        if torch.cuda.is_available():
            mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            print(f"Epoch {epoch+1:03d}: loss={loss.item():.4f}, peak memory={mem:.1f}MB")
            torch.cuda.reset_peak_memory_stats(device)
        else:
            print(f"Epoch {epoch+1:03d}: loss={loss.item():.4f}")

    ema_model = _build_ema_model(model, ema_shadow)
    return model, ema_model


def train_dit_diffusion(
    sequences: torch.Tensor,
    covariates: Optional[torch.Tensor] = None,
    num_epochs: int = 10,
    batch_size: int = 32,
    latent_length: int = 64,
    latent_dim: int = 256,
    device: str = "cuda",
    ema_decay: float = 0.9999,
    use_ema: bool = True,
) -> Tuple[
    GenotypeAutoencoder,
    TransformerDiffusionModel,
    Optional[ConditionEmbedding],
    Optional[GenotypeAutoencoder],
    Optional[TransformerDiffusionModel],
    Optional[ConditionEmbedding],
]:
    """Trains a diffusion Transformer (DiT) on genotype sequences.

    The autoencoder and Transformer backbone are trained end-to-end.  A
    DDPM scheduler with velocity prediction is used to corrupt and
    denoise latent sequences.  Conditional covariates can be used via
    embedding.

    Args:
        sequences: integer tensor (N, L) of genotype calls.
        covariates: optional float tensor (N, C_cov) for conditioning.
        num_epochs: number of training epochs.
        batch_size: mini-batch size.
        latent_length: number of tokens in the latent representation.
        latent_dim: dimension of each latent token.
        device: computation device.
        ema_decay: decay factor for EMA in (0, 1).
        use_ema: whether to track and apply EMA weights.
    """
    dataset = GenotypeDataset(sequences, covariates)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    L = sequences.shape[1]
    # Instantiate autoencoder
    ae = GenotypeAutoencoder(L, latent_length=latent_length, latent_dim=latent_dim).to(device)
    # Instantiate Transformer backbone
    dit = TransformerDiffusionModel(token_dim=latent_dim, num_layers=8, num_heads=8, mlp_ratio=4).to(device)
    # Time and condition embeddings
    time_embed = TimeEmbedding(latent_dim).to(device)
    cond_embed = None
    if covariates is not None:
        cond_embed = ConditionEmbedding(covariates.shape[1], latent_dim).to(device)
    # Define scheduler for latents
    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="v_prediction",
    )
    params = list(ae.parameters()) + list(dit.parameters())
    if cond_embed is not None:
        params += list(cond_embed.parameters())
    optimizer = torch.optim.Adam(params, lr=2e-4)
    ema_shadow: Optional[Dict[str, torch.Tensor]] = None
    if use_ema:
        if not (0.0 < ema_decay < 1.0):
            raise ValueError(f"ema_decay must be in (0, 1). Got {ema_decay}")
        ema_shadow = {
            f"ae.{name}": param.detach().clone()
            for name, param in ae.named_parameters()
            if param.requires_grad
        }
        for name, param in dit.named_parameters():
            if param.requires_grad:
                ema_shadow[f"dit.{name}"] = param.detach().clone()
        if cond_embed is not None:
            for name, param in cond_embed.named_parameters():
                if param.requires_grad:
                    ema_shadow[f"cond_embed.{name}"] = param.detach().clone()

    for epoch in range(num_epochs):
        for batch in dataloader:
            if covariates is None:
                x = batch.to(device)
                cond = None
            else:
                x, cond = batch
                x = x.to(device)
                cond = cond.to(device)
            # Autoencoder forward
            logits, z = ae(x)
            # Reconstruction loss in data space
            rec_loss = reconstruction_loss(logits, x)
            # Additional VAE losses (not implemented here)
            # Corrupt latent tokens for diffusion
            bsz = z.shape[0]
            # Permute to (B, latent_length, latent_dim)
            z_tokens = z  # Already in (B, N, D)
            t = torch.randint(0, scheduler.num_train_timesteps, (bsz,), device=device)
            noise = torch.randn_like(z_tokens)
            noisy_tokens = scheduler.add_noise(z_tokens, noise, t)
            v_target = scheduler.get_velocity(z_tokens, noise, t)
            t_emb = time_embed(t.float() / scheduler.num_train_timesteps)
            c_emb = cond_embed(cond) if cond is not None else None
            v_pred = dit(noisy_tokens, t_emb, c_emb)
            diff_loss = F.mse_loss(v_pred, v_target)
            # Optional regularizers on latent space
            tv_loss = total_variation_loss(z_tokens)
            loss = rec_loss + diff_loss + 0.001 * tv_loss
            loss.backward()
            optimizer.step()
            if use_ema and ema_shadow is not None:
                with torch.no_grad():
                    for name, param in ae.named_parameters():
                        if param.requires_grad:
                            ema_shadow[f"ae.{name}"].mul_(ema_decay).add_(param.detach(), alpha=(1.0 - ema_decay))
                    for name, param in dit.named_parameters():
                        if param.requires_grad:
                            ema_shadow[f"dit.{name}"].mul_(ema_decay).add_(param.detach(), alpha=(1.0 - ema_decay))
                    if cond_embed is not None:
                        for name, param in cond_embed.named_parameters():
                            if param.requires_grad:
                                ema_shadow[f"cond_embed.{name}"].mul_(ema_decay).add_(
                                    param.detach(), alpha=(1.0 - ema_decay)
                                )
            optimizer.zero_grad()
        if torch.cuda.is_available():
            mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            print(
                f"Epoch {epoch+1:03d}: rec={rec_loss.item():.4f}, diff={diff_loss.item():.4f}, TV={tv_loss.item():.4f}, peak_mem={mem:.1f}MB"
            )
            torch.cuda.reset_peak_memory_stats(device)
        else:
            print(
                f"Epoch {epoch+1:03d}: rec={rec_loss.item():.4f}, diff={diff_loss.item():.4f}, TV={tv_loss.item():.4f}"
            )

    ema_ae = _build_ema_model(ae, ema_shadow, prefix="ae.")
    ema_dit = _build_ema_model(dit, ema_shadow, prefix="dit.")
    ema_cond = _build_ema_model(cond_embed, ema_shadow, prefix="cond_embed.") if cond_embed is not None else None
    return ae, dit, cond_embed, ema_ae, ema_dit, ema_cond


def train_sit_flow_matching(
    sequences: torch.Tensor,
    covariates: Optional[torch.Tensor] = None,
    num_epochs: int = 10,
    batch_size: int = 32,
    latent_length: int = 64,
    latent_dim: int = 256,
    device: str = "cuda",
    ema_decay: float = 0.9999,
    use_ema: bool = True,
) -> Tuple[
    GenotypeAutoencoder,
    TransformerDiffusionModel,
    Optional[ConditionEmbedding],
    Optional[GenotypeAutoencoder],
    Optional[TransformerDiffusionModel],
    Optional[ConditionEmbedding],
]:
    """Trains a Scalable Interpolant Transformer (SiT) via flow matching.

    The same autoencoder used in the DiT case is reused to encode
    genotype sequences into latent tokens.  Instead of corrupting these
    tokens with a diffusion scheduler, we form interpolants between a
    random base latent drawn from a standard normal distribution and the
    data latent, and train the Transformer to predict the velocity field
    that transports the base to the data.  This continuous-time objective
    generalizes the diffusion loss to flow matching【185938056092858†L360-L370】.

    Args:
        sequences: integer tensor (N, L) of genotype calls.
        covariates: optional float tensor (N, C_cov) for conditioning.
        num_epochs: number of training epochs.
        batch_size: mini-batch size.
        latent_length: number of tokens in the latent representation.
        latent_dim: dimension of each latent token.
        device: computation device.
        ema_decay: decay factor for EMA in (0, 1).
        use_ema: whether to track and apply EMA weights.
    """
    dataset = GenotypeDataset(sequences, covariates)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    L = sequences.shape[1]
    ae = GenotypeAutoencoder(L, latent_length=latent_length, latent_dim=latent_dim).to(device)
    sit = TransformerDiffusionModel(token_dim=latent_dim, num_layers=8, num_heads=8, mlp_ratio=4).to(device)
    time_embed = TimeEmbedding(latent_dim).to(device)
    cond_embed = None
    if covariates is not None:
        cond_embed = ConditionEmbedding(covariates.shape[1], latent_dim).to(device)
    params = list(ae.parameters()) + list(sit.parameters())
    if cond_embed is not None:
        params += list(cond_embed.parameters())
    optimizer = torch.optim.Adam(params, lr=2e-4)
    ema_shadow: Optional[Dict[str, torch.Tensor]] = None
    if use_ema:
        if not (0.0 < ema_decay < 1.0):
            raise ValueError(f"ema_decay must be in (0, 1). Got {ema_decay}")
        ema_shadow = {
            f"ae.{name}": param.detach().clone()
            for name, param in ae.named_parameters()
            if param.requires_grad
        }
        for name, param in sit.named_parameters():
            if param.requires_grad:
                ema_shadow[f"sit.{name}"] = param.detach().clone()
        if cond_embed is not None:
            for name, param in cond_embed.named_parameters():
                if param.requires_grad:
                    ema_shadow[f"cond_embed.{name}"] = param.detach().clone()

    for epoch in range(num_epochs):
        for batch in dataloader:
            if covariates is None:
                x = batch.to(device)
                cond = None
            else:
                x, cond = batch
                x = x.to(device)
                cond = cond.to(device)
            # Autoencoder forward
            logits, z_data = ae(x)
            rec_loss = reconstruction_loss(logits, x)
            # Sample base latent from N(0, I)
            base_latent = torch.randn_like(z_data)
            bsz = z_data.shape[0]
            # Sample continuous times uniformly in [0,1]
            t = torch.rand(bsz, device=device)
            t_emb = time_embed(t)
            c_emb = cond_embed(cond) if cond is not None else None
            # Linear interpolant between base and data
            z_t = (1.0 - t).view(-1, 1, 1) * base_latent + t.view(-1, 1, 1) * z_data
            # Target velocity (derivative of interpolant)
            v_target = z_data - base_latent  # constant with respect to t for linear interpolant
            v_pred = sit(z_t, t_emb, c_emb)
            flow_loss = F.mse_loss(v_pred, v_target)
            tv_loss = total_variation_loss(z_data)
            loss = rec_loss + flow_loss + 0.001 * tv_loss
            loss.backward()
            optimizer.step()
            if use_ema and ema_shadow is not None:
                with torch.no_grad():
                    for name, param in ae.named_parameters():
                        if param.requires_grad:
                            ema_shadow[f"ae.{name}"].mul_(ema_decay).add_(param.detach(), alpha=(1.0 - ema_decay))
                    for name, param in sit.named_parameters():
                        if param.requires_grad:
                            ema_shadow[f"sit.{name}"].mul_(ema_decay).add_(param.detach(), alpha=(1.0 - ema_decay))
                    if cond_embed is not None:
                        for name, param in cond_embed.named_parameters():
                            if param.requires_grad:
                                ema_shadow[f"cond_embed.{name}"].mul_(ema_decay).add_(
                                    param.detach(), alpha=(1.0 - ema_decay)
                                )
            optimizer.zero_grad()
        if torch.cuda.is_available():
            mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            print(
                f"Epoch {epoch+1:03d}: rec={rec_loss.item():.4f}, flow={flow_loss.item():.4f}, TV={tv_loss.item():.4f}, peak_mem={mem:.1f}MB"
            )
            torch.cuda.reset_peak_memory_stats(device)
        else:
            print(
                f"Epoch {epoch+1:03d}: rec={rec_loss.item():.4f}, flow={flow_loss.item():.4f}, TV={tv_loss.item():.4f}"
            )

    ema_ae = _build_ema_model(ae, ema_shadow, prefix="ae.")
    ema_sit = _build_ema_model(sit, ema_shadow, prefix="sit.")
    ema_cond = _build_ema_model(cond_embed, ema_shadow, prefix="cond_embed.") if cond_embed is not None else None
    return ae, sit, cond_embed, ema_ae, ema_sit, ema_cond


# -----------------------------------------------------------------------------
# H5 helpers: load original genotypes + create latents via AE
# -----------------------------------------------------------------------------
def enumerate_h5_batch_paths(start_path: str, max_batches: int) -> List[str]:
    """
    Enumerate sequential H5 batch files with pattern batch00001.h5, batch00002.h5, ...
    starting from `start_path`.
    """
    import re
    d = os.path.dirname(start_path)
    m = re.search(r"batch(\d{5})\.h5$", os.path.basename(start_path))
    if not m:
        raise ValueError(f"Cannot parse batch index from {start_path}")
    start_idx = int(m.group(1))
    paths: List[str] = []
    for i in range(start_idx, start_idx + int(max_batches)):
        p = os.path.join(d, f"batch{i:05d}.h5")
        if os.path.exists(p):
            paths.append(p)
        else:
            break
    if not paths:
        raise FileNotFoundError(f"No H5 batch files found starting at {start_path}")
    return paths


def evaluate_ae_reconstruction(
    ae: GenotypeAutoencoder,
    h5_paths: List[str],
    max_batches_for_eval: int = 2,
    batch_size: int = 256,
    device: torch.device = torch.device("cuda"),
) -> None:
    """
    Evaluate AE reconstruction quality on a few H5 batches:
    X -> AE -> decode -> calls, compare to original X.
    """
    ae.eval()
    total_correct = 0.0
    total_elems = 0.0

    batches_seen = 0
    for p in h5_paths:
        if batches_seen >= max_batches_for_eval:
            break
        print(f"[AE-RECON] Evaluating on {p}")
        with h5py.File(p, "r") as f:
            X = f["X"]
            N = int(X.shape[0])
            L = int(X.shape[1])
            for s in range(0, N, batch_size):
                e = min(N, s + batch_size)
                xb = torch.from_numpy(X[s:e].astype("int64")).to(device, non_blocking=(device.type == "cuda"))
                with torch.no_grad():
                    logits, _ = ae(xb)
                    calls = logits.argmax(dim=1)  # [B, L]
                correct = (calls == xb).float().sum().item()
                elems = (calls.numel())
                total_correct += correct
                total_elems += elems

        batches_seen += 1

    if total_elems > 0:
        recon_acc = total_correct / total_elems
        print(f"[AE-RECON] Overall genotype call accuracy over evaluated batches: {recon_acc:.6f}")
    else:
        print("[AE-RECON] No elements evaluated (check h5_paths / max_batches_for_eval).")


def compute_latents_for_preview(
    ae: GenotypeAutoencoder,
    h5_path: str,
    device: torch.device = torch.device("cuda"),
    n_examples: int = 64,
) -> torch.Tensor:
    """
    Encode up to `n_examples` from a single H5 batch to latents to sanity check shapes.
    """
    print(f"[LATENTS] Computing preview latents from {h5_path}")
    with h5py.File(h5_path, "r") as f:
        X = f["X"]
        N = min(int(X.shape[0]), n_examples)
        xb = torch.from_numpy(X[:N].astype("int64")).to(device, non_blocking=(device.type == "cuda"))
        with torch.no_grad():
            _, z = ae(xb)  # z: [N, C, H, W]
    print(f"[LATENTS] Preview latent tensor shape: {tuple(z.shape)}")
    return z


@torch.no_grad()
def build_latents_h5(
    *,
    ae: torch.nn.Module,
    h5_paths: List[str],
    out_latents_h5: str,
    device: torch.device,
    batch_size: int = 256,
    latent_key: str = "Z",
    compression: Optional[str] = "gzip",
    compression_opts: int = 4,
    chunk_rows: int = 1024,
    dtype: str = "float16",
    limit_total_examples: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Stream H5 genotype batches through AE, write channel-normalized latents to a single H5 file.

    Returns:
        dict with keys: N_total, C, H, W
    """
    ae.eval()

    # First pass: determine total N across selected H5 files
    Ns = []
    Ls = []
    for p in h5_paths:
        with h5py.File(p, "r") as f:
            X = f["X"]
            Ns.append(int(X.shape[0]))
            Ls.append(int(X.shape[1]))
    if len(set(Ls)) != 1:
        raise ValueError(f"All H5 batches must have same L. Got {set(Ls)}")
    L = Ls[0]

    N_total = int(sum(Ns))
    if limit_total_examples is not None:
        N_total = min(N_total, int(limit_total_examples))

    os.makedirs(os.path.dirname(out_latents_h5), exist_ok=True)

    # Determine latent shape by running a tiny batch on first file
    with h5py.File(h5_paths[0], "r") as f:
        X = f["X"]
        n0 = min(8, int(X.shape[0]))
        xb = torch.from_numpy(X[:n0].astype("int64")).to(device, non_blocking=(device.type == "cuda"))
        _, z = ae(xb)  # expected (B, C, H, W)
        if z.dim() != 4:
            raise ValueError(f"AE latent must be 4D (B,C,H,W). Got {tuple(z.shape)}")
        _, C, H, W = z.shape

    # Create output H5 dataset
    # Use chunking over the first dimension for fast sequential write and later fast read
    h5_dtype = np.float16 if dtype == "float16" else np.float32
    chunks = (min(chunk_rows, N_total), C, H, W)

    # Write raw latents first (for numerically stable stats), then normalize in-place.
    write_idx = 0
    ch_sum = np.zeros((C,), dtype=np.float64)
    ch_sumsq = np.zeros((C,), dtype=np.float64)
    ch_count = 0.0
    with h5py.File(out_latents_h5, "w") as out_f:
        dset = out_f.create_dataset(
            latent_key,
            shape=(N_total, C, H, W),
            dtype=h5_dtype,
            chunks=chunks,
            compression=compression,
            compression_opts=compression_opts if compression is not None else None,
        )
        out_f.attrs["C"] = C
        out_f.attrs["H"] = H
        out_f.attrs["W"] = W
        out_f.attrs["L"] = L

        remaining = N_total

        for p in h5_paths:
            if remaining <= 0:
                break

            with h5py.File(p, "r") as f:
                X = f["X"]
                Np = int(X.shape[0])
                take = min(Np, remaining)

                # Iterate batches within this H5 file
                for s in tqdm(range(0, take, batch_size), desc=f"[LATENTS] {os.path.basename(p)}", leave=False):
                    e = min(take, s + batch_size)
                    xb = torch.from_numpy(X[s:e].astype("int64")).to(device, non_blocking=(device.type == "cuda"))
                    _, z = ae(xb)  # (B,C,H,W)
                    z_np = z.detach().to(torch.float32).cpu().numpy()
                    # Accumulate channel-wise moments over all pixels for normalization.
                    ch_sum += z_np.sum(axis=(0, 2, 3))
                    ch_sumsq += (z_np ** 2).sum(axis=(0, 2, 3))
                    ch_count += float(z_np.shape[0] * z_np.shape[2] * z_np.shape[3])
                    if dtype == "float16":
                        z_np = z_np.astype(np.float16, copy=False)
                    else:
                        z_np = z_np.astype(np.float32, copy=False)
                    dset[write_idx : write_idx + (e - s)] = z_np
                    write_idx += (e - s)

            remaining = N_total - write_idx

        if write_idx != N_total:
            # If we hit fewer examples than planned (e.g. missing files), resize down.
            dset.resize((write_idx, C, H, W))
            out_f.attrs["N_total"] = write_idx
            N_total = write_idx
        else:
            out_f.attrs["N_total"] = N_total

        if ch_count <= 0:
            raise RuntimeError("Failed to compute latent normalization stats: no encoded samples.")
        ch_mean = (ch_sum / ch_count).astype(np.float32)
        ch_var = np.maximum((ch_sumsq / ch_count) - (ch_mean.astype(np.float64) ** 2), 1e-12)
        ch_std = np.sqrt(ch_var).astype(np.float32)
        ch_std = np.maximum(ch_std, 1e-6).astype(np.float32)

        # Normalize the on-disk latents in-place so UNet always trains in normalized space.
        norm_chunk = max(1, int(chunk_rows))
        for s in tqdm(range(0, int(N_total), norm_chunk), desc="[LATENTS] normalize", leave=False):
            e = min(int(N_total), s + norm_chunk)
            z_blk = np.asarray(dset[s:e], dtype=np.float32)
            z_blk = (z_blk - ch_mean[None, :, None, None]) / ch_std[None, :, None, None]
            if h5_dtype == np.float16:
                z_blk = z_blk.astype(np.float16, copy=False)
            else:
                z_blk = z_blk.astype(np.float32, copy=False)
            dset[s:e] = z_blk

        out_f.attrs["latent_normalized"] = 1
        out_f.attrs["latent_norm_eps"] = float(1e-6)
        out_f.attrs["channel_mean"] = ch_mean
        out_f.attrs["channel_std"] = ch_std

    print(f"[LATENTS] Wrote latents to: {out_latents_h5}")
    print(f"[LATENTS] Shape: (N={N_total}, C={C}, H={H}, W={W}), dtype={dtype}")
    print(
        "[LATENTS] Normalized channel-wise: "
        f"mean(|mu|)={float(np.mean(np.abs(ch_mean))):.4e}, mean(std)={float(np.mean(ch_std)):.4e}"
    )
    return {
        "N_total": N_total,
        "C": C,
        "H": H,
        "W": W,
        "latent_normalized": True,
        "latent_norm_eps": float(1e-6),
        "channel_mean": ch_mean,
        "channel_std": ch_std,
    }

class LatentH5Dataset(Dataset):
    """
    Streaming dataset for latents stored in an H5 file under key latent_key.

    Each __getitem__ returns a torch.Tensor (C,H,W) float32/float16.
    """

    def __init__(self, latents_h5: str, latent_key: str = "Z", dtype: torch.dtype = torch.float16) -> None:
        super().__init__()
        self.latents_h5 = latents_h5
        self.latent_key = latent_key
        self.dtype = dtype

        # Read shape metadata once
        with h5py.File(self.latents_h5, "r") as f:
            dset = f[self.latent_key]
            self._shape = tuple(dset.shape)  # (N,C,H,W)
        if len(self._shape) != 4:
            raise ValueError(f"H5 latents must be 4D (N,C,H,W). Got {self._shape}")

    def __len__(self) -> int:
        return int(self._shape[0])

    def __getitem__(self, idx: int) -> torch.Tensor:
        # Open per worker: safe for multiprocessing DataLoader
        with h5py.File(self.latents_h5, "r") as f:
            z = f[self.latent_key][idx]  # numpy array (C,H,W)
        return torch.from_numpy(z).to(self.dtype)

def train_unet_on_latents_h5(
    *,
    latents_h5: str,
    latent_key: str = "Z",
    num_epochs: int = 5,
    batch_size: int = 32,
    lr: float = 1e-4,
    num_train_timesteps: int = 1000,
    device: torch.device,
    num_workers: int = 2,
    pin_memory: bool = True,
    amp: bool = True,
    log_every: int = 50,
    ema_decay: float = 0.9999,
    use_ema: bool = True,
    max_steps: Optional[int] = None,
) -> Tuple[UNet2DModel, Optional[UNet2DModel]]:
    """
    Train a UNet2DModel with velocity (v_prediction) parameterization on fixed latents.
    Expects latents already normalized in `build_latents_h5`.
    Optionally track an exponential moving average (EMA) of model weights.

    Returns:
        Tuple of (trained model, EMA model or None)
    """
    if max_steps is not None and max_steps <= 0:
        raise ValueError(f"max_steps must be positive when set. Got {max_steps}")

    print("[UNET] Step 1: creating latent dataset and dataloader")
    ds = LatentH5Dataset(latents_h5, latent_key=latent_key, dtype=torch.float16 if amp else torch.float32)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory and (device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )

    # Read latent shape
    print("[UNET] Step 2: reading latent tensor shape from H5")
    with h5py.File(latents_h5, "r") as f:
        C = int(f.attrs.get("C", f[latent_key].shape[1]))
        H = int(f.attrs.get("H", f[latent_key].shape[2]))
        W = int(f.attrs.get("W", f[latent_key].shape[3]))
        is_norm = bool(int(f.attrs.get("latent_normalized", 0)))
        if is_norm:
            print("[UNET] Found normalized latents with channel stats in H5 attrs.")
        else:
            print("[UNET] WARNING: latents are not marked normalized; training will use raw latent scale.")

    # UNet backbone; tune block_out_channels if you want larger/smaller model
    print("[UNET] Step 3: building UNet model")
    model = UNet2DModel(
        sample_size=H,           # assumes square; if H!=W you can still use UNet2DModel but sample_size is int.
        in_channels=C,
        out_channels=C,
        layers_per_block=2,
        block_out_channels=(128, 256, 256, 512),
        down_block_types=("DownBlock2D", "AttnDownBlock2D", "DownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "UpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
    ).to(device)

    print("[UNET] Step 4: creating diffusion scheduler")
    scheduler = DDPMScheduler(
        num_train_timesteps=num_train_timesteps,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="v_prediction",
    )

    print("[UNET] Step 5: creating optimizer and AMP scaler")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

    scaler = torch.cuda.amp.GradScaler(enabled=(amp and device.type == "cuda"))

    ema_shadow: Optional[Dict[str, torch.Tensor]] = None
    if use_ema:
        print("[UNET] Step 6: initializing EMA shadow weights")
        if not (0.0 < ema_decay < 1.0):
            raise ValueError(f"ema_decay must be in (0, 1). Got {ema_decay}")
        ema_shadow = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    model.train()
    global_step = 0
    stop_training = False
    print("[UNET] Step 7: starting training loop")
    for epoch in range(num_epochs):
        running = 0.0
        pbar = tqdm(dl, desc=f"[UNET] epoch {epoch+1}/{num_epochs}", leave=True)

        # reset peak stats
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        for batch_idx, latents in enumerate(pbar, start=1):
            if max_steps is not None and global_step >= max_steps:
                print(f"[UNET] Reached max_steps={max_steps}. Stopping training.")
                stop_training = True
                break
            print(f"[UNET] Step {global_step + 1}: epoch={epoch+1}, batch={batch_idx} begin")

            # Move to device
            latents = latents.to(device, non_blocking=(device.type == "cuda"))
            latents = latents.float() if not amp else latents  # keep fp16 under amp, cast in autocast

            bsz = latents.shape[0]
            t = torch.randint(0, scheduler.num_train_timesteps, (bsz,), device=device, dtype=torch.long)
            noise = torch.randn_like(latents)

            noisy = scheduler.add_noise(latents, noise, t)
            v_target = scheduler.get_velocity(latents, noise, t)

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(amp and device.type == "cuda")):
                v_pred = model(noisy, t).sample
                loss = F.mse_loss(v_pred, v_target)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            global_step += 1
            print(f"[UNET] Step {global_step}: optimizer update complete, loss={loss.item():.6f}")

            if use_ema and ema_shadow is not None:
                # Keep EMA update out of autograd graph for stability and speed.
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        if not param.requires_grad:
                            continue
                        ema_shadow[name].mul_(ema_decay).add_(param.detach(), alpha=(1.0 - ema_decay))
                print(f"[UNET] Step {global_step}: EMA updated")

            running += loss.item()
            if global_step % log_every == 0:
                avg = running / log_every
                running = 0.0
                if device.type == "cuda":
                    mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                    pbar.set_postfix(loss=f"{avg:.4f}", peak_mem=f"{mem:.0f}MB")
                else:
                    pbar.set_postfix(loss=f"{avg:.4f}")

        # end epoch log
        if device.type == "cuda":
            mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            print(f"[UNET] epoch {epoch+1} done. peak_mem={mem:.1f}MB")
        if stop_training:
            break

    print("[UNET] Step 8: building EMA model copy")
    ema_model = _build_ema_model(model, ema_shadow)
    print("[UNET] Step 9: returning trained model and EMA model")
    return model, ema_model


class H5GenotypeDataset(Dataset):
    """
    Map-style Dataset over many H5 files (each containing X: (N,L) int {0,1,2}).
    Returns x: (L,) long.
    """
    def __init__(self, h5_paths: List[str], x_key: str = "X") -> None:
        super().__init__()
        if len(h5_paths) == 0:
            raise ValueError("h5_paths must be non-empty.")
        self.h5_paths = list(h5_paths)
        self.x_key = x_key

        self._Ns: List[int] = []
        self._Ls: List[int] = []
        for p in self.h5_paths:
            with h5py.File(p, "r") as f:
                X = f[self.x_key]
                self._Ns.append(int(X.shape[0]))
                self._Ls.append(int(X.shape[1]))

        if len(set(self._Ls)) != 1:
            raise ValueError(f"All H5 must have same L. Got {set(self._Ls)}")
        self.L = self._Ls[0]

        # prefix sums for global indexing
        self._offsets = [0]
        s = 0
        for n in self._Ns:
            s += n
            self._offsets.append(s)  # length = len(files)+1

    def __len__(self) -> int:
        return self._offsets[-1]

    def _locate(self, idx: int) -> Tuple[int, int]:
        # find file index such that offsets[i] <= idx < offsets[i+1]
        fi = bisect.bisect_right(self._offsets, idx) - 1
        if fi < 0 or fi >= len(self.h5_paths):
            raise IndexError(idx)
        local = idx - self._offsets[fi]
        return fi, local

    def __getitem__(self, idx: int) -> torch.Tensor:
        fi, local = self._locate(int(idx))
        p = self.h5_paths[fi]
        with h5py.File(p, "r") as f:
            x = f[self.x_key][local]  # numpy (L,)
        return torch.from_numpy(x.astype("int64"))


def genotype_calls_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """
    Convert logits to integer calls (B,L) for either layout:
    - (B,L,3)  -> argmax dim=-1
    - (B,3,L)  -> argmax dim=1
    """
    if logits.dim() != 3:
        raise ValueError(f"logits must be 3D. Got {tuple(logits.shape)}")
    if logits.shape[-1] == 3:
        return logits.argmax(dim=-1)  # (B,L)
    if logits.shape[1] == 3:
        return logits.argmax(dim=1)   # (B,L)
    raise ValueError(f"Cannot infer class dim for logits shape {tuple(logits.shape)}")


@torch.no_grad()
def evaluate_ae_reconstruction_h5(
    ae: nn.Module,
    h5_paths: List[str],
    *,
    batch_size: int = 256,
    max_batches_for_eval: int = 2,
    device: torch.device = torch.device("cuda"),
) -> None:
    """
    Generic AE recon eval that works for both (B,L,3) and (B,3,L) logits.
    """
    ae.eval()
    total_correct = 0.0
    total_elems = 0.0

    seen = 0
    for p in h5_paths:
        if seen >= max_batches_for_eval:
            break
        print(f"[AE-RECON] Evaluating on {p}")
        with h5py.File(p, "r") as f:
            X = f["X"]
            N = int(X.shape[0])
            for s in range(0, N, batch_size):
                e = min(N, s + batch_size)
                xb = torch.from_numpy(X[s:e].astype("int64")).to(device, non_blocking=(device.type == "cuda"))
                logits, _ = ae(xb)
                calls = genotype_calls_from_logits(logits)
                correct = (calls == xb).float().sum().item()
                total_correct += correct
                total_elems += float(calls.numel())
        seen += 1

    if total_elems > 0:
        print(f"[AE-RECON] Call accuracy: {total_correct/total_elems:.6f}")
    else:
        print("[AE-RECON] No eval elements.")


def train_1d_autoencoder_h5(
    *,
    ae: Optional[GenotypeAutoencoder] = None,
    h5_paths: List[str],
    input_length: int,
    latent_length: int = 64,
    latent_dim: int = 256,
    embed_dim: int = 8,
    num_epochs: int = 10,
    batch_size: int = 128,
    lr: float = 2e-4,
    device: torch.device,
    num_workers: int = 2,
    pin_memory: bool = True,
    amp: bool = True,
    use_latent_channel_masking: bool = True,
    # optional domain regularizers
    lam_maf: float = 0.0,
    lam_ld: float = 0.0,
    lam_tv: float = 0.0,
    ld_window: int = 100,
    log_every: int = 50,
    ckpt_path: Optional[str] = None,
) -> GenotypeAutoencoder:
    """
    Stage-1: Train the 1D GenotypeAutoencoder on genotype H5 batches.

    Loss = CE + lam_maf*MAF + lam_ld*LD + lam_tv*TV(z)
    If `ae` is provided, training resumes from that model instead of
    initializing a new autoencoder.
    """
    ds = H5GenotypeDataset(h5_paths, x_key="X")
    if ds.L != int(input_length):
        raise ValueError(f"input_length={input_length} but H5 has L={ds.L}")
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory and (device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )

    if ae is None:
        ae = GenotypeAutoencoder(
            input_length=input_length,
            latent_length=latent_length,
            latent_dim=latent_dim,
            embed_dim=embed_dim,
        )
        print("[AE1D] Initialized new autoencoder.")
    else:
        print("[AE1D] Resuming training from provided autoencoder.")

    ae = ae.to(device)
    # Important when resuming from Stage-2 DiT workflows where AE is frozen.
    for p in ae.parameters():
        p.requires_grad = True
    if int(ae.input_length) != int(input_length):
        raise ValueError(
            f"Provided ae.input_length={ae.input_length} does not match requested input_length={input_length}."
        )
    ae.enable_structured_latent = bool(use_latent_channel_masking)

    opt = torch.optim.AdamW(ae.parameters(), lr=lr, weight_decay=1e-2)
    scaler = torch.cuda.amp.GradScaler(enabled=(amp and device.type == "cuda"))

    ae.train()
    step = 0
    for epoch in range(num_epochs):
        running = 0.0
        pbar = tqdm(dl, desc=f"[AE1D] epoch {epoch+1}/{num_epochs}", leave=True)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        for xb in pbar:
            step += 1
            xb = xb.to(device, non_blocking=(device.type == "cuda"))

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(amp and device.type == "cuda")):
                logits_full, z = ae(xb)  # logits: (B,L,3); z: (B,N,D)
                if hasattr(ae, "decode_with_latent_mask") and ae.enable_structured_latent:
                    logits = ae.decode_with_latent_mask(z, enable_masking=True)
                else:
                    logits = logits_full
                ce = reconstruction_loss(logits, xb)

                maf = maf_loss_fn(logits, xb) if lam_maf > 0 else torch.tensor(0.0, device=device)
                ld  = ld_loss_fn(logits, xb, window=ld_window) if lam_ld > 0 else torch.tensor(0.0, device=device)
                tv  = total_variation_loss(z) if lam_tv > 0 else torch.tensor(0.0, device=device)

                loss = ce + lam_maf * maf + lam_ld * ld + lam_tv * tv

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            running += loss.item()
            if step % log_every == 0:
                avg = running / log_every
                running = 0.0
                active_frac = getattr(ae, "_last_active_fraction", 1.0)
                if device.type == "cuda":
                    mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                    pbar.set_postfix(
                        loss=f"{avg:.4f}", ce=f"{ce.item():.4f}", active=f"{active_frac:.2f}", mem=f"{mem:.0f}MB"
                    )
                else:
                    pbar.set_postfix(loss=f"{avg:.4f}", ce=f"{ce.item():.4f}", active=f"{active_frac:.2f}")

        # save per-epoch if requested
        if ckpt_path is not None:
            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            torch.save(
                {
                    "model_state": ae.state_dict(),
                    "config": {
                        "input_length": input_length,
                        "latent_length": latent_length,
                        "latent_dim": latent_dim,
                        "embed_dim": embed_dim,
                        "use_latent_channel_masking": bool(use_latent_channel_masking),
                    },
                    "epoch": epoch + 1,
                },
                ckpt_path,
            )
            print(f"[AE1D] Saved checkpoint: {ckpt_path}")

    return ae


def train_dit_latent_diffusion_h5(
    *,
    ae: GenotypeAutoencoder,
    h5_paths: List[str],
    covariates: Optional[torch.Tensor] = None,  # not wired to H5 in this minimal version
    num_epochs: int = 10,
    batch_size: int = 64,
    lr: float = 2e-4,
    device: torch.device,
    num_workers: int = 2,
    pin_memory: bool = True,
    amp: bool = True,
    log_every: int = 50,
    num_train_timesteps: int = 1000,
    # Transformer config
    num_layers: int = 8,
    num_heads: int = 8,
    mlp_ratio: int = 4,
    # EMA
    ema_decay: float = 0.9999,
    use_ema: bool = True,
    ckpt_path: Optional[str] = None,
    norm_eps: float = 1e-6,
    use_latent_normalization: bool = True,
    use_udit: bool = False,
) -> Tuple[
    TransformerDiffusionModel,
    Optional[TransformerDiffusionModel],
    Dict[str, torch.Tensor],
]:
    """
    Stage-2: Freeze AE; train TransformerDiffusionModel on AE token latents
    using DDPM v-pred objective (same as UNet case but in token space).
    Optionally center/scale latents channel-wise before diffusion training.
    """
    # freeze AE
    ae.eval()
    for p in ae.parameters():
        p.requires_grad = False

    ds = H5GenotypeDataset(h5_paths, x_key="X")
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory and (device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )

    token_dim = int(ae.latent_dim)
    latent_length = int(ae.latent_length)
    if use_udit:
        dit = UDiTDiffusionModel(
            token_dim=token_dim,
            latent_length=latent_length,
            num_layers=num_layers,   # IMPORTANT: make this odd (e.g., 9, 11, 13)
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=0.0,
            use_cond_token=True,
        ).to(device)   
    else:
        dit = TransformerDiffusionModel(
            token_dim=token_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
        ).to(device)

    time_embed = TimeEmbedding(token_dim).to(device)

    cond_embed = None
    if covariates is not None:
        # If you later add covariates per sample, wire them into the dataset and loader.
        cond_embed = ConditionEmbedding(covariates.shape[1], token_dim).to(device)

    scheduler = DDPMScheduler(
        num_train_timesteps=num_train_timesteps,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="v_prediction",
    )

    params = list(dit.parameters()) + list(time_embed.parameters())
    if cond_embed is not None:
        params += list(cond_embed.parameters())

    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-2)
    scaler = torch.cuda.amp.GradScaler(enabled=(amp and device.type == "cuda"))

    if norm_eps <= 0.0:
        raise ValueError(f"norm_eps must be > 0. Got {norm_eps}")

    # ------------------------------------------------------------------
    # Optional channel-wise latent normalization stats over training set.
    # Stats are over (batch, token) dims; one mean/std per latent channel.
    # ------------------------------------------------------------------
    if use_latent_normalization:
        print("[DiT][train] Computing latent channel normalization stats...")
        ch_sum = torch.zeros(token_dim, device=device)
        ch_sq_sum = torch.zeros(token_dim, device=device)
        ch_count = 0
        with torch.no_grad():
            for xb_stats in dl:
                xb_stats = xb_stats.to(device, non_blocking=(device.type == "cuda"))
                z_stats = ae.encode(xb_stats).detach()  # (B, N, D)
                ch_sum += z_stats.sum(dim=(0, 1))
                ch_sq_sum += (z_stats ** 2).sum(dim=(0, 1))
                ch_count += int(z_stats.shape[0] * z_stats.shape[1])
        if ch_count <= 0:
            raise RuntimeError("No samples available to compute latent normalization stats.")
        latent_mean = (ch_sum / float(ch_count)).view(1, 1, -1)
        latent_var = (ch_sq_sum / float(ch_count)).view(1, 1, -1) - latent_mean ** 2
        latent_std = torch.sqrt(torch.clamp(latent_var, min=0.0) + norm_eps)
        print(
            f"[DiT][train] Latent normalization ready (channels={token_dim}, count={ch_count}, eps={norm_eps:g})"
        )
    else:
        print("[DiT][train] Latent normalization disabled; using identity normalization.")
        latent_mean = torch.zeros(1, 1, token_dim, device=device)
        latent_std = torch.ones(1, 1, token_dim, device=device)

    latent_norm_stats: Dict[str, torch.Tensor] = {
        "mean": latent_mean.detach().cpu(),
        "std": latent_std.detach().cpu(),
        "eps": torch.tensor(float(norm_eps)),
        "enabled": torch.tensor(1 if use_latent_normalization else 0, dtype=torch.int64),
    }

    ema_shadow: Optional[Dict[str, torch.Tensor]] = None
    if use_ema:
        if not (0.0 < ema_decay < 1.0):
            raise ValueError(f"ema_decay must be in (0, 1). Got {ema_decay}")
        ema_shadow = {
            name: param.detach().clone()
            for name, param in dit.named_parameters()
            if param.requires_grad
        }

    dit.train()
    step = 0
    for epoch in range(num_epochs):
        running = 0.0
        epoch_loss_sum = 0.0
        epoch_steps = 0
        pbar = tqdm(dl, desc=f"[DiT] epoch {epoch+1}/{num_epochs}", leave=True)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        for xb in pbar:
            step += 1
            xb = xb.to(device, non_blocking=(device.type == "cuda"))

            with torch.no_grad():
                z_tokens = ae.encode(xb)  # (B, N, D) tokens
                # IMPORTANT: detach to ensure AE stays frozen and to reduce graph/memory
                z_tokens = z_tokens.detach()
                # Optional channel-wise centering/scaling prior to diffusion objective.
                z_tokens_norm = (z_tokens - latent_mean) / latent_std

            bsz = z_tokens_norm.shape[0]
            t_int = torch.randint(0, scheduler.num_train_timesteps, (bsz,), device=device, dtype=torch.long)
            noise = torch.randn_like(z_tokens_norm)
            noisy = scheduler.add_noise(z_tokens_norm, noise, t_int)
            v_target = scheduler.get_velocity(z_tokens_norm, noise, t_int)

            # normalized time in [0,1] for TimeEmbedding
            t = t_int.float() / float(scheduler.num_train_timesteps)
            t_emb = time_embed(t)
            c_emb = cond_embed(None) if cond_embed is not None else None  # placeholder

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(amp and device.type == "cuda")):
                v_pred = dit(noisy, t_emb, c_emb)
                loss = F.mse_loss(v_pred, v_target)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            if use_ema and ema_shadow is not None:
                with torch.no_grad():
                    for name, param in dit.named_parameters():
                        if not param.requires_grad:
                            continue
                        ema_shadow[name].mul_(ema_decay).add_(param.detach(), alpha=(1.0 - ema_decay))

            epoch_steps += 1
            epoch_loss_sum += loss.item()
            running += loss.item()
            if step % log_every == 0:
                avg = running / log_every
                running = 0.0
                print(
                    f"[DiT][train] epoch={epoch+1}/{num_epochs} step={step} "
                    f"loss_avg={avg:.6f} loss_last={loss.item():.6f}"
                )
                if device.type == "cuda":
                    mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                    pbar.set_postfix(loss=f"{avg:.4f}", mem=f"{mem:.0f}MB")
                else:
                    pbar.set_postfix(loss=f"{avg:.4f}")

        if epoch_steps > 0:
            epoch_avg = epoch_loss_sum / float(epoch_steps)
            if device.type == "cuda":
                mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                print(
                    f"[DiT][train] epoch={epoch+1}/{num_epochs} done "
                    f"steps={epoch_steps} avg_loss={epoch_avg:.6f} peak_mem={mem:.1f}MB"
                )
            else:
                print(
                    f"[DiT][train] epoch={epoch+1}/{num_epochs} done "
                    f"steps={epoch_steps} avg_loss={epoch_avg:.6f}"
                )

        if ckpt_path is not None:
            ckpt_dir = os.path.dirname(ckpt_path)
            if ckpt_dir:
                os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_payload = {
                "model_state": dit.state_dict(),
                "time_embed_state": time_embed.state_dict(),
                "config": {
                    "token_dim": token_dim,
                    "num_layers": num_layers,
                    "num_heads": num_heads,
                    "mlp_ratio": mlp_ratio,
                    "num_train_timesteps": num_train_timesteps,
                    "ema_decay": ema_decay,
                    "use_ema": use_ema,
                    "use_latent_normalization": use_latent_normalization,
                    "norm_eps": norm_eps,
                    "arch": "UDiT" if use_udit else "DiT",
                    "latent_length": latent_length
                },
                "epoch": epoch + 1,
                "latent_norm_stats": latent_norm_stats,
            }
            if cond_embed is not None:
                ckpt_payload["cond_embed_state"] = cond_embed.state_dict()
            if ema_shadow is not None:
                ckpt_payload["ema_shadow"] = {k: v.detach().cpu() for k, v in ema_shadow.items()}
            torch.save(ckpt_payload, ckpt_path)
            print(f"[DiT][train] Saved checkpoint: {ckpt_path}")

    ema_dit = _build_ema_model(dit, ema_shadow) if use_ema else None
    # Keep learned time embedding attached to the model for later evaluation/inference helpers.
    dit.time_embed = time_embed
    dit.latent_norm_stats = {k: v.clone() if torch.is_tensor(v) else v for k, v in latent_norm_stats.items()}
    if ema_dit is not None:
        ema_dit.time_embed = copy.deepcopy(time_embed)
        ema_dit.latent_norm_stats = {k: v.clone() if torch.is_tensor(v) else v for k, v in latent_norm_stats.items()}
    return dit, ema_dit, latent_norm_stats


@torch.no_grad()
def evaluate_dit_noising_denoising_accuracy_h5(
    *,
    ae: GenotypeAutoencoder,
    dit: TransformerDiffusionModel,
    time_embed: Optional[TimeEmbedding] = None,
    latent_norm_stats: Optional[Dict[str, torch.Tensor]] = None,
    h5_paths: List[str],
    device: torch.device,
    batch_size: int = 128,
    num_workers: int = 2,
    pin_memory: bool = True,
    num_train_timesteps: int = 1000,
    noise_levels: Tuple[float, ...] = (0.10, 0.25, 0.50, 0.75, 1.00),
    max_eval_batches: Optional[int] = None,
) -> Dict[str, Dict[str, object]]:
    """
    Evaluate denoising quality in both latent and decoded spaces after fixed noising.

    For each noise level p in noise_levels:
      1) x -> z = AE.encode(x)
      2) z_t = add_noise(z, t = round(p * (T-1)))
      3) predict v with DiT and estimate z0 via v-parameterization
      4) latent metric: samplewise MSE(z0_hat, z) -> report mean +/- SE
      5) decoded metric: AE.decode(z0_hat), then build 3x3 contingency
         matrix (rows=true 0/1/2, cols=pred 0/1/2) as row-wise fractions
    """
    if num_train_timesteps <= 1:
        raise ValueError(f"num_train_timesteps must be > 1. Got {num_train_timesteps}")

    # Validate and normalize levels.
    checked_levels: List[float] = []
    for p in noise_levels:
        p_float = float(p)
        if not (0.0 < p_float <= 1.0):
            raise ValueError(f"noise_levels entries must be in (0, 1]. Got {p_float}")
        checked_levels.append(p_float)

    ae.eval()
    dit.eval()
    if time_embed is None:
        time_embed = getattr(dit, "time_embed", None)
    if time_embed is None:
        raise ValueError(
            "time_embed is required. Pass it explicitly, or call train_dit_latent_diffusion_h5 first "
            "so it is attached as dit.time_embed."
        )
    time_embed.eval()
    if latent_norm_stats is None:
        latent_norm_stats = getattr(dit, "latent_norm_stats", None)
    if latent_norm_stats is None:
        # Backward compatibility for models trained without attached stats.
        latent_mean = torch.zeros(1, 1, int(ae.latent_dim), device=device)
        latent_std = torch.ones(1, 1, int(ae.latent_dim), device=device)
    else:
        if "mean" not in latent_norm_stats or "std" not in latent_norm_stats:
            raise ValueError("latent_norm_stats must contain 'mean' and 'std'.")
        latent_mean = latent_norm_stats["mean"].to(device)
        latent_std = latent_norm_stats["std"].to(device)

    ds = H5GenotypeDataset(h5_paths, x_key="X")
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory and (device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )

    scheduler = DDPMScheduler(
        num_train_timesteps=num_train_timesteps,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="v_prediction",
    )
    alphas_cumprod = scheduler.alphas_cumprod.to(device)

    # Latent-space samplewise MSE aggregates for mean +/- standard error.
    mse_sum: Dict[float, float] = {p: 0.0 for p in checked_levels}
    mse_sq_sum: Dict[float, float] = {p: 0.0 for p in checked_levels}
    mse_n: Dict[float, int] = {p: 0 for p in checked_levels}

    # Decoded-space 3x3 contingency counts: rows=true states (0/1/2), cols=pred states (0/1/2).
    contingency_counts: Dict[float, torch.Tensor] = {
        p: torch.zeros(3, 3, dtype=torch.float64) for p in checked_levels
    }

    print(f"[DiT][eval] Starting denoise accuracy on {len(ds)} samples")
    for batch_idx, xb in enumerate(dl):
        if max_eval_batches is not None and batch_idx >= int(max_eval_batches):
            break

        xb = xb.to(device, non_blocking=(device.type == "cuda"))
        z_clean = ae.encode(xb)  # (B, N, D)
        z_clean_norm = (z_clean - latent_mean) / latent_std
        bsz = z_clean.shape[0]

        for p in checked_levels:
            # Map noise percentage p in (0,1] to diffusion timestep in [0, T-1].
            t_scalar = int(round((num_train_timesteps - 1) * p))
            t_scalar = max(0, min(num_train_timesteps - 1, t_scalar))
            t_int = torch.full((bsz,), t_scalar, device=device, dtype=torch.long)

            noise = torch.randn_like(z_clean_norm)
            z_noisy = scheduler.add_noise(z_clean_norm, noise, t_int)

            t_norm = t_int.float() / float(num_train_timesteps)
            t_emb = time_embed(t_norm)
            v_pred = dit(z_noisy, t_emb, cond_emb=None)

            # Recover z0 estimate from v-pred:
            # z_t = sqrt(a_t) * z0 + sqrt(1-a_t) * eps
            # v_t = sqrt(a_t) * eps - sqrt(1-a_t) * z0
            # => z0 = sqrt(a_t) * z_t - sqrt(1-a_t) * v_t
            a_t = alphas_cumprod[t_int].view(-1, 1, 1)
            sqrt_a = torch.sqrt(a_t)
            sqrt_one_minus_a = torch.sqrt(1.0 - a_t)
            z0_hat_norm = sqrt_a * z_noisy - sqrt_one_minus_a * v_pred
            z0_hat = z0_hat_norm * latent_std + latent_mean

            # Latent-space quality: MSE per sample over all latent pixels.
            per_sample_mse = torch.mean((z0_hat - z_clean) ** 2, dim=(1, 2))
            mse_sum[p] += float(per_sample_mse.sum().item())
            mse_sq_sum[p] += float((per_sample_mse ** 2).sum().item())
            mse_n[p] += int(per_sample_mse.shape[0])

            logits_hat = ae.decode(z0_hat)  # (B, L, 3)
            calls_hat = logits_hat.argmax(dim=-1)  # (B, L)

            # Decoded-space contingency counts.
            true_flat = xb.reshape(-1).to(torch.long)
            pred_flat = calls_hat.reshape(-1).to(torch.long)
            valid = (true_flat >= 0) & (true_flat <= 2) & (pred_flat >= 0) & (pred_flat <= 2)
            if valid.any():
                idx = true_flat[valid] * 3 + pred_flat[valid]
                batch_counts = torch.bincount(idx, minlength=9).reshape(3, 3).to(torch.float64).cpu()
                contingency_counts[p] += batch_counts

    metrics: Dict[str, Dict[str, object]] = {}
    for p in checked_levels:
        n = mse_n[p]
        if n > 0:
            mean_mse = mse_sum[p] / float(n)
            if n > 1:
                var = (mse_sq_sum[p] - float(n) * (mean_mse ** 2)) / float(n - 1)
                se_mse = math.sqrt(max(var, 0.0) / float(n))
            else:
                se_mse = float("nan")
        else:
            mean_mse = float("nan")
            se_mse = float("nan")

        counts = contingency_counts[p]
        row_sums = counts.sum(dim=1, keepdim=True)
        contingency_frac = counts / row_sums.clamp_min(1.0)
        state_acc = [float(contingency_frac[i, i].item()) for i in range(3)]

        key = f"noise_{int(round(100.0 * p))}pct"
        metrics[key] = {
            "latent_mse_mean": float(mean_mse),
            "latent_mse_se": float(se_mse),
            "decoded_contingency_fraction": contingency_frac.tolist(),
            "decoded_state_accuracy": {"0": state_acc[0], "1": state_acc[1], "2": state_acc[2]},
        }
        print(
            f"[DiT][eval] noise={100.0 * p:5.1f}% latent_mse={mean_mse:.6f} +/- {se_mse:.6f} "
            f"decoded_diag(0,1,2)=({state_acc[0]:.6f}, {state_acc[1]:.6f}, {state_acc[2]:.6f})"
        )
        print(f"[DiT][eval] noise={100.0 * p:5.1f}% contingency(rows=true, cols=pred):")
        for r in range(3):
            print(
                f"  true={r}: "
                f"{contingency_frac[r, 0].item():.6f} {contingency_frac[r, 1].item():.6f} {contingency_frac[r, 2].item():.6f}"
            )

    return metrics


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List

# -----------------------------
# Minimal transformer block (PreNorm)
# -----------------------------
class _PreNormTransformerBlock(nn.Module):
    """
    PreNorm self-attention + MLP block.
    This is intentionally simple and stable for diffusion training.
    """
    def __init__(self, dim: int, num_heads: int, mlp_ratio: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attn
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + h
        # MLP
        x = x + self.mlp(self.norm2(x))
        return x


class UDiTDiffusionModel(nn.Module):
    """
    U-ViT / "U-DiT" style denoiser for token sequences:
      - Treat time (and condition) as *tokens* (prepended).
      - Use U-shaped long skip connections between shallow and deep layers.
      - Fuse skip by concat + linear projection (best in U-ViT ablation).

    Forward signature matches your current TransformerDiffusionModel:
        out = model(tokens, time_emb, cond_emb)  # returns (B, N, D)

    Args:
        token_dim: D
        latent_length: N (number of data tokens)
        num_layers: MUST be odd >= 3 (so there is a single middle layer)
        num_heads, mlp_ratio: standard transformer params
        dropout: optional dropout (often 0 for diffusion)
        use_cond_token: whether to prepend a cond token when cond_emb is provided
    """
    def __init__(
        self,
        token_dim: int,
        latent_length: int,
        num_layers: int = 9,
        num_heads: int = 8,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
        use_cond_token: bool = True,
    ) -> None:
        super().__init__()
        if num_layers < 3 or (num_layers % 2) != 1:
            raise ValueError(f"UDiT requires odd num_layers >= 3, got {num_layers}.")

        self.token_dim = int(token_dim)
        self.latent_length = int(latent_length)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.mlp_ratio = int(mlp_ratio)
        self.use_cond_token = bool(use_cond_token)

        # Special tokens: time token always, condition token optionally
        # We'll allocate pos-embeddings for the maximum possible length:
        #   N + 1 (time) + 1 (cond)
        self.max_special = 2
        self.max_seq_len = self.latent_length + self.max_special

        # Learned 1D positional embedding (U-ViT uses learnable 1D pos emb by default).  [oai_citation:4‡ar5iv](https://ar5iv.org/abs/2209.12152)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.max_seq_len, self.token_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Down / mid / up blocks
        n_down = (num_layers - 1) // 2
        n_up = n_down
        self.n_down = n_down
        self.n_up = n_up

        self.down_blocks = nn.ModuleList([
            _PreNormTransformerBlock(self.token_dim, self.num_heads, self.mlp_ratio, dropout)
            for _ in range(n_down)
        ])
        self.mid_block = _PreNormTransformerBlock(self.token_dim, self.num_heads, self.mlp_ratio, dropout)
        self.up_blocks = nn.ModuleList([
            _PreNormTransformerBlock(self.token_dim, self.num_heads, self.mlp_ratio, dropout)
            for _ in range(n_up)
        ])

        # Skip fusion: concat + linear projection back to token_dim (best in ablation).  [oai_citation:5‡ar5iv](https://ar5iv.org/abs/2209.12152)
        self.skip_projs = nn.ModuleList([
            nn.Linear(2 * self.token_dim, self.token_dim)
            for _ in range(n_up)
        ])

        # Final projection to predict v/noise for *data tokens only*
        self.out_norm = nn.LayerNorm(self.token_dim)
        self.out_proj = nn.Linear(self.token_dim, self.token_dim)

    def _build_input_sequence(
        self,
        tokens: torch.Tensor,          # (B, N, D)
        time_emb: torch.Tensor,        # (B, 1, D)
        cond_emb: Optional[torch.Tensor] = None,  # (B, 1, D) or None
    ) -> torch.Tensor:
        B, N, D = tokens.shape
        if N != self.latent_length:
            # Keep this strict to avoid silent positional mismatch.
            raise ValueError(f"Expected tokens length N={self.latent_length}, got {N}.")
        if time_emb.shape != (B, 1, D):
            raise ValueError(f"time_emb must have shape (B,1,D)={ (B,1,D) }, got {tuple(time_emb.shape)}")

        parts: List[torch.Tensor] = [time_emb]
        use_cond = (cond_emb is not None) and self.use_cond_token
        if use_cond:
            if cond_emb.shape != (B, 1, D):
                raise ValueError(f"cond_emb must have shape (B,1,D)={ (B,1,D) }, got {tuple(cond_emb.shape)}")
            parts.append(cond_emb)
        parts.append(tokens)

        x = torch.cat(parts, dim=1)  # (B, 1(+1)+N, D)

        # Add positional embeddings (slice according to actual seq length)
        seq_len = x.shape[1]
        if seq_len > self.max_seq_len:
            raise ValueError(f"seq_len {seq_len} exceeds max_seq_len {self.max_seq_len}.")
        x = x + self.pos_embed[:, :seq_len, :]
        return x

    def forward(
        self,
        tokens: torch.Tensor,
        time_emb: torch.Tensor,
        cond_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Returns:
            v/noise prediction for data tokens only: (B, N, D)
        """
        x = self._build_input_sequence(tokens, time_emb, cond_emb)  # (B, S, D)
        skips: List[torch.Tensor] = []

        # Down path: save skip states
        for blk in self.down_blocks:
            x = blk(x)
            skips.append(x)

        # Mid
        x = self.mid_block(x)

        # Up path: fuse with corresponding skip (reverse order)
        for i, (blk, proj) in enumerate(zip(self.up_blocks, self.skip_projs)):
            skip = skips[-1 - i]
            # concat along channel dim, then project back
            x = proj(torch.cat([x, skip], dim=-1))
            x = blk(x)

        x = self.out_proj(self.out_norm(x))  # (B, S, D)

        # Drop special tokens; output only the N data tokens
        # time token always present
        start = 1
        if (cond_emb is not None) and self.use_cond_token:
            start = 2
        return x[:, start:, :]  # (B, N, D)


def train_sit_flow_matching_h5(
    *,
    ae: GenotypeAutoencoder,
    h5_paths: List[str],
    covariates: Optional[torch.Tensor] = None,  # not wired to H5 in this minimal version
    num_epochs: int = 10,
    batch_size: int = 64,
    lr: float = 2e-4,
    device: torch.device,
    num_workers: int = 2,
    pin_memory: bool = True,
    amp: bool = True,
    log_every: int = 50,
    # Transformer config
    num_layers: int = 9,          # NOTE: if use_udit=True, must be odd >= 3
    num_heads: int = 8,
    mlp_ratio: int = 4,
    # EMA
    ema_decay: float = 0.9999,
    use_ema: bool = True,
    ckpt_path: Optional[str] = None,
    # Latent normalization
    norm_eps: float = 1e-6,
    use_latent_normalization: bool = True,
    # USiT optionality (U-shaped transformer backbone)
    use_udit: bool = False,
) -> Tuple[
    nn.Module,  # sit backbone (TransformerDiffusionModel or UDiTDiffusionModel)
    Optional[nn.Module],  # ema_sit
    Dict[str, torch.Tensor],  # latent_norm_stats
]:
    """
    Stage-2 (SiT / Flow Matching): Freeze AE; train a Transformer backbone to predict
    the velocity field on latent tokens using a flow-matching objective.

    - z_data = AE.encode(x)
    - base_latent ~ N(0, I)
    - t ~ Uniform(0,1)
    - z_t = (1-t)*base + t*z_data
    - v_target = z_data - base
    - v_pred = model(z_t, time_emb(t), cond_emb)
    - loss = MSE(v_pred, v_target)

    Optionally apply channel-wise latent normalization (mean/std over batch+token dims).
    Optionally use UDiT backbone => "USiT".
    """
    # -------------------------
    # 0) Freeze AE
    # -------------------------
    ae.eval()
    for p in ae.parameters():
        p.requires_grad = False

    # -------------------------
    # 1) Data
    # -------------------------
    ds = H5GenotypeDataset(h5_paths, x_key="X")
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory and (device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )

    token_dim = int(ae.latent_dim)
    latent_length = int(ae.latent_length)

    # -------------------------
    # 2) Backbone (SiT vs USiT)
    # -------------------------
    if use_udit:
        sit = UDiTDiffusionModel(
            token_dim=token_dim,
            latent_length=latent_length,
            num_layers=num_layers,   # IMPORTANT: odd
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=0.0,
            use_cond_token=True,
        ).to(device)
    else:
        sit = TransformerDiffusionModel(
            token_dim=token_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
        ).to(device)

    time_embed = TimeEmbedding(token_dim).to(device)

    cond_embed = None
    if covariates is not None:
        cond_embed = ConditionEmbedding(covariates.shape[1], token_dim).to(device)

    params = list(sit.parameters()) + list(time_embed.parameters())
    if cond_embed is not None:
        params += list(cond_embed.parameters())

    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-2)
    scaler = torch.cuda.amp.GradScaler(enabled=(amp and device.type == "cuda"))

    if norm_eps <= 0.0:
        raise ValueError(f"norm_eps must be > 0. Got {norm_eps}")

    # ---------------------------------------------------------
    # 3) Optional latent normalization stats over training set
    # ---------------------------------------------------------
    if use_latent_normalization:
        print("[SiT][train] Computing latent channel normalization stats...")
        ch_sum = torch.zeros(token_dim, device=device)
        ch_sq_sum = torch.zeros(token_dim, device=device)
        ch_count = 0

        with torch.no_grad():
            for xb_stats in dl:
                xb_stats = xb_stats.to(device, non_blocking=(device.type == "cuda"))
                z_stats = ae.encode(xb_stats).detach()  # (B, N, D)
                ch_sum += z_stats.sum(dim=(0, 1))
                ch_sq_sum += (z_stats ** 2).sum(dim=(0, 1))
                ch_count += int(z_stats.shape[0] * z_stats.shape[1])

        if ch_count <= 0:
            raise RuntimeError("No samples available to compute latent normalization stats.")

        latent_mean = (ch_sum / float(ch_count)).view(1, 1, -1)
        latent_var = (ch_sq_sum / float(ch_count)).view(1, 1, -1) - latent_mean ** 2
        latent_std = torch.sqrt(torch.clamp(latent_var, min=0.0) + norm_eps)

        print(f"[SiT][train] Latent normalization ready (channels={token_dim}, count={ch_count}, eps={norm_eps:g})")
    else:
        print("[SiT][train] Latent normalization disabled; using identity normalization.")
        latent_mean = torch.zeros(1, 1, token_dim, device=device)
        latent_std = torch.ones(1, 1, token_dim, device=device)

    latent_norm_stats: Dict[str, torch.Tensor] = {
        "mean": latent_mean.detach().cpu(),
        "std": latent_std.detach().cpu(),
        "eps": torch.tensor(float(norm_eps)),
        "enabled": torch.tensor(1 if use_latent_normalization else 0, dtype=torch.int64),
    }

    # -------------------------
    # 4) EMA init
    # -------------------------
    ema_shadow: Optional[Dict[str, torch.Tensor]] = None
    if use_ema:
        if not (0.0 < ema_decay < 1.0):
            raise ValueError(f"ema_decay must be in (0, 1). Got {ema_decay}")
        ema_shadow = {
            name: param.detach().clone()
            for name, param in sit.named_parameters()
            if param.requires_grad
        }

    # -------------------------
    # 5) Train loop
    # -------------------------
    sit.train()
    step = 0
    for epoch in range(num_epochs):
        running = 0.0
        epoch_loss_sum = 0.0
        epoch_steps = 0
        pbar = tqdm(dl, desc=f"[SiT] epoch {epoch+1}/{num_epochs}", leave=True)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        for xb in pbar:
            step += 1
            xb = xb.to(device, non_blocking=(device.type == "cuda"))

            # -------------------------
            # Encode -> normalize
            # -------------------------
            with torch.no_grad():
                z_data = ae.encode(xb).detach()  # (B, N, D)
                z_data = (z_data - latent_mean) / latent_std

            bsz = z_data.shape[0]

            # base ~ N(0,I)
            base = torch.randn_like(z_data)

            # t ~ U(0,1)
            t = torch.rand(bsz, device=device)
            t_emb = time_embed(t)

            # cond placeholder (same status as your DiT)
            c_emb = cond_embed(None) if cond_embed is not None else None

            # linear interpolant + target velocity
            t_view = t.view(-1, 1, 1)
            z_t = (1.0 - t_view) * base + t_view * z_data
            v_target = z_data - base

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(amp and device.type == "cuda")):
                v_pred = sit(z_t, t_emb, c_emb)
                loss = F.mse_loss(v_pred, v_target)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            if use_ema and ema_shadow is not None:
                with torch.no_grad():
                    for name, param in sit.named_parameters():
                        if not param.requires_grad:
                            continue
                        ema_shadow[name].mul_(ema_decay).add_(param.detach(), alpha=(1.0 - ema_decay))

            epoch_steps += 1
            epoch_loss_sum += loss.item()
            running += loss.item()

            if step % log_every == 0:
                avg = running / log_every
                running = 0.0
                print(
                    f"[SiT][train] epoch={epoch+1}/{num_epochs} step={step} "
                    f"loss_avg={avg:.6f} loss_last={loss.item():.6f}"
                )
                if device.type == "cuda":
                    mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                    pbar.set_postfix(loss=f"{avg:.4f}", mem=f"{mem:.0f}MB")
                else:
                    pbar.set_postfix(loss=f"{avg:.4f}")

        if epoch_steps > 0:
            epoch_avg = epoch_loss_sum / float(epoch_steps)
            if device.type == "cuda":
                mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                print(
                    f"[SiT][train] epoch={epoch+1}/{num_epochs} done "
                    f"steps={epoch_steps} avg_loss={epoch_avg:.6f} peak_mem={mem:.1f}MB"
                )
            else:
                print(
                    f"[SiT][train] epoch={epoch+1}/{num_epochs} done "
                    f"steps={epoch_steps} avg_loss={epoch_avg:.6f}"
                )

        # -------------------------
        # 6) Checkpoint
        # -------------------------
        if ckpt_path is not None:
            ckpt_dir = os.path.dirname(ckpt_path)
            if ckpt_dir:
                os.makedirs(ckpt_dir, exist_ok=True)

            ckpt_payload = {
                "model_state": sit.state_dict(),
                "time_embed_state": time_embed.state_dict(),
                "config": {
                    "token_dim": token_dim,
                    "latent_length": latent_length,
                    "num_layers": num_layers,
                    "num_heads": num_heads,
                    "mlp_ratio": mlp_ratio,
                    "ema_decay": ema_decay,
                    "use_ema": use_ema,
                    "use_latent_normalization": use_latent_normalization,
                    "norm_eps": norm_eps,
                    "arch": "USiT" if use_udit else "SiT",
                },
                "epoch": epoch + 1,
                "latent_norm_stats": latent_norm_stats,
            }
            if cond_embed is not None:
                ckpt_payload["cond_embed_state"] = cond_embed.state_dict()
            if ema_shadow is not None:
                ckpt_payload["ema_shadow"] = {k: v.detach().cpu() for k, v in ema_shadow.items()}

            torch.save(ckpt_payload, ckpt_path)
            print(f"[SiT][train] Saved checkpoint: {ckpt_path}")

    # -------------------------
    # 7) Build EMA model
    # -------------------------
    ema_sit = _build_ema_model(sit, ema_shadow) if use_ema else None

    # Attach helpers for inference/eval symmetry with DiT function
    sit.time_embed = time_embed
    sit.latent_norm_stats = {k: v.clone() if torch.is_tensor(v) else v for k, v in latent_norm_stats.items()}
    if ema_sit is not None:
        ema_sit.time_embed = copy.deepcopy(time_embed)
        ema_sit.latent_norm_stats = {k: v.clone() if torch.is_tensor(v) else v for k, v in latent_norm_stats.items()}

    return sit, ema_sit, latent_norm_stats


@torch.no_grad()
def eval_output_model(
    diff_model_name,
    ae_model,
    normalized: bool = False,
    *,
    chr_no: int = 22,
    max_h5_batches: int = 2,
    max_samples: Optional[int] = None,
    batch_size: int = 128,
    num_workers: int = 2,
    pin_memory: bool = True,
    num_train_timesteps: int = 1000,
    num_inference_steps: int = 50,     # for DDIM iterative denoising (UNet/DiT/UDiT)
    num_sit_steps: int = 50,           # for ODE integration (SiT/USiT)
    ld_blocks: Optional[List[List[int]]] = None,
    # Flow inverse (backward Euler) solver knobs
    sit_fp_iters: int = 50,
    sit_fp_tol: float = 1e-6,
    noise_levels: Tuple[float, ...] = (0.10, 0.25, 0.50, 0.75, 1.00),
    compare_dir: str = "/n/home03/ahmadazim/WORKING/genGen/UKB_compare_chr22",
    h5_start_path: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    General evaluator for UNet/DiT/UDiT/SiT/USiT outputs.

    Returns:
        cycle_df: per-noise CycleMSE(p) summary + decoded contingency/accuracy
        pop_df: MAF/LD summary on independent generated draws (base ~ N(0,I) -> model -> decode)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if h5_start_path is None:
        h5_start_path = (
            f"/n/home03/ahmadazim/WORKING/genGen/UKBVQVAE/genomic_data/ae_h5/chr{chr_no}/batch00001.h5"
        )
    h5_paths = enumerate_h5_batch_paths(h5_start_path, max_batches=max_h5_batches)
    if max_samples is not None and int(max_samples) <= 0:
        raise ValueError(f"max_samples must be positive when set. Got {max_samples}")

    # Validate noise levels.
    checked_levels: List[float] = []
    for p in noise_levels:
        p_float = float(p)
        if not (0.0 < p_float <= 1.0):
            raise ValueError(f"noise_levels entries must be in (0, 1]. Got {p_float}")
        checked_levels.append(p_float)

    checkpoint_time_embed_state: Optional[Dict[str, torch.Tensor]] = None
    unet_channel_mean: Optional[torch.Tensor] = None
    unet_channel_std: Optional[torch.Tensor] = None
    unet_uses_norm = False
    model_name = str(diff_model_name).lower() if not isinstance(diff_model_name, nn.Module) else diff_model_name.__class__.__name__.lower()
    unet_family = ("unet" in model_name)

    # Resolve/load model.
    if isinstance(diff_model_name, nn.Module):
        model = diff_model_name.to(device)
    else:
        model = globals().get(model_name, None)
        if model is None:
            if unet_family:
                # Prefer job-specific files, fallback to legacy filename.
                unet_candidates = sorted(
                    glob.glob(os.path.join(compare_dir, f"unet_chr{chr_no}_vpred_job*.pt")),
                    key=os.path.getmtime,
                )
                ckpt_path = unet_candidates[-1] if len(unet_candidates) > 0 else os.path.join(
                    compare_dir, f"unet_chr{chr_no}_vpred.pt"
                )
                if not os.path.exists(ckpt_path):
                    raise FileNotFoundError(
                        f"Could not find UNet checkpoint in {compare_dir} (job-specific or legacy names)."
                    )
                payload = torch.load(ckpt_path, map_location="cpu")
                if "model_state" not in payload:
                    raise KeyError(f"UNet checkpoint missing 'model_state': {ckpt_path}")
                latent_meta = payload.get("latent_meta", {})
                C = int(latent_meta.get("C", 64))
                H = int(latent_meta.get("H", 16))
                W = int(latent_meta.get("W", 16))
                ch_mean_ckpt = latent_meta.get("channel_mean", payload.get("channel_mean", None))
                ch_std_ckpt = latent_meta.get("channel_std", payload.get("channel_std", None))
                if ch_mean_ckpt is None or ch_std_ckpt is None:
                    latents_h5_ckpt = payload.get("latents_h5", None)
                    if isinstance(latents_h5_ckpt, str) and os.path.exists(latents_h5_ckpt):
                        try:
                            with h5py.File(latents_h5_ckpt, "r") as hf_stats:
                                if ("channel_mean" in hf_stats.attrs) and ("channel_std" in hf_stats.attrs):
                                    ch_mean_ckpt = hf_stats.attrs["channel_mean"]
                                    ch_std_ckpt = hf_stats.attrs["channel_std"]
                        except Exception:
                            pass
                if ch_mean_ckpt is not None and ch_std_ckpt is not None:
                    unet_channel_mean = torch.as_tensor(ch_mean_ckpt, dtype=torch.float32, device=device).view(1, C, 1, 1)
                    unet_channel_std = torch.as_tensor(ch_std_ckpt, dtype=torch.float32, device=device).view(1, C, 1, 1)
                    unet_channel_std = torch.clamp(unet_channel_std, min=1e-6)
                    # If stats are present, default to normalized mode even when the legacy flag is missing.
                    unet_uses_norm = True
                if H != W:
                    raise ValueError(f"UNet evaluation expects square latents. Got H={H}, W={W}.")
                model = UNet2DModel(
                    sample_size=H,
                    in_channels=C,
                    out_channels=C,
                    layers_per_block=2,
                    block_out_channels=(128, 256, 256, 512),
                    down_block_types=("DownBlock2D", "AttnDownBlock2D", "DownBlock2D", "DownBlock2D"),
                    up_block_types=("UpBlock2D", "UpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
                )
                model.load_state_dict(payload["model_state"], strict=True)
                if unet_uses_norm:
                    print("[EVAL] UNet checkpoint includes latent normalization stats.")
            else:
                ckpt_name = (
                    f"{model_name}_norm_chr{chr_no}_L{int(ae_model.latent_length)}_D{int(ae_model.latent_dim)}.pt"
                    if (normalized and model_name in {"usit"})
                    else f"{model_name}_chr{chr_no}_L{int(ae_model.latent_length)}_D{int(ae_model.latent_dim)}.pt"
                )
                ckpt_path = os.path.join(compare_dir, ckpt_name)
                if not os.path.exists(ckpt_path):
                    raise FileNotFoundError(
                        f"Could not resolve model '{model_name}' from globals and checkpoint not found: {ckpt_path}"
                    )
                payload = torch.load(ckpt_path, map_location="cpu")
                token_dim = int(payload.get("latent_dim", int(ae_model.latent_dim)))
                latent_length = int(payload.get("latent_length", int(ae_model.latent_length)))
                is_udit = model_name in {"udit", "usit"}
                if is_udit:
                    model = UDiTDiffusionModel(
                        token_dim=token_dim,
                        latent_length=latent_length,
                        num_layers=9,
                        num_heads=8,
                        mlp_ratio=4,
                        dropout=0.0,
                        use_cond_token=True,
                    )
                else:
                    model = TransformerDiffusionModel(
                        token_dim=token_dim,
                        num_layers=9,
                        num_heads=8,
                        mlp_ratio=4,
                    )
                state_key = f"{model_name}_ema_state" if payload.get(f"{model_name}_ema_state") is not None else f"{model_name}_state"
                if state_key not in payload:
                    raise KeyError(f"Checkpoint missing '{state_key}': {ckpt_path}")
                raw_state = payload[state_key]
                if not isinstance(raw_state, dict):
                    raise ValueError(f"Checkpoint field '{state_key}' must be a state dict: {ckpt_path}")
                checkpoint_time_embed_state = {
                    k[len("time_embed."):]: v
                    for k, v in raw_state.items()
                    if k.startswith("time_embed.")
                }
                model_state = {
                    k: v for k, v in raw_state.items()
                    if not (k.startswith("time_embed.") or k.startswith("cond_embed."))
                }
                incompat = model.load_state_dict(model_state, strict=False)
                unexpected = list(getattr(incompat, "unexpected_keys", []))
                missing = [
                    k for k in list(getattr(incompat, "missing_keys", []))
                    if not (k.startswith("time_embed.") or k.startswith("cond_embed."))
                ]
                if len(unexpected) > 0 or len(missing) > 0:
                    raise RuntimeError(
                        f"Incompatible checkpoint/model for '{model_name}'. "
                        f"Missing keys: {missing[:10]}, Unexpected keys: {unexpected[:10]}"
                    )
                if "time_embed_state" in payload:
                    te = TimeEmbedding(token_dim).to(device)
                    te.load_state_dict(payload["time_embed_state"], strict=True)
                    model.time_embed = te
                elif checkpoint_time_embed_state:
                    te = TimeEmbedding(token_dim).to(device)
                    te.load_state_dict(checkpoint_time_embed_state, strict=True)
                    model.time_embed = te
                if "latent_norm_stats" in payload:
                    model.latent_norm_stats = payload["latent_norm_stats"]

    model = model.to(device)
    model.eval()
    ae_model = ae_model.to(device)
    ae_model.eval()

    # Token-model time embedding requirement.
    time_embed = None
    if not unet_family:
        time_embed = getattr(model, "time_embed", None)
        if time_embed is None:
            raise ValueError(
                "Model is missing attached time embedding (`model.time_embed`). "
                "Evaluate from in-memory trained models or save/load `time_embed_state` with checkpoints."
            )
        time_embed = time_embed.to(device)
        time_embed.eval()

    # Optional latent normalization (token models only).
    need_recompute_norm_stats = False
    norm_eps_eval = 1e-6
    if not unet_family:
        latent_norm_stats = getattr(model, "latent_norm_stats", None)
        if normalized:
            if latent_norm_stats is None or ("mean" not in latent_norm_stats) or ("std" not in latent_norm_stats):
                # Backward compatibility for checkpoints that did not save stats.
                # Recompute from current evaluation H5 data below.
                need_recompute_norm_stats = True
                latent_mean = None
                latent_std = None
            else:
                latent_mean = latent_norm_stats["mean"].to(device)
                latent_std = latent_norm_stats["std"].to(device)
                if "eps" in latent_norm_stats:
                    try:
                        norm_eps_eval = float(latent_norm_stats["eps"])
                    except Exception:
                        norm_eps_eval = 1e-6
        else:
            latent_mean = torch.zeros(1, 1, int(ae_model.latent_dim), device=device)
            latent_std = torch.ones(1, 1, int(ae_model.latent_dim), device=device)

    ds = H5GenotypeDataset(h5_paths, x_key="X")
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory and (device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )

    if (not unet_family) and normalized and need_recompute_norm_stats:
        print("[EVAL] latent_norm_stats missing in checkpoint; recomputing from evaluation H5 data.")
        token_dim = int(ae_model.latent_dim)
        ch_sum = torch.zeros(token_dim, device=device)
        ch_sq_sum = torch.zeros(token_dim, device=device)
        ch_count = 0
        for xb_stats in dl:
            xb_stats = xb_stats.to(device, non_blocking=(device.type == "cuda"))
            z_stats = ae_model.encode(xb_stats).detach()  # (B, N, D)
            ch_sum += z_stats.sum(dim=(0, 1))
            ch_sq_sum += (z_stats ** 2).sum(dim=(0, 1))
            ch_count += int(z_stats.shape[0] * z_stats.shape[1])
        if ch_count <= 0:
            raise RuntimeError("Failed to recompute latent normalization stats: no samples available.")
        latent_mean = (ch_sum / float(ch_count)).view(1, 1, -1)
        latent_var = (ch_sq_sum / float(ch_count)).view(1, 1, -1) - latent_mean ** 2
        latent_std = torch.sqrt(torch.clamp(latent_var, min=0.0) + float(norm_eps_eval))

    # Training-time scheduler (used for add_noise statistics / alpha_bar lookup)
    train_scheduler = DDPMScheduler(
        num_train_timesteps=num_train_timesteps,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="v_prediction",
    )
    alphas_cumprod = train_scheduler.alphas_cumprod.to(device)

    # Inference-time sampler timesteps: we will implement deterministic DDIM updates ourselves
    ddim = DDIMScheduler(
        num_train_timesteps=num_train_timesteps,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="v_prediction",
        clip_sample=False,
    )
    ddim.set_timesteps(num_inference_steps, device=device)

    is_sit_family = model_name in {"sit", "usit"}

    # ---------------------------
    # Reverse-sample KL(q0 || N(0,I)) accumulators (diagonal Gaussian fit)
    # We accumulate per-dimension mean/var in the *model working space*:
    #   - SiT: token space after optional normalization (base is ~N(0,I))
    #   - DiT/UDiT: token space after optional normalization (base is ~N(0,I))
    #   - UNet: latent space (DDPM terminal is ~N(0,I) approximately)
    # ---------------------------
    rev_sum: Optional[torch.Tensor] = None       # (D,)
    rev_sq_sum: Optional[torch.Tensor] = None    # (D,)
    rev_count: int = 0

    # ---------------------------
    # Helpers
    # ---------------------------
    def _time_embed_token(t_int: torch.Tensor) -> torch.Tensor:
        # training used t_norm = t_int / num_train_timesteps (not num_train_timesteps-1)
        t_norm = t_int.float() / float(num_train_timesteps)
        return time_embed(t_norm)

    def _v_pred(z: torch.Tensor, t_int_or_float: torch.Tensor) -> torch.Tensor:
        """
        Returns v_pred with same shape as z in the *model working space*.
        - UNet: model(z, t_int).sample
        - Token models: model(z, time_embed(t_norm), None)
        - Flow models: model(z, time_embed(t_float), None)  (t_float in [0,1])
        """
        if unet_family:
            # t is int64 timesteps
            return model(z, t_int_or_float).sample
        if is_sit_family:
            # t is float in [0,1]
            t_emb = time_embed(t_int_or_float)
            return model(z, t_emb, cond_emb=None)
        # token diffusion
        t_emb = _time_embed_token(t_int_or_float)
        return model(z, t_emb, cond_emb=None)

    def _ddim_x0_eps_from_v(z_t: torch.Tensor, v_t: torch.Tensor, a_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        For v-pred parameterization:
          x0 = sqrt(a) * z_t - sqrt(1-a) * v
          eps = sqrt(1-a) * z_t + sqrt(a) * v
        a_t must be broadcastable to z_t.
        """
        sqrt_a = torch.sqrt(a_t)
        sqrt_oma = torch.sqrt(1.0 - a_t)
        x0 = sqrt_a * z_t - sqrt_oma * v_t
        eps = sqrt_oma * z_t + sqrt_a * v_t
        return x0, eps

    def _broadcast_alpha(a: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # a: (B,) -> broadcast to z shape
        if z.dim() == 4:
            return a.view(-1, 1, 1, 1)
        return a.view(-1, 1, 1)

    def _ddim_step_deterministic(z_t: torch.Tensor, t_cur: int, t_next: int) -> torch.Tensor:
        """
        Deterministic DDIM-style update using model-predicted v:
          z_{t_next} = sqrt(a_next) * x0_pred + sqrt(1-a_next) * eps_pred
        with x0_pred, eps_pred computed from (z_t, v_pred, a_cur).
        Works for both UNet latents and token latents.
        """
        B = z_t.shape[0]
        t_cur_tensor = torch.full((B,), int(t_cur), device=z_t.device, dtype=torch.long)
        v_t = _v_pred(z_t, t_cur_tensor)
        a_cur = _broadcast_alpha(alphas_cumprod[t_cur_tensor], z_t)
        a_next = _broadcast_alpha(
            alphas_cumprod[torch.full((B,), int(t_next), device=z_t.device, dtype=torch.long)],
            z_t,
        )
        x0, eps = _ddim_x0_eps_from_v(z_t, v_t, a_cur)
        z_next = torch.sqrt(a_next) * x0 + torch.sqrt(1.0 - a_next) * eps
        return z_next

    def _pick_ddim_t_target(p: float) -> int:
        """
        Map noise fraction p in (0,1] to a target training timestep.
        """
        t_start = int(round((num_train_timesteps - 1) * float(p)))
        return max(0, min(num_train_timesteps - 1, t_start))

    def _ddpm_q_sample(z0: torch.Tensor, t_int: int) -> torch.Tensor:
        """
        Forward noising kernel q(z_t | z_0) using the *training* DDPM scheduler.
        This is the correct "forward process" corruption for denoising diffusion models.
        """
        B = z0.shape[0]
        t = torch.full((B,), int(t_int), device=z0.device, dtype=torch.long)
        eps = torch.randn_like(z0)
        return train_scheduler.add_noise(z0, eps, t)

    def _pick_ddim_start_index(t_target: int) -> int:
        """
        DDIM timesteps are descending. We want the first index with timestep <= t_target.
        """
        timesteps = ddim.timesteps  # 1D descending tensor of ints
        idx_candidates = (timesteps <= int(t_target)).nonzero(as_tuple=False)
        return int(idx_candidates[0].item()) if idx_candidates.numel() > 0 else 0

    def _ddim_denoise_from(z_t: torch.Tensor, start_idx: int) -> torch.Tensor:
        """
        Deterministic denoise along ddim.timesteps[start_idx:] down to 0.
        """
        timesteps = ddim.timesteps
        z = z_t
        for i in range(start_idx, timesteps.numel() - 1):
            t_cur = int(timesteps[i].item())
            t_next = int(timesteps[i + 1].item())
            z = _ddim_step_deterministic(z, t_cur=t_cur, t_next=t_next)
        return z  # should correspond to t=0 end

    def _ddim_invert_to(z0: torch.Tensor, t_target: int) -> torch.Tensor:
        """
        Deterministic "learned corruption" via DDIM inversion:
        push z0 at t=0 forward to z_{t_target} using model itself.
        """
        timesteps_desc = ddim.timesteps
        timesteps_asc = torch.flip(timesteps_desc, dims=[0])  # ascending

        # find the last index where timestep <= t_target, so we end at t_target (or nearest below/at)
        idx_end_candidates = (timesteps_asc <= int(t_target)).nonzero(as_tuple=False)
        if idx_end_candidates.numel() == 0:
            return z0
        idx_end = int(idx_end_candidates[-1].item())

        z = z0
        for i in range(0, idx_end):
            t_cur = int(timesteps_asc[i].item())
            t_next = int(timesteps_asc[i + 1].item())
            z = _ddim_step_deterministic(z, t_cur=t_cur, t_next=t_next)
        return z

    def _flow_backward_euler(z_t: torch.Tensor, t_hi: float, t_lo: float, n_steps: int) -> torch.Tensor:
        """
        Deterministic inverse segment for SiT/USiT:
          integrate backward from t_hi -> t_lo using implicit (backward) Euler
          z_{k-1} = z_k - dt * v(z_{k-1}, t_{k-1})
        solved by fixed-point iteration.
        """
        if n_steps < 1:
            return z_t
        device_ = z_t.device
        B = z_t.shape[0]
        dt = (t_hi - t_lo) / float(n_steps)
        z = z_t
        for k in range(n_steps):
            t_prev = t_hi - dt * (k + 1)  # descending
            t_prev_tensor = torch.full((B,), float(t_prev), device=device_, dtype=torch.float32)
            # fixed point: x = z - dt * v(x, t_prev)
            x = z
            for _ in range(int(sit_fp_iters)):
                v = _v_pred(x, t_prev_tensor)
                x_new = z - dt * v
                if (x_new - x).pow(2).mean().sqrt().item() < float(sit_fp_tol):
                    x = x_new
                    break
                x = x_new
            z = x
        return z

    def _flow_forward_euler(z_t: torch.Tensor, t_lo: float, t_hi: float, n_steps: int) -> torch.Tensor:
        """
        Deterministic forward segment for SiT/USiT:
          dz/dt = v(z,t), explicit Euler.
        """
        if n_steps < 1:
            return z_t
        device_ = z_t.device
        B = z_t.shape[0]
        dt = (t_hi - t_lo) / float(n_steps)
        z = z_t
        for k in range(n_steps):
            t_cur = t_lo + dt * k
            t_cur_tensor = torch.full((B,), float(t_cur), device=device_, dtype=torch.float32)
            v = _v_pred(z, t_cur_tensor)
            z = z + dt * v
        return z

    # ---------------------------
    # Stats accumulators for MAF/LD
    # ---------------------------
    L = int(getattr(ae_model, "input_length", None) or getattr(ae_model, "L", None) or 0)
    if L <= 0:
        # fallback: infer from first H5 batch later
        pass

    maf_sum_real: Optional[torch.Tensor] = None
    maf_cnt_real: Optional[torch.Tensor] = None
    maf_sum_gen: Optional[torch.Tensor] = None
    maf_cnt_gen: Optional[torch.Tensor] = None

    # LD accumulators per block: store sum_x (m,), xtx (m,m), n
    def _init_ld_acc(blocks: List[List[int]]) -> List[Dict[str, Any]]:
        acc: List[Dict[str, Any]] = []
        for idxs in blocks:
            m = len(idxs)
            acc.append(
                {
                    "idxs": list(map(int, idxs)),
                    "sum": torch.zeros(m, dtype=torch.float64),
                    "xtx": torch.zeros(m, m, dtype=torch.float64),
                    "n": 0,
                }
            )
        return acc

    ld_acc_real: Optional[List[Dict[str, Any]]] = _init_ld_acc(ld_blocks) if ld_blocks is not None else None
    ld_acc_gen: Optional[List[Dict[str, Any]]] = _init_ld_acc(ld_blocks) if ld_blocks is not None else None

    def _update_maf(maf_sum: torch.Tensor, maf_cnt: torch.Tensor, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # X: (B,L) int64 with values 0/1/2 (possibly missing outside [0,2])
        Xl = X.to(torch.int64)
        valid = (Xl >= 0) & (Xl <= 2)
        maf_sum += (Xl.clamp(0, 2).to(torch.float64) * valid.to(torch.float64)).sum(dim=0)
        maf_cnt += valid.to(torch.float64).sum(dim=0)
        return maf_sum, maf_cnt

    def _update_ld(ld_acc: List[Dict[str, Any]], X: torch.Tensor) -> None:
        # X: (B,L) int64 values 0/1/2; assume blocks small enough to do xtx.
        Xf = X.to(torch.float64)
        B = int(Xf.shape[0])
        for b in ld_acc:
            idxs = b["idxs"]
            G = Xf[:, idxs]  # (B,m)
            b["sum"] += G.sum(dim=0).cpu()
            b["xtx"] += (G.t().mm(G)).cpu()
            b["n"] += B

    def _finalize_ld(ld_acc: List[Dict[str, Any]]) -> List[torch.Tensor]:
        """
        Returns list of LD matrices (r^2) per block as float64 CPU tensors.
        """
        out: List[torch.Tensor] = []
        for b in ld_acc:
            n = int(b["n"])
            if n <= 1:
                out.append(torch.zeros_like(b["xtx"]))
                continue
            mean = b["sum"] / float(n)  # (m,)
            xtx = b["xtx"] / float(n)   # (m,m)
            cov = xtx - mean.view(-1, 1) * mean.view(1, -1)
            var = torch.diag(cov).clamp_min(1e-12)
            denom = torch.sqrt(var.view(-1, 1) * var.view(1, -1)).clamp_min(1e-12)
            corr = cov / denom
            r2 = corr.pow(2)
            out.append(r2)
        return out

    # Cycle metrics accumulators
    mse_sum: Dict[float, float] = {p: 0.0 for p in checked_levels}
    mse_sq_sum: Dict[float, float] = {p: 0.0 for p in checked_levels}
    mse_n: Dict[float, int] = {p: 0 for p in checked_levels}
    contingency_counts: Dict[float, torch.Tensor] = {
        p: torch.zeros(3, 3, dtype=torch.float64) for p in checked_levels
    }
    samples_seen = 0

    for xb in dl:
        if max_samples is not None and samples_seen >= int(max_samples):
            break
        xb = xb.to(device, non_blocking=(device.type == "cuda"))
        if max_samples is not None:
            remaining = int(max_samples) - samples_seen
            if remaining <= 0:
                break
            if xb.shape[0] > remaining:
                xb = xb[:remaining]
        # Always compute clean latent (for MSE) from AE encode/forward.
        # For UNet-family, assume ae_model(x) returns (logits, z_latent) where z_latent is what diffusion denoises.
        if unet_family:
            logits_clean, z_clean = ae_model(xb)
        else:
            z_clean = ae_model.encode(xb)
        bsz = z_clean.shape[0]
        samples_seen += int(bsz)

        # --- accumulate REAL MAF/LD from xb (decoded space) ---
        if maf_sum_real is None:
            maf_sum_real = torch.zeros(xb.shape[1], dtype=torch.float64)
            maf_cnt_real = torch.zeros(xb.shape[1], dtype=torch.float64)
        maf_sum_real, maf_cnt_real = _update_maf(maf_sum_real, maf_cnt_real, xb.detach().cpu())
        if ld_acc_real is not None:
            _update_ld(ld_acc_real, xb.detach().cpu())

        if unet_family:
            if z_clean.dim() != 4:
                raise ValueError(f"UNet path expects 4D latents (B,C,H,W). Got {tuple(z_clean.shape)}.")
            if unet_uses_norm and (unet_channel_mean is not None) and (unet_channel_std is not None):
                z_clean_work = (z_clean - unet_channel_mean) / unet_channel_std
            else:
                z_clean_work = z_clean
        else:
            z_clean_work = (z_clean - latent_mean) / latent_std

        # --------------------------
        # Reverse-sample once per batch (NOT per p) and accumulate moments.
        # "Reverse sample" means: deterministically map data -> base.
        # --------------------------
        if is_sit_family:
            # Flow: data at t=1, base at t=0
            z_rev = _flow_backward_euler(
                z_clean_work, t_hi=1.0, t_lo=0.0, n_steps=max(1, int(num_sit_steps))
            )
        else:
            # Diffusion: data at t=0, base approx at *large t* under the *forward* noising kernel q(z_t|z_0).
            # Use DDPM forward noising (not learned/DDIM inversion).
            t_max = int(ddim.timesteps[0].item())  # typically near num_train_timesteps-1
            z_rev = _ddpm_q_sample(z_clean_work, t_int=t_max)

        # Flatten per-dimension and accumulate on CPU float64 for numerical stability.
        z_rev_flat = z_rev.reshape(z_rev.shape[0], -1).detach()
        if rev_sum is None:
            D = int(z_rev_flat.shape[1])
            rev_sum = torch.zeros(D, dtype=torch.float64)
            rev_sq_sum = torch.zeros(D, dtype=torch.float64)
        zr = z_rev_flat.to(torch.float64).cpu()
        rev_sum += zr.sum(dim=0)
        rev_sq_sum += (zr * zr).sum(dim=0)
        rev_count += int(zr.shape[0])

        for p in checked_levels:
            # --------------------------
            # Cycle evaluation:
            #   - SiT/USiT: reverse-flow to t0 then forward-flow back to data
            #   - Diffusion (UNet/DiT/UDiT): forward-noise with q(z_t|z_0) then denoise with DDIM
            # --------------------------
            if is_sit_family:
                # flow time is [0,1], data at t=1. noise fraction p => go to t0 = 1-p.
                t0 = float(1.0 - p)
                t0 = max(0.0, min(1.0, t0))
                # steps proportional to interval length p
                n_seg = max(1, int(round(float(num_sit_steps) * float(p))))
                # invert: 1 -> t0 (backward Euler), then forward: t0 -> 1
                z_corrupt = _flow_backward_euler(z_clean_work, t_hi=1.0, t_lo=t0, n_steps=n_seg)
                z_recon = _flow_forward_euler(z_corrupt, t_lo=t0, t_hi=1.0, n_steps=n_seg)
                z0_hat_norm = z_recon
                z0_hat = z0_hat_norm * latent_std + latent_mean
                per_sample_mse = torch.mean((z0_hat - z_clean) ** 2, dim=(1, 2))
            else:
                # diffusion models: corrupt via *forward* noising q(z_t|z_0), then denoise via DDIM
                t_target = _pick_ddim_t_target(p)              # training-time t in [0, T-1]
                start_idx = _pick_ddim_start_index(t_target)   # nearest DDIM index with timestep <= t_target
                t0 = int(ddim.timesteps[start_idx].item())     # actual DDIM timestep we will start denoising from

                # forward corruption under q(z_t|z_0) at timestep t0 (aligned to the DDIM grid)
                z_corrupt = _ddpm_q_sample(z_clean_work, t_int=t0)

                # reconstruction: DDIM denoise from that grid timestep down to 0
                z_recon = _ddim_denoise_from(z_corrupt, start_idx=start_idx)
                if unet_family:
                    if unet_uses_norm and (unet_channel_mean is not None) and (unet_channel_std is not None):
                        z0_hat = z_recon * unet_channel_std + unet_channel_mean
                    else:
                        z0_hat = z_recon
                    per_sample_mse = torch.mean((z0_hat - z_clean) ** 2, dim=(1, 2, 3))
                else:
                    z0_hat_norm = z_recon
                    z0_hat = z0_hat_norm * latent_std + latent_mean
                    per_sample_mse = torch.mean((z0_hat - z_clean) ** 2, dim=(1, 2))

            mse_sum[p] += float(per_sample_mse.sum().item())
            mse_sq_sum[p] += float((per_sample_mse ** 2).sum().item())
            mse_n[p] += int(per_sample_mse.shape[0])

            # Decode the denoised latent for decoded accuracy (ALL model families).
            # This fixes the previous UNet bug where you were using clean logits.
            try:
                logits_hat = ae_model.decode(z0_hat)
            except Exception:
                # Fallback: if your AE uses a different decoder API, handle it here.
                # e.g., logits_hat = ae_model.decode_latents(z0_hat)
                raise RuntimeError(
                    "Failed to decode denoised latent. Implement ae_model.decode(z_latent) "
                    "or adjust this eval to your AE decoder API."
                )
            calls_hat = genotype_calls_from_logits(logits_hat)
            true_flat = xb.reshape(-1).to(torch.long)
            pred_flat = calls_hat.reshape(-1).to(torch.long)
            valid = (true_flat >= 0) & (true_flat <= 2) & (pred_flat >= 0) & (pred_flat <= 2)
            if valid.any():
                idx = true_flat[valid] * 3 + pred_flat[valid]
                batch_counts = torch.bincount(idx, minlength=9).reshape(3, 3).to(torch.float64).cpu()
                contingency_counts[p] += batch_counts

    # --------------------------
    # Independent generation for MAF/LD
    # --------------------------
    n_gen_total = int(samples_seen)
    if n_gen_total <= 0:
        raise RuntimeError("No evaluation samples seen; cannot compute population metrics.")

    # init gen accumulators
    maf_sum_gen = torch.zeros_like(maf_sum_real) if maf_sum_real is not None else None
    maf_cnt_gen = torch.zeros_like(maf_cnt_real) if maf_cnt_real is not None else None

    def _decode_to_calls(z_latent: torch.Tensor) -> torch.Tensor:
        logits = ae_model.decode(z_latent)
        return genotype_calls_from_logits(logits)

    def _generate_one_batch(B: int) -> torch.Tensor:
        """
        Returns X_gen calls (B,L) int64 sampled independently from base ~ N(0,I) via the model.
        """
        if is_sit_family:
            # base in normalized latent space
            if not unet_family:
                z = torch.randn(B, int(ae_model.latent_length), int(ae_model.latent_dim), device=device)
                # integrate t:0->1
                z = _flow_forward_euler(z, t_lo=0.0, t_hi=1.0, n_steps=max(1, int(num_sit_steps)))
                z_lat = z * latent_std + latent_mean
            else:
                raise ValueError("SiT/USiT path should not be UNet-family in this codebase.")
            return _decode_to_calls(z_lat).detach().cpu()
        else:
            # diffusion sampling: start from pure noise at largest DDIM timestep, denoise to 0
            if unet_family:
                # infer latent shape from UNet config
                C = int(model.config.in_channels)
                H = int(model.config.sample_size)
                W = int(model.config.sample_size)
                z = torch.randn(B, C, H, W, device=device)
                z = _ddim_denoise_from(z, start_idx=0)  # full path from max timestep to 0
                if unet_uses_norm and (unet_channel_mean is not None) and (unet_channel_std is not None):
                    z_lat = z * unet_channel_std + unet_channel_mean
                else:
                    z_lat = z
            else:
                z = torch.randn(B, int(ae_model.latent_length), int(ae_model.latent_dim), device=device)
                z = _ddim_denoise_from(z, start_idx=0)
                z_lat = z * latent_std + latent_mean
            return _decode_to_calls(z_lat).detach().cpu()

    # generate in batches and accumulate MAF/LD
    gen_done = 0
    while gen_done < n_gen_total:
        B = min(int(batch_size), n_gen_total - gen_done)
        Xg = _generate_one_batch(B)  # CPU int64
        gen_done += B
        if maf_sum_gen is not None and maf_cnt_gen is not None:
            maf_sum_gen, maf_cnt_gen = _update_maf(maf_sum_gen, maf_cnt_gen, Xg)
        if ld_acc_gen is not None:
            _update_ld(ld_acc_gen, Xg)

    # --------------------------
    # Build outputs
    # --------------------------
    cycle_rows: List[Dict[str, Any]] = []
    decoded_rows: List[Dict[str, Any]] = []
    for p in checked_levels:
        n = mse_n[p]
        if n > 0:
            mean_mse = mse_sum[p] / float(n)
            if n > 1:
                var = (mse_sq_sum[p] - float(n) * (mean_mse ** 2)) / float(n - 1)
                se_mse = math.sqrt(max(var, 0.0) / float(n))
            else:
                se_mse = float("nan")
        else:
            mean_mse = float("nan")
            se_mse = float("nan")

        counts = contingency_counts[p]
        row_sums = counts.sum(dim=1, keepdim=True)
        frac = counts / row_sums.clamp_min(1.0)
        noise_pct = int(round(100.0 * p))

        cycle_rows.append(
            {
                "model": model_name,
                "normalized": bool(normalized),
                "noise_pct": noise_pct,
                "cycle_mse_mean": float(mean_mse),
                "cycle_mse_se": float(se_mse),
                "n_samples": int(n),
            }
        )
        for true_state in range(3):
            decoded_rows.append(
                {
                    "model": model_name,
                    "normalized": bool(normalized),
                    "noise_pct": noise_pct,
                    "true_state": true_state,
                    "pred_0": float(frac[true_state, 0].item()),
                    "pred_1": float(frac[true_state, 1].item()),
                    "pred_2": float(frac[true_state, 2].item()),
                    "state_acc": float(frac[true_state, true_state].item()),
                }
            )

    cycle_df = pd.DataFrame(cycle_rows)
    decoded_df = pd.DataFrame(decoded_rows)

    # --------------------------
    # Population metrics: MAF + LD
    # --------------------------
    if maf_sum_real is None or maf_cnt_real is None or maf_sum_gen is None or maf_cnt_gen is None:
        raise RuntimeError("Failed to accumulate MAF stats.")

    maf_real = (maf_sum_real / maf_cnt_real.clamp_min(1.0)) / 2.0
    maf_gen = (maf_sum_gen / maf_cnt_gen.clamp_min(1.0)) / 2.0
    maf_real_np = maf_real.numpy()
    maf_gen_np = maf_gen.numpy()

    maf_mse = float(((maf_gen_np - maf_real_np) ** 2).mean())
    sse = float(((maf_gen_np - maf_real_np) ** 2).sum())
    sst = float(((maf_real_np - maf_real_np.mean()) ** 2).sum())
    maf_r2 = float(1.0 - (sse / sst)) if sst > 0 else float("nan")

    ld_l2 = float("nan")
    ld_block_norms: Optional[List[float]] = None
    if ld_blocks is not None:
        if ld_acc_real is None or ld_acc_gen is None:
            raise RuntimeError("LD blocks provided but LD accumulators not initialized.")
        ld_real_list = _finalize_ld(ld_acc_real)
        ld_gen_list = _finalize_ld(ld_acc_gen)
        if len(ld_real_list) != len(ld_gen_list):
            raise RuntimeError("LD finalize mismatch.")
        sq = 0.0
        ld_block_norms = []
        for A, B in zip(ld_real_list, ld_gen_list):
            d = (B - A).pow(2).sum().item()
            sq += float(d)
            ld_block_norms.append(float((B - A).pow(2).sum().sqrt().item()))
        ld_l2 = float(math.sqrt(max(sq, 0.0)))

    pop_row: Dict[str, Any] = {
        "model": model_name,
        "normalized": bool(normalized),
        "n_real": int(samples_seen),
        "n_gen": int(n_gen_total),
        "maf_mse": maf_mse,
        "maf_r2": maf_r2,
        "ld_l2": ld_l2,
    }
    if ld_block_norms is not None:
        pop_row["ld_block_l2_list"] = ld_block_norms

    # --------------------------
    # Reverse KL: KL(q0 || N(0,I)) where q0 is diagonal Gaussian fit to z_rev
    # --------------------------
    if rev_sum is None or rev_sq_sum is None or rev_count <= 0:
        pop_row["rev_kl_diag"] = float("nan")
        pop_row["rev_kl_diag_perdim"] = float("nan")
    else:
        mu = (rev_sum / float(rev_count))                       # (D,)
        ex2 = (rev_sq_sum / float(rev_count))                   # (D,)
        var = (ex2 - mu * mu).clamp_min(1e-12)                  # (D,)
        # # KL(N(mu,var) || N(0,1)) summed over dims
        # kl = 0.5 * (var + mu * mu - 1.0 - torch.log(var)).sum()
        # pop_row["rev_kl_diag"] = float(kl.item())
        # pop_row["rev_kl_diag_perdim"] = float((kl / float(mu.numel())).item())

        # Two KL directions between Gaussians (diagonal):
        #   KL(q || p0) where q ~ N(mu,var), p0 ~ N(0,I)
        kl_q_p0 = 0.5 * (var + mu * mu - 1.0 - torch.log(var)).sum()
        #   KL(p0 || q)  (THIS matches Lemma 5.2 direction KL(p0 || p1^rev) under assumptions)
        inv_var = (1.0 / var).clamp_max(1e12)
        kl_p0_q = 0.5 * (inv_var + mu * mu * inv_var - 1.0 + torch.log(var)).sum()

        pop_row["rev_kl_q_p0_diag"] = float(kl_q_p0.item())
        pop_row["rev_kl_p0_q_diag"] = float(kl_p0_q.item())
        pop_row["rev_kl_p0_q_diag_perdim"] = float((kl_p0_q / float(mu.numel())).item())

    pop_df = pd.DataFrame([pop_row])

    # Return: (1) cycle+decoded (merged later if you want), (2) population summary
    # If you want one table, merge cycle_df with decoded_df via keys.
    return cycle_df.merge(decoded_df, on=["model", "normalized", "noise_pct"], how="left"), pop_df