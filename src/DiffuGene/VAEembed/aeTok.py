from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


def find_best_ck(x: int, max_c: int = 5, *, min_k: Optional[int] = None) -> Tuple[int, int]:
    """Pick (c,k) so c*2^k is close to x with optional k lower-bound."""
    best_c, best_k = 1, 0
    best_err = abs(1.0 - x)
    for c in range(1, max_c + 1):
        k = 0
        while True:
            val = c * (1 << k)
            err = abs(val - x)
            if err < best_err:
                best_err = err
                best_c, best_k = c, k
            if val > 10 * x:
                break
            k += 1
    if min_k is not None and best_k < int(min_k):
        best_k = int(min_k)
        c_est = int(round(float(x) / float(1 << best_k)))
        best_c = min(max(1, c_est), max_c)
    return best_c, best_k


def pixel_unshuffle_1d(x: torch.Tensor, factor: int = 2) -> torch.Tensor:
    bsz, channels, length = x.shape
    if length % factor != 0:
        raise ValueError(f"Sequence length {length} is not divisible by factor {factor}.")
    new_len = length // factor
    x = x.view(bsz, channels, new_len, factor)
    x = x.permute(0, 1, 3, 2).contiguous()
    return x.view(bsz, channels * factor, new_len)


def pixel_shuffle_1d(x: torch.Tensor, factor: int = 2) -> torch.Tensor:
    bsz, channels, length = x.shape
    if channels % factor != 0:
        raise ValueError(f"Number of channels {channels} is not divisible by factor {factor}.")
    new_ch = channels // factor
    x = x.view(bsz, new_ch, factor, length)
    x = x.permute(0, 1, 3, 2).contiguous()
    return x.view(bsz, new_ch, length * factor)


class PixelUnshuffleDownSample1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, factor: int = 2) -> None:
        super().__init__()
        if out_channels != in_channels * factor:
            raise ValueError(
                f"Expected out_channels={in_channels*factor}, got {out_channels}"
            )
        self.factor = int(factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return pixel_unshuffle_1d(x, self.factor)


class PixelShuffleUpSample1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, factor: int = 2) -> None:
        super().__init__()
        if in_channels != out_channels * factor:
            raise ValueError(
                f"Expected in_channels={out_channels*factor}, got {in_channels}"
            )
        self.factor = int(factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return pixel_shuffle_1d(x, self.factor)


class ResidualDownsample1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, factor: int = 2) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=factor,
            padding=1,
            bias=False,
        )
        self.shortcut = PixelUnshuffleDownSample1D(in_channels, out_channels, factor=factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x) + self.shortcut(x)


class ResidualUpsample1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, factor: int = 2) -> None:
        super().__init__()
        self.deconv = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size=factor,
            stride=factor,
            bias=False,
        )
        self.shortcut = PixelShuffleUpSample1D(in_channels, out_channels, factor=factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.deconv(x) + self.shortcut(x)


class TokenAutoencoder1D(nn.Module):
    """
    Deterministic 1D genotype autoencoder producing token latents.

    - Input: integer genotypes x in {0,1,2}, shape (B, L)
    - Latent: tokens z, shape (B, latent_length, latent_dim)
    - Output: logits over 3 genotypes, shape (B, L, 3)
    """

    def __init__(
        self,
        input_length: int,
        latent_length: int = 64,
        latent_dim: int = 256,
        embed_dim: int = 8,
        max_c: int = 5,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_length = int(input_length)
        self.latent_length = int(latent_length)
        self.latent_dim = int(latent_dim)
        self.embed_dim = int(embed_dim)
        self.dropout = nn.Dropout(float(dropout))

        if self.latent_length <= 0 or (self.latent_length & (self.latent_length - 1)) != 0:
            raise ValueError(f"latent_length must be a positive power of two. Got {self.latent_length}")

        self.target_exp = int(math.log2(self.latent_length))
        c, k = find_best_ck(self.input_length, max_c=int(max_c), min_k=self.target_exp)
        self.c = int(c)
        self.L1 = int(k)
        self.target_len = int(self.c * (1 << self.L1))
        self.pad_1d = self.target_len > self.input_length
        self.truncate_1d = self.target_len < self.input_length

        self.num_down = int(self.L1 - self.target_exp)
        if self.num_down < 0:
            raise ValueError(f"Invalid downsample depth from L1={self.L1} to target_exp={self.target_exp}")

        if self.latent_dim % (1 << self.num_down) != 0:
            raise ValueError(
                f"latent_dim={self.latent_dim} must be divisible by 2^num_down={1 << self.num_down}"
            )
        self.start_channels = int(self.latent_dim // (1 << self.num_down))

        self.input_embed = nn.Conv1d(4, self.embed_dim, kernel_size=1, bias=False)
        self.conv_reduce = nn.Conv1d(
            self.embed_dim,
            self.start_channels,
            kernel_size=self.c,
            stride=self.c,
            bias=False,
        )

        down_blocks: List[ResidualDownsample1D] = []
        ch = self.start_channels
        for _ in range(self.num_down):
            down_blocks.append(ResidualDownsample1D(ch, ch * 2, factor=2))
            ch = ch * 2
        self.down_blocks = nn.ModuleList(down_blocks)
        if ch != self.latent_dim:
            raise RuntimeError(f"Encoder channel mismatch: got {ch}, expected {self.latent_dim}")

        up_blocks: List[ResidualUpsample1D] = []
        ch = self.latent_dim
        for _ in range(self.num_down):
            up_blocks.append(ResidualUpsample1D(ch, ch // 2, factor=2))
            ch = ch // 2
        self.up_blocks = nn.ModuleList(up_blocks)
        if ch != self.start_channels:
            raise RuntimeError(f"Decoder channel mismatch: got {ch}, expected {self.start_channels}")

        self.deconv_expand = nn.ConvTranspose1d(
            self.start_channels,
            self.embed_dim,
            kernel_size=self.c,
            stride=self.c,
            bias=False,
        )
        self.output_proj = nn.Conv1d(self.embed_dim, 3, kernel_size=1, bias=True)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        bsz, length = x.shape
        x_onehot = F.one_hot(x.long(), num_classes=3).float()
        x_dosage = x.float().unsqueeze(-1)
        x4 = torch.cat([x_onehot, x_dosage], dim=-1).permute(0, 2, 1).contiguous()
        h = self.input_embed(x4)
        h = self.dropout(h)

        if self.pad_1d:
            h = F.pad(h, (0, self.target_len - length))
        elif self.truncate_1d:
            h = h[..., : self.target_len]

        h = self.conv_reduce(h)
        for block in self.down_blocks:
            h = block(h)
        return h.permute(0, 2, 1).contiguous()

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = z.permute(0, 2, 1).contiguous()
        for block in self.up_blocks:
            h = block(h)
        h = self.deconv_expand(h)

        if self.pad_1d:
            h = h[..., : self.input_length]
        elif self.truncate_1d:
            pad_len = self.input_length - self.target_len
            if pad_len > 0:
                h = F.pad(h, (0, pad_len))
        logits = self.output_proj(h)
        return logits.permute(0, 2, 1).contiguous()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        logits = self.decode(z)
        return logits, z


def reconstruction_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Cross-entropy loss for genotype reconstruction."""
    bsz, length, n_classes = logits.shape
    return F.cross_entropy(logits.view(bsz * length, n_classes), target.view(bsz * length), reduction="mean")


@dataclass
class TokenAEConfig:
    input_length: int
    latent_length: int = 64
    latent_dim: int = 256
    embed_dim: int = 8
    max_c: int = 5
    dropout: float = 0.0
    lr: float = 2e-4
    betas: Tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.0
    grad_clip: float = 1.0


def build_token_ae(cfg: TokenAEConfig) -> Tuple[nn.Module, torch.optim.Optimizer]:
    model = TokenAutoencoder1D(
        input_length=int(cfg.input_length),
        latent_length=int(cfg.latent_length),
        latent_dim=int(cfg.latent_dim),
        embed_dim=int(cfg.embed_dim),
        max_c=int(cfg.max_c),
        dropout=float(cfg.dropout),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.lr),
        betas=cfg.betas,
        weight_decay=float(cfg.weight_decay),
    )
    return model, optimizer