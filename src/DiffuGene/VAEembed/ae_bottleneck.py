"""
Deterministic autoencoder for per-SNP genotype sequences.

1D encoder → 2D latent → mirrored decoder. Returns (logits, z).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)

# Global constant controlling latent bottleneck compression:
# bottleneck_channels = latent_channels // BOTTLENECK_DIV
# Example: latent_channels=64 -> DIV=2 => 32, DIV=4 => 16
BOTTLENECK_DIV: int = 2
# BOTTLENECK_DIV: int = 4
# Minimum number of channels to keep when sampling a mask in bottleneck space
MIN_MASK_CHANNELS: int = 16
# MIN_MASK_CHANNELS: int = 4

def find_best_ck(x: int, max_c: int = 5, *, min_k: Optional[int] = None) -> Tuple[int, int]:
    """Pick (c,k) so c*2^k ≈ x (optionally enforce k ≥ min_k)."""
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


def local_ld_penalty(x_data: torch.Tensor, x_hat: torch.Tensor, window: int = 128) -> torch.Tensor:
    """LD penalty: mean Frobenius norm between windowed correlation matrices of x_data and x_hat."""
    B, L = x_data.shape
    total = 0.0
    num = 0
    # Slide a window of length window across the sequence with stride
    # equal to the window length.  For each window compute the
    # correlation matrices and accumulate the Frobenius norm of their
    # difference.
    for s in range(0, L - window + 1, window):
        e = s + window
        xd = x_data[:, s:e]
        xr = x_hat[:, s:e]
        xd = xd - xd.mean(dim=0, keepdim=True)
        xr = xr - xr.mean(dim=0, keepdim=True)
        cov_d = (xd.T @ xd) / (B - 1 + 1e-6)
        cov_r = (xr.T @ xr) / (B - 1 + 1e-6)
        std_d = torch.sqrt(torch.diag(cov_d) + 1e-6)
        std_r = torch.sqrt(torch.diag(cov_r) + 1e-6)
        corr_d = cov_d / (std_d[:, None] * std_d[None, :] + 1e-6)
        corr_r = cov_r / (std_r[:, None] * std_r[None, :] + 1e-6)
        total = total + torch.norm(corr_r - corr_d, p='fro')
        num += 1
    return total / max(num, 1)


def pixel_unshuffle_1d(x: torch.Tensor, factor: int = 2) -> torch.Tensor:
    """1D pixel-unshuffle: (B,C,L) → (B,C*factor,L//factor). factor must divide L."""
    B, C, L = x.shape
    if L % factor != 0:
        raise ValueError(f"Sequence length {L} is not divisible by factor {factor}.")
    new_L = L // factor
    # reshape+permute
    x = x.view(B, C, new_L, factor)
    x = x.permute(0, 1, 3, 2).contiguous()  # (B, C, factor, new_L)
    x = x.view(B, C * factor, new_L)
    return x


def pixel_shuffle_1d(x: torch.Tensor, factor: int = 2) -> torch.Tensor:
    """Inverse of pixel_unshuffle_1d: (B,C,L) → (B,C//factor,L*factor). C must be divisible by factor."""
    B, C, L = x.shape
    if C % factor != 0:
        raise ValueError(f"Number of channels {C} is not divisible by factor {factor}.")
    new_C = C // factor
    # reshape+permute
    x = x.view(B, new_C, factor, L)
    x = x.permute(0, 1, 3, 2).contiguous()  # (B, new_C, L, factor)
    x = x.view(B, new_C, L * factor)
    return x


# class PixelUnshuffleChannelAveragingDownSample1D(nn.Module):
#     """Shortcut: unshuffle along length, average channel groups to match out_channels."""
#     def __init__(self, in_channels: int, out_channels: int, factor: int = 2) -> None:
#         super().__init__()
#         self.in_channels = int(in_channels)
#         self.out_channels = int(out_channels)
#         self.factor = int(factor)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         skip = pixel_unshuffle_1d(x, self.factor)  # (B, inC*factor, L//factor)
#         if skip.size(1) != self.out_channels:
#             B, C_skip, L_new = skip.shape
#             group_size = C_skip // self.out_channels
#             skip = skip.view(B, self.out_channels, group_size, L_new).mean(dim=2)
#         return skip
class PixelUnshuffleDownSample1D(nn.Module):
    """Shortcut: 1D pixel_unshuffle without averaging. Requires out_channels == in_channels * factor."""
    def __init__(self, in_channels: int, out_channels: int, factor: int = 2) -> None:
        super().__init__()
        assert out_channels == in_channels * factor, f"1D non-averaging shortcut requires out_channels={in_channels*factor}, got {out_channels}"
        self.factor = int(factor)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return pixel_unshuffle_1d(x, self.factor)  # (B, inC*factor, L//factor)


# class PixelShuffleChannelReplicateUpSample1D(nn.Module):
#     """Shortcut: shuffle channels into length; replicate channels to match out_channels. No norm/act."""
#     def __init__(self, in_channels: int, out_channels: int, factor: int = 2) -> None:
#         super().__init__()
#         self.in_channels = int(in_channels)
#         self.out_channels = int(out_channels)
#         self.factor = int(factor)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         skip = pixel_shuffle_1d(x, self.factor)  # (B, inC//factor, L*factor)
#         if skip.size(1) != self.out_channels:
#             B, C_skip, L_new = skip.shape
#             if self.out_channels % C_skip != 0:
#                 raise ValueError(f"Cannot replicate {C_skip}→{self.out_channels} channels.")
#             dup = self.out_channels // C_skip
#             skip = skip.repeat_interleave(dup, dim=1)
#         return skip
class PixelShuffleUpSample1D(nn.Module):
    """Shortcut: 1D pixel_shuffle without replication. Requires in_channels == out_channels * factor."""
    def __init__(self, in_channels: int, out_channels: int, factor: int = 2) -> None:
        super().__init__()
        assert in_channels == out_channels * factor, f"1D non-replicating shortcut requires in_channels={out_channels*factor}, got {in_channels}"
        self.factor = int(factor)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return pixel_shuffle_1d(x, self.factor)  # (B, inC//factor, L*factor)

class ResidualDownsample1D(nn.Module):
    """Main path: Conv(stride=2). Shortcut: pixel_unshuffle (no averaging in 1d)."""
    def __init__(self, in_channels: int, out_channels: int, factor: int = 2) -> None:
        super().__init__()
        self.factor = int(factor)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=self.factor, padding=1, bias=False)
        # self.gn1   = nn.GroupNorm(num_groups=min(32, out_channels), num_channels=out_channels)
        # self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        # self.gn2   = nn.GroupNorm(num_groups=min(32, out_channels), num_channels=out_channels)
        # self.shortcut = PixelUnshuffleChannelAveragingDownSample1D(in_channels, out_channels, factor=self.factor)
        self.shortcut = PixelUnshuffleDownSample1D(in_channels, out_channels, factor=self.factor)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv1(x) + self.shortcut(x)


class ResidualUpsample1D(nn.Module):
    """Main path: Deconv(stride=2). Shortcut: pixel_shuffle (no replication in 1d)."""
    def __init__(self, in_channels: int, out_channels: int, factor: int = 2) -> None:
        super().__init__()
        self.factor = int(factor)
        self.deconv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=self.factor, stride=self.factor, bias=False)
        # self.gn1    = nn.GroupNorm(num_groups=min(32, out_channels), num_channels=out_channels)
        # self.conv2  = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        # self.gn2    = nn.GroupNorm(num_groups=min(32, out_channels), num_channels=out_channels)
        # self.shortcut = PixelShuffleChannelReplicateUpSample1D(in_channels, out_channels, factor=self.factor)
        self.shortcut = PixelShuffleUpSample1D(in_channels, out_channels, factor=self.factor)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.deconv(x) + self.shortcut(x)


class PixelUnshuffleChannelAveragingDownSample2D(nn.Module):
    """Shortcut: 2D pixel_unshuffle then average channel groups to match out_channels. No norm/act."""
    def __init__(self, in_channels: int, out_channels: int, factor: int = 2) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.factor = int(factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = F.pixel_unshuffle(x, self.factor)  # (B, inC*factor^2, H//f, W//f)
        if skip.size(1) != self.out_channels:
            B, C_skip, Hn, Wn = skip.shape
            group_size = C_skip // self.out_channels
            skip = skip.view(B, self.out_channels, group_size, Hn, Wn).mean(dim=2)
        return skip

class ResidualDownsample2D(nn.Module):
    """Main path: Conv(stride=2)→GN→SiLU→Conv→GN. Shortcut: pixel_unshuffle(+avg). Sum without post-activation."""
    def __init__(self, in_channels: int, out_channels: int, factor: int = 2) -> None:
        super().__init__()
        self.factor = int(factor)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=self.factor, padding=1, bias=False)
        # self.gn1   = nn.GroupNorm(num_groups=min(32, out_channels), num_channels=out_channels)
        # self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        # self.gn2   = nn.GroupNorm(num_groups=min(32, out_channels), num_channels=out_channels)
        self.shortcut = PixelUnshuffleChannelAveragingDownSample2D(in_channels, out_channels, factor=self.factor)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        skip = self.shortcut(x)
        z = y + skip
        return z


class PixelShuffleChannelReplicateUpSample2D(nn.Module):
    """Shortcut: 2D pixel_shuffle then replicate channels to match out_channels. No norm/act."""
    def __init__(self, in_channels: int, out_channels: int, factor: int = 2) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.factor = int(factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = F.pixel_shuffle(x, self.factor)  # (B, inC//f^2, H*f, W*f)
        if skip.size(1) != self.out_channels:
            B, C_skip, Hn, Wn = skip.shape
            if self.out_channels % C_skip != 0:
                raise ValueError(f"Cannot replicate {C_skip}→{self.out_channels} channels.")
            dup = self.out_channels // C_skip
            skip = skip.repeat_interleave(dup, dim=1)
        return skip

class ResidualUpsample2D(nn.Module):
    """Main path: Deconv(stride=2)→GN→SiLU→Conv→GN. Shortcut: pixel_shuffle(+replicate). Sum without post-activation."""
    def __init__(self, in_channels: int, out_channels: int, factor: int = 2) -> None:
        super().__init__()
        self.factor = int(factor)
        # kernel_size=2*factor, stride=factor, padding=factor//2 mirrors downsample receptive field
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2*self.factor, stride=self.factor, padding=self.factor//2, bias=False)
        # self.gn1    = nn.GroupNorm(num_groups=min(32, out_channels), num_channels=out_channels)
        # self.conv2  = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        # self.gn2    = nn.GroupNorm(num_groups=min(32, out_channels), num_channels=out_channels)
        self.shortcut = PixelShuffleChannelReplicateUpSample2D(in_channels, out_channels, factor=self.factor)
  
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.deconv(x)
        skip = self.shortcut(x)
        z = y + skip
        return z


class GenotypeAutoencoder(nn.Module):
    """Deterministic AE: 1D encode → 2D latent → mirrored decoder.

    Returns (logits, z). Masking prevents padded positions from affecting loss.
    Args: input_length, K1, K2, C, embed_dim.
    """

    def __init__(self, input_length: int, K1: int, K2: int, C: int, *, embed_dim: int = 8, beta_kl: float = 1e-6) -> None:
        super().__init__()
        # Store base-space parameters
        self.input_length = int(input_length)
        self.embed_dim = int(embed_dim)
        # Default KL coefficient for variational penalty (kept tiny for backward compatibility)
        self.beta_kl = float(beta_kl)
        
        # Enable DC-AE 1.5 style structured latent space
        self.enable_structured_latent: bool = True
        self._last_masked_channels: int = int(C)
        # Buffers to hold latest variational statistics for loss computation
        self._last_mu: Optional[torch.Tensor] = None
        self._last_logvar: Optional[torch.Tensor] = None

        def _log2_int(n: int, name: str) -> int:
            if n <= 0 or (n & (n - 1)) != 0:
                raise ValueError(f"{name} must be a positive power of two, got {n}")
            return int(math.log2(n))
        self.K1_size = int(K1)
        self.K2_size = int(K2)
        self.C  = int(C)
        self.K1 = _log2_int(self.K1_size, "K1")
        self.K2 = _log2_int(self.K2_size, "K2")
        self.C_exp = _log2_int(self.C, "C")
        
        # ---------------- 1D encoder/decoder ----------------
        # Compute (c, k) such that c * 2**k approx L
        c, k = find_best_ck(self.input_length)
        self.c = int(c)
        self.L1 = int(k)
        self.target_len = self.c * (1 << self.L1)
        
        # Flags to indicate whether we need to compress or pad the input
        self.compress_1d = self.target_len < self.input_length
        self.pad_1d = self.target_len > self.input_length
        
        self.length_after_conv = 1 << self.L1     # = 2^L1
        self.M1D = self.K1_size                   # 1D target square side M = 2^K1
        self.num_down1d = self.L1 - self.K1       # 1D residual steps and initial channel exponent M1
        if self.num_down1d < 0:
            raise ValueError(f"Negative num_down1d: K1={self.K1} cannot exceed L1={self.L1}.")
        M1_exp = 2*self.K1 - self.L1
        if M1_exp < 0:
            raise ValueError(
                f"Negative init_1d_channels: 2*K1 - L1 must be > 0; got 2*{self.K1}-{self.L1}={M1_exp}. "
                "Choose larger K1 or different (c,k)."
            )
        self.init_1d_channels = 1 << M1_exp     # = 2^M1

        # Input embedding: map one-hot + dosage (4) → embed_dim (e.g., 8)        
        self.input_embed = nn.Conv1d(4, self.embed_dim, kernel_size=1, bias=False)
        
        # Stride-c reduction to length 2^L1, projecting to 2^M1 channels
        self.conv_reduce = nn.Conv1d(self.embed_dim, self.init_1d_channels, kernel_size=self.c, stride=self.c, bias=False)
        
        # Build 1D downsampling blocks: exactly (L1−K1) steps; each halves length and doubles channels.
        down1d_blocks: List[ResidualDownsample1D] = []
        in_ch = self.init_1d_channels
        for _ in range(self.num_down1d):
            out_ch = in_ch * 2               # no averaging; skip uses pixel_unshuffle
            down1d_blocks.append(ResidualDownsample1D(in_ch, out_ch))
            in_ch = out_ch
        if in_ch != self.M1D:                # Sanity check: final 1D channels = 2^K1
            raise RuntimeError(f"1D encoder channels {in_ch} != 2^K1={self.M1D}")
        self.down1d_blocks = nn.ModuleList(down1d_blocks)
        
        # 1D upsampling blocks: exact inverse (double length; halve channels)
        up1d_blocks: List[ResidualUpsample1D] = []
        chs: List[int] = [self.init_1d_channels]
        tmp = self.init_1d_channels
        for _ in range(self.num_down1d):
            tmp = tmp * 2
            chs.append(tmp)
        for i in reversed(range(self.num_down1d)):
            up1d_blocks.append(ResidualUpsample1D(chs[i+1], chs[i]))
        self.up1d_blocks = nn.ModuleList(up1d_blocks)
        
        # Invert stride-c reduction
        self.deconv_expand = nn.ConvTranspose1d(self.init_1d_channels, self.embed_dim, kernel_size=self.c, stride=self.c, bias=False)

        # ---------------- 2D encoder/decoder ----------------
        # Steps: from 2^K1 to 2^K2, steps2 = K1−K2
        self.k2d_steps = self.K1 - self.K2
        self.M2D = self.K2_size
        self.latent_channels = self.C      # final 2D channels = 2^C
        M2_exp = self.C_exp - self.k2d_steps    # Initial 2D channels: 2^M2, where M2 = C − (K1−K2)
        if M2_exp < 0:
            raise ValueError(f"M2 exponent C-(K1-K2) must be ≥0; got {self.C_exp}-({self.K1}-{self.K2})={M2_exp}")
        self.init_2d_channels = 1 << M2_exp
        
        # Initial 2D projection 
        self.proj2d = nn.Conv2d(1, self.init_2d_channels, kernel_size=1, bias=False)

        # Build the stack of 2D downsampling blocks
        down2d_blocks: List[ResidualDownsample2D] = []
        in_ch2d = self.init_2d_channels
        for i in range(self.k2d_steps):
            out_ch2d = self.latent_channels if i == self.k2d_steps - 1 else in_ch2d * 2
            down2d_blocks.append(ResidualDownsample2D(in_ch2d, out_ch2d))
            in_ch2d = out_ch2d
        if in_ch2d != self.latent_channels:
            raise RuntimeError(f"2D encoder channels {in_ch2d} != C={self.latent_channels}")
        self.down2d_blocks = nn.ModuleList(down2d_blocks)

        # --------- Variational bottleneck and latent heads (compress → heads → sample → expand) ---------
        # Compress channels to C//BOTTLENECK_DIV before heads; expand back to C before decoding
        # Use Identity if division would not reduce channels
        div = max(1, int(BOTTLENECK_DIV))
        proposed_bottleneck = max(1, self.latent_channels // div)
        if proposed_bottleneck < self.latent_channels:
            self.bottleneck_channels = proposed_bottleneck
            self.to_latent = nn.Conv2d(self.latent_channels, self.bottleneck_channels, kernel_size=1, bias=False)
            self.from_latent = nn.Conv2d(self.bottleneck_channels, self.latent_channels, kernel_size=1, bias=False)
        else:
            self.bottleneck_channels = self.latent_channels
            self.to_latent = nn.Identity()
            self.from_latent = nn.Identity()
        # Heads that output μ and logσ² in the bottleneck space
        self.latent_mu = nn.Conv2d(self.bottleneck_channels, self.bottleneck_channels, kernel_size=1, bias=True)
        self.latent_logvar = nn.Conv2d(self.bottleneck_channels, self.bottleneck_channels, kernel_size=1, bias=True)
        self._init_latent_heads()

        # Build the corresponding 2D upsampling blocks in reverse order
        up2d_blocks: List[ResidualUpsample2D] = []
        in_ch2d = self.latent_channels
        for i in range(self.k2d_steps):
            out_ch2d = self.init_2d_channels if (i == self.k2d_steps - 1) else (in_ch2d // 2)
            up2d_blocks.append(ResidualUpsample2D(in_ch2d, out_ch2d))
            in_ch2d = out_ch2d
        self.up2d_blocks = nn.ModuleList(up2d_blocks)
        
        # Projection back to a single channel before 1D decoding
        self.unproj2d = nn.Conv2d(self.init_2d_channels, 1, kernel_size=1, bias=False)
        
        # Final projection to logits over three genotype states
        self.out_proj = nn.Conv1d(self.embed_dim, 3, kernel_size=1, bias=True)

    def _init_latent_heads(self) -> None:
        # Initialize heads: mu ≈ identity, logvar small negative bias (var ~ exp(-4))
        with torch.no_grad():
            if isinstance(self.latent_mu, nn.Conv2d):
                self.latent_mu.weight.zero_()
                Cb = self.bottleneck_channels
                for c in range(Cb):
                    self.latent_mu.weight[c, c, 0, 0] = 1.0
                self.latent_mu.bias.zero_()
            if isinstance(self.latent_logvar, nn.Conv2d):
                self.latent_logvar.weight.zero_()
                self.latent_logvar.bias.fill_(-4.0)

    def _sample_latent_channel_mask(
        self, device: torch.device
    ) -> Tuple[torch.Tensor, int]:
        """
        Sample a channel-wise mask for the latent bottleneck (C/4 channels).
        Picks c' from a small set of fractions of the bottleneck size and keeps the first c' channels.
        """
        C = int(self.bottleneck_channels)

        if (not getattr(self, "enable_structured_latent", False)) or C <= 0:
            mask = torch.ones(1, C, 1, 1, device=device)
            return mask, C

        # Define a few valid active-channel counts, e.g. for C=16: {4, 8, 12, 16}
        step = max(4, C // 4)
        c_values = list(range(step, C + 1, step))
        # Enforce a minimum masking count (e.g., 16) but allow full usage if C < minimum
        min_c = min(int(MIN_MASK_CHANNELS), C)
        c_values = [c for c in c_values if c >= min_c and c <= C]
        if len(c_values) == 0:
            mask = torch.ones(1, C, 1, 1, device=device)
            return mask, C

        idx = torch.randint(len(c_values), (1,), device=device).item()
        c_prime = int(c_values[idx])

        mask = torch.zeros(1, C, 1, 1, device=device)
        mask[:, :c_prime, :, :] = 1.0
        return mask, c_prime

    def decode_with_latent_mask(
        self,
        z: torch.Tensor,
        *,
        enable_masking: bool = True,
    ) -> torch.Tensor:
        """
        Decode with channel-wise masking.
        During eval / inference, we typically keep enable_masking=False
        and use all channels.
        """
        if (not self.training) or (not enable_masking):
            return self.decode(z)
        mask, c_prime = self._sample_latent_channel_mask(z.device)
        self._last_masked_channels = int(c_prime)
        z_masked = z * mask
        return self.decode(z_masked)

    def _prepare_input(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Embed integer genotypes and pad/compress to target_len; return (h, mask)."""
        B, L = x.shape
        # Build 4-ch input (one-hot[3]+dosage)
        x_onehot = F.one_hot(x.long(), num_classes=3).float()  # (B,L,3)
        x_dosage = x.float().unsqueeze(-1)  # (B,L,1)
        x4 = torch.cat([x_onehot, x_dosage], dim=-1)  # (B,L,4)
        x4 = x4.permute(0, 2, 1).contiguous()  # (B,4,L)
        h = self.input_embed(x4)  # (B, embed_dim, L)
        mask: Optional[torch.Tensor] = None
        if self.compress_1d:
            # resize to target_len
            h = F.interpolate(h, size=self.target_len, mode="linear", align_corners=False)
        elif self.pad_1d:
            # right-pad to target_len
            pad_len = self.target_len - L
            h = F.pad(h, (0, pad_len))
            # mask: 1 for real positions, 0 for padded
            mask = torch.zeros((B, self.target_len), dtype=torch.float32, device=x.device)
            mask[:, :L] = 1.0
        return h, mask

    def _downsample_1d(self, h: torch.Tensor, mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Reduce to length 2^L1 and apply (L1-K1) residual downsample blocks; update mask."""
        # stride-c reduce: (B,embed_dim,L) → (B,2^M1,2^L1)
        h = self.conv_reduce(h)
        # downsample mask
        if mask is not None:
            # max-pool by factor c
            B, L = mask.shape
            new_len = L // self.c
            mask_view = mask.view(B, new_len, self.c)
            mask = mask_view.max(dim=2).values  # (B, new_len)
        # 1D downsample stack
        for down in self.down1d_blocks:
            h = down(h)
            if mask is not None:
                # halve mask length per block
                Bm, Lm = mask.shape
                new_len = Lm // 2
                mask_view = mask.view(Bm, new_len, 2)
                mask = mask_view.max(dim=2).values
        return h, mask

    def _upsample_1d(self, h: torch.Tensor) -> torch.Tensor:
        """Invert 1D downsampling (double length; halve channels)."""
        # Iterate through the upsample blocks in order
        for up in self.up1d_blocks:
            h = up(h)
        return h

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode→decode and return (logits, z)."""
        h, mask = self._prepare_input(x)              # prepare input
        h1d, mask1d = self._downsample_1d(h, mask)    # 1D encode
        B, C1, M = h1d.shape
        
        
        h2d = h1d.permute(0, 2, 1).unsqueeze(1)       # 1D→2D
        h2d = self.proj2d(h2d)                        # 2D proj
        for down_block in self.down2d_blocks:         # 2D down
            h2d = down_block(h2d)
        z_full = h2d                                   # full-channel latent (internal only)

        # Variational bottleneck: compress → heads → reparameterize → expand
        h_bottleneck = self.to_latent(h2d)
        mu = self.latent_mu(h_bottleneck)
        logvar = self.latent_logvar(h_bottleneck)
        self._last_mu = mu
        self._last_logvar = logvar
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z_bottleneck = mu + std * eps                 # exposed latent (C/4 channels)
        z_for_decode = self.from_latent(z_bottleneck) # expand to full channels for decoder

        for up_block in self.up2d_blocks:             # 2D up
            z_for_decode = up_block(z_for_decode)
        h2d = self.unproj2d(z_for_decode)
        h1d_up = h2d.squeeze(1).permute(0, 2, 1).contiguous()  # 2D→1D
        h1d_up = self._upsample_1d(h1d_up)
        h1d_up = self.deconv_expand(h1d_up)  
        
        if self.pad_1d:
            h1d_up = h1d_up[..., : self.input_length]
        elif self.compress_1d:
            h1d_up = F.interpolate(h1d_up, size=self.input_length, mode="linear", align_corners=False)
        
        logits = self.out_proj(h1d_up)
        return logits, z_bottleneck

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        # Accept either bottleneck (C/4) or full (C) latent; expand if needed
        if z.dim() != 4:
            raise ValueError("decode expects a 4D latent tensor (B,C,H,W)")
        C_in = z.size(1)
        if C_in == self.bottleneck_channels:
            h2d = self.from_latent(z)
        elif C_in == self.latent_channels:
            h2d = z
        else:
            raise ValueError(f"decode received latent with channels={C_in}, expected {self.bottleneck_channels} (bottleneck) or {self.latent_channels} (full)")
        for up_block in self.up2d_blocks:
            h2d = up_block(h2d)
        h2d = self.unproj2d(h2d)
        h1d_up = h2d.squeeze(1).permute(0, 2, 1).contiguous()  # (B, C1D, L1)
        h1d_up = self._upsample_1d(h1d_up)
        h1d_up = self.deconv_expand(h1d_up)
        if self.pad_1d:
            h1d_up = h1d_up[..., : self.input_length]
        elif self.compress_1d:
            h1d_up = F.interpolate(h1d_up, size=self.input_length, mode="linear", align_corners=False)
        return self.out_proj(h1d_up)

    def kl_divergence(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # KL[q(z|x) || N(0, I)] averaged over batch
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        kl = kl.sum(dim=(1, 2, 3)).mean()
        return kl


    def loss_function(
        self,
        logits: torch.Tensor,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
        *,
        beta_kl: float = 1e-6,
        maf_lambda: float = 0.0,
        ld_lambda: float = 0.0,
        ld_window: int = 128,
    ) -> Tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Masked cross-entropy with optional MAF/LD penalties. Returns (loss, metrics)."""
        # Masked cross-entropy
        ce = F.cross_entropy(logits, x.long(), reduction="none")  # (B,L)
        if mask is not None:
            ce = ce * mask
            recon_loss = ce.sum() / (mask.sum() + 1e-6)
        else:
            recon_loss = ce.mean()
        total_loss = recon_loss
        # Compute auxiliary penalties if requested
        metrics = {"recon": recon_loss.detach()}
        # Compute expected dosage from logits: sum_k k * p_k
        probs = torch.softmax(logits, dim=1)  # (B,3,L)
        class_values = torch.tensor([0.0, 1.0, 2.0], device=logits.device).view(1, 3, 1)
        x_hat = (probs * class_values).sum(dim=1)  # (B,L)
        if maf_lambda and maf_lambda > 0.0:
            # Minor allele frequency differences
            maf_data = x.float().mean(dim=0) / 2.0
            maf_recon = x_hat.mean(dim=0) / 2.0
            maf_pen = torch.mean(torch.abs(maf_recon - maf_data))
            total_loss = total_loss + maf_lambda * maf_pen
            metrics["maf"] = maf_pen.detach()
        else:
            metrics["maf"] = torch.tensor(0.0, device=logits.device)
        if ld_lambda and ld_lambda > 0.0 and x.size(1) >= ld_window:
            # LD structure penalty
            ld_pen = local_ld_penalty(x.float(), x_hat, window=ld_window)
            total_loss = total_loss + ld_lambda * ld_pen
            metrics["ld"] = ld_pen.detach()
        else:
            metrics["ld"] = torch.tensor(0.0, device=logits.device)

        # KL penalty (variational term) using latest stored mu/logvar
        used_beta = float(beta_kl) if beta_kl is not None else float(self.beta_kl)
        mu = getattr(self, "_last_mu", None)
        logvar = getattr(self, "_last_logvar", None)
        if used_beta > 0.0 and (mu is not None) and (logvar is not None):
            kl = self.kl_divergence(mu, logvar)
            total_loss = total_loss + used_beta * kl
            metrics["kl"] = kl.detach()
            metrics["beta"] = torch.tensor(used_beta, device=logits.device)
        else:
            metrics["kl"] = torch.tensor(0.0, device=logits.device)
            metrics["beta"] = torch.tensor(float(used_beta), device=logits.device)

        return total_loss, metrics


# ---------------------------------------------------------------------
# Config, builder, and trainer for the AE
# ---------------------------------------------------------------------

@dataclass
class VAEConfig:
    # structure (base space; must be powers of two)
    input_length: int
    K1: int  # e.g., 512
    K2: int  # e.g., 32
    C:  int  # e.g., 64
    embed_dim: int = 8

    # training params
    lr: float = 2e-4
    betas: Tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    ld_lambda: float = 0.0
    maf_lambda: float = 0.0
    ld_window: int = 128


def build_vae(cfg: VAEConfig) -> Tuple[nn.Module, torch.optim.Optimizer]:
    """
    Build the deterministic AE from VAEConfig and return (model, optimizer).
    The model's structure is fully determined by:
        (cfg.input_length, cfg.K1, cfg.K2, cfg.C, cfg.embed_dim)
    Returns:
        model: GenotypeAutoencoder
        optimizer: AdamW initialized with cfg.{lr, betas, weight_decay}
    """
    model = GenotypeAutoencoder(
        input_length=int(cfg.input_length),
        K1=int(cfg.K1),
        K2=int(cfg.K2),
        C=int(cfg.C),
        embed_dim=int(cfg.embed_dim),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(cfg.lr), betas=cfg.betas, weight_decay=float(cfg.weight_decay)
    )
    return model, optimizer


def train_vae(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    device: Optional[torch.device] = None,
    num_epochs: int = 10,
    grad_clip: Optional[float] = 1.0,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    val_dataloader: Optional[torch.utils.data.DataLoader] = None,
    plateau_min_rel_improve: float = 0.005,
    plateau_patience: int = 3,
    plateau_mse_threshold: float = 0.01,
    maf_lambda: float = 0.0,
    ld_lambda: float = 0.0,
    ld_window: int = 128,
    autocast_device: Optional[str] = "cuda",  # "cuda" or "cpu"
) -> Dict[str, Any]:
    """
    Train the GenotypeAutoencoder on integer genotype batches of shape (B, L).
    Prints compact epoch metrics and returns checkpoints for the best model
    (by validation MSE) and the final state.
    Returns:
        {
          "best_state_dict":  CPU copy of the best model params,
          "best_meta":        {"epoch": int, "val_mse": float},
          "last_state_dict":  CPU copy of the last model params,
          "last_meta":        {"epoch": int, "val_mse": float},
        }
    """
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if device is not None:
        model.to(device)
    model.train()

    @torch.no_grad()
    def _eval_validation_mse(model_eval: nn.Module, loader: torch.utils.data.DataLoader) -> float:
        if loader is None:
            return float("inf")
        was_training = model_eval.training
        model_eval.eval()
        total_mse = 0.0
        total_n = 0
        for xb in loader:
            xb = xb.to(device) if device is not None else xb
            logits, _z = model_eval(xb)
            probs = torch.softmax(logits, dim=1)         # (B,3,L)
            x_hat = probs[:, 1, :] + 2.0 * probs[:, 2, :]  # expected dosage
            mse = torch.mean((x_hat - xb.float()) ** 2).item()
            total_mse += mse * xb.size(0)
            total_n += xb.size(0)
        if was_training:
            model_eval.train()
        return total_mse / max(1, total_n)

    global_step = 0
    best_val_mse = float("inf")
    best_epoch = 0
    epochs_no_improve = 0
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_meta: Dict[str, Any] = {}

    for epoch in range(1, int(num_epochs) + 1):
        sum_loss = 0.0
        sum_recon = 0.0
        sum_maf = 0.0
        sum_ld = 0.0
        count = 0

        for xb in dataloader:
            xb = xb.to(device) if device is not None else xb
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=(autocast_device or "cuda"),
                dtype=torch.bfloat16 if (autocast_device == "cuda") else torch.float32,
                enabled=(autocast_device == "cuda" and (device is not None and device.type == "cuda")),
            ):
                logits_full, z = model(xb)
                if hasattr(model, "decode_with_latent_mask") and getattr(
                    model, "enable_structured_latent", False
                ):
                    logits = model.decode_with_latent_mask(z, enable_masking=True)
                else:
                    logits = logits_full

                loss, metrics = model.loss_function(
                    logits, xb, None, maf_lambda=maf_lambda, ld_lambda=ld_lambda, ld_window=ld_window
                )

            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            optimizer.step()
            global_step += 1

            bs = xb.size(0)
            sum_loss += float(loss.item()) * bs
            sum_recon += float(metrics["recon"].item()) * bs
            sum_maf += float(metrics["maf"].item()) * bs
            sum_ld += float(metrics["ld"].item()) * bs
            count += bs

        if scheduler is not None:
            # some schedulers (e.g., CosineAnnealingLR) are stepped per-epoch
            if hasattr(scheduler, "step") and not isinstance(
                scheduler, torch.optim.lr_scheduler.OneCycleLR
            ):
                try:
                    scheduler.step()
                except Exception:
                    pass

        # Epoch summary
        mean_loss = sum_loss / max(1, count)
        mean_recon = sum_recon / max(1, count)
        mean_maf = sum_maf / max(1, count)
        mean_ld = sum_ld / max(1, count)
        lr_value = optimizer.param_groups[0].get("lr", 0.0)
        logger.info(
            f"Epoch {epoch}/{num_epochs}: "
            f"loss={mean_loss:.4f} | recon={mean_recon:.4f} | "
            f"maf={mean_maf:.5f} | ld={mean_ld:.5f} | lr={float(lr_value):.6f}"
        )

        # Validation + early stopping
        if val_dataloader is not None:
            val_mse = _eval_validation_mse(model, val_dataloader)
            logger.info(f"Epoch {epoch}/{num_epochs}: val_mse={val_mse:.6f}")
        else:
            val_mse = float("inf")
        first = (epoch == 1)
        improved = val_mse < best_val_mse * (1.0 - float(plateau_min_rel_improve))
        if first or improved:
            best_val_mse = val_mse
            best_epoch = epoch
            epochs_no_improve = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_meta = {"epoch": int(epoch), "val_mse": float(val_mse)}
        else:
            epochs_no_improve += 1

        if best_val_mse <= float(plateau_mse_threshold) and epochs_no_improve >= int(plateau_patience):
            logger.info(
                f"Early stopping: no ≥{plateau_min_rel_improve*100:.2f}% relative improvement for "
                f"{plateau_patience} epochs (best val MSE={best_val_mse:.6f})."
            )
            break

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()

    last_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    last_meta = {"epoch": int(epoch), "val_mse": float(_eval_validation_mse(model, val_dataloader) if val_dataloader else float("inf"))}
    return {
        "best_state_dict": best_state if best_state is not None else last_state,
        "best_meta": best_meta if best_meta else {"epoch": int(best_epoch), "val_mse": float(best_val_mse)},
        "last_state_dict": last_state,
        "last_meta": last_meta,
    }