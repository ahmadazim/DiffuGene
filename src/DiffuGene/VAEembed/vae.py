"""
This module defines the SNPVQVAE model which implements a variant of
vector-quantised variational autoencoder (VQ-VAE) tailored for per-SNP
genotype sequences. The model strictly follows a deterministic
downsampling/up-sampling strategy outlined by the user:

1. Determine a pair of integers c and k such that c * 2**k is
   closest to the input length L. A helper function find_best_ck is
   declared but not implemented here; it should return (c, k) for a
   given L.
2. If c * 2**k < L, the input is compressed via linear interpolation
   down to c * 2**k. If c * 2**k > L, the input is right-padded
   with zeros (a mask is carried forward for padded positions) up to
   c * 2**k.
3. A 1D convolution with kernel size c and stride c then maps
   length c * 2**k down to length 2**k. The output channel
   dimension is fixed at eight.
4. If k is odd, an additional stride-2 convolution halves the length
   to 2**(k-1). This layer preserves the eight channels.
5. The resulting 1D representation is reordered using a Morton (Z-order)
   curve into a square S x S grid (S = 2**floor(k/2)) with
   eight channels.
6. A stack of stride-2 2D convolutions further downscales the spatial
   dimensions to latent_grid_dim x latent_grid_dim while projecting
   into latent_dim feature channels. A residual vector quantiser
   compresses the latent representation. The decoder mirrors the
   encoder: 2D transposed convolutions restore the spatial resolution,
   the Morton ordering is undone, and transposed 1D convolutions (plus
   interpolation or cropping) recover the original sequence length.

Only the configuration parameters input_length, latent_dim and
latent_grid_dim are used from the provided cfg. All other
convolutional parameters in earlier versions are deliberately ignored.

The code is organised into small helper modules for clarity. See the
`SNPVQVAE` docstring for further details.
"""

from __future__ import annotations

import math
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

class EMACodebook(nn.Module):
    """
    Exponential moving average (EMA) codebook for vector quantization.
    Attributes:
        code_dim: Dimensionality of code vectors (d).
        num_codes: Number of distinct code vectors (K).
        decay: Exponential decay rate for the EMA updates.
        eps: Small constant to avoid division by zero.
    """

    def __init__(self, code_dim: int, num_codes: int, decay: float = 0.99, eps: float = 1e-5):
        super().__init__()
        self.code_dim = code_dim
        self.num_codes = num_codes
        self.decay = decay
        self.eps = eps
        # Initialize embedding weights
        embed = torch.randn(num_codes, code_dim) * 0.05
        self.embedding = nn.Parameter(embed, requires_grad=False)  # EMA updates only
        # Buffers for EMA updates
        self.register_buffer("ema_cluster_size", torch.zeros(num_codes))
        self.register_buffer("ema_embed", torch.zeros(num_codes, code_dim))

    def forward(self, z_e: torch.Tensor, beta_commit: float = 0.25) -> Tuple[torch.Tensor, torch.Tensor, torch.LongTensor, Dict[str, torch.Tensor]]:
        """
        Quantize the continuous latents z_e by nearest-neighbor lookup
        in the codebook, apply a straight-through estimator for backprop,
        and perform EMA updates to the codebook entries.
        Args:
            z_e: Tensor of shape `(B, d, T)` representing continuous
                 latent codes.
            beta_commit: Weight of the commitment loss term.
        Returns:
            z_q_st: Straight-through quantized tensor of shape `(B, d, T)`.
            commit_loss: Scalar commitment loss.
            indices: Tensor of shape `(B, T)` of codebook indices for
                     each position.
            stats: Dictionary containing codebook usage statistics,
                   including 'perplexity' and 'usage_hist'.
        """
        B, d, T = z_e.shape
        assert d == self.code_dim, f"code_dim mismatch: got {d}, expected {self.code_dim}"
        flat = z_e.permute(0, 2, 1).contiguous().view(B * T, d)  # [BT, d]
        BT = flat.size(0)
        K = self.num_codes
        e_sq = (self.embedding * self.embedding).sum(dim=1)  # [K]
        # To avoid OOM, chunk distance computation if needed
        max_elems = 128_000_000  # threshold for total elements
        elems = BT * K
        if elems <= max_elems:
            x_sq = (flat * flat).sum(dim=1, keepdim=True)  # [BT, 1]
            dist = x_sq - 2 * flat @ self.embedding.T + e_sq.unsqueeze(0)  # [BT, K]
            indices = torch.argmin(dist, dim=1)  # [BT]
        else:
            chunk = max(1, max_elems // max(K, 1))
            idx_chunks = []
            for s in range(0, BT, chunk):
                f = flat[s:s + chunk]  # [c, d]
                x_sq_c = (f * f).sum(dim=1, keepdim=True)  # [c, 1]
                dist_c = x_sq_c - 2 * f @ self.embedding.T + e_sq.unsqueeze(0)  # [c, K]
                idx_chunks.append(torch.argmin(dist_c, dim=1))
            indices = torch.cat(idx_chunks, dim=0)  # [BT]
        z_q = self.embedding[indices]  # [BT, d]
        z_q = z_q.view(B, T, d).permute(0, 2, 1).contiguous()  # [B, d, T]
        # Match activation dtype (autocast-safe)
        if z_q.dtype != z_e.dtype:
            z_q = z_q.to(dtype=z_e.dtype)
        # Straight-through estimator
        z_q_st = z_e + (z_q - z_e).detach()
        # Commitment loss
        commit_loss = beta_commit * F.mse_loss(z_e, z_q.detach(), reduction='mean')
        # EMA updates (only in training mode)
        if self.training:
            # Cast to EMA buffer dtypes to avoid dtype mismatch under autocast
            counts = torch.bincount(indices, minlength=self.num_codes).to(self.ema_cluster_size.dtype)  # [K]
            embed_sum = torch.zeros_like(self.ema_embed)  # [K, d] (float32 by default)
            flat_ema = flat.detach().to(embed_sum.dtype)
            embed_sum.index_add_(0, indices, flat_ema)  # sum of z_e per code
            self.ema_cluster_size.mul_(self.decay).add_(counts * (1.0 - self.decay))
            self.ema_embed.mul_(self.decay).add_(embed_sum * (1.0 - self.decay))
            # Laplace smoothing to avoid dead codes
            n = self.ema_cluster_size + self.eps * self.num_codes
            embed_normalized = self.ema_embed / n.unsqueeze(1)
            self.embedding.data.copy_(embed_normalized)
        # Usage statistics
        with torch.no_grad():
            counts = torch.bincount(indices, minlength=self.num_codes).float()  # [K]
            probs = counts / counts.sum().clamp_min(1.0)
            perplexity = torch.exp(-(probs * (probs.clamp_min(1e-12)).log()).sum())
            usage_hist = counts / counts.sum().clamp_min(1.0)
        return z_q_st, commit_loss, indices.view(B, T), {
            "perplexity": perplexity.detach(),
            "usage_hist": usage_hist.detach(),
        }


class ResidualVQ(nn.Module):
    """
    Residual vector quantizer: apply multiple EMA codebooks sequentially
    to encode residuals. This allows more precise reconstruction with
    smaller codebooks per level.
    """
    def __init__(self, code_dim: int, num_codes: int, num_quantizers: int = 2, decay: float = 0.99, eps: float = 1e-5):
        super().__init__()
        assert num_quantizers >= 1
        self.levels = nn.ModuleList([
            EMACodebook(code_dim, num_codes, decay=decay, eps=eps)
            for _ in range(num_quantizers)
        ])

    def forward(self, z_e: torch.Tensor, beta_commit: float = 0.25) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], List[Dict[str, torch.Tensor]]]:
        """
        Apply each codebook to the residual, accumulate quantized codes,
        and return the sum along with the total commitment loss and
        per-level statistics.

        Args:
            z_e: Tensor of shape `(B, d, T)` representing continuous
                 latents.
            beta_commit: Commitment loss weight passed to each codebook.

        Returns:
            z_q_total_st: Sum of quantized codes (straight-through) of
                           shape `(B, d, T)`.
            total_commit_loss: Scalar total commitment loss across levels.
            indices_per_level: List of tensors containing codebook
                               indices for each level.
            stats_per_level: List of dictionaries with perplexity and
                               usage_hist for each level.
        """
        residual = z_e
        z_q_sum = torch.zeros_like(z_e)
        commit_total = torch.tensor(0.0, device=z_e.device)
        indices_list: List[torch.Tensor] = []
        stats_list: List[Dict[str, torch.Tensor]] = []
        for codebook in self.levels:
            z_q_l, commit_l, idx_l, stats_l = codebook(residual, beta_commit=beta_commit)
            z_q_sum = z_q_sum + z_q_l
            commit_total = commit_total + commit_l
            indices_list.append(idx_l)
            stats_list.append(stats_l)
            residual = (residual - z_q_l).detach() + z_q_l  # keep current gradients
        z_q_total_st = z_e + (z_q_sum - z_e).detach()
        return z_q_total_st, commit_total, indices_list, stats_list

def find_best_ck(x: int, max_c: int = 5, *, min_k: Optional[int] = None, force_even_k: bool = True) -> Tuple[int, int]:
    """Pick (c,k) with c∈[1..max_c] and integer k so c*2^k approximates x.
    Optionally enforce k >= min_k and even parity to avoid a second halving conv.
    """
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
        # refine c to keep c*2^k roughly near x
        c_est = int(round(float(x) / float(1 << best_k)))
        best_c = min(max(1, c_est), max_c)
    # Enforce even k if requested (to avoid extra halving conv)
    if force_even_k and (best_k % 2) == 1:
        best_k += 1
    return best_c, best_k

def compute_morton2_order(dim: int) -> torch.LongTensor:
    if dim <= 0:
        raise ValueError("dim must be positive")

    def _decode_pow2(morton: int, bits: int) -> Tuple[int, int]:
        x = y = 0
        for i in range(bits):
            b = (morton >> (2 * i)) & 0x3
            x |= ((b >> 0) & 1) << i
            y |= ((b >> 1) & 1) << i
        return x, y

    bits = max(1, math.ceil(math.log2(dim)))
    M = 1 << bits

    coords: List[Tuple[int, int]] = []
    for morton in range(M * M):
        x, y = _decode_pow2(morton, bits)
        if x < dim and y < dim:
            coords.append((x, y))
            if len(coords) == dim * dim:
                break

    if len(coords) != dim * dim:
        raise RuntimeError("Failed to generate full generalized Morton ordering.")

    n = dim * dim
    order = torch.empty(n, dtype=torch.long)
    for rank, (x, y) in enumerate(coords):
        pos = x * dim + y
        order[pos] = rank
    return order

def local_ld_penalty(x_data: torch.Tensor, x_hat: torch.Tensor, window: int = 128) -> torch.Tensor:
    """Frobenius norm difference of within-window correlation matrices."""
    B, L = x_data.shape
    total = 0.0
    num = 0
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

class ResBlock2D(nn.Module):
    """A simple residual block for 2D feature maps."""

    def __init__(self, ch: int, k: int = 3) -> None:
        super().__init__()
        p = k // 2
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, kernel_size=k, padding=p, bias=False),
            nn.GroupNorm(num_groups=min(32, ch), num_channels=ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(ch, ch, kernel_size=k, padding=p, bias=False),
            nn.GroupNorm(num_groups=min(32, ch), num_channels=ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(self.block(x) + x)


class ResBlock1D(nn.Module):
    """A simple residual block for 1D feature maps."""

    def __init__(self, ch: int, k: int = 3) -> None:
        super().__init__()
        p = k // 2
        self.block = nn.Sequential(
            nn.Conv1d(ch, ch, kernel_size=k, padding=p, bias=False),
            nn.GroupNorm(num_groups=min(32, ch), num_channels=ch),
            nn.GELU(),
            nn.Conv1d(ch, ch, kernel_size=k, padding=p, bias=False),
            nn.GroupNorm(num_groups=min(32, ch), num_channels=ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.block(x) + x)


class Down2D(nn.Module):
    """Stack of stride-2 2D convolutions with residual blocks."""

    def __init__(self, ch_in: int, ch_latent: int, steps: int) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        ch = ch_in
        # First project channels to latent_dim if necessary
        if ch != ch_latent:
            layers += [
                nn.Conv2d(ch, ch_latent, kernel_size=1, bias=False),
                nn.GroupNorm(num_groups=min(32, ch_latent), num_channels=ch_latent),
                nn.SiLU(inplace=True),
            ]
            ch = ch_latent
        for _ in range(steps):
            layers += [
                nn.Conv2d(ch, ch, kernel_size=3, stride=2, padding=1, bias=False),
                nn.GroupNorm(num_groups=min(32, ch), num_channels=ch),
                nn.SiLU(inplace=True),
                ResBlock2D(ch, k=3),
            ]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Up2D(nn.Module):
    """Stack of stride-2 transpose 2D convolutions with residual blocks."""

    def __init__(self, ch_latent: int, ch_out: int, steps: int) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        ch = ch_latent
        for _ in range(steps):
            layers += [
                nn.ConvTranspose2d(ch, ch, kernel_size=4, stride=2, padding=1, bias=False),
                nn.GroupNorm(num_groups=min(32, ch), num_channels=ch),
                nn.SiLU(inplace=True),
                ResBlock2D(ch, k=3),
            ]
        # Project channels back to desired output channels if needed
        if ch != ch_out:
            layers += [
                nn.Conv2d(ch, ch_out, kernel_size=1, bias=False),
                nn.GroupNorm(num_groups=min(32, ch_out), num_channels=ch_out),
                nn.SiLU(inplace=True),
            ]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SNPVQVAE(nn.Module):
    """Vector-quantised variational autoencoder for SNP genotype sequences.
    1. Determine c and k so that c * 2**k is the closest power-of-two multiple to
       the input length L. The helper find_best_ck(L) should return
       these values. c must be an integer >=1 and k a non-negative
       integer.
    2. If c * 2**k < L, compress the sequence to length c * 2**k via
       linear interpolation; otherwise pad with zeros up to that length and
       carry a mask for the padded positions.
    3. Apply a 1D convolution with kernel size and stride equal to c,
       mapping length c * 2**k down to 2**k. The output channel
       dimension is fixed at eight.
    4. If k is odd, perform a second 1D convolution with kernel
       size 2 and stride 2 to halve the length. The resulting
       length will be 2**floor(k/2).
    5. Reorder the 1D sequence into a square grid via Morton ordering.
    6. Downsample the grid to latent_grid_dim x latent_grid_dim using a
       stack of stride-2 2D convolutions while projecting into
       latent_dim channels. Quantise and decode by mirroring
       these steps, ultimately producing logits over {0,1,2} for each
       original input position.
    Only cfg.input_length, cfg.latent_dim and cfg.latent_grid_dim
    are consumed; all other configuration fields are ignored.
    """

    def __init__(self, cfg: VQVAEConfig) -> None:
        super().__init__()
        # Retain entire configuration for access in forward/loss
        self.cfg = cfg
        # Extract and validate core configuration parameters
        self.L: int = int(cfg.input_length)
        self.latent_dim: int = int(cfg.latent_dim)
        self.latent_grid_dim: int = int(cfg.latent_grid_dim)
        self.hidden_1d: int = int(cfg.hidden_1d_channels)
        self.hidden_2d: int = int(cfg.hidden_2d_channels)
        self.layers_at_final: int = int(cfg.layers_at_final)

        if self.latent_grid_dim <= 0 or (self.latent_grid_dim & (self.latent_grid_dim - 1)) != 0:
            raise ValueError("latent_grid_dim must be a positive power of two")
        if self.latent_dim <= 0:
            raise ValueError("latent_dim must be positive")

        # Determine (c, k) such that c * 2^k approximates L, with k bounded
        # so that S = 2^{floor(k/2)} >= cfg.latent_grid_dim. Also prefer even k.
        try:
            lgd_req = int(max(1, int(cfg.latent_grid_dim)))
            min_k_req = int(math.ceil(2 * math.log2(lgd_req)))
            if (min_k_req % 2) == 1:
                min_k_req += 1
            # Guard against odd-optimal k halving: add +2 buffer
            min_k_req += 2
        except Exception:
            min_k_req = None
        c, k = find_best_ck(self.L, min_k=min_k_req, force_even_k=True)
        self.c: int = c
        self.k: int = k

        # Compute the target intermediate length (c * 2^k)
        target_len: int = c * (2 ** k)
        self.target_len: int = target_len
        # Flags for compress or pad
        self.compress_1d: bool = (target_len < self.L)
        self.pad_1d: bool = (target_len > self.L)

        # Compute the length after convolution(s)
        # After conv with stride=c: length = target_len // c == 2**k
        length_after_conv: int = 2 ** k
        # If k is odd, second conv halves length again
        if k % 2 == 1:
            length_after_conv = length_after_conv // 2
        self.length_after_conv: int = length_after_conv

        # Determine square dimension S for Morton ordering
        # length_after_conv should be a power of two; S = 2**floor(k/2)
        if length_after_conv <= 0 or (length_after_conv & (length_after_conv - 1)) != 0:
            raise ValueError("Length after convolution must be a positive power of two")
        # side length for square
        S = 1 << (int(math.log2(length_after_conv)) // 2)
        if S * S != length_after_conv:
            raise ValueError(
                f"Computed square side {S} does not satisfy S*S = {length_after_conv}; check c, k logic"
            )
        self.S: int = S

        # Validate that S >= latent_grid_dim and S % latent_grid_dim == 0
        if self.S < self.latent_grid_dim:
            raise ValueError(
                f"Square side S={self.S} must be >= latent_grid_dim={self.latent_grid_dim}"
            )
        if (self.S % self.latent_grid_dim) != 0:
            raise ValueError(
                f"S ({self.S}) must be divisible by latent_grid_dim ({self.latent_grid_dim})"
            )
        # Number of 2D downsampling steps
        self.k2d: int = int(math.log2(self.S // self.latent_grid_dim))

        # Precompute Morton order and its inverse for S
        order = compute_morton2_order(self.S)  # [S*S]
        if not isinstance(order, torch.Tensor) or order.numel() != self.S * self.S:
            raise ValueError("compute_morton2_order returned invalid order tensor")
        order = order.long()
        inv_order = torch.empty_like(order)
        inv_order[order] = torch.arange(order.numel(), dtype=torch.long)
        self.register_buffer("order_2d", order, persistent=False)
        self.register_buffer("inv_order_2d", inv_order, persistent=False)

        # Token embedding: map 3 genotype classes to hidden_1d embedding channels
        self.token_embed = nn.Embedding(3, self.hidden_1d)

        # Convolution to reduce length from target_len to 2**k
        # Kernel size and stride both equal to c; channels stay at hidden_1d
        self.conv1d_reduce = nn.Conv1d(self.hidden_1d, self.hidden_1d, kernel_size=c, stride=c, bias=False)
        self.resblock1d = ResBlock1D(self.hidden_1d, k=3)
        # Optional second conv when k is odd
        if k % 2 == 1:
            self.conv1d_extra = nn.Conv1d(self.hidden_1d, self.hidden_1d, kernel_size=2, stride=2, bias=False)
            self.has_extra_conv = True
        else:
            self.conv1d_extra = None
            self.has_extra_conv = False

        # Transposed 1D convolution layers mirror the forward convs
        self.deconv1d_expand = nn.ConvTranspose1d(self.hidden_1d, self.hidden_1d, kernel_size=c, stride=c, bias=False)
        if k % 2 == 1:
            self.deconv1d_extra = nn.ConvTranspose1d(self.hidden_1d, self.hidden_1d, kernel_size=2, stride=2, bias=False)
        else:
            self.deconv1d_extra = None

        # Project 2D features into hidden_2d channels before downsampling
        self.proj2d = nn.Conv2d(self.hidden_1d, self.hidden_2d, kernel_size=1, bias=False)
        # Downsample in 2D to latent_grid_dim x latent_grid_dim with hidden_2d channels
        self.down2d = Down2D(ch_in=self.hidden_2d, ch_latent=self.hidden_2d, steps=self.k2d)
        # Optional extra convolutional layers at final latent grid resolution (encoder side)
        if self.layers_at_final > 0:
            self.final2d_encoder = nn.Sequential(*[ResBlock2D(self.hidden_2d, k=3) for _ in range(self.layers_at_final)])
        else:
            self.final2d_encoder = nn.Identity()
        # Project to latent_dim channels right before quantizer
        self.pre_quant_proj = nn.Conv2d(self.hidden_2d, self.latent_dim, kernel_size=1, bias=False)
        # Stabilize scale before quantization
        self.pre_quant_norm = nn.GroupNorm(num_groups=min(32, self.latent_dim), num_channels=self.latent_dim)
        # Quantiser (provided externally)
        self.quantizer = ResidualVQ(
            code_dim=self.latent_dim,
            num_codes=int(cfg.codebook_size),
            num_quantizers=int(cfg.num_quantizers),
            decay=float(cfg.ema_decay),
            eps=float(cfg.ema_eps),
        )
        # Mirror: after quantizer, project back to hidden_2d channels, optional final layers, then upsample 2D back to S x S, and project to hidden_1d
        self.post_quant_proj = nn.Conv2d(self.latent_dim, self.hidden_2d, kernel_size=1, bias=False)
        # Stabilize scale after dequantization
        self.post_quant_norm = nn.GroupNorm(num_groups=min(32, self.latent_dim), num_channels=self.latent_dim)
        if self.layers_at_final > 0:
            self.final2d_decoder = nn.Sequential(*[ResBlock2D(self.hidden_2d, k=3) for _ in range(self.layers_at_final)])
        else:
            self.final2d_decoder = nn.Identity()
        self.up2d = Up2D(ch_latent=self.hidden_2d, ch_out=self.hidden_1d, steps=self.k2d)

        # Final projection to logits
        self.out_head = nn.Sequential(
            nn.Conv1d(self.hidden_1d, self.hidden_1d, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(32, self.hidden_1d), num_channels=self.hidden_1d),
            nn.GELU(),
            nn.Conv1d(self.hidden_1d, 3, kernel_size=1, bias=True),
        )

    # ---------------------------------------------------------------------
    # Helper methods for encoding

    def _prepare_1d(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Convert integer genotype sequence x into an embedded 1D tensor of
        shape [B, 8, target_len]. Performs padding or interpolation to
        reach the intermediate length target_len. Returns the
        prepared 1D tensor and an optional mask tensor of shape
        [B, target_len], where padded positions are marked with 0.
        Args:
            x: Input tensor of shape [B, L] with values in {0,1,2}.
        Returns:
            x_prepared: [B, 8, target_len]
            mask: Optional[torch.Tensor] of shape [B, target_len] or None
        """
        B, L = x.shape
        # Embed to [B, L, 8] then permute to [B,8,L]
        e = self.token_embed(x.long())
        h = e.permute(0, 2, 1).contiguous()  # [B, 8, L]
        mask = None
        if self.compress_1d:
            # Compress via linear interpolation to target_len
            h = F.interpolate(
                h, size=self.target_len, mode="linear", align_corners=False
            )
            # No mask needed; compressed positions are considered valid
        elif self.pad_1d:
            # Pad with zeros to target_len
            pad_len = self.target_len - L
            if pad_len > 0:
                h = F.pad(h, (0, pad_len))
                # Mask marks original data positions as 1 and padded as 0
                mask = torch.zeros((B, self.target_len), dtype=torch.float32, device=h.device)
                mask[:, :L] = 1.0
        else:
            # Exact match; nothing to do
            pass
        return h, mask

    def _downsample_1d(self, h: torch.Tensor, mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply the 1D convolutional downsampling steps (stride=c and optional
        stride=2) and update the mask accordingly. Input h is
        [B,8,target_len], output is [B,8,length_after_conv]. The
        returned mask, if provided, is downsampled by taking a max over
        the corresponding stride windows.
        """
        # First conv: stride=c
        h = self.conv1d_reduce(h)
        h = self.resblock1d(h)
        # Downsample mask if present
        if mask is not None:
            # Reduce mask by grouping consecutive `c` positions
            B, L = mask.shape
            new_len = mask.shape[1] // self.c
            # Reshape to [B, new_len, c] and take max
            mask_view = mask.view(B, new_len, self.c)
            mask = mask_view.max(dim=2).values
        # Second conv if k is odd
        if self.has_extra_conv:
            h = self.conv1d_extra(h)
            if mask is not None:
                # Halve the mask length: group pairs and take max
                B, Lm = mask.shape
                new_len = Lm // 2
                mask = mask.view(B, new_len, 2).max(dim=2).values
        return h, mask

    def _to_square_and_reorder(self, h: torch.Tensor, mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Reorder the 1D representation into a square via Morton ordering.
        Input h is [B,8,length_after_conv]. Output h2d is
        [B,8,S,S]. The mask is reordered and reshaped similarly to
        [B,S,S], or None if no mask was provided.
        """
        B, C, L = h.shape
        if L != self.S * self.S:
            raise RuntimeError(
                f"Length after 1D convs ({L}) does not match S*S ({self.S * self.S})"
            )
        # Reorder using Morton order along the last dimension
        order = self.order_2d.to(h.device)
        h_seq = h  # [B,8,L]
        h_seq = h_seq[..., order]
        h2d = h_seq.view(B, C, self.S, self.S)
        mask2d: Optional[torch.Tensor] = None
        if mask is not None:
            m_seq = mask  # [B,L]
            m_seq = m_seq[..., order]
            mask2d = m_seq.view(B, self.S, self.S)
        return h2d, mask2d

    def _downsample_2d(self, h2d: torch.Tensor, mask2d: Optional[torch.Tensor]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Downsample the 2D grid to latent_grid_dim x latent_grid_dim and
        project to latent_dim channels. Also downsamples the mask by
        aggregating validity across 2x2 windows.
        """
        # Project channels first
        h = self.proj2d(h2d)
        # Downsample via stride-2 convs
        h = self.down2d(h)
        # Optional extra conv layers at final resolution
        h = self.final2d_encoder(h)
        # Project to latent_dim for quantizer input
        h = self.pre_quant_proj(h)
        h = self.pre_quant_norm(h)
        # Downsample mask via average pooling > 0.5 if provided
        mask_tokens: Optional[torch.Tensor] = None
        if mask2d is not None:
            # Each stride-2 conv halves each spatial dimension; pool mask accordingly
            m = mask2d
            for _ in range(self.k2d):
                # 2x2 non-overlapping regions: if any element is valid, mark as 1
                m = F.avg_pool2d(m.unsqueeze(1), kernel_size=2, stride=2).squeeze(1)
                m = (m > 0.0).float()
            # Flatten mask to [B, T*T]
            mask_tokens = m.view(m.size(0), -1)
        return h, mask_tokens

    def encode_tokens(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Encode the input integer genotype sequence x into a latent
        continuous representation and produce a mask for valid token
        positions. Returns z_e_seq of shape [B,latent_dim,T*T] and
        mask_tokens of shape [B,T*T] (or None).
        """
        # Prepare 1D representation
        h1d, mask1d = self._prepare_1d(x)
        # Apply 1D convolutional downsampling
        h1d, mask1d = self._downsample_1d(h1d, mask1d)
        # Reorder and reshape to square
        h2d, mask2d = self._to_square_and_reorder(h1d, mask1d)
        # Downsample 2D to latent grid and get mask tokens
        h_latent, mask_tokens = self._downsample_2d(h2d, mask2d)
        # Flatten spatial dims to [B, latent_dim, T*T]
        B, C, H, W = h_latent.shape
        if H != self.latent_grid_dim or W != self.latent_grid_dim:
            raise RuntimeError(
                f"2D encoder output has shape ({H},{W}) but expected ({self.latent_grid_dim},{self.latent_grid_dim})"
            )
        z_e_seq = h_latent.view(B, C, H * W)
        return z_e_seq, mask_tokens

    # ---------------------------------------------------------------------
    # Decoding helpers

    def _upsample_2d(self, z_q_seq: torch.Tensor) -> torch.Tensor:
        """
        Convert quantised tokens of shape [B,latent_dim,T*T] back to
        a 2D grid [B,hidden_1d,S,S] via 2D transposed convolutions."""
        B = z_q_seq.size(0)
        C = z_q_seq.size(1)
        # Reshape to [B,C,T,T]
        grid_dim = self.latent_grid_dim
        h2d = z_q_seq.view(B, C, grid_dim, grid_dim)
        # Project from latent_dim to hidden_2d and apply optional final layers
        # Symmetric normalization after dequantization
        h2d = self.post_quant_norm(h2d)
        h2d = self.post_quant_proj(h2d)
        h2d = self.final2d_decoder(h2d)
        # Upsample via 2D transposed convs; output [B,hidden_1d,S,S]
        h2d_up = self.up2d(h2d)
        return h2d_up

    def _from_square_and_reorder(self, h2d: torch.Tensor) -> torch.Tensor:
        """
        Reorder the 2D grid back into a 1D sequence using the inverse
        Morton order. Input h2d is [B,8,S,S]. Returns
        [B,8,length_after_conv].
        """
        B, C, H, W = h2d.shape
        if H != self.S or W != self.S:
            raise RuntimeError(
                f"Expected decoded 2D shape ({self.S},{self.S}), got ({H},{W})"
            )
        # Flatten to [B,8,S*S] and reorder
        h_seq = h2d.view(B, C, H * W)
        inv = self.inv_order_2d.to(h2d.device)
        h_seq = h_seq[..., inv]
        return h_seq

    def _upsample_1d(self, h: torch.Tensor) -> torch.Tensor:
        """
        Mirror the 1D downsampling: apply transposed convs and final
        interpolate or crop to restore the original length L. Input
        h is [B,8,length_after_conv]. Returns [B,8,L].
        """
        # Mirror optional extra conv if used
        if self.has_extra_conv:
            if self.deconv1d_extra is None:
                raise RuntimeError("deconv1d_extra missing despite has_extra_conv")
            h = self.deconv1d_extra(h)
        # Transposed conv with stride=c
        h = self.deconv1d_expand(h)
        # At this point length should be target_len
        # If we padded originally, crop to L; if we compressed, interpolate back to L
        B, C, L_curr = h.shape
        if self.pad_1d:
            # Crop padded part
            if L_curr < self.L:
                raise RuntimeError(
                    f"Decoded length {L_curr} is smaller than original length {self.L}"
                )
            h = h[..., : self.L]
        elif self.compress_1d:
            # Upsample back to original length
            if L_curr != self.target_len:
                raise RuntimeError(
                    f"Expected length {self.target_len} after deconv, got {L_curr}"
                )
            h = F.interpolate(
                h, size=self.L, mode="linear", align_corners=False
            )
        else:
            if L_curr != self.L:
                raise RuntimeError(
                    f"Expected decoded length {self.L}, got {L_curr}"
                )
        return h

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Encode x and return quantisable latent representation along
        with a mask for valid token positions."""
        return self.encode_tokens(x)

    def decode_logits(self, z_q_seq: torch.Tensor) -> torch.Tensor:
        """Decode quantised tokens z_q_seq into logits over
        {0,1,2} for each input position. The returned tensor has
        shape [B,3,L].
        """
        # 2D upsample to [B,8,S,S]
        h2d_up = self._upsample_2d(z_q_seq)
        # Inverse Morton order and reshape to [B,8,length_after_conv]
        h1d = self._from_square_and_reorder(h2d_up)
        # Upsample to original 1D length
        h1d_up = self._upsample_1d(h1d)
        # Final 1D head to logits
        logits3 = self.out_head(h1d_up)
        return logits3

    def forward(
        self, x: torch.Tensor, beta_commit: Optional[float] = None
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        List[torch.Tensor],
        List[Dict[str, torch.Tensor]],
        Optional[torch.Tensor],
    ]:
        """Run the full VQ-VAE forward pass: encode, quantise and decode.

        Returns a tuple consisting of the reconstructed logits, the
        quantised latent representation, the commitment loss, the list of
        indices per residual quantisation layer, the list of running
        statistics for the quantiser and the token mask (or None).
        """
        z_e_seq, mask_tokens = self.encode_tokens(x)
        z_q_seq, commit_loss, indices_list, stats_list = self.quantizer(
            z_e_seq, beta_commit=(self.cfg.beta_commit if beta_commit is None else beta_commit)
        )
        logits3 = self.decode_logits(z_q_seq)
        return logits3, z_q_seq, commit_loss, indices_list, stats_list, mask_tokens

    def loss_function(
        self,
        logits3: torch.Tensor,
        x: torch.Tensor,
        commit_loss: torch.Tensor,
        mask_tokens: Optional[torch.Tensor],
        *,
        ld_lambda: Optional[float] = None,
        maf_lambda: Optional[float] = None,
        ld_window: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute the training loss for a batch.

        The loss is the sum of masked cross-entropy reconstruction loss,
        commitment loss from quantisation and optional auxiliary penalties
        for minor allele frequency (MAF) and local linkage disequilibrium
        (LD).

        Args:
            logits3: [B,3,L] reconstructed logits.
            x: [B,L] true genotype labels.
            commit_loss: scalar commitment loss returned by the quantiser.
            mask_tokens: optional [B,T*T] mask for valid latent tokens. This
                implementation does not currently use it directly but
                retains the parameter for forward compatibility.
            ld_lambda: weight for the LD penalty (overrides cfg.ld_lambda).
            maf_lambda: weight for the MAF penalty (overrides cfg.maf_lambda).
            ld_window: window size for LD penalty (overrides cfg.ld_window).

        Returns:
            loss: scalar tensor
            metrics: dictionary of individual loss terms for logging
        """
        cfg = self.cfg
        # Override with provided lambdas if specified
        ld_lambda = cfg.ld_lambda if ld_lambda is None else ld_lambda
        maf_lambda = cfg.maf_lambda if maf_lambda is None else maf_lambda
        ld_window = cfg.ld_window if ld_window is None else ld_window
        device = logits3.device

        # Masked cross-entropy over positions
        ce_per_pos = F.cross_entropy(logits3, x.long(), reduction="none")  # [B,L]
        recon: torch.Tensor = ce_per_pos.mean()

        # Auxiliary penalties
        aux_maf = torch.tensor(0.0, device=device)
        aux_ld = torch.tensor(0.0, device=device)
        if maf_lambda and maf_lambda > 0:
            pi = torch.softmax(logits3, dim=1)  # [B,3,L]
            x_hat = pi[:, 1, :] + 2.0 * pi[:, 2, :]  # Expected genotype
            maf_data = x.float().mean(dim=0) / 2.0
            maf_recon = x_hat.mean(dim=0) / 2.0
            aux_maf = maf_lambda * torch.mean(torch.abs(maf_recon - maf_data))
        if ld_lambda and ld_lambda > 0 and x.size(1) >= (ld_window or 0):
            aux_ld = ld_lambda * local_ld_penalty(x.float(), x_hat, window=ld_window)

        loss = recon + commit_loss + aux_maf + aux_ld
        return loss, {
            "recon": recon.detach(),
            "commit": commit_loss.detach(),
            "aux_maf": aux_maf.detach(),
            "aux_ld": aux_ld.detach(),
        }


# ---------------------------------------------------------------------
# Training & Eval Helpers
# ---------------------------------------------------------------------

@torch.no_grad()
def eval_batch_categorical(model: SNPVQVAE, x: torch.Tensor, device=None, verbose=True) -> Dict[str, any]:
    """
    Evaluate the model on a batch of data. Computes genotype accuracy,
    dosage MSE, and diagnostic statistics. Does not compute loss.

    Args:
        model: The SNPVQVAE model.
        x: Input batch of shape `(B, L)` with values in {0,1,2}.
        device: Optional device to move the input to (defaults to
                model's device).
        verbose: If True, prints summary statistics to stdout.

    Returns:
        A dictionary containing evaluation metrics and diagnostic
        tensors.
    """
    if device is None:
        device = next(model.parameters()).device
    x = x.to(device)
    x_long = x.long()
    # Forward pass
    logits3, z_q_seq, commit_loss, indices_list, stats_list, mask_tokens = model(x)
    # Ensure logits have correct shape
    if logits3.dim() != 3:
        raise ValueError(f"Expected logits to have 3 dims [B, 3, L], got {logits3.shape}.")
    probs = torch.softmax(logits3, dim=1)
    class_dim = 1
    B, _, L = probs.shape
    # Accuracy (exact genotype)
    pred = probs.argmax(dim=class_dim)  # [B, L]
    acc = (pred == x_long).float().mean().item()
    # MSE on dosage using expected value under categorical
    class_values = torch.tensor([0.0, 1.0, 2.0], device=probs.device).view(1, 3, 1)
    x_hat = (probs * class_values).sum(dim=class_dim)  # [B, L]
    mse = F.mse_loss(x_hat, x.float()).item()
    # Diagnostics on incorrect sites
    err_mask = (pred != x_long)
    num_err = int(err_mask.sum().item())
    true_probs = probs.gather(class_dim, x_long.unsqueeze(class_dim)).squeeze(class_dim)
    pred_probs = probs.gather(class_dim, pred.unsqueeze(class_dim)).squeeze(class_dim)
    margin = pred_probs - true_probs
    if num_err > 0:
        mean_true_prob_err = true_probs[err_mask].mean().item()
        mean_margin_err = margin[err_mask].mean().item()
        top2 = probs.topk(k=2, dim=class_dim)
        in_top2 = (top2.indices == x_long.unsqueeze(class_dim)).any(dim=class_dim)
        top2_rate_err = in_top2[err_mask].float().mean().item()
        conf = {}
        for c in [0, 1, 2]:
            mask_c = (x_long == c) & err_mask
            conf[f"err_true_{c}"] = int(mask_c.sum().item())
    else:
        mean_true_prob_err = float('nan')
        mean_margin_err = float('nan')
        top2_rate_err = float('nan')
        conf = {f"err_true_{c}": 0 for c in [0, 1, 2]}
    perpl = [s["perplexity"].item() for s in stats_list] if stats_list else None
    if verbose:
        base = f"[Eval] acc={acc:.4f} | mse={mse:.6f} | errors={num_err}"
        if commit_loss is not None and torch.is_tensor(commit_loss):
            base += f" | commit={commit_loss.item():.5f}"
        if perpl is not None:
            ptxt = " | ".join([f"P{li+1}={perpl[li]:.1f}" for li in range(len(perpl))])
            base += f" | {ptxt}"
        logger.info(base)
        if num_err > 0:
            logger.info(
                f"       mean true prob on errors={mean_true_prob_err:.4f} | "
                f"mean margin (pred-true) on errors={mean_margin_err:.4f} | "
                f"true-in-top2 (errors)={top2_rate_err:.4f}"
            )
            logger.info(f"       confusion on errors: {conf}")
            err_idx = err_mask.nonzero(as_tuple=False)
            k = min(5, err_idx.size(0))
            if k > 0:
                logger.info("       sample error sites (batch, pos):")
                for i in range(k):
                    b, pos = err_idx[i].tolist()
                    p_vec = probs[b, :, pos].tolist()
                    logger.info(
                        f"         ex{i}: b={b}, pos={pos}, true={x_long[b,pos].item()}, "
                        f"pred={pred[b,pos].item()}, probs={[round(v, 4) for v in p_vec]}"
                    )
    metrics = {
        "acc": acc,
        "mse": mse,
        "num_errors": num_err,
        "mean_true_prob_on_errors": mean_true_prob_err,
        "mean_margin_pred_minus_true_on_errors": mean_margin_err,
        "top2_contains_true_on_errors": top2_rate_err,
        **conf,
    }
    return {
        "acc": acc,
        "mse": mse,
        "commit": commit_loss.item() if commit_loss is not None and torch.is_tensor(commit_loss) else None,
        "perplexities": perpl,
        "usage_hists": [s["usage_hist"].cpu() for s in stats_list] if stats_list else None,
        "metrics": metrics,
        "pred": pred,
        "probs": probs,
        "x_hat": x_hat,
        "true_probs": true_probs,
        "pred_probs": pred_probs,
        "err_mask": err_mask,
    }


def train_vqvae(model: SNPVQVAE,
                dataloader: torch.utils.data.DataLoader,
                optimizer: torch.optim.Optimizer,
                device: Optional[torch.device] = None,
                num_epochs: int = 10,
                grad_clip: float = 1.0,
                scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
                beta_warmup_steps: int = 0) -> None:
    """
    Train the VQ-VAE model on a dataloader. Prints out aggregate
    statistics per epoch.
    Args:
        model: The model to train.
        dataloader: PyTorch DataLoader providing batches of shape `(B,L)`.
        optimizer: Optimizer for updating model parameters.
        device: Optional device to run the training on.
        num_epochs: Number of epochs.
        grad_clip: Gradient clipping threshold (set None or ≤0 to disable).
    """
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    if device is not None:
        model.to(device)
    model.train()
    
    global_step = 0
    for epoch in range(num_epochs):
        tot_loss = tot_recon = tot_commit = tot_maf = tot_ld = 0.0
        tot_n = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        # for batch in dataloader:
            x = batch
            x = x.to(device) if device is not None else x
            optimizer.zero_grad(set_to_none=True)
            # logits3, z_q_seq, commit_loss, _, _, mask_tokens = model(x)
            # loss, metrics = model.loss_function(logits3, x, commit_loss, mask_tokens)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                if beta_warmup_steps and beta_warmup_steps > 0:
                    frac = min(1.0, float(global_step + 1) / float(beta_warmup_steps))
                    beta_now = frac * model.cfg.beta_commit
                else:
                    beta_now = model.cfg.beta_commit
                logits3, z_q_seq, commit_loss, _, _, mask_tokens = model(x, beta_commit=beta_now)
                loss, metrics = model.loss_function(logits3, x, commit_loss, mask_tokens)
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            global_step += 1
            B = x.size(0)
            tot_n += B
            tot_loss += loss.item() * B
            tot_recon += metrics["recon"].item() * B
            tot_commit += metrics["commit"].item() * B
            tot_maf += metrics["aux_maf"].item() * B
            tot_ld += metrics["aux_ld"].item() * B
        if scheduler is not None:
            scheduler.step()
        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"loss={tot_loss/tot_n:.4f}, recon={tot_recon/tot_n:.4f}, "
              f"commit={tot_commit/tot_n:.4f}, aux_maf={tot_maf/tot_n:.5f}, "
              f"aux_ld={tot_ld/tot_n:.5f}, lr={scheduler.get_last_lr()[0]:.6f}, beta={beta_now:.4f}")
        # Clear GPU cache to free memory between epochs
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, 'ipc_collect'):
                torch.cuda.ipc_collect()


# ---------------------------------------------------------------------
# Convenience: build model + optimizer with sane defaults
# ---------------------------------------------------------------------

@dataclass
class VQVAEConfig:
    input_length: int
    latent_dim: int 
    codebook_size: int
    num_quantizers: int
    ema_decay: float
    ema_eps: float
    beta_commit: float
    ld_lambda: float
    maf_lambda: float
    ld_window: int
    latent_grid_dim: int
    hidden_1d_channels: int
    hidden_2d_channels: int
    layers_at_final: int

def build_vqvae(input_length: int,
                latent_dim: int = 64,
                codebook_size: int = 1024,
                num_quantizers: int = 2,
                beta_commit: float = 0.25,
                lr: float = 2e-4,
                latent_grid_dim: int = 16, 
                ld_lambda: float = 1e-3,
                maf_lambda: float = 1e-3,
                ld_window: int = 128,
                ema_decay: float = 0.9,
                ema_eps: float = 1e-5,
                hidden_1d_channels: int = 8,
                hidden_2d_channels: Optional[int] = None,
                layers_at_final: int = 0) -> Tuple[SNPVQVAE, torch.optim.Optimizer]:
    if hidden_2d_channels is None:
        hidden_2d_channels = latent_dim
    cfg = VQVAEConfig(
        input_length=input_length,
        latent_dim=latent_dim,
        codebook_size=codebook_size,
        num_quantizers=num_quantizers,
        beta_commit=beta_commit,
        ema_decay=ema_decay,
        ema_eps=ema_eps,
        ld_lambda=ld_lambda,
        maf_lambda=maf_lambda,
        ld_window=ld_window,
        latent_grid_dim=latent_grid_dim,
        hidden_1d_channels=hidden_1d_channels,
        hidden_2d_channels=hidden_2d_channels,
        layers_at_final=layers_at_final,
    )
    model = SNPVQVAE(cfg)
    optim = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    return model, optim