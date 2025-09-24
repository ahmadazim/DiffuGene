"""
vqvae_genomics.py
==================

An implementation of a Vector-Quantized Variational Autoencoder (VQ‑VAE)
tailored for genomic data. Unlike typical image VQ-VAEs, this model
treats the input 1D sequence as a 2D grid using a Morton (Z-order)
curve, embeds it via a small 2D convolutional encoder, and learns a
discrete latent representation with a codebook. It mirrors the
encoder with a 2D transposed convolution decoder to reconstruct the
original sequence logits. The design supports residual vector
quantization (RVQ) levels, EMA codebook updates, and optional
auxiliary losses to encourage realistic allele frequency (MAF) and
local linkage disequilibrium (LD) patterns.

Key components:

- **Morton 2D order mapping** to reorder the 1D sequence into a 2D grid
  preserving locality.
- **2D convolutional encoder** that downsamples the padded grid from
  size S×S to 16×16, producing `latent_dim` channels.
- **EMA codebook** for discrete latent vectors with residual VQ levels.
- **2D transpose convolutional decoder** that mirrors the encoder to
  reconstruct a categorical distribution over {0,1,2} for each site.
- **Masked loss** that ignores padded positions when computing the
  reconstruction loss.

This file defines the model and training helpers. To build a model
instance, call ``build_vqvae`` with the desired configuration.
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Morton (Z-order) mapping: 2D <-> 1D for dim=d (N = d^2 tokens)
# ---------------------------------------------------------------------

def morton2_index_to_xy(idx: int, dim: int) -> Tuple[int, int]:
    """
    Generalized Morton (Z-order) decoding for arbitrary dim≥1.
    We enumerate classic Morton order over an enclosing M×M grid
    with M = 2^ceil(log2(dim)), and skip any (x,y) with x>=dim or y>=dim.
    The idx here is the rank within the filtered sequence: 0 <= idx < dim*dim.
    Intended mainly for validation / single-lookups; use compute_morton2_order(dim)
    to construct the full permutation efficiently.
    """
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
    target_rank = idx
    rank = 0
    for morton in range(M * M):
        x, y = _decode_pow2(morton, bits)
        if x < dim and y < dim:
            if rank == target_rank:
                return x, y
            rank += 1
    raise IndexError(f"generalized Morton idx {idx} out of range for dim={dim}")


def compute_morton2_order(dim: int) -> torch.LongTensor:
    """
    Build the generalized Morton permutation 'order' of length dim*dim.
    We enumerate classic Morton over M×M (M = 2^ceil(log2(dim))) and
    keep only coords with x<dim and y<dim. 'order' maps row-major
    pos=x*dim+y --> generalized Morton rank in [0, dim*dim).
    """
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


def compute_square_size_and_k(L: int, min_hw: int = 16) -> Tuple[int, int]:
    """
    Given a 1D input length ``L``, compute the minimal square side
    length ``S`` and the number of downsampling steps ``k`` such that
    padding the input to length ``S*S`` allows it to be reshaped to a
    square grid of side ``S``, and after ``k`` successive stride-2
    downsampling operations, the spatial dimensions reach
    ``(min_hw, min_hw)``.

    The function ensures that ``S`` is of the form ``min_hw * 2^k``
    (hence always a power of two multiple of ``min_hw``) and
    ``S*S >= L``.

    Args:
        L: Original 1D sequence length.
        min_hw: The target spatial height/width after downsampling.
                 Defaults to 16.

    Returns:
        A tuple ``(S, k)`` where ``S`` is the side length of the padded
        square grid and ``k`` is the number of downsampling steps.
    """
    if L <= 0:
        raise ValueError("Input length must be positive")
    # Pick S as the next even number >= ceil(sqrt(L)) and >= min_hw.
    root = math.ceil(math.sqrt(L))
    if root % 2 != 0:
        root += 1
    S = max(min_hw, root)
    # Compute k such that S = min_hw * 2^k (for down/up sampling depth)
    # If S is not an exact power-of-two multiple of min_hw, choose the smallest k with min_hw*2^k >= S
    k = 0
    cur = min_hw
    while cur < S:
        cur *= 2
        k += 1
    return S, k


# ---------------------------------------------------------------------
# Utilities: local LD penalty
# ---------------------------------------------------------------------

def local_ld_penalty(x_data: torch.Tensor, x_hat: torch.Tensor, window: int = 128) -> torch.Tensor:
    """
    Compute a penalty based on the difference between the local LD
    correlation matrices of the true data and the reconstruction.

    Args:
        x_data: Tensor of shape `(B, L)` representing the true genotypes
                (0/1/2 encoded).
        x_hat: Tensor of shape `(B, L)` representing the reconstructed
                expected dosage (float).
        window: Size of the window used to compute local correlation
                matrices. Defaults to 128.

    Returns:
        A scalar tensor representing the Frobenius norm of the
        difference between correlation matrices averaged over windows.
    """
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


# ---------------------------------------------------------------------
# Residual block (2D) and EMA codebook / Residual VQ
# ---------------------------------------------------------------------

class ResBlock2D(nn.Module):
    """
    A simple 2D residual block consisting of two convolutional layers
    separated by GroupNorm and SiLU activation. The block adds its
    input to the output and applies a final activation.
    """
    def __init__(self, ch: int, k: int = 3, groups: int = 32):
        super().__init__()
        pad = k // 2
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, kernel_size=k, padding=pad, bias=False),
            nn.GroupNorm(num_groups=min(groups, ch), num_channels=ch),
            nn.SiLU(),
            nn.Conv2d(ch, ch, kernel_size=k, padding=pad, bias=False),
            nn.GroupNorm(num_groups=min(groups, ch), num_channels=ch),
        )
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


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
        Quantize the continuous latents ``z_e`` by nearest-neighbor lookup
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
        # Straight-through estimator
        z_q_st = z_e + (z_q - z_e).detach()
        # Commitment loss
        commit_loss = beta_commit * F.mse_loss(z_e.detach(), z_q, reduction='mean')
        # EMA updates (only in training mode)
        if self.training:
            counts = torch.bincount(indices, minlength=self.num_codes).to(flat.dtype)  # [K]
            embed_sum = torch.zeros_like(self.ema_embed)  # [K, d]
            embed_sum.index_add_(0, indices, flat)  # sum of z_e per code
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


# ---------------------------------------------------------------------
# VQ‑VAE 2D Model
# ---------------------------------------------------------------------

@dataclass
class VQVAEConfig:
    input_length: int
    latent_dim: int = 64  # d = latent channels
    codebook_size: int = 1024  # K = number of codes
    num_quantizers: int = 2  # RVQ levels
    ema_decay: float = 0.9
    ema_eps: float = 1e-5
    hidden_channels: int = 64  # C0: initial conv channels
    width_mult_per_stage: float = 1.0
    beta_commit: float = 0.25
    ld_lambda: float = 1e-3
    maf_lambda: float = 0.0
    ld_window: int = 128
    # Spatial grid dimension for latent tokens (must be a power of two).
    # The final latent representation will have shape (latent_grid_dim x latent_grid_dim).
    latent_grid_dim: int = 16
    # Optional explicit encoder/decoder specifications
    init_down_kernel: int = 0
    init_down_stride: int = 0
    init_down_padding: int = 0
    init_down_out_channels: int = 0
    keep_layers_at_T: int = 0
    dec_up_kernel: int = 0
    dec_up_stride: int = 0
    dec_up_padding: int = 0
    dec_up_output_padding: int = 0
    dec_up_out_channels: int = 0


class SNPVQVAE(nn.Module):
    """
    VQ‑VAE model for per-SNP genotype sequences using a 2D convolutional
    backbone. The model pads the 1D input to a square of size ``S x S``
    (where ``S`` is the minimal power-of-two multiple of 16 such that
    ``S^2 >= input_length``) and reorders it using a Morton (Z-order)
    curve. It then embeds the one-hot encoded input, applies a series
    of stride-2 convolutional blocks to reduce the spatial dimensions
    to 16×16, quantizes the features using a residual VQ, and mirrors
    the process to reconstruct the logits over {0,1,2} for each input
    position.
    """
    def __init__(self, cfg: VQVAEConfig):
        super().__init__()
        self.cfg = cfg
        # Validate that latent_grid_dim is a power of two
        if cfg.latent_grid_dim <= 0 or (cfg.latent_grid_dim & (cfg.latent_grid_dim - 1)) != 0:
            raise ValueError("latent_grid_dim must be a positive power of two")
        # Compute the square side length S and number of downsampling steps k
        # S will be the smallest square grid such that S >= latent_grid_dim and S^2 >= input_length
        # with S = latent_grid_dim * 2^k
        S, k = compute_square_size_and_k(cfg.input_length, min_hw=cfg.latent_grid_dim)
        self.S = S
        self.k = k
        self.L_pad = S * S
        # Precompute Morton order for dimension S
        order2 = compute_morton2_order(S)
        inv_order2 = torch.empty_like(order2)
        inv_order2[order2] = torch.arange(order2.numel(), dtype=torch.long)
        self.register_buffer("order_2d", order2)
        self.register_buffer("inv_order_2d", inv_order2)
        # Initial embedding conv: one-hot 3 channels to C0 channels
        # self.embed = nn.Conv2d(3, cfg.hidden_channels, kernel_size=1, bias=True)
        self.token_embed = nn.Embedding(3, cfg.hidden_channels)
        # Downsampling conv stages (automatic) + adaptive pooling to target latent grid
        T = cfg.latent_grid_dim
        H, W = self.S, self.S
        self.enc_in_H, self.enc_in_W = H, W
        enc_layers: List[nn.Module] = []
        ch = cfg.hidden_channels
        if cfg.init_down_kernel > 0 and cfg.init_down_stride > 0 and cfg.init_down_out_channels > 0:
            out_h = (H + 2 * cfg.init_down_padding - cfg.init_down_kernel) // cfg.init_down_stride + 1
            out_w = (W + 2 * cfg.init_down_padding - cfg.init_down_kernel) // cfg.init_down_stride + 1
            if out_h != T or out_w != T:
                raise ValueError(
                    f"init_down params map ({H},{W})->({out_h},{out_w}), expected ({T},{T})."
                )
            enc_layers += [
                nn.Conv2d(ch, cfg.init_down_out_channels, kernel_size=cfg.init_down_kernel,
                          stride=cfg.init_down_stride, padding=cfg.init_down_padding, bias=False),
                nn.GroupNorm(num_groups=min(32, cfg.init_down_out_channels), num_channels=cfg.init_down_out_channels),
                nn.SiLU(),
                ResBlock2D(cfg.init_down_out_channels, k=3),
            ]
            ch = cfg.init_down_out_channels
            for _ in range(max(0, cfg.keep_layers_at_T)):
                enc_layers += [
                    nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=False),
                    nn.GroupNorm(num_groups=min(32, ch), num_channels=ch),
                    nn.SiLU(),
                    ResBlock2D(ch, k=3),
                ]
            enc_layers += [
                nn.Conv2d(ch, cfg.latent_dim, kernel_size=1, bias=False),
                nn.GroupNorm(num_groups=max(1, min(32, cfg.latent_dim)), num_channels=cfg.latent_dim),
                nn.SiLU(),
                ResBlock2D(cfg.latent_dim, k=3),
            ]
            self.encoder = nn.Sequential(*enc_layers)
        else:
            raise ValueError(
                "No pooling path allowed. Provide explicit init_down_* and keep_layers_at_T to map S→T exactly."
            )
        # VQ quantizer
        self.quantizer = ResidualVQ(
            code_dim=cfg.latent_dim,
            num_codes=cfg.codebook_size,
            num_quantizers=cfg.num_quantizers,
            decay=cfg.ema_decay,
            eps=cfg.ema_eps,
        )
        # Decoder upsample layers (automatic); final interpolate to exact encoder input grid
        self.dec_hidden = cfg.hidden_channels
        dec_layers: List[nn.Module] = []
        ch_dec = cfg.latent_dim
        if cfg.dec_up_kernel > 0 and cfg.dec_up_stride > 0 and cfg.dec_up_out_channels > 0:
            # Project to decoder channels at T×T, then keep capacity via keep_layers_at_T blocks
            dec_layers += [
                nn.Conv2d(ch_dec, cfg.dec_up_out_channels, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(num_groups=max(1, min(32, cfg.dec_up_out_channels)), num_channels=cfg.dec_up_out_channels),
                nn.SiLU(inplace=True),
                ResBlock2D(cfg.dec_up_out_channels, k=3),
            ]
            for _ in range(max(0, cfg.keep_layers_at_T)):
                dec_layers += [
                    nn.Conv2d(cfg.dec_up_out_channels, cfg.dec_up_out_channels, kernel_size=3, padding=1, bias=False),
                    nn.GroupNorm(num_groups=max(1, min(32, cfg.dec_up_out_channels)), num_channels=cfg.dec_up_out_channels),
                    nn.SiLU(inplace=True),
                    ResBlock2D(cfg.dec_up_out_channels, k=3),
                ]
            dec_layers += [
                nn.ConvTranspose2d(cfg.dec_up_out_channels, cfg.dec_up_out_channels,
                                   kernel_size=cfg.dec_up_kernel, stride=cfg.dec_up_stride,
                                   padding=cfg.dec_up_padding,
                                   output_padding=cfg.dec_up_output_padding,
                                   bias=False),
                nn.GroupNorm(num_groups=max(1, min(32, cfg.dec_up_out_channels)), num_channels=cfg.dec_up_out_channels),
                nn.SiLU(inplace=True),
            ]
            ch_dec = cfg.dec_up_out_channels
            self.decoder_up = nn.Sequential(*dec_layers)
        else:
            H_dec, W_dec = T, T
            while (H_dec * 2) <= self.enc_in_H and (W_dec * 2) <= self.enc_in_W:
                ch_next = max(1, int(round(ch_dec / cfg.width_mult_per_stage)))
                dec_layers += [
                    nn.ConvTranspose2d(ch_dec, ch_next, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False),
                    nn.GroupNorm(num_groups=max(1, min(32, ch_next)), num_channels=ch_next),
                    nn.SiLU(inplace=True),
                    ResBlock2D(ch_next, k=3),
                ]
                ch_dec = ch_next
                H_dec *= 2
                W_dec *= 2
            self.decoder_up = nn.Sequential(*dec_layers)
        # Final head: map hidden channels to 3 logits (for classes 0, 1, 2)
        self.out_head = nn.Sequential(
            nn.Conv2d(ch_dec, ch_dec, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=max(1, min(32, ch_dec)), num_channels=ch_dec),
            nn.SiLU(inplace=True),
            nn.Conv2d(ch_dec, 3, kernel_size=1),
        )

        # Log architecture summary
        logger.info(
            f"[VQ-VAE] S={self.S}, L_pad={self.L_pad}, latent_grid_dim={T}, enc_in=({self.enc_in_H},{self.enc_in_W})"
        )
        logger.info(
            f"[VQ-VAE] Encoder stages={len(enc_layers)} | Decoder stages={len(dec_layers)} | hidden_ch={self.cfg.hidden_channels}"
        )

    def pad_and_reorder(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pad the input sequence to length S^2, reorder it using Morton
        order, and return the reordered one-hot encoded grid along with a
        mask indicating valid positions.

        Args:
            x: Tensor of shape `(B, L)` with values in {0,1,2}.

        Returns:
            x_ohe_reordered: Tensor of shape `(B, 3, S, S)` containing
                              the one-hot encoded, Morton-reordered
                              and padded input.
            mask: Tensor of shape `(B, L_pad)` where mask[i] = 1 for
                  positions corresponding to the original sequence and 0
                  for padded positions.
        """
        B, L = x.shape
        device = x.device
        # Pad to S^2 (self.L_pad)
        pad_len = self.L_pad - L
        # x_pad: [B, L_pad]
        if pad_len > 0:
            x_pad = F.pad(x, (0, pad_len))
        else:
            x_pad = x
        # Build mask: 1 for original data, 0 for padded
        mask_cpu = torch.zeros(B, self.L_pad, device=torch.device('cpu'), dtype=torch.float32)
        mask_cpu[:, :L] = 1.0
        # Reorder using Morton order: order_2d maps row-major positions to Morton indices
        # x_pad[:, order_2d] places original sequence in row-major order such that
        # row-major positions correspond to Morton indices.
        reorder_dev = self.order_2d.to(device)
        x_reordered = x_pad[:, reorder_dev]  # [B, S*S]
        # Keep mask on CPU
        reorder_cpu = self.order_2d.cpu()
        mask_reordered = mask_cpu[:, reorder_cpu]  # [B, S*S]
        # One-hot encode: values in {0,1,2}
        # [B, S*S, 3]
        # x_onehot = F.one_hot(x_reordered.long(), num_classes=3).float()
        # # Reshape to [B, S, S, 3] then permute to [B, 3, S, S]
        # x_ohe_grid = x_onehot.view(B, self.S, self.S, 3).permute(0, 3, 1, 2).contiguous()
        # return x_ohe_grid, mask_reordered
        # Embed directly: [B, S*S, C0] -> [B, C0, S, S]
        e = self.token_embed(x_reordered.long())              # [B, S*S, C0]
        x_emb_grid = e.view(B, self.S, self.S, -1).permute(0, 3, 1, 2).contiguous()
        return x_emb_grid, mask_reordered

    def encode_tokens(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode the input sequence ``x`` to continuous latent tokens and
        produce a mask for valid positions.

        Args:
            x: Tensor of shape `(B, L)` with values in {0,1,2}.

        Returns:
            z_e_seq: Continuous latent tokens of shape `(B, d, N)` where
                     `d` is latent_dim and `N`=16*16.
            mask: Tensor of shape `(B, N)` derived from the original
                  sequence mask, flattened in row-major order on the
                  downsampled grid. This is used downstream to weight
                  the loss.
        """
        # Reorder and one-hot encode
        x_ohe_grid, mask_reordered = self.pad_and_reorder(x)  # [B,3,S,S], [B,S*S]
        # Embed and downsample
        # h = self.embed(x_ohe_grid)  # [B, C0, S, S]
        h = x_ohe_grid
        h = self.encoder(h)  # [B, d, latent_grid_dim, latent_grid_dim]
        # Flatten spatial dims (row-major) to tokens
        # latent_grid_dim x latent_grid_dim = final spatial resolution
        B, C, H, W = h.shape
        z_e_seq = h.view(B, C, H * W)  # [B, d, latent_grid_dim^2]
        # With explicit mapping, encoder output grid must equal T×T; thus mask can pass-through
        if (H != self.cfg.latent_grid_dim) or (W != self.cfg.latent_grid_dim):
            raise ValueError(
                f"Encoder grid {(H,W)} != ({self.cfg.latent_grid_dim},{self.cfg.latent_grid_dim}); no pooling permitted."
            )
        mask_tokens = mask_reordered
        return z_e_seq, mask_tokens

    def quantize(self, z_e_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], List[Dict[str, torch.Tensor]]]:
        return self.quantizer(z_e_seq, beta_commit=self.cfg.beta_commit)

    def decode_logits(self, z_q_seq: torch.Tensor) -> torch.Tensor:
        """
        Decode the quantized latents into logits over {0,1,2} for each
        position in the original sequence.

        Args:
            z_q_seq: Quantized latent tokens of shape `(B, d, 16*16)`.
            pad: Unused; kept for API compatibility.

        Returns:
            logits3: Tensor of shape `(B, 3, L)` representing logits
                     over the classes for each of the original
                     positions. Padded positions (if any) are
                     truncated.
        """
        B = z_q_seq.size(0)
        # Reshape tokens back to [B,d,latent_grid_dim,latent_grid_dim]
        grid_dim = self.cfg.latent_grid_dim
        h = z_q_seq.view(B, z_q_seq.size(1), grid_dim, grid_dim)
        # Decode upsample
        h = self.decoder_up(h)
        # Ensure spatial matches encoder input grid exactly
        if (h.shape[-2] != self.enc_in_H) or (h.shape[-1] != self.enc_in_W):
            h = F.interpolate(h, size=(self.enc_in_H, self.enc_in_W), mode="bilinear", align_corners=False)
        # Project to 3 logits
        y = self.out_head(h)  # [B, 3, S, S]
        # Flatten row-major [B, 3, S*S]
        y_flat = y.view(B, 3, -1)
        # Reorder back to original sequence: apply inverse Morton mapping
        inv_order = self.inv_order_2d.to(y_flat.device)
        y_orig = y_flat[:, :, inv_order]  # [B, 3, L_pad]
        # Truncate to original length (we drop padded positions)
        L = self.cfg.input_length
        logits3 = y_orig[:, :, :L]
        return logits3

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor], List[Dict[str, torch.Tensor]], torch.Tensor]:
        """
        Run the full VQ‑VAE forward pass: encode, quantize, and decode.

        Args:
            x: Input tensor of shape `(B, L)` with values in {0,1,2}.

        Returns:
            logits3: Reconstructed logits of shape `(B, 3, L)`.
            z_q_seq: Quantized latent tokens `(B, d, 16*16)`.
            commit_loss: Scalar commitment loss.
            indices_list: List of index tensors per RVQ level.
            stats_list: List of statistics per RVQ level.
            mask_tokens: Mask for tokens (shape `(B,16*16)`) indicating
                        which token positions correspond to valid input
                        positions. (Useful for future extensions.)
        """
        z_e_seq, mask_tokens = self.encode_tokens(x)
        z_q_seq, commit_loss, indices_list, stats_list = self.quantize(z_e_seq)
        logits3 = self.decode_logits(z_q_seq)
        return logits3, z_q_seq, commit_loss, indices_list, stats_list, mask_tokens

    def loss_function(self,
                      logits3: torch.Tensor,
                      x: torch.Tensor,
                      commit_loss: torch.Tensor,
                      mask_tokens: torch.Tensor,
                      *,
                      ld_lambda: Optional[float] = None,
                      maf_lambda: Optional[float] = None,
                      ld_window: Optional[int] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute the loss for a batch. The loss consists of the
        cross-entropy reconstruction loss (masked to ignore padded
        positions), the commitment loss from quantization, and optional
        auxiliary penalties on allele frequency (MAF) and local LD.

        Args:
            logits3: Reconstructed logits of shape `(B,3,L)`.
            x: True genotypes of shape `(B,L)` with values in {0,1,2}.
            commit_loss: Scalar commitment loss.
            mask_tokens: Mask for valid positions on the token grid
                         flattened to `(B,16*16)`. Not used here but
                         retained for extensibility.
            ld_lambda: Weight for the local LD penalty (defaults to
                       cfg.ld_lambda).
            maf_lambda: Weight for the MAF penalty (defaults to
                        cfg.maf_lambda).
            ld_window: Window size for LD penalty (defaults to
                        cfg.ld_window).

        Returns:
            Tuple of (loss, metrics) where metrics is a dictionary
            containing individual loss terms for logging.
        """
        cfg = self.cfg
        ld_lambda = cfg.ld_lambda if ld_lambda is None else ld_lambda
        maf_lambda = cfg.maf_lambda if maf_lambda is None else maf_lambda
        ld_window = cfg.ld_window if ld_window is None else ld_window
        device = logits3.device
        # Cross-entropy per position (unreduced)
        ce_per_pos = F.cross_entropy(logits3, x.long(), reduction='none')  # [B,L]
        # Normalize without allocating a large mask tensor on GPU
        recon = ce_per_pos.mean()
        # Auxiliary penalties
        aux_maf = torch.tensor(0.0, device=device)
        aux_ld = torch.tensor(0.0, device=device)
        if (maf_lambda > 0) or (ld_lambda > 0):
            pi = torch.softmax(logits3, dim=1)  # [B,3,L]
            x_hat = pi[:, 1, :] + 2.0 * pi[:, 2, :]  # E[X]
            if maf_lambda > 0:
                maf_data = x.float().mean(dim=0) / 2.0
                maf_recon = x_hat.mean(dim=0) / 2.0
                aux_maf = maf_lambda * torch.mean(torch.abs(maf_recon - maf_data))
            if ld_lambda > 0 and x.size(1) >= ld_window:
                aux_ld = ld_lambda * local_ld_penalty(x.float(), x_hat, window=ld_window)
        # Total loss
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
                grad_clip: float = 1.0) -> None:
    """
    Train the VQ‑VAE model on a dataloader. Prints out aggregate
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
    
    for epoch in range(num_epochs):
        tot_loss = tot_recon = tot_commit = tot_maf = tot_ld = 0.0
        tot_n = 0
        for batch in tqdm(dataloader, desc="Epoch {epoch+1}/{num_epochs}"):
            x = batch
            x = x.to(device) if device is not None else x
            optimizer.zero_grad(set_to_none=True)
            # logits3, z_q_seq, commit_loss, _, _, mask_tokens = model(x)
            # loss, metrics = model.loss_function(logits3, x, commit_loss, mask_tokens)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits3, z_q_seq, commit_loss, _, _, mask_tokens = model(x)
                loss, metrics = model.loss_function(logits3, x, commit_loss, mask_tokens)
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            B = x.size(0)
            tot_n += B
            tot_loss += loss.item() * B
            tot_recon += metrics["recon"].item() * B
            tot_commit += metrics["commit"].item() * B
            tot_maf += metrics["aux_maf"].item() * B
            tot_ld += metrics["aux_ld"].item() * B
        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"loss={tot_loss/tot_n:.4f}, recon={tot_recon/tot_n:.4f}, "
              f"commit={tot_commit/tot_n:.4f}, aux_maf={tot_maf/tot_n:.5f}, aux_ld={tot_ld/tot_n:.5f}")
        # Clear GPU cache to free memory between epochs
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, 'ipc_collect'):
                torch.cuda.ipc_collect()


# ---------------------------------------------------------------------
# Convenience: build model + optimizer with sane defaults
# ---------------------------------------------------------------------

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
                ema_decay: float = 0.99,
                ema_eps: float = 1e-5, 
                hidden_channels: int = 64,
                width_mult_per_stage: float = 1.0,
                # NEW: explicit S->T and T->~S mapping knobs
                init_down_kernel: int = 0,
                init_down_stride: int = 0,
                init_down_padding: int = 0,
                init_down_out_channels: int = 0,
                keep_layers_at_T: int = 0,
                dec_up_kernel: int = 0,
                dec_up_stride: int = 0,
                dec_up_padding: int = 0,
                dec_up_output_padding: int = 0,
                dec_up_out_channels: int = 0) -> Tuple[SNPVQVAE, torch.optim.Optimizer]:
    """
    Construct a SNPVQVAE model and its optimizer with default hyperparameters.

    Args:
        input_length: Length of the input sequence (number of SNPs).
        latent_dim: Number of latent channels.
        codebook_size: Number of codes in the VQ codebook.
        num_quantizers: Number of residual VQ levels.
        beta_commit: Commitment loss weight.
        lr: Learning rate for the Adam optimizer.

    Returns:
        A tuple `(model, optimizer)` ready for training.
    """
    cfg = VQVAEConfig(
        input_length=input_length,
        latent_dim=latent_dim,
        codebook_size=codebook_size,
        num_quantizers=num_quantizers,
        beta_commit=beta_commit,
        hidden_channels=hidden_channels,
        width_mult_per_stage=width_mult_per_stage,
        ema_decay=ema_decay,
        ema_eps=ema_eps,
        ld_lambda=ld_lambda,
        maf_lambda=maf_lambda,
        ld_window=ld_window,
        latent_grid_dim=latent_grid_dim,
        init_down_kernel=init_down_kernel,
        init_down_stride=init_down_stride,
        init_down_padding=init_down_padding,
        init_down_out_channels=init_down_out_channels,
        keep_layers_at_T=keep_layers_at_T,
        dec_up_kernel=dec_up_kernel,
        dec_up_stride=dec_up_stride,
        dec_up_padding=dec_up_padding,
        dec_up_output_padding=dec_up_output_padding,
        dec_up_out_channels=dec_up_out_channels,
    )
    model = SNPVQVAE(cfg)
    optim = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    return model, optim