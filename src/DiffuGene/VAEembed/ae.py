from __future__ import annotations

import math
import logging
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# =====================================================================
# Standalone utility functions
# =====================================================================

def local_ld_penalty(x_data: torch.Tensor, x_hat: torch.Tensor, window: int = 128) -> torch.Tensor:
    """LD penalty: mean Frobenius norm between local correlation matrices."""
    bsz, length = x_data.shape
    total = 0.0
    num = 0
    for s in range(0, length - window + 1, window):
        e = s + window
        xd = x_data[:, s:e]
        xr = x_hat[:, s:e]
        xd = xd - xd.mean(dim=0, keepdim=True)
        xr = xr - xr.mean(dim=0, keepdim=True)
        cov_d = (xd.T @ xd) / (bsz - 1 + 1e-6)
        cov_r = (xr.T @ xr) / (bsz - 1 + 1e-6)
        std_d = torch.sqrt(torch.diag(cov_d) + 1e-6)
        std_r = torch.sqrt(torch.diag(cov_r) + 1e-6)
        corr_d = cov_d / (std_d[:, None] * std_d[None, :] + 1e-6)
        corr_r = cov_r / (std_r[:, None] * std_r[None, :] + 1e-6)
        total = total + torch.norm(corr_r - corr_d, p="fro")
        num += 1
    return total / max(num, 1)


def latent_tv_loss_1d(
    z: torch.Tensor, mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Total-variation smoothness penalty along the token axis.
    z: (B, T, D)
    mask: (B, T) optional float {0,1} — when provided, only transitions
          between two real (unpadded) tokens contribute to the penalty.
    Returns scalar.
    """
    dt = (z[:, 1:, :] - z[:, :-1, :]) ** 2  # (B, T-1, D)
    if mask is not None:
        edge_mask = (mask[:, :-1] * mask[:, 1:]).unsqueeze(-1)  # (B, T-1, 1)
        dt = dt * edge_mask
        return dt.sum() / (edge_mask.sum() * z.shape[-1] + 1e-8)
    return dt.mean()


def find_best_ck(x: int, min_c: int = 1, max_c: int = 5, *, min_k: Optional[int] = None) -> Tuple[int, int]:
    """Pick (c,k) so c*2^k is close to x with optional k lower-bound."""
    best_c, best_k = 1, 0
    best_err = abs(1.0 - x)
    for c in range(min_c, max_c + 1):
        k = 0
        while True:
            val = c * (1 << k)
            # no truncation
            if val < x:
                k += 1
                continue
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


# =====================================================================
# 1D pixel shuffle / unshuffle helpers and residual blocks
# =====================================================================

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


class PixelUnshuffleChannelAveragingDownSample1D(nn.Module):
    """
    Shortcut: unshuffle along length, then average channel groups to match out_channels.

    Input:  (B, inC, L)
    Unshuf: (B, inC*factor, L//factor)
    Then if needed average groups to (B, outC, L//factor)
    """
    def __init__(self, in_channels: int, out_channels: int, factor: int = 2) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.factor = int(factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = pixel_unshuffle_1d(x, self.factor)  # (B, inC*factor, L//factor)
        if skip.size(1) != self.out_channels:
            B, C_skip, L_new = skip.shape
            if C_skip % self.out_channels != 0:
                raise ValueError(f"Cannot average {C_skip}→{self.out_channels} channels (non-integer group).")
            group_size = C_skip // self.out_channels
            skip = skip.view(B, self.out_channels, group_size, L_new).mean(dim=2)
        return skip


class PixelShuffleChannelReplicateUpSample1D(nn.Module):
    """
    Shortcut: shuffle channels into length; replicate channels to match out_channels.

    Input:  (B, inC, L)
    Shuf:   (B, inC//factor, L*factor)
    Then if needed replicate channels to (B, outC, L*factor)
    """
    def __init__(self, in_channels: int, out_channels: int, factor: int = 2) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.factor = int(factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = pixel_shuffle_1d(x, self.factor)  # (B, inC//factor, L*factor)
        if skip.size(1) != self.out_channels:
            B, C_skip, L_new = skip.shape
            if self.out_channels % C_skip != 0:
                raise ValueError(f"Cannot replicate {C_skip}→{self.out_channels} channels.")
            dup = self.out_channels // C_skip
            skip = skip.repeat_interleave(dup, dim=1)
        return skip


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


class ResidualDownsample1D_Average(nn.Module):
    """
    Main path: Conv1d stride=factor to out_channels
    Shortcut: pixel_unshuffle then channel-group average to out_channels

    Use this when you want to downsample length by `factor` but *not* force out_channels == in_channels*factor
    (e.g., using DC-AE style averaging compression).
    """
    def __init__(self, in_channels: int, out_channels: int, factor: int = 2) -> None:
        super().__init__()
        self.factor = int(factor)
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=self.factor,
            padding=1,
            bias=False,
        )
        self.shortcut = PixelUnshuffleChannelAveragingDownSample1D(
            in_channels, out_channels, factor=self.factor
        )

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


class ResidualUpsample1D_Replicate(nn.Module):
    """
    Main path: ConvTranspose1d stride=factor to out_channels
    Shortcut: pixel_shuffle then channel replicate to out_channels

    Use this when you want to upsample length by `factor` but *not* force in_channels == out_channels*factor.
    """
    def __init__(self, in_channels: int, out_channels: int, factor: int = 2) -> None:
        super().__init__()
        self.factor = int(factor)
        self.deconv = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size=self.factor,
            stride=self.factor,
            bias=False,
        )
        self.shortcut = PixelShuffleChannelReplicateUpSample1D(
            in_channels, out_channels, factor=self.factor
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.deconv(x) + self.shortcut(x)


# =====================================================================
# TokenAutoencoder1D
# =====================================================================

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
        self.compress_1d = self.target_len < self.input_length

        self.num_down = int(self.L1 - self.target_exp)
        if self.num_down < 0:
            raise ValueError(f"Invalid downsample depth from L1={self.L1} to target_exp={self.target_exp}")

        self.start_channels = self.embed_dim

        # -- Structured latent training (channel masking) --
        self.enable_structured_latent: bool = True
        self.structured_latent_min_active: int = 16
        self.structured_latent_step: int = 4
        self._last_latent_mask: Optional[torch.Tensor] = None
        self._last_masked_channels: int = self.latent_dim
        self._last_active_fraction: float = 1.0

        # -- Internal mask bookkeeping (for padded inputs) --
        self._last_mask_1d: Optional[torch.Tensor] = None

        # -- Encoder layers --
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

        # ---- Channel-compress to latent_dim (DC-AE style) ----
        self.post_down_channel_blocks = nn.ModuleList()
        self._post_down_channels_start = int(ch)

        if ch > self.latent_dim:
            while ch > self.latent_dim:
                if ch % 2 != 0:
                    raise ValueError(f"Cannot halve odd channel count ch={ch} toward latent_dim={self.latent_dim}.")
                ch2 = ch // 2
                if ch2 < self.latent_dim:
                    raise ValueError(f"Overshoot: halving {ch} gives {ch2} < latent_dim={self.latent_dim}.")
                self.post_down_channel_blocks.append(
                    ResidualDownsample1D_Average(ch, ch2, factor=1)
                )
                ch = ch2
        elif ch < self.latent_dim:
            self.post_down_channel_blocks.append(
                nn.Conv1d(ch, self.latent_dim, kernel_size=1, bias=False)
            )
            ch = self.latent_dim

        self.encoder_out_channels = int(ch)

        # ---- Invert channel-compress at decoder entry ----
        self.pre_up_channel_blocks = nn.ModuleList()
        ch_start = int(self._post_down_channels_start)

        if ch_start > self.latent_dim:
            ch_dec = self.latent_dim
            while ch_dec < ch_start:
                ch2 = ch_dec * 2
                if ch2 > ch_start:
                    raise ValueError(f"Overshoot: doubling {ch_dec} gives {ch2} > ch_start={ch_start}.")
                self.pre_up_channel_blocks.append(
                    ResidualUpsample1D_Replicate(ch_dec, ch2, factor=1)
                )
                ch_dec = ch2
            if ch_dec != ch_start:
                raise RuntimeError(f"Decoder channel build mismatch: got {ch_dec}, expected {ch_start}")
        elif ch_start < self.latent_dim:
            self.pre_up_channel_blocks.append(
                nn.Conv1d(self.latent_dim, ch_start, kernel_size=1, bias=False)
            )

        self.from_latent = nn.Identity()

        ch = ch_start
        up_blocks: List[ResidualUpsample1D] = []
        for _ in range(self.num_down):
            up_blocks.append(ResidualUpsample1D(ch, ch // 2, factor=2))
            ch = ch // 2
        self.up_blocks = nn.ModuleList(up_blocks)

        self.deconv_expand = nn.ConvTranspose1d(
            self.start_channels,
            self.embed_dim,
            kernel_size=self.c,
            stride=self.c,
            bias=False,
        )
        self.output_proj = nn.Conv1d(self.embed_dim, 3, kernel_size=1, bias=True)

    # ----------------------------------------------------------------
    # Input preparation: embed + pad / compress to target_len
    # ----------------------------------------------------------------

    def _prepare_input(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Embed integer genotypes and pad or compress to target_len.

        Returns:
            h:    (B, embed_dim, target_len)
            mask: (B, target_len) float {0,1} when padded, else None
        """
        B, L = x.shape
        x_onehot = F.one_hot(x.long(), num_classes=3).float()
        x_dosage = x.float().unsqueeze(-1)
        x4 = torch.cat([x_onehot, x_dosage], dim=-1).permute(0, 2, 1).contiguous()
        h = self.input_embed(x4)  # (B, embed_dim, L)

        mask: Optional[torch.Tensor] = None
        if self.compress_1d:
            h = F.interpolate(h, size=self.target_len, mode="linear", align_corners=False)
        elif self.pad_1d:
            h = F.pad(h, (0, self.target_len - L))
            mask = torch.zeros((B, self.target_len), dtype=torch.float32, device=x.device)
            mask[:, :L] = 1.0
        return h, mask

    # ----------------------------------------------------------------
    # Encoder internals (with mask propagation)
    # ----------------------------------------------------------------

    def _encode_from_embed_internal(
        self, h: torch.Tensor, mask: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Encode from embedded representation through conv_reduce → down_blocks → to_latent.
        Propagates mask through stride-c and factor-2 downsampling.

        Args:
            h:    (B, embed_dim, target_len)
            mask: (B, target_len) or None
        Returns:
            z:         (B, latent_length, latent_dim)
            mask_down: (B, latent_length) or None
        """
        h = self.conv_reduce(h)
        if mask is not None:
            B, Lm = mask.shape
            new_len = Lm // self.c
            mask = mask.view(B, new_len, self.c).max(dim=2).values

        for block in self.down_blocks:
            h = block(h)
            if mask is not None:
                Bm, Lm = mask.shape
                new_len = Lm // 2
                mask = mask.view(Bm, new_len, 2).max(dim=2).values

        for blk in self.post_down_channel_blocks:
            h = blk(h)

        z = h.permute(0, 2, 1).contiguous()  # (B, T, D)
        return z, mask

    # ----------------------------------------------------------------
    # Public encode / decode / forward
    # ----------------------------------------------------------------

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode genotypes to token latent.  Returns z: (B, latent_length, latent_dim)."""
        h, mask = self._prepare_input(x)
        h = self.dropout(h)
        z, mask_down = self._encode_from_embed_internal(h, mask)
        self._last_mask_1d = mask_down
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode token latent to logits.  Returns (B, L, 3)."""
        h = z.permute(0, 2, 1).contiguous()
        h = self.from_latent(h)
        for blk in self.pre_up_channel_blocks:
            h = blk(h)
        for block in self.up_blocks:
            h = block(h)
        h = self.deconv_expand(h)

        if self.pad_1d:
            h = h[..., : self.input_length]
        elif self.compress_1d:
            h = F.interpolate(h, size=self.input_length, mode="linear", align_corners=False)

        logits = self.output_proj(h)
        return logits.permute(0, 2, 1).contiguous()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (logits, z).
        During training, latent channel masking is applied before decoding.
        The returned z is always the *unmasked* latent.
        """
        z = self.encode(x)
        logits = self.decode_with_latent_mask(z, enable_masking=True)
        return logits, z

    # ----------------------------------------------------------------
    # Structured latent channel masking
    # ----------------------------------------------------------------

    def _sample_latent_channel_mask(self, device: torch.device) -> Tuple[torch.Tensor, int]:
        """
        Sample a channel-wise prefix mask for the latent.

        Returns:
            mask:    (1, latent_dim, 1) float {0,1}
            c_prime: number of active (front) channels
        """
        D = self.latent_dim
        if not self.enable_structured_latent or D < self.structured_latent_min_active:
            return torch.ones(1, D, 1, device=device), D

        c_values = list(range(
            self.structured_latent_min_active,
            D + 1,
            self.structured_latent_step,
        ))
        if len(c_values) == 0:
            return torch.ones(1, D, 1, device=device), D

        idx = torch.randint(len(c_values), (1,), device=device).item()
        c_prime = int(c_values[idx])

        mask = torch.zeros(1, D, 1, device=device)
        mask[:, :c_prime, :] = 1.0
        return mask, c_prime

    def decode_with_latent_mask(
        self, z: torch.Tensor, *, enable_masking: bool = True
    ) -> torch.Tensor:
        """
        Decode with optional channel masking of the latent.

        During eval or when masking is disabled: standard decode.
        During training with masking enabled: zero out a random suffix of
        latent channels before decoding.
        """
        if not self.training or not enable_masking:
            self._last_latent_mask = None
            self._last_active_fraction = 1.0
            return self.decode(z)

        mask, c_prime = self._sample_latent_channel_mask(z.device)
        self._last_masked_channels = c_prime
        self._last_active_fraction = float(c_prime) / float(self.latent_dim)
        self._last_latent_mask = mask

        # z: (B, T, D) — mask (1, D, 1) → permute to (1, 1, D) for broadcast
        z_masked = z * mask.permute(0, 2, 1)
        return self.decode(z_masked)

    # ----------------------------------------------------------------
    # Analysis helpers
    # ----------------------------------------------------------------

    def decode_expected_dosage(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent z to expected dosage in [0, 2].
        z: (B, T, D) → x_hat: (B, L)
        """
        logits = self.decode(z)  # (B, L, 3)
        probs = torch.softmax(logits, dim=-1)
        class_values = torch.tensor([0.0, 1.0, 2.0], device=z.device).view(1, 1, 3)
        return (probs * class_values).sum(dim=-1)

    @torch.no_grad()
    def encode_from_embed(
        self, h: torch.Tensor, mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Encode from an already-embedded representation (no gradient).

        h:    (B, embed_dim, target_len) — typically from _prepare_input
        mask: (B, target_len) or None
        Returns z: (B, latent_length, latent_dim)
        """
        z, _ = self._encode_from_embed_internal(h, mask)
        return z

    def encode_from_embed_grad(
        self, h: torch.Tensor, mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Encode from an already-embedded representation (with gradient).
        Used by stage-2 stable penalty.
        """
        z, _ = self._encode_from_embed_internal(h, mask)
        return z

    # ----------------------------------------------------------------
    # Loss function
    # ----------------------------------------------------------------

    def loss_function(
        self,
        logits: torch.Tensor,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
        *,
        maf_lambda: float = 0.0,
        ld_lambda: float = 0.0,
        ld_window: int = 128,
        beta_kl: Optional[float] = None,
    ) -> Tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # logits: (B, L, 3)
        bsz, length, n_classes = logits.shape
        ce = F.cross_entropy(
            logits.view(bsz * length, n_classes),
            x.view(bsz * length).long(),
            reduction="none",
        ).view(bsz, length)
        if mask is not None:
            if mask.shape != ce.shape:
                raise ValueError(
                    f"Mask shape {mask.shape} is not compatible with CE shape {ce.shape}; "
                    "mask must be in base space (B, L)."
                )
            ce = ce * mask
            recon_loss = ce.sum() / (mask.sum() + 1e-6)
        else:
            recon_loss = ce.mean()

        total_loss = recon_loss
        metrics = {"recon": recon_loss.detach()}

        probs = torch.softmax(logits, dim=-1)  # (B, L, 3)
        class_values = torch.tensor([0.0, 1.0, 2.0], device=logits.device).view(1, 1, 3)
        x_hat = (probs * class_values).sum(dim=-1)  # (B, L)
        if maf_lambda and maf_lambda > 0.0:
            maf_data = x.float().mean(dim=0) / 2.0
            maf_recon = x_hat.mean(dim=0) / 2.0
            maf_pen = torch.mean(torch.abs(maf_recon - maf_data))
            total_loss = total_loss + maf_lambda * maf_pen
            metrics["maf"] = maf_pen.detach()
        else:
            metrics["maf"] = torch.tensor(0.0, device=logits.device)

        if ld_lambda and ld_lambda > 0.0 and x.size(1) >= ld_window:
            ld_pen = local_ld_penalty(x.float(), x_hat, window=ld_window)
            total_loss = total_loss + ld_lambda * ld_pen
            metrics["ld"] = ld_pen.detach()
        else:
            metrics["ld"] = torch.tensor(0.0, device=logits.device)

        used_beta = float(beta_kl) if beta_kl is not None else 0.0
        metrics["kl"] = torch.tensor(0.0, device=logits.device)
        metrics["beta"] = torch.tensor(used_beta, device=logits.device)
        return total_loss, metrics


# =====================================================================
# Standalone loss / metric helpers
# =====================================================================

def reconstruction_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Cross-entropy loss for genotype reconstruction."""
    bsz, length, n_classes = logits.shape
    return F.cross_entropy(logits.view(bsz * length, n_classes), target.view(bsz * length), reduction="mean")


# =====================================================================
# Evaluation hook: latent penalties (TV / robust / stable)
# =====================================================================

@torch.no_grad()
def eval_latent_penalties_tokenae(
    model: TokenAutoencoder1D,
    loader: torch.utils.data.DataLoader,
    *,
    latent_noise_std_eval: float,
    embed_noise_std_eval: float,
    max_batches: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """
    Evaluate TV, robust, and stable penalties on a data loader.

    Returns dict with keys: tv, robust, stable, stable_norm.
    """
    if loader is None:
        return {
            "tv": float("nan"), "robust": float("nan"),
            "stable": float("nan"), "stable_norm": float("nan"),
        }

    was_training = model.training
    model.eval()

    total_tv = 0.0
    total_robust = 0.0
    total_stable = 0.0
    total_stable_norm = 0.0
    total_n = 0
    batches_seen = 0

    for xb in loader:
        xb = xb.to(device) if device is not None else xb
        B = xb.size(0)
        total_n += B

        h, mask = model._prepare_input(xb)
        z, mask_down = model._encode_from_embed_internal(h, mask)  # (B, T, D)

        # 1) TV smoothness (mask-aware for padded inputs)
        tv_val = latent_tv_loss_1d(z, mask=mask_down)

        # 2) Robust: MSE(x, decode(z + noise))
        if latent_noise_std_eval > 0.0:
            z_pert = z + latent_noise_std_eval * torch.randn_like(z)
        else:
            z_pert = z
        x_hat_pert = model.decode_expected_dosage(z_pert)
        robust_val = F.mse_loss(x_hat_pert, xb.float())

        # 3) Stable: MSE(z_clean, z_noisy) with noise in embedding space
        if embed_noise_std_eval > 0.0:
            h_noisy = h + embed_noise_std_eval * torch.randn_like(h)
        else:
            h_noisy = h
        z_noisy = model.encode_from_embed(h_noisy, mask)
        stable_val = F.mse_loss(z, z_noisy)
        stable_norm_val = stable_val / (embed_noise_std_eval ** 2 + 1e-8)

        total_tv += float(tv_val.item()) * B
        total_robust += float(robust_val.item()) * B
        total_stable += float(stable_val.item()) * B
        total_stable_norm += float(stable_norm_val.item()) * B
        batches_seen += 1
        if max_batches is not None and batches_seen >= max_batches:
            break

    if was_training:
        model.train()

    if total_n == 0:
        return {
            "tv": float("nan"), "robust": float("nan"),
            "stable": float("nan"), "stable_norm": float("nan"),
        }
    return {
        "tv": total_tv / total_n,
        "robust": total_robust / total_n,
        "stable": total_stable / total_n,
        "stable_norm": total_stable_norm / total_n,
    }


# =====================================================================
# Config, builder
# =====================================================================

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
    # Stage-2 regularization
    maf_lambda: float = 0.0
    ld_lambda: float = 0.0
    ld_window: int = 128
    tv_lambda: float = 0.0
    robust_lambda: float = 0.0
    stable_lambda: float = 0.0
    latent_noise_std: float = 0.05
    embed_noise_std: float = 0.05
    stage2_start_frac: float = 0.75
    latent_eval_max_batches: int = 10


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


# =====================================================================
# Training loop with stage-2 regularization
# =====================================================================

def train_token_ae(
    model: TokenAutoencoder1D,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    device: Optional[torch.device] = None,
    num_epochs: int = 100,
    grad_clip: Optional[float] = 1.0,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    val_dataloader: Optional[torch.utils.data.DataLoader] = None,
    plateau_min_rel_improve: float = 0.005,
    plateau_patience: int = 3,
    plateau_mse_threshold: float = 0.01,
    maf_lambda: float = 0.0,
    ld_lambda: float = 0.0,
    ld_window: int = 128,
    autocast_device: Optional[str] = "cuda",
    tv_lambda: float = 0.0,
    robust_lambda: float = 0.0,
    stable_lambda: float = 0.0,
    latent_noise_std: float = 0.05,
    embed_noise_std: float = 0.05,
    stage2_start_frac: float = 0.75,
    latent_eval_max_batches: int = 10,
) -> Dict[str, Any]:
    """
    Train TokenAutoencoder1D on integer genotype batches of shape (B, L).

    Stage 1 (first portion of epochs): standard CE + MAF/LD + structured
        latent masking.
    Stage 2 (from stage2_start_frac onward): add latent regularizers:
        - TV smoothness along token axis
        - Robust: MSE(x, decode(z + noise))
        - Stable: MSE(z_clean, z_noisy) for embedding noise

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

    # ------------------------------------------------------------------
    # Validation MSE helper
    # ------------------------------------------------------------------
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
            logits, _ = model_eval(xb)  # (B, L, 3)
            probs = torch.softmax(logits, dim=-1)
            x_hat = probs[..., 1] + 2.0 * probs[..., 2]
            mse = torch.mean((x_hat - xb.float()) ** 2).item()
            total_mse += mse * xb.size(0)
            total_n += xb.size(0)
        if was_training:
            model_eval.train()
        return total_mse / max(1, total_n)

    # ------------------------------------------------------------------
    # Stage-2 schedule
    # ------------------------------------------------------------------
    stage2_frac = max(0.0, min(1.0, float(stage2_start_frac)))
    stage2_start_epoch = max(1, min(int(num_epochs), int(math.floor(stage2_frac * float(num_epochs))) + 1))
    logger.info(
        "Stage-2 latent regularization (start_frac=%.2f) will run from epoch %d to %d "
        "(tv_lambda=%.3e, robust_lambda=%.3e, stable_lambda=%.3e)",
        stage2_frac, stage2_start_epoch, int(num_epochs),
        tv_lambda, robust_lambda, stable_lambda,
    )

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

        stage2_active = (
            epoch >= stage2_start_epoch
            and (tv_lambda > 0.0 or robust_lambda > 0.0 or stable_lambda > 0.0)
        )

        for xb in dataloader:
            xb = xb.to(device) if device is not None else xb
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=(autocast_device or "cuda"),
                dtype=torch.bfloat16 if autocast_device == "cuda" else torch.float32,
                enabled=(autocast_device == "cuda" and device is not None and device.type == "cuda"),
            ):
                logits, z = model(xb)

                loss, metrics = model.loss_function(
                    logits, xb, None,
                    maf_lambda=maf_lambda,
                    ld_lambda=ld_lambda,
                    ld_window=ld_window,
                )

                # ----- Stage-2 latent regularizers -----
                if stage2_active:
                    stage2_loss = torch.tensor(0.0, device=xb.device)

                    # Clean (no-dropout) embed + encode once for all penalties
                    h_emb, mask_emb = model._prepare_input(xb)
                    z_clean, mask_down = model._encode_from_embed_internal(
                        h_emb, mask_emb
                    )

                    if tv_lambda > 0.0:
                        tv_val = latent_tv_loss_1d(z_clean, mask=mask_down)
                        stage2_loss = stage2_loss + tv_lambda * tv_val

                    if robust_lambda > 0.0 and latent_noise_std > 0.0:
                        z_pert = z_clean + latent_noise_std * torch.randn_like(z_clean)
                        x_hat_pert = model.decode_expected_dosage(z_pert)
                        robust_val = F.mse_loss(x_hat_pert, xb.float())
                        stage2_loss = stage2_loss + robust_lambda * robust_val

                    if stable_lambda > 0.0 and embed_noise_std > 0.0:
                        h_noisy = h_emb + embed_noise_std * torch.randn_like(h_emb)
                        z_noisy = model.encode_from_embed_grad(h_noisy, mask_emb)
                        stable_val = F.mse_loss(z_clean, z_noisy)
                        stage2_loss = stage2_loss + stable_lambda * stable_val

                    loss = loss + stage2_loss

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
            if hasattr(scheduler, "step") and not isinstance(
                scheduler, torch.optim.lr_scheduler.OneCycleLR
            ):
                try:
                    scheduler.step()
                except Exception:
                    pass

        mean_loss = sum_loss / max(1, count)
        mean_recon = sum_recon / max(1, count)
        mean_maf = sum_maf / max(1, count)
        mean_ld = sum_ld / max(1, count)
        lr_value = optimizer.param_groups[0].get("lr", 0.0)
        logger.info(
            "Epoch %d/%d: loss=%.4f | recon=%.4f | maf=%.5f | ld=%.5f | "
            "lr=%.6f | stage2_active=%s",
            epoch, num_epochs, mean_loss, mean_recon, mean_maf, mean_ld,
            float(lr_value), stage2_active,
        )

        # Validation + early stopping
        if val_dataloader is not None:
            val_mse = _eval_validation_mse(model, val_dataloader)
            logger.info("Epoch %d/%d: val_mse=%.6f", epoch, num_epochs, val_mse)
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
                "Early stopping: no >=%.2f%% relative improvement for %d epochs "
                "(best val MSE=%.6f).",
                plateau_min_rel_improve * 100, plateau_patience, best_val_mse,
            )
            break

        # Evaluate latent penalties (train + val)
        train_pen = eval_latent_penalties_tokenae(
            model, dataloader,
            latent_noise_std_eval=latent_noise_std,
            embed_noise_std_eval=embed_noise_std,
            max_batches=latent_eval_max_batches,
            device=device,
        )
        if val_dataloader is not None:
            val_pen = eval_latent_penalties_tokenae(
                model, val_dataloader,
                latent_noise_std_eval=latent_noise_std,
                embed_noise_std_eval=embed_noise_std,
                max_batches=latent_eval_max_batches,
                device=device,
            )
        else:
            val_pen = {
                "tv": float("nan"), "robust": float("nan"),
                "stable": float("nan"), "stable_norm": float("nan"),
            }

        logger.info(
            "Epoch %d/%d latent penalties: "
            "train_tv=%.6e | train_robust=%.6e | train_stable=%.6e | train_stable_norm=%.6e | "
            "val_tv=%.6e | val_robust=%.6e | val_stable=%.6e | val_stable_norm=%.6e",
            epoch, num_epochs,
            train_pen["tv"], train_pen["robust"], train_pen["stable"], train_pen["stable_norm"],
            val_pen["tv"], val_pen["robust"], val_pen["stable"], val_pen["stable_norm"],
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()

    last_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    last_meta = {
        "epoch": int(epoch),
        "val_mse": float(_eval_validation_mse(model, val_dataloader) if val_dataloader else float("inf")),
    }
    return {
        "best_state_dict": best_state if best_state is not None else last_state,
        "best_meta": best_meta if best_meta else {"epoch": int(best_epoch), "val_mse": float(best_val_mse)},
        "last_state_dict": last_state,
        "last_meta": last_meta,
    }
