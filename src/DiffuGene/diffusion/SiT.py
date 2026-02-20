from __future__ import annotations

import math
from typing import Optional, List, Tuple

import torch
import torch.nn as nn


class _PreNormTransformerBlock(nn.Module):
    """PreNorm self-attention + MLP block."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


class TimeEmbedding(nn.Module):
    """Scalar t -> token embedding of shape (B, 1, D)."""

    def __init__(self, dim: int, fourier_dim: Optional[int] = None) -> None:
        super().__init__()
        self.dim = int(dim)
        self.fourier_dim = int(fourier_dim or dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.fourier_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() != 1:
            raise ValueError(f"Expected 1D timestep tensor (B,), got {tuple(t.shape)}")
        half = self.fourier_dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=t.dtype) / max(1, half - 1)
        )
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if emb.size(-1) < self.fourier_dim:
            emb = torch.cat([emb, torch.zeros_like(emb[:, : self.fourier_dim - emb.size(-1)])], dim=-1)
        return self.mlp(emb).unsqueeze(1)


class ConditionEmbedding(nn.Module):
    """Covariate vector -> condition token (B, 1, D)."""

    def __init__(self, cond_dim: int, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        if c is None:
            raise ValueError("ConditionEmbedding.forward received None.")
        return self.net(c).unsqueeze(1)


class TransformerDiffusionModel(nn.Module):
    """Standard SiT-style transformer over latent tokens."""

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
        self.token_dim = int(token_dim)
        self.latent_length = int(latent_length)
        self.use_cond_token = bool(use_cond_token)
        self.max_special = 2
        self.max_seq_len = self.latent_length + self.max_special
        self.pos_embed = nn.Parameter(torch.zeros(1, self.max_seq_len, self.token_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.blocks = nn.ModuleList(
            [
                _PreNormTransformerBlock(self.token_dim, num_heads, mlp_ratio, dropout)
                for _ in range(num_layers)
            ]
        )
        self.out_norm = nn.LayerNorm(self.token_dim)
        self.out_proj = nn.Linear(self.token_dim, self.token_dim)

    def _build_input(
        self,
        tokens: torch.Tensor,
        time_emb: torch.Tensor,
        cond_emb: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, int]:
        bsz, n, dim = tokens.shape
        if n != self.latent_length:
            raise ValueError(f"Expected latent_length={self.latent_length}, got {n}")
        if time_emb.shape != (bsz, 1, dim):
            raise ValueError(f"time_emb must be {(bsz, 1, dim)}, got {tuple(time_emb.shape)}")
        parts: List[torch.Tensor] = [time_emb]
        start = 1
        if cond_emb is not None and self.use_cond_token:
            if cond_emb.shape != (bsz, 1, dim):
                raise ValueError(f"cond_emb must be {(bsz, 1, dim)}, got {tuple(cond_emb.shape)}")
            parts.append(cond_emb)
            start = 2
        parts.append(tokens)
        x = torch.cat(parts, dim=1)
        seq_len = x.shape[1]
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len={self.max_seq_len}")
        x = x + self.pos_embed[:, :seq_len, :]
        return x, start

    def forward(
        self,
        tokens: torch.Tensor,
        time_emb: torch.Tensor,
        cond_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x, start = self._build_input(tokens, time_emb, cond_emb)
        for blk in self.blocks:
            x = blk(x)
        x = self.out_proj(self.out_norm(x))
        return x[:, start:, :]


class UDiTDiffusionModel(nn.Module):
    """U-shaped token transformer with long skip connections (USiT)."""

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
        self.use_cond_token = bool(use_cond_token)
        self.max_special = 2
        self.max_seq_len = self.latent_length + self.max_special
        self.pos_embed = nn.Parameter(torch.zeros(1, self.max_seq_len, self.token_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        n_down = (num_layers - 1) // 2
        self.down_blocks = nn.ModuleList(
            [_PreNormTransformerBlock(self.token_dim, num_heads, mlp_ratio, dropout) for _ in range(n_down)]
        )
        self.mid_block = _PreNormTransformerBlock(self.token_dim, num_heads, mlp_ratio, dropout)
        self.up_blocks = nn.ModuleList(
            [_PreNormTransformerBlock(self.token_dim, num_heads, mlp_ratio, dropout) for _ in range(n_down)]
        )
        self.skip_projs = nn.ModuleList([nn.Linear(2 * self.token_dim, self.token_dim) for _ in range(n_down)])
        self.out_norm = nn.LayerNorm(self.token_dim)
        self.out_proj = nn.Linear(self.token_dim, self.token_dim)

    def _build_input(
        self,
        tokens: torch.Tensor,
        time_emb: torch.Tensor,
        cond_emb: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, int]:
        bsz, n, dim = tokens.shape
        if n != self.latent_length:
            raise ValueError(f"Expected latent_length={self.latent_length}, got {n}")
        if time_emb.shape != (bsz, 1, dim):
            raise ValueError(f"time_emb must be {(bsz, 1, dim)}, got {tuple(time_emb.shape)}")
        parts: List[torch.Tensor] = [time_emb]
        start = 1
        if cond_emb is not None and self.use_cond_token:
            if cond_emb.shape != (bsz, 1, dim):
                raise ValueError(f"cond_emb must be {(bsz, 1, dim)}, got {tuple(cond_emb.shape)}")
            parts.append(cond_emb)
            start = 2
        parts.append(tokens)
        x = torch.cat(parts, dim=1)
        seq_len = x.shape[1]
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len={self.max_seq_len}")
        x = x + self.pos_embed[:, :seq_len, :]
        return x, start

    def forward(
        self,
        tokens: torch.Tensor,
        time_emb: torch.Tensor,
        cond_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x, start = self._build_input(tokens, time_emb, cond_emb)
        skips: List[torch.Tensor] = []
        for blk in self.down_blocks:
            x = blk(x)
            skips.append(x)
        x = self.mid_block(x)
        for i, (blk, proj) in enumerate(zip(self.up_blocks, self.skip_projs)):
            x = proj(torch.cat([x, skips[-1 - i]], dim=-1))
            x = blk(x)
        x = self.out_proj(self.out_norm(x))
        return x[:, start:, :]


class SiTFlowModel(nn.Module):
    """
    Flow-matching SiT/USiT wrapper with a UNet-compatible forward signature.
    Supports either token latents (B,N,D) or 2D latents (B,C,H,W).
    """

    def __init__(
        self,
        token_dim: int,
        latent_length: int,
        cond_dim: Optional[int] = None,
        num_layers: int = 9,
        num_heads: int = 8,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
        use_udit: bool = False,
    ) -> None:
        super().__init__()
        self.token_dim = int(token_dim)
        self.latent_length = int(latent_length)
        self.conditional = cond_dim is not None and int(cond_dim) > 0

        self.time_embed = TimeEmbedding(self.token_dim)
        self.cond_embed = ConditionEmbedding(int(cond_dim), self.token_dim) if self.conditional else None
        if self.conditional:
            self.null_cond_emb = nn.Parameter(torch.randn(1, self.token_dim))
        self.backbone: nn.Module
        if use_udit:
            self.backbone = UDiTDiffusionModel(
                token_dim=self.token_dim,
                latent_length=self.latent_length,
                num_layers=num_layers,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                use_cond_token=True,
            )
        else:
            self.backbone = TransformerDiffusionModel(
                token_dim=self.token_dim,
                latent_length=self.latent_length,
                num_layers=num_layers,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                use_cond_token=True,
            )

    def _flatten_tokens(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[Tuple[int, int, int]]]:
        if x.dim() == 3:
            bsz, n, d = x.shape
            if n != self.latent_length or d != self.token_dim:
                raise ValueError(
                    f"Expected token input (B,{self.latent_length},{self.token_dim}), got {tuple(x.shape)}"
                )
            return x, None
        if x.dim() == 4:
            bsz, c, h, w = x.shape
            if c != self.token_dim:
                raise ValueError(f"Expected channel dim C={self.token_dim} for 2D input, got C={c}")
            n = h * w
            if n != self.latent_length:
                raise ValueError(
                    f"Expected flattened latent_length={self.latent_length} from H*W, got H*W={n}"
                )
            tokens = x.permute(0, 2, 3, 1).reshape(bsz, n, c).contiguous()
            return tokens, (c, h, w)
        raise ValueError(f"Unsupported latent shape {tuple(x.shape)}. Expected 3D or 4D tensor.")

    def _restore_shape(self, tokens: torch.Tensor, spatial_meta: Optional[Tuple[int, int, int]]) -> torch.Tensor:
        if spatial_meta is None:
            return tokens
        c, h, w = spatial_meta
        bsz = tokens.shape[0]
        return tokens.view(bsz, h, w, c).permute(0, 3, 1, 2).contiguous()

    def _forward_single(self, x: torch.Tensor, t: torch.Tensor, c: Optional[torch.Tensor]) -> torch.Tensor:
        tokens, spatial_meta = self._flatten_tokens(x)
        t_emb = self.time_embed(t)
        c_emb = self.cond_embed(c) if (self.cond_embed is not None and c is not None) else None
        v_tokens = self.backbone(tokens, t_emb, c_emb)
        return self._restore_shape(v_tokens, spatial_meta)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        c: Optional[torch.Tensor] = None,
        cfg_drop_prob: Optional[float] = None,
        return_pair: bool = False,
    ):
        if not self.conditional:
            if cfg_drop_prob is not None and return_pair:
                y = self._forward_single(x, t, None)
                return y, y
            return self._forward_single(x, t, None)

        if c is None:
            raise ValueError("Conditional SiT forward expects covariates `c`.")

        if cfg_drop_prob is None:
            return self._forward_single(x, t, c)

        bsz = x.shape[0]
        cond_emb = self.cond_embed(c)  # (B,1,D)
        null_emb = self.null_cond_emb.view(1, 1, -1).expand(bsz, 1, -1).to(cond_emb.dtype)
        mask = (torch.rand(bsz, device=x.device) < float(cfg_drop_prob)).view(bsz, 1, 1)
        dropped_emb = torch.where(mask, null_emb, cond_emb)

        tokens, spatial_meta = self._flatten_tokens(x)
        t_emb = self.time_embed(t)

        x_in = torch.cat([tokens, tokens], dim=0)
        t_in = torch.cat([t_emb, t_emb], dim=0)
        e_in = torch.cat([dropped_emb, cond_emb], dim=0)
        v_all = self.backbone(x_in, t_in, e_in)
        v_uncond, v_cond = v_all.chunk(2, dim=0)
        out_uncond = self._restore_shape(v_uncond, spatial_meta)
        out_cond = self._restore_shape(v_cond, spatial_meta)
        if return_pair:
            return out_uncond, out_cond
        return out_cond
