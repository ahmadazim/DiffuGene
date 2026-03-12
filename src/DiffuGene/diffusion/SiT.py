from __future__ import annotations

import math
from typing import Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Feature-wise RMS normalization."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x / rms) * self.weight


def _has_sdpa() -> bool:
    return hasattr(F, "scaled_dot_product_attention")


class Attention(nn.Module):
    """UViT-style multi-head attention with SDPA fast-path fallback."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        proj_bias: bool = True,
    ) -> None:
        super().__init__()
        dim = int(dim)
        num_heads = int(num_heads)
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(float(attn_drop))
        self.attn_drop_p = float(attn_drop)

        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(float(proj_drop))
        self.use_sdpa = _has_sdpa()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, dim = x.shape
        qkv = self.qkv(x)
        qkv = qkv.view(bsz, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.use_sdpa:
            drop_p = self.attn_drop_p if self.training else 0.0
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=drop_p)
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            out = attn @ v

        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, dim)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class MLP(nn.Module):
    """ViT-style feed-forward MLP."""

    def __init__(self, dim: int, mlp_ratio: float = 4.0, drop: float = 0.0) -> None:
        super().__init__()
        dim = int(dim)
        hidden = int(dim * float(mlp_ratio))
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(float(drop))
        self.fc2 = nn.Linear(hidden, dim)
        self.drop2 = nn.Dropout(float(drop))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class UViTBlock(nn.Module):
    """
    UViT pre-norm block with optional in-block skip fusion.

    If skip is enabled:
      x = skip_linear(cat([x, skip], -1))
    then:
      x = x + attn(norm1(x))
      x = x + mlp(norm2(x))
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        mlp_drop: float = 0.0,
        skip: bool = False,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            proj_bias=True,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim=dim, mlp_ratio=mlp_ratio, drop=mlp_drop)
        self.skip_linear = nn.Linear(2 * dim, dim) if skip else None

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.skip_linear is not None:
            if skip is None:
                raise ValueError("UViTBlock configured with skip=True but got skip=None.")
            x = self.skip_linear(torch.cat([x, skip], dim=-1))
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class _PreNormTransformerBlock(nn.Module):
    """PreNorm self-attention + MLP block using UViT components."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
        qkv_bias: bool = False,
    ) -> None:
        super().__init__()
        self.block = UViTBlock(
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=float(mlp_ratio),
            qkv_bias=qkv_bias,
            attn_drop=dropout,
            proj_drop=dropout,
            mlp_drop=dropout,
            skip=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


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
        hidden_dim: int,
        num_layers: int = 9,
        num_heads: int = 8,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
        qkv_bias: bool = False,
        use_cond_token: bool = True,
    ) -> None:
        super().__init__()
        self.token_dim = int(token_dim)
        self.latent_length = int(latent_length)
        self.hidden_dim = int(hidden_dim)
        self.use_cond_token = bool(use_cond_token)
        self.max_special = 2
        self.max_seq_len = self.latent_length + self.max_special
        self.pos_embed = nn.Parameter(torch.zeros(1, self.max_seq_len, self.hidden_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.blocks = nn.ModuleList(
            [
                _PreNormTransformerBlock(
                    self.hidden_dim,
                    num_heads,
                    mlp_ratio,
                    dropout,
                    qkv_bias=qkv_bias,
                )
                for _ in range(num_layers)
            ]
        )
        self.out_norm = nn.LayerNorm(self.hidden_dim)
        self.out_proj = nn.Linear(self.hidden_dim, self.hidden_dim)

    def _build_input(
        self,
        tokens: torch.Tensor,
        time_emb: torch.Tensor,
        cond_emb: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, int]:
        bsz, n, dim = tokens.shape
        if n != self.latent_length:
            raise ValueError(f"Expected latent_length={self.latent_length}, got {n}")
        if dim != self.hidden_dim:
            raise ValueError(f"Expected tokens last-dim hidden_dim={self.hidden_dim}, got {dim}")
        if time_emb.shape != (bsz, 1, self.hidden_dim):
            raise ValueError(
                f"time_emb must be {(bsz, 1, self.hidden_dim)}, got {tuple(time_emb.shape)}"
            )
        parts: List[torch.Tensor] = [time_emb]
        start = 1
        if cond_emb is not None and self.use_cond_token:
            if cond_emb.shape != (bsz, 1, self.hidden_dim):
                raise ValueError(
                    f"cond_emb must be {(bsz, 1, self.hidden_dim)}, got {tuple(cond_emb.shape)}"
                )
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


class USiTmodel(nn.Module):
    """UViT/U-SiT style U-shaped transformer backbone."""

    def __init__(
        self,
        hidden_dim: int,
        latent_length: int,
        num_layers: int = 9,
        num_heads: int = 8,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
        qkv_bias: bool = False,
        use_cond_token: bool = True,
    ) -> None:
        super().__init__()
        if num_layers < 3 or (num_layers % 2) != 1:
            raise ValueError(f"USiTmodel requires odd num_layers >= 3, got {num_layers}.")
        self.hidden_dim = int(hidden_dim)
        self.latent_length = int(latent_length)
        self.use_cond_token = bool(use_cond_token)
        self.max_special = 2
        self.max_seq_len = self.latent_length + self.max_special
        self.pos_embed = nn.Parameter(torch.zeros(1, self.max_seq_len, self.hidden_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        half = num_layers // 2
        self.in_blocks = nn.ModuleList(
            [
                UViTBlock(
                    dim=self.hidden_dim,
                    num_heads=num_heads,
                    mlp_ratio=float(mlp_ratio),
                    qkv_bias=qkv_bias,
                    attn_drop=dropout,
                    proj_drop=dropout,
                    mlp_drop=dropout,
                    skip=False,
                )
                for _ in range(half)
            ]
        )
        self.mid_block = UViTBlock(
            dim=self.hidden_dim,
            num_heads=num_heads,
            mlp_ratio=float(mlp_ratio),
            qkv_bias=qkv_bias,
            attn_drop=dropout,
            proj_drop=dropout,
            mlp_drop=dropout,
            skip=False,
        )
        self.out_blocks = nn.ModuleList(
            [
                UViTBlock(
                    dim=self.hidden_dim,
                    num_heads=num_heads,
                    mlp_ratio=float(mlp_ratio),
                    qkv_bias=qkv_bias,
                    attn_drop=dropout,
                    proj_drop=dropout,
                    mlp_drop=dropout,
                    skip=True,
                )
                for _ in range(half)
            ]
        )
        self.norm = nn.LayerNorm(self.hidden_dim)
        self.out_proj = nn.Linear(self.hidden_dim, self.hidden_dim)

    def _build_input(
        self,
        tokens: torch.Tensor,
        time_emb: torch.Tensor,
        cond_emb: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, int]:
        bsz, n, dim = tokens.shape
        if n != self.latent_length:
            raise ValueError(f"Expected latent_length={self.latent_length}, got {n}")
        if dim != self.hidden_dim:
            raise ValueError(f"Expected tokens last-dim hidden_dim={self.hidden_dim}, got {dim}")
        if time_emb.shape != (bsz, 1, self.hidden_dim):
            raise ValueError(
                f"time_emb must be {(bsz, 1, self.hidden_dim)}, got {tuple(time_emb.shape)}"
            )
        parts: List[torch.Tensor] = [time_emb]
        start = 1
        if cond_emb is not None and self.use_cond_token:
            if cond_emb.shape != (bsz, 1, self.hidden_dim):
                raise ValueError(f"cond_emb must be {(bsz, 1, self.hidden_dim)}, got {tuple(cond_emb.shape)}")
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
        for blk in self.in_blocks:
            x = blk(x)
            skips.append(x)
        x = self.mid_block(x)
        for blk in self.out_blocks:
            x = blk(x, skip=skips.pop())
        x = self.out_proj(self.norm(x))
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
        hidden_dim: Optional[int] = None,
        cond_dim: Optional[int] = None,
        num_layers: int = 9,
        num_heads: int = 8,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
        qkv_bias: bool = False,
        use_udit: bool = False,
        source_token_ids: Optional[torch.Tensor] = None,
        num_sources: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.token_dim = int(token_dim)
        self.hidden_dim = int(hidden_dim if hidden_dim is not None else token_dim)
        self.latent_length = int(latent_length)
        # Flow SiT/USiT is unconditional by design for efficiency.
        self.conditional = False

        self.time_embed = TimeEmbedding(self.hidden_dim)
        self.cond_embed = None
        self.source_token_ids: Optional[torch.Tensor]
        self.num_sources = 0
        self.in_proj = nn.Linear(self.token_dim, self.hidden_dim, bias=True)
        # self.pre_backbone_norm = RMSNorm(self.hidden_dim)
        self.pre_backbone_norm = nn.Identity()

        if source_token_ids is not None:
            source_ids = source_token_ids.to(dtype=torch.long).view(-1)
            if source_ids.numel() != self.latent_length:
                raise ValueError(
                    f"source_token_ids must have length {self.latent_length}, got {source_ids.numel()}"
                )
            inferred_sources = int(source_ids.max().item()) + 1 if source_ids.numel() > 0 else 0
            if num_sources is None:
                self.num_sources = inferred_sources
            else:
                self.num_sources = int(num_sources)
                if self.num_sources < inferred_sources:
                    raise ValueError(
                        f"num_sources={self.num_sources} cannot cover max source id "
                        f"{inferred_sources - 1}"
                    )
            self.register_buffer("source_token_ids", source_ids, persistent=True)
            self.source_embed = nn.Embedding(self.num_sources, self.hidden_dim)
            self.adapter_norm = nn.LayerNorm(self.hidden_dim)
            self.adapter_weight = nn.Parameter(
                torch.empty(self.num_sources, self.hidden_dim, self.hidden_dim)
            )
            self.adapter_bias = nn.Parameter(torch.zeros(self.num_sources, self.hidden_dim))
            self.adapter_gate = nn.Parameter(torch.zeros(self.num_sources))
            nn.init.xavier_uniform_(self.adapter_weight)
        else:
            self.source_token_ids = None
            self.source_embed = None
            self.adapter_norm = None
            self.adapter_weight = None
            self.adapter_bias = None
            self.adapter_gate = None
        self.backbone: nn.Module
        if use_udit:
            self.backbone = USiTmodel(
                hidden_dim=self.hidden_dim,
                latent_length=self.latent_length,
                num_layers=num_layers,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                qkv_bias=qkv_bias,
                use_cond_token=False,
            )
        else:
            self.backbone = TransformerDiffusionModel(
                token_dim=self.token_dim,
                latent_length=self.latent_length,
                hidden_dim=self.hidden_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                qkv_bias=qkv_bias,
                use_cond_token=False,
            )
        self.out_proj = nn.Linear(self.hidden_dim, self.token_dim, bias=True)

    def _prepare_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"_prepare_tokens expects last-dim hidden_dim={self.hidden_dim}, got {tokens.shape[-1]}"
            )
        if self.source_token_ids is None:
            return self.pre_backbone_norm(tokens)
        source_ids = self.source_token_ids.to(device=tokens.device)
        if source_ids.numel() != tokens.shape[1]:
            raise ValueError(
                f"Token/source length mismatch: tokens={tokens.shape[1]} vs source ids={source_ids.numel()}"
            )
        x = tokens + self.source_embed(source_ids).unsqueeze(0).to(tokens.dtype)
        y = self.adapter_norm(x)
        weight = self.adapter_weight[source_ids]
        bias = self.adapter_bias[source_ids]
        gate = self.adapter_gate[source_ids].view(1, -1, 1).to(x.dtype)
        delta = torch.einsum("bld,ldh->blh", y, weight) + bias.unsqueeze(0)
        x = x + gate * delta
        return self.pre_backbone_norm(x)

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
        tokens = self.in_proj(tokens)
        tokens = self._prepare_tokens(tokens)
        t_emb = self.time_embed(t)
        v_hidden = self.backbone(tokens, t_emb, None)
        v_tokens = self.out_proj(v_hidden)
        return self._restore_shape(v_tokens, spatial_meta)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        c: Optional[torch.Tensor] = None,
        cfg_drop_prob: Optional[float] = None,
        return_pair: bool = False,
    ):
        if return_pair:
            y = self._forward_single(x, t, None)
            return y, y
        return self._forward_single(x, t, None)
