from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


SharedForwardFn = Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]


@dataclass
class Stage2PenaltyConfigTok:
    tv_lambda: float = 0.0
    robust_lambda: float = 0.0
    stable_lambda: float = 0.0
    latent_noise_std: float = 0.05
    embed_noise_std: float = 0.05

    def is_active(self) -> bool:
        return any(v > 0.0 for v in (self.tv_lambda, self.robust_lambda, self.stable_lambda))


def latent_tv_loss_1d(z: torch.Tensor) -> torch.Tensor:
    """Total variation penalty over token axis for z=(B,N,D)."""
    if z.dim() != 3:
        raise ValueError(f"Expected (B,N,D), got {tuple(z.shape)}")
    return torch.mean(torch.abs(z[:, 1:, :] - z[:, :-1, :]))


def decode_expected_dosage_token(ae: nn.Module, z: torch.Tensor) -> torch.Tensor:
    logits = ae.decode(z)  # (B,L,3)
    probs = torch.softmax(logits, dim=-1)
    return probs[..., 1] + 2.0 * probs[..., 2]


def compute_shared_head_penalties_tok(
    ae: nn.Module,
    shared_forward: SharedForwardFn,
    z_input: torch.Tensor,
    z_dec: torch.Tensor,
    x_batch: torch.Tensor,
    penalty_cfg: Optional[Stage2PenaltyConfigTok],
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    zero = torch.zeros(1, device=z_dec.device, dtype=z_dec.dtype).squeeze(0)
    metrics = {
        "tv": zero.detach().clone(),
        "robust": zero.detach().clone(),
        "stable": zero.detach().clone(),
        "stable_norm": zero.detach().clone(),
    }
    if penalty_cfg is None or not penalty_cfg.is_active():
        return zero, metrics

    loss = zero.clone()
    if penalty_cfg.tv_lambda > 0.0:
        tv = latent_tv_loss_1d(z_dec)
        loss = loss + penalty_cfg.tv_lambda * tv
        metrics["tv"] = tv.detach()

    if penalty_cfg.robust_lambda > 0.0 and penalty_cfg.latent_noise_std > 0.0:
        z_pert = z_dec + penalty_cfg.latent_noise_std * torch.randn_like(z_dec)
        x_hat = decode_expected_dosage_token(ae, z_pert)
        robust = F.mse_loss(x_hat, x_batch.float())
        loss = loss + penalty_cfg.robust_lambda * robust
        metrics["robust"] = robust.detach()

    if penalty_cfg.stable_lambda > 0.0 and penalty_cfg.embed_noise_std > 0.0:
        _, z_dec_noisy = shared_forward(z_input + penalty_cfg.embed_noise_std * torch.randn_like(z_input))
        stable = F.mse_loss(z_dec, z_dec_noisy)
        loss = loss + penalty_cfg.stable_lambda * stable
        metrics["stable"] = stable.detach()
        metrics["stable_norm"] = stable.detach() / (penalty_cfg.embed_noise_std**2 + 1e-8)
    return loss, metrics


class FiLM1D(nn.Module):
    """
    Shared token head for z=(B,N,D): LayerNorm + FiLM modulation by chromosome id.
    """

    def __init__(self, token_dim: int, film_dim: int = 32, num_chromosomes: int = 22):
        super().__init__()
        self.norm = nn.LayerNorm(token_dim)
        self.embed = nn.Embedding(num_embeddings=num_chromosomes, embedding_dim=film_dim)
        self.gamma = nn.Linear(film_dim, token_dim)
        self.beta = nn.Linear(film_dim, token_dim)
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def forward(self, x: torch.Tensor, chrom_id: Union[int, torch.Tensor]) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"FiLM1D expects (B,N,D), got {tuple(x.shape)}")
        x = self.norm(x)
        if isinstance(chrom_id, int):
            chrom_id = torch.full((x.size(0),), int(chrom_id), dtype=torch.long, device=x.device)
        elif chrom_id.dim() == 0:
            chrom_id = chrom_id.expand(x.size(0))
        chrom_id = chrom_id.to(device=x.device, dtype=torch.long)
        emb = self.embed(chrom_id)
        gamma = self.gamma(emb).unsqueeze(1)  # (B,1,D)
        beta = self.beta(emb).unsqueeze(1)    # (B,1,D)
        return x * (1 + gamma) + beta


class HomogenizedTokenAE(nn.Module):
    def __init__(self, ae_list):
        super().__init__()
        self.aes = nn.ModuleList(ae_list)
        token_dim = int(ae_list[0].latent_dim)
        self.encode_head = FiLM1D(token_dim)
        self.decode_head = FiLM1D(token_dim)

        for ae in self.aes:
            for p in ae.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor, chrom_id: int):
        ae = self.aes[chrom_id]
        logits, z = ae(x)
        bsz = z.size(0)
        chrom_vec = torch.full((bsz,), int(chrom_id), dtype=torch.long, device=z.device)
        z_hom = self.encode_head(z, chrom_vec)
        z_dec = self.decode_head(z_hom, chrom_vec)
        logits_out = ae.decode(z_dec)
        return logits_out, z_hom
