from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ae import latent_tv_loss, decode_expected_dosage


SharedForwardFn = Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]


@dataclass
class Stage2PenaltyConfig:
    """Container for latent-stage penalties applied during shared-head training."""

    tv_lambda: float = 0.0
    robust_lambda: float = 0.0
    stable_lambda: float = 0.0
    latent_noise_std: float = 0.05
    embed_noise_std: float = 0.05

    def is_active(self) -> bool:
        return any(
            val > 0.0
            for val in (self.tv_lambda, self.robust_lambda, self.stable_lambda)
        )


def compute_shared_head_penalties(
    ae: "GenotypeAutoencoder",
    shared_forward: SharedForwardFn,
    z_input: torch.Tensor,
    z_dec: torch.Tensor,
    x_batch: torch.Tensor,
    penalty_cfg: Optional[Stage2PenaltyConfig],
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Mirror the stage-2 penalties from train.py but applied to the shared latent heads.

    Args:
        ae: Frozen GenotypeAutoencoder used for decoding.
        shared_forward: Callable(latent) -> (z_hom, z_dec) using the shared heads.
        z_input: Original latent tensor emitted by the chromosome AE before heads.
        z_dec: Output latent fed into ae.decode from the current forward pass.
        x_batch: Ground-truth genotypes (B, L) as integer tensor.
        penalty_cfg: Stage2PenaltyConfig or None.

    Returns:
        (penalty_loss, metrics) where penalty_loss is a scalar tensor added to the
        optimization objective and metrics contains detached penalty values.
    """
    zero = torch.zeros(1, device=z_dec.device, dtype=z_dec.dtype).squeeze(0)
    metrics = {
        "tv": zero.detach().clone(),
        "robust": zero.detach().clone(),
        "stable": zero.detach().clone(),
        "stable_norm": zero.detach().clone(),
    }

    if penalty_cfg is None or not penalty_cfg.is_active():
        return zero, metrics

    penalty_loss = zero.clone()

    if penalty_cfg.tv_lambda > 0.0:
        tv_val = latent_tv_loss(z_dec)
        penalty_loss = penalty_loss + penalty_cfg.tv_lambda * tv_val
        metrics["tv"] = tv_val.detach()

    if penalty_cfg.robust_lambda > 0.0 and penalty_cfg.latent_noise_std > 0.0:
        noise_lat = penalty_cfg.latent_noise_std * torch.randn_like(z_dec)
        z_pert = z_dec + noise_lat
        x_hat = decode_expected_dosage(ae, z_pert)
        robust_val = F.mse_loss(x_hat, x_batch.float())
        penalty_loss = penalty_loss + penalty_cfg.robust_lambda * robust_val
        metrics["robust"] = robust_val.detach()

    if penalty_cfg.stable_lambda > 0.0 and penalty_cfg.embed_noise_std > 0.0:
        noise_input = penalty_cfg.embed_noise_std * torch.randn_like(z_input)
        _, z_dec_noisy = shared_forward(z_input + noise_input)
        stable_val = F.mse_loss(z_dec, z_dec_noisy)
        penalty_loss = penalty_loss + penalty_cfg.stable_lambda * stable_val
        metrics["stable"] = stable_val.detach()
        metrics["stable_norm"] = stable_val.detach() / (
            penalty_cfg.embed_noise_std**2 + 1e-8
        )

    return penalty_loss, metrics

class FiLM2D(nn.Module):
    """
    Shared FiLM-based 1x1 Conv + GroupNorm + modulation for 2D latents.
    """
    def __init__(self, in_channels, film_dim=16, num_groups=8):
        super().__init__()
        # Initialize 1×1 conv as identity
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        with torch.no_grad():
            eye = torch.eye(in_channels).view(in_channels, in_channels, 1, 1)
            self.conv.weight.copy_(eye)
        # GroupNorm
        self.norm = nn.GroupNorm(num_groups=min(num_groups, in_channels), num_channels=in_channels)
        # FiLM embedding for chromosome ID
        self.embed = nn.Embedding(num_embeddings=22, embedding_dim=film_dim)
        self.gamma = nn.Linear(film_dim, in_channels)
        self.beta  = nn.Linear(film_dim, in_channels)
        # Initialize FiLM to identity
        nn.init.zeros_(self.gamma.weight); nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight);  nn.init.zeros_(self.beta.bias)

    def forward(self, x: torch.Tensor, chrom_id: Union[int, torch.Tensor]):
        # x: (B, C, H, W)
        x = self.conv(x)
        x = self.norm(x)
        if isinstance(chrom_id, int):
            chrom_id = torch.full((x.size(0),), chrom_id, dtype=torch.long, device=x.device)
        elif chrom_id.dim() == 0:
            chrom_id = chrom_id.expand(x.size(0))
        chrom_id = chrom_id.to(device=x.device, dtype=torch.long)

        embedding = self.embed(chrom_id)  # (B, film_dim)
        gamma = self.gamma(embedding).view(-1, x.size(1), 1, 1)
        beta  = self.beta(embedding).view(-1, x.size(1), 1, 1)
        return x * (1 + gamma) + beta


class HomogenizedAE(nn.Module):
    def __init__(self, ae_list):
        super().__init__()
        self.aes = nn.ModuleList(ae_list)
        # Assume all AEs have the same latent channel count
        latent_channels = ae_list[0].latent_channels
        self.encode_head = FiLM2D(latent_channels)
        self.decode_head = FiLM2D(latent_channels)

        # Freeze all AE parameters
        for ae in self.aes:
            for p in ae.parameters():
                p.requires_grad = False

    def forward(self, x, chrom_id):
        # x: (B, L) integer genotypes, chrom_id: scalar index for chromosome
        ae = self.aes[chrom_id]
        logits, z = ae(x)
        b = z.size(0)
        chrom_vec = torch.full((b,), int(chrom_id), dtype=torch.long, device=z.device)
        z_hom  = self.encode_head(z, chrom_vec)
        z_dec  = self.decode_head(z_hom, chrom_vec)
        logits_out = ae.decode(z_dec)
        return logits_out, z_hom


def train_shared_heads(models, dataloaders, epochs, device):
    model = HomogenizedAE(models).to(device)
    optimizer = torch.optim.AdamW(
        list(model.encode_head.parameters()) + list(model.decode_head.parameters()),
        lr=2e-3, weight_decay=0.01
    )
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for chrom_id, loader in enumerate(dataloaders):
            for batch in loader:
                batch = batch.to(device)
                logits, z_hom = model(batch, chrom_id)
                loss = F.cross_entropy(logits, batch.long())
                # optional latent MSE loss: encourage minimal distortion in latent space
                loss_lat = F.mse_loss(z_hom, model.aes[chrom_id](batch)[1])
                total = loss + 0.1 * loss_lat
                total.backward()
                optimizer.step()
                optimizer.zero_grad()
                total_loss += total.item()
        print(f"Epoch {epoch+1}: loss={total_loss:.4f}")
    return model