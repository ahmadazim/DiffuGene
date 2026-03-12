import os
import argparse
import glob
from typing import List, Optional, Tuple, Dict, Any
import bisect
from dataclasses import asdict

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import h5py
import torch.nn.functional as F

import sys
sys.path.append('/n/home03/ahmadazim/WORKING/genGen/DiffuGene/src')
from DiffuGene.VAEembed.ae import GenotypeAutoencoder, VAEConfig


class GenotypeVAE(GenotypeAutoencoder):
    """
    Variational extension of GenotypeAutoencoder.

    - Reuses the entire AE backbone (1D/2D encoder + decoder).
    - Adds conv_mu / conv_logvar heads on top of the latent z (B,C,H,W).
    - Samples z via reparameterization and decodes with the existing `decode`.
    """

    def __init__(
        self,
        *args,
        beta_kl: float = 0.0,
        bottleneck_channels: int = 16,
        **kwargs,
    ) -> None:
        """
        bottleneck_channels:
            If None or equal to self.latent_channels (C), the VAE operates
            directly in the original latent space (no bottleneck).
            If < C, we apply a learned 1x1 conv bottleneck:
                h2d (B,C,H,W) -> compress -> (B,Bc,H,W)
                sample in bottleneck space
                -> expand -> (B,C,H,W) -> decode
        """
        super().__init__(*args, **kwargs)
        self.beta_kl = float(beta_kl)

        # Original AE latent channels (e.g. 64)
        C_full = int(self.latent_channels)
        Bc = int(bottleneck_channels)
        if Bc > C_full:
            raise ValueError(f"bottleneck_channels ({Bc}) cannot exceed latent_channels ({C_full})")
        self.bottleneck_channels = Bc

        if Bc == C_full:
            self.compress = nn.Identity()
            self.expand = nn.Identity()
        else:
            self.compress = nn.Conv2d(C_full, Bc, kernel_size=1, bias=False)
            self.expand = nn.Conv2d(Bc, C_full, kernel_size=1, bias=False)

        self.mu_head = nn.Conv2d(
            self.bottleneck_channels, self.bottleneck_channels, kernel_size=1, bias=True
        )
        self.logvar_head = nn.Conv2d(
            self.bottleneck_channels, self.bottleneck_channels, kernel_size=1, bias=True
        )

        # Initialize mu_head close to identity, logvar_head to small variance
        self._init_heads()

    def _init_heads(self) -> None:
        with torch.no_grad():
            # mu_head ≈ identity
            self.mu_head.weight.zero_()
            C = self.bottleneck_channels
            for c in range(C):
                self.mu_head.weight[c, c, 0, 0] = 1.0
            self.mu_head.bias.zero_()

            # logvar_head starts with small negative bias => var ~ exp(-4) ~ 0.018
            self.logvar_head.weight.zero_()
            self.logvar_head.bias.fill_(-4.0)

    def encode_latent(self, x: torch.Tensor):
        """
        Run the original encoder path, but instead of taking z=h2d directly,
        we run it through mu/logvar heads and reparameterize.
        Returns: (z, mu, logvar, mask)
        """
        # This mirrors your original forward up to `z = h2d`
        h, mask = self._prepare_input(x)              # (B, embed_dim, L), mask
        h1d, mask1d = self._downsample_1d(h, mask)    # (B, C1, M)
        B, C1, M = h1d.shape

        h2d = h1d.permute(0, 2, 1).unsqueeze(1)       # (B, 1, M, C1)
        h2d = self.proj2d(h2d)
        for down_block in self.down2d_blocks:
            h2d = down_block(h2d)

        h_bottleneck = self.compress(h2d)
        mu = self.mu_head(h_bottleneck)
        logvar = self.logvar_head(h_bottleneck)

        # Reparameterize
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        
        z_bottleneck = mu + std * eps

        return z_bottleneck, mu, logvar, mask

    def forward(self, x: torch.Tensor):
        """
        VAE forward: encode -> sample -> decode
        Returns: (logits, z, mu, logvar, mask)
        """
        z_bottleneck, mu, logvar, mask = self.encode_latent(x)
        z_full = self.expand(z_bottleneck)
        logits = self.decode(z_full)
        return logits, z_bottleneck, mu, logvar, mask

    def decode_full(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode a latent tensor to a logits tensor.
        """
        z_full = self.expand(z)
        logits = self.decode(z_full)
        return logits

    def kl_divergence(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        KL[q(z|x) || N(0,I)] averaged over batch.
        """
        # Standard VAE KL: -0.5 * sum(1 + logσ² - μ² - σ²)
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        # Sum over all latent dims, mean over batch
        kl = kl.sum(dim=(1, 2, 3)).mean()
        return kl

    def vae_loss_function(
        self,
        logits: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        mask: Optional[torch.Tensor],
        *,
        beta_kl: Optional[float] = None,
        maf_lambda: float = 0.0,
        ld_lambda: float = 0.0,
        ld_window: int = 128,
    ) -> Tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        VAE loss = AE recon/auxiliary loss + beta * KL.
        Reuses the existing AE.loss_function for recon/MAF/LD, then adds KL.
        """
        # Use the AE's existing loss_function to get recon + maf/ld penalties
        base_loss, metrics = super().loss_function(
            logits, x, None, maf_lambda=maf_lambda, ld_lambda=ld_lambda, ld_window=ld_window
        )

        kl = self.kl_divergence(mu, logvar)
        beta = float(beta_kl) if beta_kl is not None else self.beta_kl
        total_loss = base_loss + beta * kl

        metrics["kl"] = kl.detach()
        metrics["beta"] = torch.tensor(beta, device=logits.device)
        metrics["total"] = total_loss.detach()
        return total_loss, metrics

# def freeze_backbone_keep_vae_heads_trainable(model: GenotypeVAE) -> None:
#     for name, param in model.named_parameters():
#         # Only train the new VAE heads
#         if name.startswith("mu_head") or name.startswith("logvar_head"):
#             param.requires_grad = True
#         else:
#             param.requires_grad = False

def load_vae_from_ae_checkpoint(
    checkpoint_path: str,
    device: torch.device,
    beta_kl: float = 0.0,
    bottleneck_channels: int = 16,
) -> Tuple[GenotypeVAE, Dict[str, Any]]:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    cfg_dict = ckpt["config"]
    cfg = VAEConfig(**cfg_dict)

    # Build VAE with SAME structural config as AE
    vae = GenotypeVAE(
        input_length=cfg.input_length,
        K1=cfg.K1,
        K2=cfg.K2,
        C=cfg.C,
        embed_dim=cfg.embed_dim,
        beta_kl=beta_kl,
        bottleneck_channels=bottleneck_channels,
    )

    # Load AE weights into the shared backbone; ignore missing mu/logvar heads
    incompat = vae.load_state_dict(ckpt["model_state"], strict=False)
    if incompat.missing_keys:
        print("[VAE] Missing keys (expected: new mu/logvar heads/bottleneck heads):", incompat.missing_keys)
    if incompat.unexpected_keys:
        print("[VAE] Unexpected keys:", incompat.unexpected_keys)

    vae.to(device)
    
    # # freeze the backbone and only train the VAE heads
    # freeze_backbone_keep_vae_heads_trainable(vae)

    return vae, cfg_dict

def train_vae_variational(
    model: GenotypeVAE,
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
    autocast_device: Optional[str] = "cuda",
    beta_max: float = 1.0,
    beta_warmup_epochs: int = 5,
) -> Dict[str, Any]:
    """
    VAE training that reuses the AE's reconstruction pipeline and adds KL with
    a beta warmup schedule over `beta_warmup_epochs`.
    """
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if device is not None:
        model.to(device)
    model.train()

    @torch.no_grad()
    def _eval_validation_mse(model_eval: GenotypeVAE, loader: torch.utils.data.DataLoader) -> float:
        if loader is None:
            return float("inf")
        was_training = model_eval.training
        model_eval.eval()
        total_mse = 0.0
        total_n = 0
        for xb in loader:
            xb = xb.to(device) if device is not None else xb
            logits, z, mu, logvar, mask = model_eval(xb)
            probs = torch.softmax(logits, dim=1)  # (B,3,L)
            x_hat = probs[:, 1, :] + 2.0 * probs[:, 2, :]
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
        # beta schedule: linearly ramp 0 -> beta_max over beta_warmup_epochs
        if epoch <= beta_warmup_epochs:
            beta = beta_max * (epoch / float(beta_warmup_epochs))
        else:
            beta = beta_max
        model.beta_kl = beta  # store in the model (used as default in vae_loss_function)

        sum_loss = 0.0
        sum_recon = 0.0
        sum_maf = 0.0
        sum_ld = 0.0
        sum_kl = 0.0
        count = 0

        for xb in dataloader:
            xb = xb.to(device) if device is not None else xb
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=(autocast_device or "cuda"),
                dtype=torch.bfloat16 if (autocast_device == "cuda") else torch.float32,
                enabled=(autocast_device == "cuda" and (device is not None and device.type == "cuda")),
            ):
                logits, z, mu, logvar, mask = model(xb)
                loss, metrics = model.vae_loss_function(
                    logits,
                    xb,
                    mu,
                    logvar,
                    mask,
                    beta_kl=beta,
                    maf_lambda=maf_lambda,
                    ld_lambda=ld_lambda,
                    ld_window=ld_window,
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
            sum_kl += float(metrics["kl"].item()) * bs
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
        mean_kl = sum_kl / max(1, count)
        lr_value = optimizer.param_groups[0].get("lr", 0.0)
        print(
            f"[VAE] Epoch {epoch}/{num_epochs}: "
            f"loss={mean_loss:.4f} | recon={mean_recon:.4f} | "
            f"maf={mean_maf:.5f} | ld={mean_ld:.5f} | kl={mean_kl:.5f} | "
            f"beta={beta:.4f} | lr={float(lr_value):.6f}"
        )

        val_mse = _eval_validation_mse(model, val_dataloader) if val_dataloader is not None else float("inf")
        first = (epoch == 1)
        # improved = val_mse < best_val_mse * (1.0 - float(plateau_min_rel_improve))  # hashed out: no relative improvement check
        improved = val_mse < best_val_mse
        if first or improved:
            best_val_mse = val_mse
            best_epoch = epoch
            epochs_no_improve = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_meta = {"epoch": int(epoch), "val_mse": float(val_mse)}
        else:
            epochs_no_improve += 1

        # if best_val_mse <= float(plateau_mse_threshold) and epochs_no_improve >= int(plateau_patience):
        #     print(
        #         f"[VAE][EarlyStopping] no ≥{plateau_min_rel_improve*100:.2f}% relative improvement for "
        #         f"{plateau_patience} epochs (best val MSE={best_val_mse:.6f})."
        #     )
        #     break

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
