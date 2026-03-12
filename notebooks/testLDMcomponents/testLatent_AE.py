#!/usr/bin/env python

import os
import sys
import argparse
from typing import Tuple, List

import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F

# Make project src importable
sys.path.append('/n/home03/ahmadazim/WORKING/genGen/DiffuGene/src')
from DiffuGene.VAEembed.ae import GenotypeAutoencoder, VAEConfig  # type: ignore


# ------------------------
# AE loading
# ------------------------

def load_ae_from_checkpoint(ae_ckpt_path: str, device: torch.device) -> Tuple[GenotypeAutoencoder, VAEConfig]:
    """
    Load a trained AE checkpoint that includes a 'config' and a 'model_state'.
    """
    payload = torch.load(ae_ckpt_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"AE checkpoint must be a dict payload: {ae_ckpt_path}")
    cfg_dict = payload.get("config")
    state = payload.get("model_state")
    if cfg_dict is None or state is None:
        raise KeyError(f"AE checkpoint missing 'config' or 'model_state': {ae_ckpt_path}")
    cfg = VAEConfig(**cfg_dict)
    ae = GenotypeAutoencoder(
        input_length=cfg.input_length,
        K1=cfg.K1,
        K2=cfg.K2,
        C=cfg.C,
        embed_dim=cfg.embed_dim,
    )
    missing = ae.load_state_dict(state, strict=True)
    if getattr(missing, "missing_keys", None) or getattr(missing, "unexpected_keys", None):
        raise RuntimeError(f"AE state mismatch: {missing}")
    ae.to(device).eval()
    for p in ae.parameters():
        p.requires_grad = False
    return ae, cfg


# ------------------------
# Helpers: encode-only + losses
# ------------------------

@torch.no_grad()
def encode_from_embed(
    ae: GenotypeAutoencoder,
    h: torch.Tensor,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    """
    Encode from an already-embedded 1D representation h to the 2D latent z.

    Mirrors the encoder part of GenotypeAutoencoder.forward, but stops at z.
    """
    # 1D downsample
    h1d, _ = ae._downsample_1d(h, mask)  # type: ignore[attr-defined]
    # 1D -> 2D
    h2d = h1d.permute(0, 2, 1).unsqueeze(1)  # (B,1,M1D,2^K1)
    h2d = ae.proj2d(h2d)                     # type: ignore[attr-defined]
    # 2D down blocks
    for down_block in ae.down2d_blocks:      # type: ignore[attr-defined]
        h2d = down_block(h2d)
    z = h2d
    return z


@torch.no_grad()
def encode_only(ae: GenotypeAutoencoder, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pure encoder: x -> (z, h_embed, mask).

    - z: 2D latent (B, C_latent, K2, K2)
    - h_embed: embedded 1D representation before encoder (B, embed_dim, L_eff)
    - mask: optional length mask, or None
    """
    # Prepare input (one-hot + dosage -> embedding)
    h, mask = ae._prepare_input(x)  # type: ignore[attr-defined]
    z = encode_from_embed(ae, h, mask)
    return z, h, mask


def latent_tv_loss(z: torch.Tensor) -> torch.Tensor:
    """
    Total variation (TV) smoothness penalty on 2D latent.
    z: (B, C, H, W)
    Returns scalar mean over batch.
    """
    dh = (z[:, :, 1:, :] - z[:, :, :-1, :]) ** 2
    dw = (z[:, :, :, 1:] - z[:, :, :, :-1]) ** 2
    return (dh.mean() + dw.mean())


@torch.no_grad()
def decode_expected_dosage(ae: GenotypeAutoencoder, z: torch.Tensor) -> torch.Tensor:
    """
    Decode latent z with AE and return expected dosage E[X | logits].
    """
    logits = ae.decode(z)  # (B,3,L)
    probs = torch.softmax(logits, dim=1)  # (B,3,L)
    class_values = torch.tensor([0.0, 1.0, 2.0], device=z.device).view(1, 3, 1)
    x_hat = (probs * class_values).sum(dim=1)  # (B,L)
    return x_hat


# ------------------------
# H5 helpers
# ------------------------

def enumerate_h5_batch_paths(start_path: str, max_batches: int) -> List[str]:
    d = os.path.dirname(start_path)
    base = os.path.basename(start_path)
    m = None
    import re
    m = re.search(r"batch(\d{5})\.h5$", base)
    if not m:
        raise ValueError(f"Cannot parse batch index from {start_path}")
    start_idx = int(m.group(1))
    paths: List[str] = []
    for i in range(start_idx, start_idx + int(max_batches)):
        p = os.path.join(d, f"batch{i:05d}.h5")
        if os.path.exists(p):
            paths.append(p)
        else:
            break
    if not paths:
        raise FileNotFoundError(f"No H5 batch files found starting at {start_path}")
    return paths


# ------------------------
# Main analysis
# ------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze latent-space penalties for a trained GenotypeAutoencoder")
    parser.add_argument("--ae-model-path", type=str, required=True, help="Path to trained AE checkpoint (.pt)")
    parser.add_argument("--h5-batch-path", type=str, required=True,
                        help="Path to one H5 batch file, e.g., .../vqvae_h5_cache/chr22/batch00001.h5")
    parser.add_argument("--max-batches", type=int, default=1,
                        help="Max number of sequential H5 batches to include starting from the provided one")
    parser.add_argument("--encode-batch-size", type=int, default=128)
    parser.add_argument("--n-eval", type=int, default=10000,
                        help="Max number of samples to use for penalty estimates")
    parser.add_argument("--latent-noise-std", type=float, default=0.05,
                        help="Std dev of Gaussian noise in latent space for robust-subspace penalty")
    parser.add_argument("--embed-noise-std", type=float, default=0.05,
                        help="Std dev of Gaussian noise added to embedded input for stable-subspace penalty")
    parser.add_argument("--output-metrics", type=str, default=None,
                        help="Optional path to save metrics as a .npz file")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # 1) Load AE
    print(f"[INFO] Loading AE from {args.ae_model_path}")
    ae, cfg = load_ae_from_checkpoint(args.ae_model_path, device=device)
    print(f"[INFO] Loaded AE with input_length={cfg.input_length}, K1={cfg.K1}, K2={cfg.K2}, C={cfg.C}")

    # 2) Enumerate H5 batch files
    h5_paths = enumerate_h5_batch_paths(args.h5_batch_path, int(args.max_batches))
    print(f"[INFO] Found {len(h5_paths)} H5 batch file(s) starting from {args.h5_batch_path}")

    # 3) Iterate and accumulate penalties
    total_tv = 0.0
    total_robust = 0.0
    total_stable = 0.0
    total_stable_norm = 0.0  # normalized by embed_noise_std^2
    total_n = 0

    sigma_z = float(args.latent_noise_std)
    sigma_emb = float(args.embed_noise_std)

    for h5_path in h5_paths:
        print(f"[INFO] Reading {h5_path}")
        with h5py.File(h5_path, "r") as f:
            X = f["X"]
            N, L = int(X.shape[0]), int(X.shape[1])
            print(f"    H5 dataset shape: N={N}, L={L}")
            start = 0
            while start < N and total_n < args.n_eval:
                end = min(N, start + args.encode_batch_size, start + (args.n_eval - total_n))
                xb_np = X[start:end].astype("int64")  # (B,L)
                start = end

                xb = torch.from_numpy(xb_np).to(device)
                B = xb.size(0)

                # 3.1 Encode using full AE forward to get z for TV + robust penalty
                with torch.no_grad():
                    logits_full, z = ae(xb)  # z: (B, C_latent, K2, K2)

                # Patch smoothness (TV) on z
                tv_batch = latent_tv_loss(z)

                # Robust subspace: MSE X vs D(z + noise_z)
                noise_z = torch.randn_like(z) * sigma_z
                z_pert = z + noise_z
                x_hat_pert = decode_expected_dosage(ae, z_pert)  # (B,L)
                robust_batch = ((x_hat_pert - xb.float()) ** 2).mean()

                # Stable subspace: MSE(z_clean, z_noisy) where noise is added to embedding
                # Encode-only path from embedded input
                with torch.no_grad():
                    h_embed, mask = ae._prepare_input(xb)  # type: ignore[attr-defined]
                    z_clean = encode_from_embed(ae, h_embed, mask)
                    noise_emb = torch.randn_like(h_embed) * sigma_emb
                    h_noisy = h_embed + noise_emb
                    z_noisy = encode_from_embed(ae, h_noisy, mask)
                    stable_batch = ((z_noisy - z_clean) ** 2).mean()
                    stable_batch_norm = stable_batch / (sigma_emb ** 2 + 1e-12)

                # Accumulate weighted by batch size
                total_tv += float(tv_batch.item()) * B
                total_robust += float(robust_batch.item()) * B
                total_stable += float(stable_batch.item()) * B
                total_stable_norm += float(stable_batch_norm.item()) * B
                total_n += B

                print(
                    f"    Processed {total_n} samples "
                    f"(batch B={B}): TV={tv_batch.item():.6e}, "
                    f"robust={robust_batch.item():.6e}, "
                    f"stable={stable_batch.item():.6e}, "
                    f"stable_norm={stable_batch_norm.item():.6e}"
                )

                if total_n >= args.n_eval:
                    break

        if total_n >= args.n_eval:
            break

    if total_n == 0:
        print("[ERROR] No samples processed; check H5 path and n-eval.")
        return

    mean_tv = total_tv / total_n
    mean_robust = total_robust / total_n
    mean_stable = total_stable / total_n
    mean_stable_norm = total_stable_norm / total_n

    print("\n================ LATENT PENALTY SUMMARY ================")
    print(f"Total samples evaluated: {total_n}")
    print(f"Patch TV (latent smoothness)        : {mean_tv:.6e}")
    print(f"Robust subspace MSE (X, D(E+δz))    : {mean_robust:.6e}")
    print(f"Stable subspace MSE (z, z_noisy)    : {mean_stable:.6e}")
    print(f"Stable subspace (normalized)        : {mean_stable_norm:.6e}")
    print("========================================================\n")

    if args.output_metrics is not None:
        out_path = args.output_metrics
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.savez(
            out_path,
            n_samples=total_n,
            patch_tv=mean_tv,
            robust_mse=mean_robust,
            stable_mse=mean_stable,
            stable_mse_norm=mean_stable_norm,
            latent_noise_std=sigma_z,
            embed_noise_std=sigma_emb,
        )
        print(f"[INFO] Saved metrics to {out_path}")


if __name__ == "__main__":
    main()