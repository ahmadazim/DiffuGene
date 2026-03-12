#!/usr/bin/env python

import os
import sys
import argparse
import math
import re
from typing import Tuple, List, Dict, Any, Sequence, Optional

import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from diffusers import UNet2DModel, DDPMScheduler, DDIMScheduler
from timm.utils import ModelEmaV3
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm

# -------------------------------------------------------------------------
# Project imports
# -------------------------------------------------------------------------
sys.path.append("/n/home03/ahmadazim/WORKING/genGen/DiffuGene/src")
from DiffuGene.VAEembed.ae import GenotypeAutoencoder, VAEConfig  # type: ignore

# -------------------------------------------------------------------------
# Basic device setup
# -------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[SETUP] Using device: {device}")


# -------------------------------------------------------------------------
# I/O + DATA HELPERS
# -------------------------------------------------------------------------
class MemmapLatentDataset(Dataset):
    """
    Simple dataset that reads latents from a memmap on the fly.
    Shape: (N, C, H, W) with dtype float32.
    """

    def __init__(self, memmap_path: str, shape: Tuple[int, int, int, int]) -> None:
        self.memmap_path = memmap_path
        self.shape = shape
        self.mm = np.memmap(memmap_path, dtype="float32", mode="r", shape=shape)

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        x = self.mm[idx]  # (C, H, W)
        return torch.from_numpy(np.array(x, copy=False))


def create_memmap(path: str, shape: Tuple[int, int, int, int]) -> np.memmap:
    mm = np.memmap(path, dtype="float32", mode="w+", shape=shape)
    return mm


def save_shape_file(memmap_path: str, shape: Tuple[int, ...]) -> None:
    shape_file = memmap_path.replace(".npy", "_shape.txt")
    with open(shape_file, "w") as f:
        f.write(",".join(map(str, shape)))


def _enumerate_h5_batch_paths(start_path: str, max_batches: int) -> List[str]:
    """
    Given a starting H5 path with 'batchXXXXX.h5' pattern,
    enumerate up to max_batches consecutive batch files.
    """
    d = os.path.dirname(start_path)
    m = re.search(r"batch(\d{5})\.h5$", os.path.basename(start_path))
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


def encode_h5s_to_memmap(
    ae: GenotypeAutoencoder,
    h5_paths: List[str],
    memmap_out: str,
    device: torch.device,
    encode_batch_size: int = 128,
) -> Tuple[str, Tuple[int, int, int, int]]:
    """
    Encode raw genotype H5 batches into latent-space memmap using the given AE.
    Returns (memmap_path, shape).
    """
    # Pass 1: count total N
    total_N = 0
    for p in h5_paths:
        with h5py.File(p, "r") as f:
            X = f["X"]
            total_N += int(X.shape[0])

    C, M = int(ae.latent_channels), int(ae.M2D)
    shape = (total_N, C, M, M)
    mm = create_memmap(memmap_out, shape)

    # Pass 2: encode
    offset = 0
    for p in h5_paths:
        print(f"[ENCODE] {p} -> memmap rows [{offset}, ...)")
        with h5py.File(p, "r") as f:
            X = f["X"]
            N = int(X.shape[0])
            for s in range(0, N, encode_batch_size):
                e = min(N, s + encode_batch_size)
                xb = torch.from_numpy(X[s:e].astype("int64")).to(
                    device, non_blocking=(device.type == "cuda")
                )
                with torch.no_grad():
                    logits, z = ae(xb)
                    z_cpu = z.detach().to("cpu").float().numpy()
                mm[offset + s : offset + e] = z_cpu
                del xb, logits, z, z_cpu
        offset += N

    del mm
    save_shape_file(memmap_out, shape)
    print(f"[ENCODE] Saved memmap to {memmap_out} with shape {shape}")
    return memmap_out, shape


# -------------------------------------------------------------------------
# AUTOENCODER LOADING + BASIC CHECKS
# -------------------------------------------------------------------------
def load_autoencoder(ae_ckpt_path: str) -> GenotypeAutoencoder:
    """
    Load GenotypeAutoencoder from checkpoint saved with VAEConfig + model_state.
    """
    payload = torch.load(ae_ckpt_path, map_location="cpu")
    cfg_dict = payload.get("config")
    state = payload.get("model_state")
    if cfg_dict is None or state is None:
        raise RuntimeError(f"Checkpoint {ae_ckpt_path} missing config or model_state.")

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

    print(f"[AE] Loaded AE from {ae_ckpt_path}")
    if "best_meta" in payload:
        print(f"[AE] best_meta: {payload['best_meta']}")
    if "last_meta" in payload:
        print(f"[AE] last_meta: {payload['last_meta']}")
    return ae


def check_ae_reconstruction(
    ae: GenotypeAutoencoder,
    h5_batch_path: str,
    max_samples: int = 512,
) -> None:
    """
    Quick AE reconstruction check: hard-call accuracy vs original genotypes.
    """
    print(f"[AE] Reconstruction check using {h5_batch_path}, max_samples={max_samples}")
    with h5py.File(h5_batch_path, "r") as f:
        X = f["X"]
        X = X[:max_samples]
        xb = torch.from_numpy(X.astype("int64")).to(
            device, non_blocking=(device.type == "cuda")
        )
        with torch.no_grad():
            logits, z = ae(xb)
        recon = ae.decode(z)
        hard_calls = torch.argmax(logits, dim=1)
        acc = (hard_calls == xb).float().mean().item()
        print(f"[AE] Hard-call reconstruction accuracy over {X.shape[0]} samples: {acc:.6f}")


# -------------------------------------------------------------------------
# LATENT-SPACE DIAGNOSTICS (ORIGINAL LATENTS)
# -------------------------------------------------------------------------
def gaussian_kl_1d(mu0: np.ndarray, s0: np.ndarray, mu1: float, s1: float) -> np.ndarray:
    """
    KL(N(mu0, s0^2) || N(mu1, s1^2)) in 1D, elementwise.
    """
    v0, v1 = s0**2, s1**2
    return 0.5 * (np.log(v1 / v0) + (v0 + (mu0 - mu1) ** 2) / v1 - 1.0)


def latent_original_stats_and_plots(
    memmap_path: str,
    mm_shape: Tuple[int, int, int, int],
    max_samples: int = 2048,
    max_dims_for_kl: int = 16384,
) -> Dict[str, Any]:
    """
    Compute + plot diagnostics of the *original* latent space only (no diffusion).
    - KL to N(0,1) for random subset of dims
    - Histograms of the worst KL dims vs N(0,1)
    - Histograms for lowest / highest variance dims vs N(0,1)
    - Per-channel spatial variance profiles (non-plot summary)
    Returns a dict with basic stats for later use.
    """
    print(f"[LATENT-ORIG] Loading original latents from {memmap_path}")
    mm = np.memmap(memmap_path, dtype="float32", mode="r", shape=mm_shape)
    N = min(max_samples, mm_shape[0])
    orig_lat = np.asarray(mm[:N])  # (N, C, H, W)
    _, C, H, W = orig_lat.shape
    orig_flat = orig_lat.reshape(N, -1)
    D = orig_flat.shape[1]
    print(f"[LATENT-ORIG] Using N={N}, C={C}, H={H}, W={W}, d={D}")

    rng = np.random.default_rng(123)
    n_dims = min(D, max_dims_for_kl)
    dims = rng.choice(D, size=n_dims, replace=False)

    mu_o = orig_flat[:, dims].mean(axis=0)
    sd_o = orig_flat[:, dims].std(axis=0) + 1e-12
    kl_to_std = gaussian_kl_1d(mu_o, sd_o, 0.0, 1.0)

    std_df = pd.DataFrame(
        {
            "dim": dims,
            "mu_o": mu_o,
            "sd_o": sd_o,
            "kl_to_std": kl_to_std,
        }
    ).sort_values(by="kl_to_std", ascending=False)

    print("\n[LATENT-ORIG] Top-5 dims by KL(latent_dim || N(0,1)):")
    for i in range(min(5, len(std_df))):
        row = std_df.iloc[i]
        print(
            f"  dim {int(row['dim']):5d} | mu={row['mu_o']:.3f}, "
            f"sd={row['sd_o']:.3f}, KL_to_std={row['kl_to_std']:.4f}"
        )

    print("\n[LATENT-ORIG] Lowest-5 dims by KL(latent_dim || N(0,1)):")
    for i in range(1, min(6, len(std_df))):
        row = std_df.iloc[-i]
        print(
            f"  dim {int(row['dim']):5d} | mu={row['mu_o']:.3f}, "
            f"sd={row['sd_o']:.3f}, KL_to_std={row['kl_to_std']:.4f}"
        )

    # ---- Figure 1: Worst KL dims vs N(0,1) ----
    fig, axes = plt.subplots(4, 5, figsize=(20, 10))
    axes = axes.ravel()
    n_to_plot = min(20, len(std_df))
    for i in range(n_to_plot):
        dim = int(std_df.iloc[i]["dim"])
        ax = axes[i]
        data = orig_flat[:, dim]
        ax.hist(data, bins=50, alpha=0.6, density=True, label="latent dim")

        x_min, x_max = data.min(), data.max()
        xs = np.linspace(x_min, x_max, 200)
        ax.plot(xs, norm.pdf(xs, loc=0.0, scale=1.0), label="N(0,1)")
        ax.set_title(f"Dim {dim}, KL_to_std={std_df.iloc[i]['kl_to_std']:.4f}")
        ax.legend()
    fig.suptitle("Original latents: dims farthest from N(0,1)", y=1.02)
    plt.tight_layout()
    plt.show()

    # ---- Variance ranking ----
    var_all = orig_flat.var(axis=0)
    var_sorted_idx = np.argsort(var_all)
    k = 10
    low_var_dims = var_sorted_idx[:k]
    high_var_dims = var_sorted_idx[-k:]

    print("\n[LATENT-ORIG] Lowest-variance dims:")
    for d in low_var_dims:
        print(f"  dim {int(d):5d} | var={var_all[d]:.5f}")

    print("\n[LATENT-ORIG] Highest-variance dims:")
    for d in high_var_dims[::-1]:
        print(f"  dim {int(d):5d} | var={var_all[d]:.5f}")

    dims_var_fig = list(low_var_dims) + list(high_var_dims)
    fig, axes = plt.subplots(4, 5, figsize=(20, 10))
    axes = axes.ravel()
    n_to_plot = min(20, len(dims_var_fig))
    for i in range(n_to_plot):
        dim = int(dims_var_fig[i])
        ax = axes[i]
        data = orig_flat[:, dim]
        ax.hist(data, bins=50, alpha=0.6, density=True, label="latent dim")
        x_min, x_max = data.min(), data.max()
        xs = np.linspace(x_min, x_max, 200)
        ax.plot(xs, norm.pdf(xs, loc=0.0, scale=1.0), label="N(0,1)")
        rank_type = "low-var" if i < k else "high-var"
        ax.set_title(f"{rank_type} dim {dim}\nvar={var_all[dim]:.5f}")
        ax.legend()
    fig.suptitle("Original latents: lowest/highest variance dims vs N(0,1)", y=1.02)
    plt.tight_layout()
    plt.show()

    # ---- Per-channel spatial variance map (printed summary) ----
    sigmas_hat = orig_lat.var(axis=0)  # (C, H, W)
    global_std = float(np.sqrt(orig_flat.var(axis=0).mean()))
    print(f"[LATENT-ORIG] Global sqrt(mean variance) over all voxels: {global_std:.6f}")

    return {
        "N": N,
        "C": C,
        "H": H,
        "W": W,
        "orig_lat": orig_lat,
        "orig_flat": orig_flat,
        "global_std": global_std,
        "sigmas_hat": sigmas_hat,
    }


# -------------------------------------------------------------------------
# DIFFUSION MODEL + SAMPLING
# -------------------------------------------------------------------------
class LatentUNET2D(nn.Module):
    """
    Unconditional UNet for HxW latents with C channels.
    This wraps diffusers.UNet2DModel with a conv in/out.
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        base_channels: int = 256,
        sample_size: int = 16,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Conv2d(input_channels, base_channels, kernel_size=3, padding=1)
        self.unet = UNet2DModel(
            sample_size=sample_size,
            in_channels=base_channels,
            out_channels=base_channels,
            layers_per_block=2,
            block_out_channels=[base_channels, 2 * base_channels, 3 * base_channels],
            down_block_types=["DownBlock2D", "DownBlock2D", "AttnDownBlock2D"],
            up_block_types=["AttnUpBlock2D", "UpBlock2D", "UpBlock2D"],
        )
        self.output_proj = nn.Conv2d(base_channels, output_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        y = self.unet(h, t).sample
        return self.output_proj(y)


def build_ddpm_scheduler(device: torch.device) -> DDPMScheduler:
    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_schedule="linear",
        beta_start=1e-4,
        beta_end=0.02,
        clip_sample=True,
        clip_sample_range=10.0,
    )
    scheduler.config.prediction_type = "epsilon"
    scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(device)
    scheduler.betas = scheduler.betas.to(device)
    return scheduler


def build_ddim_scheduler(
    num_inference_steps: int,
    device: torch.device,
) -> DDIMScheduler:
    sched = DDIMScheduler(
        num_train_timesteps=1000,
        beta_schedule="linear",
        beta_start=1e-4,
        beta_end=0.02,
        clip_sample=True,
        clip_sample_range=10.0,
    )
    sched.config.prediction_type = "epsilon"
    sched.set_timesteps(num_inference_steps=num_inference_steps, device=device)
    return sched


@torch.no_grad()
def sample_uncond_ddim(
    model: nn.Module,
    C: int,
    H: int,
    W: int,
    num_samples: int,
    num_steps: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Unconditional DDIM sampling from N(0, I) in latent space.
    """
    sched = build_ddim_scheduler(num_inference_steps=num_steps, device=device)
    x = torch.randn(num_samples, C, H, W, device=device).to(memory_format=torch.channels_last)
    model.eval()
    for t in sched.timesteps:
        t_vec = t.expand(num_samples).to(device)
        eps = model(x, t_vec)
        step = sched.step(model_output=eps, timestep=t, sample=x)
        x = step.prev_sample
    return x


# -------------------------------------------------------------------------
# DIFFUSION DIAGNOSTICS
# -------------------------------------------------------------------------
def _hist_probs(x: np.ndarray, bins: int, lo: float, hi: float) -> np.ndarray:
    hist, _ = np.histogram(x, bins=bins, range=(lo, hi))
    p = hist.astype(np.float64)
    p += 1e-12
    p /= p.sum()
    return p


def _js_div(p: np.ndarray, q: np.ndarray) -> float:
    m = 0.5 * (p + q)
    p = np.maximum(p, 1e-12)
    q = np.maximum(q, 1e-12)
    m = np.maximum(m, 1e-12)
    return 0.5 * (
        np.sum(p * (np.log(p) - np.log(m))) + np.sum(q * (np.log(q) - np.log(m)))
    )


def js_marginal_plots(
    orig_flat: np.ndarray,
    gen_flat: np.ndarray,
    max_dims_for_js: int = 4096,
    num_to_plot: int = 20,
) -> None:
    """
    JS divergence between orig vs gen for random subset of dims,
    with histograms of worst/best dims.
    """
    N, D = orig_flat.shape
    rng = np.random.default_rng(123)
    n_dims = min(D, max_dims_for_js)
    dims = rng.choice(D, size=n_dims, replace=False)

    js_vals = []
    for d in dims:
        xo = orig_flat[:, d]
        xg = gen_flat[:, d]
        lo = float(min(xo.min(), xg.min()))
        hi = float(max(xo.max(), xg.max()))
        if lo == hi:
            js_vals.append(0.0)
            continue
        p = _hist_probs(xo, bins=50, lo=lo, hi=hi)
        q = _hist_probs(xg, bins=50, lo=lo, hi=hi)
        js_vals.append(_js_div(p, q))
    js_vals = np.array(js_vals, dtype=np.float64)
    worst_js_idx = np.argsort(js_vals)[::-1]
    best_js_idx = np.argsort(js_vals)

    print(
        f"[JS] Mean JS over {n_dims} dims = {js_vals.mean():.6f}, "
        f"median = {np.median(js_vals):.6f}, max = {js_vals.max():.6f}"
    )

    k_plot = min(num_to_plot, len(dims))

    # Worst dims
    fig, axes = plt.subplots(4, 5, figsize=(20, 10))
    axes = axes.ravel()
    for i in range(k_plot):
        d = int(dims[worst_js_idx[i]])
        ax = axes[i]
        ax.hist(orig_flat[:, d], bins=50, alpha=0.5, label="orig", density=True)
        ax.hist(gen_flat[:, d], bins=50, alpha=0.5, label="gen", density=True)
        ax.set_title(f"dim {d} | JS={js_vals[worst_js_idx[i]]:.4f}")
        ax.legend()
    fig.suptitle("JS worst dims: orig vs gen", y=1.02)
    plt.tight_layout()
    plt.show()

    # Best dims
    fig, axes = plt.subplots(4, 5, figsize=(20, 10))
    axes = axes.ravel()
    for i in range(k_plot):
        d = int(dims[best_js_idx[i]])
        ax = axes[i]
        ax.hist(orig_flat[:, d], bins=50, alpha=0.5, label="orig", density=True)
        ax.hist(gen_flat[:, d], bins=50, alpha=0.5, label="gen", density=True)
        ax.set_title(f"dim {d} | JS={js_vals[best_js_idx[i]]:.4f}")
        ax.legend()
    fig.suptitle("JS best dims: orig vs gen", y=1.02)
    plt.tight_tight_layout = plt.tight_layout
    plt.show()


def variance_comparison_plots(
    orig_flat: np.ndarray,
    gen_flat_dict: Dict[int, np.ndarray],
    C: int,
    N_gen: int,
    orig_lat: np.ndarray,
    gen_lat_dict: Dict[int, np.ndarray],
) -> None:
    """
    Compare voxel-wise variances across orig vs generated latents
    for various DDIM lengths.
    - Histogram of voxel variances
    - Per-channel variance profiles across spatial positions
    - Two example channels
    """
    # Histogram of voxel variances
    plt.figure(figsize=(8, 5))
    plt.hist(orig_flat.var(axis=0), bins=100, alpha=0.5, label="Orig")
    for steps, flat in gen_flat_dict.items():
        plt.hist(
            flat.var(axis=0),
            bins=100,
            alpha=0.5,
            label=f"Gen ({steps} steps)",
        )
    plt.xlabel("Latent voxel variance")
    plt.ylabel("Frequency")
    plt.title("Histogram of latent voxel variances")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Per-channel variance vs spatial index, 2x2 grid
    fig, ax = plt.subplots(2, 2, figsize=(10, 8))
    steps_sorted = sorted(gen_lat_dict.keys())
    for c in range(C):
        ax[0, 0].plot(orig_lat[:, c].reshape(N_gen, -1).var(axis=0))
        ax[0, 1].plot(gen_lat_dict[steps_sorted[0]][:, c].reshape(N_gen, -1).var(axis=0))
        ax[1, 0].plot(gen_lat_dict[steps_sorted[1]][:, c].reshape(N_gen, -1).var(axis=0))
        ax[1, 1].plot(gen_lat_dict[steps_sorted[2]][:, c].reshape(N_gen, -1).var(axis=0))
    ax[0, 0].set_title("Orig")
    ax[0, 1].set_title(f"Gen {steps_sorted[0]} steps")
    ax[1, 0].set_title(f"Gen {steps_sorted[1]} steps")
    ax[1, 1].set_title(f"Gen {steps_sorted[2]} steps")
    for axi in ax.ravel():
        axi.set_xlabel("Voxel index (flattened HxW)")
        axi.set_ylabel("Variance")
    plt.tight_layout()
    plt.show()

    # Two example channels (e.g., 1 and 40 if available)
    example_channels = [1, 40] if C > 40 else [0, min(C - 1, 1)]
    fig, axes = plt.subplots(len(example_channels), 1, figsize=(10, 4 * len(example_channels)))
    if len(example_channels) == 1:
        axes = [axes]  # make iterable
    for ax, c in zip(axes, example_channels):
        ax.plot(orig_lat[:, c].reshape(N_gen, -1).var(axis=0), label="Orig")
        for steps in steps_sorted:
            ax.plot(
                gen_lat_dict[steps][:, c].reshape(N_gen, -1).var(axis=0),
                label=f"{steps} steps",
            )
        ax.set_title(f"Channel {c}")
        ax.set_xlabel("Voxel index (flattened HxW)")
        ax.set_ylabel("Variance")
        ax.legend()
    plt.tight_layout()
    plt.show()

    # Global sqrt(mean variance)
    print("[VAR-GLOBAL] sqrt(mean variance) per setting:")
    print(f"  Orig: {np.sqrt(orig_flat.var(axis=0).mean()):.6f}")
    for steps, flat in gen_flat_dict.items():
        print(f"  Gen ({steps} steps): {np.sqrt(flat.var(axis=0).mean()):.6f}")


def one_step_mse_diagnostics(
    model: nn.Module,
    scheduler: DDPMScheduler,
    train_loader: DataLoader,
    C: int,
    H: int,
    W: int,
    max_batches: int = 4,
    max_samples: int = 2048,
    t_step: int = 20,
) -> Dict[str, Any]:
    """
    One-step diagnostics:
    - x0 MSE vs t
    - noise (eps) MSE vs t
    """
    # Collect a batch of real latents
    xs = []
    total = 0
    for b_idx, x_cpu in enumerate(train_loader):
        x = x_cpu.to(device=device, non_blocking=(device.type == "cuda")).float()
        xs.append(x)
        total += x.size(0)
        if b_idx + 1 >= max_batches or total >= max_samples:
            break
    x0 = torch.cat(xs, dim=0)
    if x0.size(0) > max_samples:
        x0 = x0[:max_samples]
    N = x0.size(0)
    print(f"[1STEP-MSE] Using N={N} for one-step MSE diagnostics.")

    T = int(scheduler.config.num_train_timesteps)
    t_values = list(range(0, T, t_step))
    if (T - 1) not in t_values:
        t_values.append(T - 1)

    mse_per_t = []
    noise_mse_per_t = []

    with torch.no_grad():
        for t_scalar in t_values:
            t_batch = torch.full((N,), t_scalar, device=device, dtype=torch.long)
            noise = torch.randn_like(x0)
            x_t = scheduler.add_noise(x0, noise, t_batch)
            eps_hat = model(x_t.to(memory_format=torch.channels_last), t_batch)

            alpha_bar = scheduler.alphas_cumprod[t_batch].view(N, 1, 1, 1).to(device)
            x0_hat = (x_t - torch.sqrt(1.0 - alpha_bar) * eps_hat) / torch.sqrt(
                alpha_bar + 1e-12
            )

            mse = F.mse_loss(x0_hat, x0, reduction="mean").item()
            noise_mse = F.mse_loss(noise, eps_hat, reduction="mean").item()
            mse_per_t.append(mse)
            noise_mse_per_t.append(noise_mse)

            print(
                f"[1STEP-MSE] t={t_scalar:4d} | x0 MSE={mse:.6f} | "
                f"eps MSE={noise_mse:.6f}"
            )

    mse_per_t = np.array(mse_per_t, dtype=np.float64)
    noise_mse_per_t = np.array(noise_mse_per_t, dtype=np.float64)
    print(f"[1STEP-MSE] Mean x0 MSE over t-grid = {mse_per_t.mean():.6f}")
    print(f"[1STEP-MSE] Mean eps MSE over t-grid = {noise_mse_per_t.mean():.6f}")

    # Plot both on one figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(t_values, mse_per_t, marker="o")
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("x0 MSE")
    axes[0].set_title("One-step x0 MSE vs t")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t_values, noise_mse_per_t, marker="o")
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("eps MSE")
    axes[1].set_title("One-step eps MSE vs t")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return {
        "x0": x0,
        "t_values": t_values,
        "mse_per_t": mse_per_t,
        "noise_mse_per_t": noise_mse_per_t,
    }


def inspect_schedule(scheduler: DDPMScheduler) -> None:
    with torch.no_grad():
        betas = scheduler.betas.cpu()
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        T = alpha_bar.shape[0]
        amp = torch.sqrt((1.0 - alpha_bar) / (alpha_bar + 1e-20))

        print(f"[SCHED] num_train_timesteps = {T}")
        print(f"[SCHED] alpha_bar[0]  = {alpha_bar[0].item():.6f}")
        print(f"[SCHED] alpha_bar[-1] = {alpha_bar[-1].item():.6e}")
        print(f"[SCHED] min alpha_bar = {alpha_bar.min().item():.6e}")
        print(f"[SCHED] max amplification factor = {amp.max().item():.3e}")
        print(f"[SCHED] amp at t=0   = {amp[0].item():.3e}")
        print(f"[SCHED] amp at t=T-1 = {amp[-1].item():.3e}")


def compare_mse_to_amp(
    mse_x0_per_t: np.ndarray,
    mse_eps_per_t: np.ndarray,
    scheduler: DDPMScheduler,
) -> None:
    betas = scheduler.betas.cpu()
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    amp = torch.sqrt((1.0 - alpha_bar) / (alpha_bar + 1e-20))
    theo_x0 = (amp**2) * torch.from_numpy(mse_eps_per_t)

    for t in [0, 50, 100, 200, 400, 600, 800, 999]:
        if t >= len(alpha_bar):
            break
        print(
            f"[SCHED-COMP] t={t:4d} | "
            f"emp_x0={mse_x0_per_t[t]:.4f} | "
            f"theo_x0≈{theo_x0[t].item():.4f} | "
            f"eps_MSE={mse_eps_per_t[t]:.4f} | "
            f"amp={amp[t].item():.2e}"
        )


@torch.no_grad()
def diagnose_full_chain(
    model: nn.Module,
    scheduler: DDPMScheduler,
    x0_batch: torch.Tensor,
    device: torch.device,
    num_inference_steps: int = 1000,
) -> float:
    """
    Forward to T-1 and then run reverse chain with given num_inference_steps
    (DDPM forward, DDIM-like reverse using DDPMScheduler's step).
    """
    model.eval()
    x0 = x0_batch.to(device)
    B = x0.size(0)

    timesteps = torch.tensor(
        [scheduler.config.num_train_timesteps - 1],
        device=device,
        dtype=torch.long,
    ).expand(B)
    noise = torch.randn_like(x0)
    x_T = scheduler.add_noise(x0, noise, timesteps)

    scheduler.set_timesteps(num_inference_steps=num_inference_steps, device=device)
    x = x_T
    for t in scheduler.timesteps:
        t_vec = t.expand(B).to(device)
        eps = model(x, t_vec)
        step = scheduler.step(model_output=eps, timestep=t, sample=x)
        x = step.prev_sample

    mse = torch.mean((x - x0) ** 2).item()
    print(
        f"[FULL-CHAIN] MSE after reverse chain with {num_inference_steps} steps "
        f"= {mse:.6f}"
    )
    return mse


@torch.no_grad()
def diagnose_truncated_T(
    model: nn.Module,
    scheduler: DDPMScheduler,
    x0_batch: torch.Tensor,
    device: torch.device,
    T_max_list: Sequence[int] = (200, 400, 600, 800, 999),
    steps_per_chain: int = 100,
) -> None:
    """
    For each T_max, diffuse to T_max, then run reverse chain restricted to [T_max, 0]
    using steps_per_chain inference steps.
    """
    model.eval()
    x0 = x0_batch.to(device)
    B = x0.size(0)

    for T_max in T_max_list:
        T_max = min(T_max, scheduler.config.num_train_timesteps - 1)
        t_forward = torch.tensor([T_max], device=device, dtype=torch.long).expand(B)
        noise = torch.randn_like(x0)
        x_T = scheduler.add_noise(x0, noise, t_forward)

        timesteps = torch.linspace(T_max, 0, steps_per_chain, device=device).long()
        x = x_T.clone()
        for t in timesteps:
            t_vec = t.expand(B).to(device)
            eps = model(x, t_vec)
            step = scheduler.step(model_output=eps, timestep=t, sample=x)
            x = step.prev_sample

        mse = torch.mean((x - x0) ** 2).item()
        print(f"[TRUNC-T] T_max={T_max:4d}, steps={steps_per_chain:4d} | MSE={mse:.6f}")


def data_driven_diffusion_diagnostics(
    x0: torch.Tensor,
    target_snrs: Sequence[float] = (1e-2, 1e-3),
    alphas: Sequence[float] = (0.1, 0.2),
) -> Dict[str, Any]:
    """
    Data-driven diagnostics for diffusion hyperparameters:
    - E||X||^2, feature-wise variances
    - median NN distance^2
    - suggested sigma_max for target final SNRs
    - heuristic K_min lower bounds
    """
    with torch.no_grad():
        x0_cpu = x0.detach().to("cpu", non_blocking=True)
        N, C, H, W = x0_cpu.shape
        d = C * H * W
        print(f"[DIAG] N={N}, C={C}, H={H}, W={W}, d={d}")

        x_flat = x0_cpu.view(N, -1)
        feature_mean = x_flat.mean(dim=0)
        feature_var = x_flat.var(dim=0, unbiased=False)
        norm_sq_per_sample = x_flat.pow(2).sum(dim=1)
        E_norm_sq = norm_sq_per_sample.mean().item()

        print(f"[DIAG] E[||X||^2] (empirical) = {E_norm_sq:.4f}")
        print(f"[DIAG] Mean per-feature variance = {feature_var.mean().item():.4f}")
        print(
            f"[DIAG] Min/Max per-feature variance = "
            f"{feature_var.min().item():.4f} / {feature_var.max().item():.4f}"
        )

        # NN distances
        max_for_nn = 2000
        n_nn = min(N, max_for_nn)
        if n_nn < 2:
            print("[DIAG] Not enough samples for NN diagnostics.")
            delta_data_sq = float("nan")
            delta_data = float("nan")
        else:
            x_nn = x_flat[:n_nn]
            print(f"[DIAG] Computing pairwise NN distances with n={n_nn} ...")
            dist = torch.cdist(x_nn, x_nn, p=2)
            dist_sq = dist.pow(2)
            dist_sq.fill_diagonal_(float("inf"))
            nn_dist_sq, _ = dist_sq.min(dim=1)
            delta_data_sq = nn_dist_sq.median().item()
            delta_data = math.sqrt(delta_data_sq)
            print(f"[DIAG] Median NN distance^2 = {delta_data_sq:.6f}")
            print(f"[DIAG] Median NN distance   = {delta_data:.6f}")

        # sigma_max from SNR
        sigma_max2_dict: Dict[float, float] = {}
        sigma_max_dict: Dict[float, float] = {}
        for snr_final in target_snrs:
            sigma2 = E_norm_sq / (d * snr_final)
            sigma = math.sqrt(sigma2)
            sigma_max2_dict[snr_final] = sigma2
            sigma_max_dict[snr_final] = sigma
            print(
                f"[DIAG] For target final SNR={snr_final:g}: "
                f"sigma_max^2 ≈ {sigma2:.4f}, sigma_max ≈ {sigma:.4f}"
            )

        # K_min
        if not math.isnan(delta_data_sq) and delta_data_sq > 0.0:
            for snr_final in target_snrs:
                sigma2 = sigma_max2_dict[snr_final]
                for alpha in alphas:
                    K_min = sigma2 * d / (alpha * delta_data_sq)
                    print(
                        f"[DIAG] K_min (SNR_final={snr_final:g}, alpha={alpha:.2f}) "
                        f"≈ {K_min:.1f}"
                    )
        else:
            print("[DIAG] Skipping K_min diagnostics (Δ_data^2 unavailable).")

        return {
            "N": N,
            "C": C,
            "H": H,
            "W": W,
            "d": d,
            "E_norm_sq": E_norm_sq,
            "feature_mean": feature_mean,
            "feature_var": feature_var,
            "delta_data_sq": delta_data_sq,
            "delta_data": delta_data,
            "sigma_max2_from_SNR": sigma_max2_dict,
            "sigma_max_from_SNR": sigma_max_dict,
        }


# -------------------------------------------------------------------------
# MAIN ONE-SHOT FUNCTION
# -------------------------------------------------------------------------
def run_latent_diffusion_diagnostics(
    chr_no: int,
    name: str,
    ae_ckpt_path: str,
    unet_ckpt_path: str,
    h5_batch_path: str,
    memmap_out_path: str,
    max_encode_batches: int = 1,
    encode_batch_size: int = 256,
    train_batch_size: int = 256,
    train_num_workers: int = 8,
    max_samples_latent_stats: int = 2048,
    num_gen_samples: int = 2048,
) -> None:
    """
    High-level entry point:
    For a given (AE, UNet, latent space), run:
      - AE reconstruction check
      - encode H5 -> latent memmap
      - original latent-space diagnostics (vs N(0,1), var ranks)
      - UNet/DDPM/DDIM diagnostics (JS, MSE vs t, full-chain, truncated-T)
      - variance comparison across DDIM lengths
      - data-driven diffusion diagnostics for σ_max, K_min, etc.
    Produces several figures (with subplots) and prints diagnostics to stdout.
    """
    print("=" * 80)
    print(
        f"[RUN] Latent + diffusion diagnostics | chr={chr_no}, name={name}, "
        f"AE={os.path.basename(ae_ckpt_path)}, UNet={os.path.basename(unet_ckpt_path)}"
    )
    print("=" * 80)

    # 1) Load AE and run reconstruction check
    ae = load_autoencoder(ae_ckpt_path)
    check_ae_reconstruction(ae, h5_batch_path, max_samples=512)

    # 2) Encode H5 -> memmap (or reuse if already exists)
    if not os.path.exists(memmap_out_path):
        h5_paths = _enumerate_h5_batch_paths(h5_batch_path, int(max_encode_batches))
        print(f"[RUN] Encoding {len(h5_paths)} H5 batches to memmap")
        memmap_path, mm_shape = encode_h5s_to_memmap(
            ae=ae,
            h5_paths=h5_paths,
            memmap_out=memmap_out_path,
            device=device,
            encode_batch_size=encode_batch_size,
        )
    else:
        # Attempt to read shape file
        shape_file = memmap_out_path.replace(".npy", "_shape.txt")
        if not os.path.exists(shape_file):
            raise FileNotFoundError(f"Shape file not found for {memmap_out_path}")
        with open(shape_file, "r") as f:
            shape = tuple(map(int, f.read().strip().split(",")))
        memmap_path, mm_shape = memmap_out_path, shape  # type: ignore
        print(f"[RUN] Reusing existing memmap {memmap_path} with shape {mm_shape}")

    # Build loader over latents
    ds = MemmapLatentDataset(memmap_path, mm_shape)
    train_loader = DataLoader(
        ds,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=train_num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # 3) Original latent diagnostics
    orig_stats = latent_original_stats_and_plots(
        memmap_path=memmap_path,
        mm_shape=mm_shape,
        max_samples=max_samples_latent_stats,
    )
    orig_lat = orig_stats["orig_lat"]
    orig_flat = orig_stats["orig_flat"]
    N_orig, C, H, W = orig_lat.shape

    # 4) Build UNet + EMA + scheduler, load weights
    model = LatentUNET2D(
        input_channels=C,
        output_channels=C,
        base_channels=256,
        sample_size=H,
    ).to(device)
    model = model.to(memory_format=torch.channels_last)

    ema_src = model
    for p in ema_src.parameters():
        p.data = p.data.float()
    ema = ModelEmaV3(ema_src, decay=0.99, device="cpu")
    for p in ema.module.parameters():
        p.data = p.data.float()

    print(f"[RUN] Loading UNet/EMA weights from {unet_ckpt_path}")
    ckpt = torch.load(unet_ckpt_path, map_location="cpu")
    w = ckpt.get("weights")
    e = ckpt.get("ema")
    if w is not None:
        model.load_state_dict(w, strict=True)
    if e is not None:
        ema.load_state_dict(e, strict=True)
    ema.eval().to(device)

    scheduler = build_ddpm_scheduler(device)
    inspect_schedule(scheduler)

    # 5) Generate samples with various DDIM lengths
    N_gen = min(num_gen_samples, N_orig)
    print(f"[RUN] Generating N_gen={N_gen} samples at multiple DDIM lengths")
    gen_steps_list = [50, 100, 200]
    gen_lat_dict: Dict[int, np.ndarray] = {}
    gen_flat_dict: Dict[int, np.ndarray] = {}
    for steps in gen_steps_list:
        gen = sample_uncond_ddim(
            ema, C=C, H=H, W=W, num_samples=N_gen, num_steps=steps, device=device
        )
        gen_np = gen.detach().to("cpu").float().numpy()
        gen_lat_dict[steps] = gen_np
        gen_flat_dict[steps] = gen_np.reshape(N_gen, -1)
        print(f"[RUN] Generated samples for {steps} steps, shape={gen_np.shape}")

    # Subsample orig for JS/variance to match N_gen
    orig_lat_sub = orig_lat[:N_gen]
    orig_flat_sub = orig_flat[:N_gen]

    # 6) JS marginal diagnostics (use e.g. 100-step samples as "canonical")
    js_marginal_plots(orig_flat_sub, gen_flat_dict[50])

    # 7) Variance comparison plots across steps
    variance_comparison_plots(
        orig_flat=orig_flat_sub,
        gen_flat_dict=gen_flat_dict,
        C=C,
        N_gen=N_gen,
        orig_lat=orig_lat_sub,
        gen_lat_dict=gen_lat_dict,
    )

    # 8) One-step MSE vs t for the EMA model
    mse_diag = one_step_mse_diagnostics(
        model=ema,
        scheduler=scheduler,
        train_loader=train_loader,
        C=C,
        H=H,
        W=W,
        max_batches=4,
        max_samples=2048,
        t_step=20,
    )
    # compare_mse_to_amp(
    #     mse_x0_per_t=mse_diag["mse_per_t"],
    #     mse_eps_per_t=mse_diag["noise_mse_per_t"],
    #     scheduler=scheduler,
    # )

    # 9) Full-chain and truncated-T diagnostics on a subset of x0
    x0_for_chain = mse_diag["x0"][:256]
    for num_steps in [50, 100, 200]:
        diagnose_full_chain(
            model=ema,
            scheduler=scheduler,
            x0_batch=x0_for_chain,
            device=device,
            num_inference_steps=num_steps,
        )
    diagnose_truncated_T(
        model=ema,
        scheduler=scheduler,
        x0_batch=x0_for_chain,
        device=device,
        T_max_list=(200, 400, 600, 800, 999),
        steps_per_chain=100,
    )

    # 10) Data-driven diffusion diagnostics (σ_max, K_min, etc.)
    data_driven_diffusion_diagnostics(x0_for_chain)

    print("=" * 80)
    print("[RUN] Diagnostics complete.")
    print("=" * 80)


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Latent + diffusion diagnostics for a given AE + UNet pair."
    )
    parser.add_argument("--chr_no", type=int, required=True)
    parser.add_argument("--name", type=str, required=True, help="Label for model/latent set")
    parser.add_argument("--ae_ckpt_path", type=str, required=True)
    parser.add_argument("--unet_ckpt_path", type=str, required=True)
    parser.add_argument("--h5_batch_path", type=str, required=True)
    parser.add_argument("--memmap_out_path", type=str, required=True)
    parser.add_argument("--max_encode_batches", type=int, default=1)
    parser.add_argument("--encode_batch_size", type=int, default=256)
    parser.add_argument("--train_batch_size", type=int, default=256)
    parser.add_argument("--train_num_workers", type=int, default=8)
    parser.add_argument("--max_samples_latent_stats", type=int, default=2048)
    parser.add_argument("--num_gen_samples", type=int, default=2048)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_latent_diffusion_diagnostics(
        chr_no=args.chr_no,
        name=args.name,
        ae_ckpt_path=args.ae_ckpt_path,
        unet_ckpt_path=args.unet_ckpt_path,
        h5_batch_path=args.h5_batch_path,
        memmap_out_path=args.memmap_out_path,
        max_encode_batches=args.max_encode_batches,
        encode_batch_size=args.encode_batch_size,
        train_batch_size=args.train_batch_size,
        train_num_workers=args.train_num_workers,
        max_samples_latent_stats=args.max_samples_latent_stats,
        num_gen_samples=args.num_gen_samples,
    )