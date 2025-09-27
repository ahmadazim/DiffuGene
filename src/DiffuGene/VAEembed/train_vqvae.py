#!/usr/bin/env python

import os
import argparse
import glob
from typing import List
import bisect

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import h5py

from .vae import SNPVQVAE, build_vqvae, train_vqvae as train_vqvae_fn, find_best_ck
from ..utils.file_utils import read_bim_file
from ..utils import setup_logging, get_logger


logger = get_logger(__name__)


def log_model_summary(model: SNPVQVAE, x_sample: torch.Tensor, device: torch.device) -> None:
    """Minimal forward shape logging compatible with the new VAE API."""
    model.eval()
    x_sample = x_sample.to(device)
    try:
        with torch.no_grad():
            z_e_seq, mask_tokens = model.encode_tokens(x_sample)
            logger.info(f"[ModelSummary] z_e_seq: {tuple(z_e_seq.shape)} | mask_tokens={None if mask_tokens is None else tuple(mask_tokens.shape)}")
            z_q_seq, commit_loss, _, _ = model.quantizer(z_e_seq, beta_commit=model.cfg.beta_commit)
            logger.info(f"[ModelSummary] z_q_seq: {tuple(z_q_seq.shape)} | commit={commit_loss.item():.6f}")
            logits3 = model.decode_logits(z_q_seq)
            logger.info(f"[ModelSummary] logits3: {tuple(logits3.shape)}")
    except Exception as e:
        logger.warning(f"[ModelSummary] Failed to log shapes due to: {e}")
    model.train()


class H5ChromosomeDataset(Dataset):
    """Dataset for a single chromosome from H5 caches.
    Directory structure: <h5_dir>/chr<no>/batchXXXXX.h5 with datasets X (B, L_chr), iid (B), bp (L_chr).
    """
    def __init__(self, h5_dir: str, chromosome: int):
        self.h5_dir = h5_dir
        self.chromosome = int(chromosome)
        # Discover batch ids within the chromosome directory
        self.chr_dir = os.path.join(h5_dir, f"chr{self.chromosome}")
        batch_files = sorted(glob.glob(os.path.join(self.chr_dir, "batch*.h5")))
        if not batch_files:
            raise FileNotFoundError(f"No H5 caches found in {self.chr_dir}")
        self.batches = [os.path.splitext(os.path.basename(p))[0] for p in batch_files]  # batchXXXXX
        # Determine feature length L for this chromosome
        with h5py.File(os.path.join(self.chr_dir, f"{self.batches[0]}.h5"), 'r') as f:
            self.chrom_length = int(f['X'].shape[1])
        # Determine per-batch sample sizes
        self.batch_sizes = []
        for bname in self.batches:
            fpath = os.path.join(self.chr_dir, f"{bname}.h5")
            with h5py.File(fpath, 'r') as f:
                rows = int(f['X'].shape[0])
            if rows <= 0:
                raise ValueError(f"Empty batch detected for {bname}")
            self.batch_sizes.append(rows)
        # Build cumulative sizes for fast index mapping (variable batch sizes)
        self.cum_sizes = []
        total = 0
        for sz in self.batch_sizes:
            total += sz
            self.cum_sizes.append(total)

    @property
    def total_len(self) -> int:
        return self.chrom_length

    def __len__(self):
        return self.cum_sizes[-1]

    def _load_sample(self, batch_idx: int, sample_idx_in_batch: int) -> torch.Tensor:
        p = os.path.join(self.chr_dir, f"{self.batches[batch_idx]}.h5")
        with h5py.File(p, 'r') as f:
            x = f['X'][sample_idx_in_batch]  # (L_chr,)
            return torch.from_numpy(x.astype('int64'))

    def __getitem__(self, idx: int) -> torch.Tensor:
        # Map global index to (batch_idx, sample_idx) with variable batch sizes
        if idx < 0 or idx >= self.cum_sizes[-1]:
            raise IndexError(idx)
        batch_idx = bisect.bisect_right(self.cum_sizes, idx)
        prev_cum = 0 if batch_idx == 0 else self.cum_sizes[batch_idx - 1]
        sample_idx = idx - prev_cum
        x = self._load_sample(batch_idx, sample_idx)
        return x


def main():
    p = argparse.ArgumentParser(description="Train per-chromosome VQ-VAE on H5 caches")
    p.add_argument("--h5-dir", required=True)
    p.add_argument("--chromosome", type=int, required=True)
    p.add_argument("--bim", type=str, default=None, help="Optional BIM file to infer input length for this chromosome")
    p.add_argument("--latent-dim", type=int, default=64)
    p.add_argument("--codebook-size", type=int, default=1024)
    p.add_argument("--num-quantizers", type=int, default=2)
    p.add_argument("--beta-commit", type=float, default=0.25)
    p.add_argument("--latent-grid-dim", type=int, default=16)
    p.add_argument("--hidden-1d-channels", type=int, default=8)
    p.add_argument("--hidden-2d-channels", type=int, default=64)
    p.add_argument("--layers-at-final", type=int, default=0)
    p.add_argument("--ema-decay", type=float, default=0.99)
    p.add_argument("--ld-lambda", type=float, default=1e-3)
    p.add_argument("--maf-lambda", type=float, default=0.0)
    p.add_argument("--ld-window", type=int, default=128)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--save-path", required=True)
    args = p.parse_args()

    setup_logging()

    # Build per-chromosome dataset
    dataset = H5ChromosomeDataset(args.h5_dir, args.chromosome)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)

    # Infer input length (L) either from BIM across selected chromosomes or from H5 shapes
    if args.bim:
        # Input length for this chromosome from BIM
        bim_chr = read_bim_file(args.bim, int(args.chromosome))
        L = int(bim_chr.shape[0])
    else:
        # Use H5 feature length for this chromosome
        L = dataset.total_len

    # Report dataset stats
    total_records = len(dataset)
    logger.info(
        f"[VQ-VAE Train] chr={args.chromosome} | samples={total_records} | features(L)={L} | batch_size={args.batch_size} | "
        f"num_h5_batches={len(dataset.batches)}"
    )

    # Enforce k lower bound so that S >= desired latent_grid_dim
    # We want S = 2^{floor(k/2)} >= latent_grid_dim -> k >= 2*log2(latent_grid_dim)
    import math as _math
    min_k_needed = int(_math.ceil(2 * _math.log2(max(1, int(args.latent_grid_dim)))))
    if (min_k_needed % 2) == 1:
        min_k_needed += 1
    # Add +2 to guard against odd-k halving path
    min_k_needed += 2
    c_tmp, k_tmp = find_best_ck(L, min_k=min_k_needed, force_even_k=True)
    # No need to clamp latent_grid_dim now; S computed in the model will be >= requested
    latent_grid_dim_eff = int(args.latent_grid_dim)

    model, optim = build_vqvae(
        input_length=L,
        latent_grid_dim=int(latent_grid_dim_eff),
        latent_dim=args.latent_dim,
        codebook_size=args.codebook_size,
        num_quantizers=args.num_quantizers,
        beta_commit=args.beta_commit,
        lr=args.lr,
        ld_lambda=args.ld_lambda,
        maf_lambda=args.maf_lambda,
        ld_window=args.ld_window,
        ema_decay=args.ema_decay,
        hidden_1d_channels=int(args.hidden_1d_channels),
        hidden_2d_channels=int(args.hidden_2d_channels),
        layers_at_final=int(args.layers_at_final),
    )

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith('cuda') else 'cpu')
    model.to(device)
    model.train()

    # Log a one-batch model summary with user-provided params
    try:
        # Prepare a small sample batch from dataset
        first_batch = next(iter(DataLoader(dataset, batch_size=min(4, max(1, args.batch_size)), shuffle=False)))
        log_model_summary(model, first_batch, device)
        logger.info(
            f"[ModelParams] L={model.cfg.input_length}, S={model.S}, latent_grid_dim={model.cfg.latent_grid_dim}, "
            f"latent_dim={model.cfg.latent_dim}, hidden_1d={model.cfg.hidden_1d_channels}, hidden_2d={model.cfg.hidden_2d_channels}, "
            f"layers_at_final={model.cfg.layers_at_final}, num_quantizers={model.cfg.num_quantizers}, codebook_size={model.cfg.codebook_size}"
        )
    except Exception as e:
        logger.warning(f"Skipped detailed model summary due to: {e}")

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=int(args.epochs), eta_min=max(3e-5, float(args.lr)*1e-2))
    # Delegate training to shared helper 
    train_vqvae_fn(model, loader, optim, device=device, num_epochs=int(args.epochs), scheduler=scheduler)

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save({
        'model_state': model.state_dict(),
        'config': model.cfg.__dict__,
    }, args.save_path)
    logger.info(f"Saved VQ-VAE to {args.save_path}")


if __name__ == "__main__":
    main()


