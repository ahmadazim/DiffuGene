import os
import argparse
import glob
from typing import List, Optional, Tuple, Dict, Any
import bisect
from dataclasses import asdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import h5py

from .vae import (
    GenotypeAutoencoder,
    VAEConfig,
    build_vae,
    train_vae as train_vae_fn,
    find_best_ck,
)
from ..utils import setup_logging, get_logger


logger = get_logger(__name__)


def log_model_summary(model: GenotypeAutoencoder, x_sample: torch.Tensor, device: torch.device) -> None:
    model.eval()
    x_sample = x_sample.to(device)
    try:
        with torch.no_grad():
            logits3, z = model(x_sample)
            logger.info(
                "[ModelSummary] logits: %s | latent z: %s",
                tuple(logits3.shape),
                tuple(z.shape),
            )
            # also log internal c,k, dims
            logger.info(
                "[ModelSummary] c=%d, L1=%d, target_len=%d, after_conv=%d, M1D=%d, "
                "M2D=%d, latent_channels=%d",
                model.c, model.L1, model.target_len, model.length_after_conv,
                model.M1D, model.M2D, model.latent_channels
            )
    except Exception as e:
        logger.warning(f"[ModelSummary] Failed to log shapes due to: {e}")
    model.train()


class H5ChromosomeDataset(Dataset):
    """Dataset for a single chromosome from H5 caches.
    Directory structure: <h5_dir>/chr<no>/batchXXXXX.h5 with datasets X (B, L_chr), iid (B), bp (L_chr).
    """
    def __init__(self, h5_dir: str, chromosome: int, load_num_batches: int = None):
        self.h5_dir = h5_dir
        self.chromosome = int(chromosome)
        # Discover batch ids within the chromosome directory
        self.chr_dir = os.path.join(h5_dir, f"chr{self.chromosome}")
        batch_files = sorted(glob.glob(os.path.join(self.chr_dir, "batch*.h5")))
        if not batch_files:
            raise FileNotFoundError(f"No H5 caches found in {self.chr_dir}")
        if load_num_batches is not None:
            batch_files = batch_files[:load_num_batches]
        self.batches = [os.path.splitext(os.path.basename(p))[0] for p in batch_files]  # batchXXXXX
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
    p = argparse.ArgumentParser(description="Train per-chromosome AE on H5 caches")
    p.add_argument("--h5-dir", required=True)
    p.add_argument("--chromosome", type=int, required=True)
    # --- AE structure (powers of two) ---
    p.add_argument("--spatial1d", type=int, required=True, help="1D spatial size (K1), power of two (e.g., 512)")
    p.add_argument("--spatial2d", type=int, required=True, help="2D spatial size (K2), power of two (e.g., 32)")
    p.add_argument("--latent-channels", type=int, default=64, help="Final 2D channels C (power of two; e.g., 64)")
    p.add_argument("--embed-dim", type=int, default=8, help="1D embed channels (keep 8 unless you know what you’re doing)")
    # --- training ---
    p.add_argument("--ld-lambda", type=float, default=0.0)
    p.add_argument("--maf-lambda", type=float, default=0.0)
    p.add_argument("--ld-window", type=int, default=128)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--val-h5-dir", type=str, required=True, help="Validation H5 cache root with chr*/batch*.h5")
    # Early stopping
    p.add_argument("--plateau-min-rel-improve", type=float, default=0.005)
    p.add_argument("--plateau-patience", type=int, default=3)
    p.add_argument("--plateau-mse-threshold", type=float, default=0.01)
    p.add_argument("--save-path", required=True)
    args = p.parse_args()

    setup_logging()

    # Build datasets/loaders
    dataset = H5ChromosomeDataset(args.h5_dir, args.chromosome)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_dataset = H5ChromosomeDataset(args.val_h5_dir, args.chromosome)
    val_loader = DataLoader(val_dataset, batch_size=max(256, args.batch_size), shuffle=False, num_workers=4, pin_memory=True)

    L = dataset.total_len

    # Report dataset stats
    total_records = len(dataset)
    logger.info(
        f"[AE Train] chr={args.chromosome} | samples={total_records} | features(L)={L} | batch_size={args.batch_size} | "
        f"num_h5_batches={len(dataset.batches)}"
    )

    # Build model + optimizer
    cfg = VAEConfig(
        input_length=L,
        K1=int(args.spatial1d),
        K2=int(args.spatial2d),
        C=int(args.latent_channels),
        embed_dim=int(args.embed_dim),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    model, optim = build_vae(cfg)

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    model.to(device)
    model.train()

    # Log a one-batch model summary
    try:
        first_batch = next(iter(DataLoader(dataset, batch_size=min(4, max(1, args.batch_size)), shuffle=False)))
        log_model_summary(model, first_batch, device)
        logger.info(
            f"[ModelParams] L={model.input_length} | K1_size={model.K1_size} (K1={model.K1}) | "
            f"K2_size={model.K2_size} (K2={model.K2}) | C_channels={model.C} (C_exp={model.C_exp}) | embed_dim={model.embed_dim}"
        )
    except Exception as e:
        logger.warning(f"Skipped detailed model summary due to: {e}")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=int(args.epochs), eta_min=max(3e-5, float(args.lr) * 1e-2)
    )
    
    # Train
    res = train_vae_fn(
        model,
        loader,
        optim,
        device=device,
        num_epochs=int(args.epochs),
        grad_clip=float(args.grad_clip),
        scheduler=scheduler,
        val_dataloader=val_loader,
        plateau_min_rel_improve=float(args.plateau_min_rel_improve),
        plateau_patience=int(args.plateau_patience),
        plateau_mse_threshold=float(args.plateau_mse_threshold),
        maf_lambda=float(args.maf_lambda),
        ld_lambda=float(args.ld_lambda),
        ld_window=int(args.ld_window),
    )

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    # Save best checkpoint
    best_path = args.save_path
    best_payload = {
        "model_state": res["best_state_dict"],
        "config": asdict(cfg),
        "meta": {
            "c": int(getattr(model, "c", -1)),
            "L1": int(getattr(model, "L1", -1)),
            "target_len": int(getattr(model, "target_len", -1)),
            "length_after_conv": int(getattr(model, "length_after_conv", -1)),
            "M1D": int(getattr(model, "M1D", -1)),
            "M2D": int(getattr(model, "M2D", -1)),
            "K1_size": int(getattr(model, "K1_size", -1)),
            "K2_size": int(getattr(model, "K2_size", -1)),
            "C_channels": int(getattr(model, "C", -1)),
        },
        "training_args": {
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "device": args.device,
            "chromosome": int(args.chromosome),
            "val_mse": float(res["best_meta"].get("val_mse", float("nan"))),
            "best_epoch": int(res["best_meta"].get("epoch", args.epochs)),
        },
        "best_meta": res["best_meta"],
    }
    torch.save(best_payload, best_path)
    logger.info(f"Saved best model to {best_path} (val MSE={best_payload['training_args']['val_mse']:.6f})")


if __name__ == "__main__":
    main()


