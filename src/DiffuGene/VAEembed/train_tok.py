#!/usr/bin/env python
from __future__ import annotations

import argparse
import bisect
import glob
import os
from dataclasses import asdict
from typing import Dict

import h5py
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    from .aeTok import TokenAEConfig, build_token_ae
    from ..utils import setup_logging, get_logger
except Exception:
    import sys
    sys.path.append('/n/home03/ahmadazim/WORKING/genGen/DiffuGene/src')
    from DiffuGene.VAEembed.aeTok import TokenAEConfig, build_token_ae
    from DiffuGene.utils import setup_logging, get_logger


logger = get_logger(__name__)


class H5ChromosomeDataset(Dataset):
    def __init__(self, h5_dir: str, chromosome: int, load_num_batches: int | None = None):
        self.chromosome = int(chromosome)
        self.chr_dir = os.path.join(h5_dir, f"chr{self.chromosome}")
        batch_files = sorted(glob.glob(os.path.join(self.chr_dir, "batch*.h5")))
        if not batch_files:
            raise FileNotFoundError(f"No H5 caches found in {self.chr_dir}")
        if load_num_batches is not None:
            batch_files = batch_files[:load_num_batches]
        self.batches = [os.path.splitext(os.path.basename(p))[0] for p in batch_files]
        with h5py.File(batch_files[0], "r") as f:
            self.chrom_length = int(f["X"].shape[1])
        self.batch_sizes = []
        self.cum_sizes = []
        total = 0
        for b in self.batches:
            with h5py.File(os.path.join(self.chr_dir, f"{b}.h5"), "r") as f:
                rows = int(f["X"].shape[0])
            self.batch_sizes.append(rows)
            total += rows
            self.cum_sizes.append(total)

    def __len__(self) -> int:
        return self.cum_sizes[-1]

    def _load_sample(self, batch_idx: int, sample_idx: int) -> torch.Tensor:
        p = os.path.join(self.chr_dir, f"{self.batches[batch_idx]}.h5")
        with h5py.File(p, "r") as f:
            x = f["X"][sample_idx]
        return torch.from_numpy(x.astype("int64"))

    def __getitem__(self, idx: int) -> torch.Tensor:
        if idx < 0 or idx >= self.cum_sizes[-1]:
            raise IndexError(idx)
        batch_idx = bisect.bisect_right(self.cum_sizes, idx)
        prev = 0 if batch_idx == 0 else self.cum_sizes[batch_idx - 1]
        sample_idx = idx - prev
        return self._load_sample(batch_idx, sample_idx)


@torch.no_grad()
def _eval_val(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    maf_lambda: float = 0.0,
    ld_lambda: float = 0.0,
    ld_window: int = 128,
    beta_kl: float = 0.0,
) -> Dict[str, float]:
    model.eval()
    total_recon = 0.0
    total_maf = 0.0
    total_ld = 0.0
    total_kl = 0.0
    total_mse = 0.0
    total_n = 0
    for xb in loader:
        xb = xb.to(device)
        logits, _ = model(xb)
        loss, metrics = model.loss_function(
            logits,
            xb,
            None,
            maf_lambda=maf_lambda,
            ld_lambda=ld_lambda,
            ld_window=ld_window,
            beta_kl=beta_kl,
        )
        probs = torch.softmax(logits, dim=-1)
        x_hat = probs[..., 1] + 2.0 * probs[..., 2]
        mse = torch.mean((x_hat - xb.float()) ** 2)
        bsz = xb.size(0)
        total_recon += float(metrics["recon"].item()) * bsz
        total_maf += float(metrics["maf"].item()) * bsz
        total_ld += float(metrics["ld"].item()) * bsz
        total_kl += float(metrics["kl"].item()) * bsz
        total_mse += float(mse.item()) * bsz
        total_n += bsz
    model.train()
    if total_n == 0:
        return {
            "recon": float("nan"),
            "maf": float("nan"),
            "ld": float("nan"),
            "kl": float("nan"),
            "mse": float("nan"),
            "loss": float("nan"),
        }
    return {
        "recon": total_recon / total_n,
        "maf": total_maf / total_n,
        "ld": total_ld / total_n,
        "kl": total_kl / total_n,
        "mse": total_mse / total_n,
        "loss": (total_recon + total_maf + total_ld) / total_n,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Train per-chromosome TokenAutoencoder1D on H5 caches.")
    p.add_argument("--h5-dir", required=True)
    p.add_argument("--val-h5-dir", required=True)
    p.add_argument("--chromosome", type=int, required=True)
    p.add_argument("--latent-length", type=int, required=True)
    p.add_argument("--latent-dim", type=int, default=256)
    p.add_argument("--embed-dim", type=int, default=8)
    p.add_argument("--max-c", type=int, default=5)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--maf-lambda", type=float, default=0.0)
    p.add_argument("--ld-lambda", type=float, default=0.0)
    p.add_argument("--ld-window", type=int, default=128)
    p.add_argument("--beta-kl", type=float, default=0.0, help="Kept for parity; token AE currently deterministic.")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--save-path", required=True)
    args = p.parse_args()

    setup_logging()
    train_ds = H5ChromosomeDataset(args.h5_dir, args.chromosome)
    val_ds = H5ChromosomeDataset(args.val_h5_dir, args.chromosome)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=max(256, args.batch_size), shuffle=False, num_workers=4, pin_memory=True)

    cfg = TokenAEConfig(
        input_length=int(train_ds.chrom_length),
        latent_length=int(args.latent_length),
        latent_dim=int(args.latent_dim),
        embed_dim=int(args.embed_dim),
        max_c=int(args.max_c),
        dropout=float(args.dropout),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        grad_clip=float(args.grad_clip),
    )
    model, optimizer = build_token_ae(cfg)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(args.epochs), eta_min=max(3e-5, float(args.lr) * 1e-2)
    )

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    model.to(device).train()

    best = {"epoch": 0, "val_mse": float("inf"), "state": None}
    for epoch in range(1, int(args.epochs) + 1):
        sum_loss = 0.0
        sum_recon = 0.0
        sum_maf = 0.0
        sum_ld = 0.0
        sum_kl = 0.0
        n = 0
        for xb in train_loader:
            xb = xb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(xb)
            loss, metrics = model.loss_function(
                logits,
                xb,
                None,
                maf_lambda=float(args.maf_lambda),
                ld_lambda=float(args.ld_lambda),
                ld_window=int(args.ld_window),
                beta_kl=float(args.beta_kl),
            )
            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            optimizer.step()
            sum_loss += float(loss.item()) * xb.size(0)
            sum_recon += float(metrics["recon"].item()) * xb.size(0)
            sum_maf += float(metrics["maf"].item()) * xb.size(0)
            sum_ld += float(metrics["ld"].item()) * xb.size(0)
            sum_kl += float(metrics["kl"].item()) * xb.size(0)
            n += xb.size(0)
        scheduler.step()

        val = _eval_val(
            model,
            val_loader,
            device,
            maf_lambda=float(args.maf_lambda),
            ld_lambda=float(args.ld_lambda),
            ld_window=int(args.ld_window),
            beta_kl=float(args.beta_kl),
        )
        logger.info(
            "[TOK-AE] epoch=%d/%d train_loss=%.6f train_recon=%.6f train_maf=%.6f train_ld=%.6f train_kl=%.6f "
            "val_recon=%.6f val_maf=%.6f val_ld=%.6f val_kl=%.6f val_mse=%.6f",
            epoch,
            int(args.epochs),
            (sum_loss / max(1, n)),
            (sum_recon / max(1, n)),
            (sum_maf / max(1, n)),
            (sum_ld / max(1, n)),
            (sum_kl / max(1, n)),
            val["recon"],
            val["maf"],
            val["ld"],
            val["kl"],
            val["mse"],
        )
        if val["mse"] < best["val_mse"]:
            best["val_mse"] = float(val["mse"])
            best["epoch"] = int(epoch)
            best["state"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    payload = {
        "model_state": best["state"] if best["state"] is not None else model.state_dict(),
        "config": asdict(cfg),
        "meta": {
            "chromosome": int(args.chromosome),
            "best_epoch": int(best["epoch"]),
            "best_val_mse": float(best["val_mse"]),
            "latent_length": int(args.latent_length),
            "latent_dim": int(args.latent_dim),
        },
    }
    torch.save(payload, args.save_path)
    logger.info("Saved token AE checkpoint to %s", args.save_path)


if __name__ == "__main__":
    main()
