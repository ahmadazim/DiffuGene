import os
import argparse
import glob
from typing import List, Optional, Tuple, Dict, Any
import bisect
from dataclasses import asdict
import time

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import h5py
import torch.nn.functional as F

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__))))

from DiffuGene.VAEembed.vae import (
    GenotypeAutoencoder,
    VAEConfig,
    build_vae,
    train_vae as train_vae_fn,
    find_best_ck,
)
from DiffuGene.VAEembed.train import H5ChromosomeDataset, log_model_summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# load command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--chrom", type=int, required=True)
parser.add_argument("--K1", type=int, required=True)
parser.add_argument("--K2", type=int, required=True)
parser.add_argument("--C", type=int, required=True)
args = parser.parse_args()

h5_dir = "/n/home03/ahmadazim/WORKING/genGen/UKBVQVAE/genomic_data/vqvae_h5_cache/"
chrom = args.chrom
bs = 128
K1 = args.K1
K2 = args.K2
C = args.C
embed_dim = 8
lr = 2e-3
weight_decay = 0.1
epochs = 15


dataset = H5ChromosomeDataset(h5_dir, chrom, load_num_batches=1)
loader = DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=0, pin_memory=True)
L = dataset.total_len

total_records = len(dataset)
print(
    f"[AE Train] chr={chrom} | samples={total_records} | features(L)={L} | batch_size={bs} | "
    f"num_h5_batches={len(dataset.batches)}"
)

cfg = VAEConfig(
    input_length=L,
    K1=int(K1),
    K2=int(K2),
    C=int(C),
    embed_dim=int(embed_dim),
    lr=float(lr),
    weight_decay=float(weight_decay),
)
model, optim = build_vae(cfg)
model.to(device)
model.train()

first_batch = next(iter(DataLoader(dataset, batch_size=min(4, max(1, bs)), shuffle=False)))
log_model_summary(model, first_batch, device)
print(
    f"[ModelParams] L={model.input_length} | K1={model.K1} | K2={model.K2} | C={model.C} | embed_dim={model.embed_dim}"
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optim, T_max=int(epochs), eta_min=max(3e-5, float(lr) * 1e-2)
)

if device.type == "cuda":
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    train_start = time.perf_counter()

res = train_vae_fn(
    model,
    loader,
    optim,
    device=device,
    num_epochs=int(epochs),
    scheduler=scheduler, 
    maf_lambda=1e-3,
    ld_lambda=1e-3,
)

torch.cuda.synchronize() if device.type == "cuda" else None
train_time_s = time.perf_counter() - train_start
train_peak_mb = (torch.cuda.max_memory_allocated() / (1024**2)) if device.type == "cuda" else None
train_peak_reserved_mb = (torch.cuda.max_memory_reserved() / (1024**2)) if device.type == "cuda" else None
if device.type == "cuda":
    torch.cuda.reset_peak_memory_stats()

# evaluate on the validation set
val_h5_dir = "/n/home03/ahmadazim/WORKING/genGen/UKBVQVAE/genomic_data/val_h5_cache/"
val_dataset = H5ChromosomeDataset(val_h5_dir, chrom, load_num_batches=1)
val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False)
xb = next(iter(val_loader)).to(device)
enc = model(xb)
logits3 = enc[0]
softmax3 = F.softmax(logits3, dim=1)
xhat = softmax3.argmax(dim=1)


# -------------------------------------------------------------------
# Save + Eval (runtime, peak mem, validation diagnostics)
# -------------------------------------------------------------------
import json
from datetime import datetime
from dataclasses import asdict

@torch.no_grad()
def eval_batch_autoencoder(model: GenotypeAutoencoder, x: torch.Tensor, device=None, verbose: bool = True) -> Dict[str, Any]:
    """
    Evaluate AE on one batch. Returns accuracy, dosage MSE, CE, and error-site diagnostics.
    """
    if device is None:
        device = next(model.parameters()).device
    x = x.to(device)
    x_long = x.long()

    logits3, _ = model(x)                         # (B,3,L)
    if logits3.dim() != 3:
        raise ValueError(f"Expected logits [B,3,L], got {tuple(logits3.shape)}")
    probs = torch.softmax(logits3, dim=1)         # (B,3,L)
    B, _, L = probs.shape

    # Predictions & dosage
    pred = probs.argmax(dim=1)                    # (B,L)
    class_values = torch.tensor([0.0, 1.0, 2.0], device=probs.device).view(1, 3, 1)
    x_hat = (probs * class_values).sum(dim=1)     # (B,L)

    # Core metrics
    acc = (pred == x_long).float().mean().item()
    mse = F.mse_loss(x_hat, x.float()).item()
    ce  = F.cross_entropy(logits3, x_long, reduction="mean").item()

    # Error diagnostics
    err_mask = (pred != x_long)                   # (B,L)
    num_err  = int(err_mask.sum().item())
    true_probs = probs.gather(1, x_long.unsqueeze(1)).squeeze(1)  # (B,L)
    pred_probs = probs.gather(1, pred.unsqueeze(1)).squeeze(1)    # (B,L)
    margin = pred_probs - true_probs

    if num_err > 0:
        mean_true_prob_err = true_probs[err_mask].mean().item()
        mean_pred_prob_err = pred_probs[err_mask].mean().item()
        mean_margin_err    = margin[err_mask].mean().item()
        top2 = probs.topk(k=2, dim=1)
        in_top2 = (top2.indices == x_long.unsqueeze(1)).any(dim=1)  # (B,L)
        top2_rate_err = in_top2[err_mask].float().mean().item()
        conf = {f"err_true_{c}": int(((x_long == c) & err_mask).sum().item()) for c in [0,1,2]}
    else:
        mean_true_prob_err = float("nan")
        mean_pred_prob_err = float("nan")
        mean_margin_err    = float("nan")
        top2_rate_err      = float("nan")
        conf = {f"err_true_{c}": 0 for c in [0,1,2]}

    if verbose:
        print(
            f"[Eval] acc={acc:.4f} | mse={mse:.6f} | ce={ce:.6f} | errors={num_err} | "
            f"trueProb(err)={mean_true_prob_err:.4f} | predProb(err)={mean_pred_prob_err:.4f} | "
            f"margin(err)={mean_margin_err:.4f} | top2_true(err)={top2_rate_err:.4f}"
        )
        if num_err > 0:
            print(f"[Eval] confusion on errors: {conf}")

    return {
        "acc": acc, "mse": mse, "ce": ce,
        "num_errors": num_err,
        "mean_true_prob_on_errors": mean_true_prob_err,
        "mean_pred_prob_on_errors": mean_pred_prob_err,
        "mean_margin_pred_minus_true_on_errors": mean_margin_err,
        "top2_contains_true_on_errors": top2_rate_err,
        **conf,
        "pred": pred, "probs": probs, "x_hat": x_hat, "err_mask": err_mask,
    }


@torch.no_grad()
def eval_on_loader(model: GenotypeAutoencoder, loader: DataLoader, device, max_batches: int = 5, verbose: bool = True) -> Dict[str, Any]:
    """
    Evaluate across a few batches; returns aggregated metrics.
    """
    model.eval()
    agg = {
        "sites": 0, "errors": 0,
        "sum_mse": 0.0, "sum_ce": 0.0, "sum_acc_sites": 0.0,
        "sum_true_prob_err": 0.0, "sum_pred_prob_err": 0.0, "sum_margin_err": 0.0, "sum_top2_true_err": 0.0,
        "err_true_0": 0, "err_true_1": 0, "err_true_2": 0,
    }
    batches_done = 0
    for xb in loader:
        xb = xb.to(device)
        out = eval_batch_autoencoder(model, xb, device=device, verbose=False)
        B, L = xb.shape
        sites = B * L

        agg["sites"] += sites
        agg["errors"] += out["num_errors"]
        agg["sum_mse"] += out["mse"]
        agg["sum_ce"]  += out["ce"]
        agg["sum_acc_sites"] += out["acc"] * sites

        if out["num_errors"] > 0:
            agg["sum_true_prob_err"] += out["mean_true_prob_on_errors"] * out["num_errors"]
            agg["sum_pred_prob_err"] += out["mean_pred_prob_on_errors"] * out["num_errors"]
            agg["sum_margin_err"]    += out["mean_margin_pred_minus_true_on_errors"] * out["num_errors"]
            if not np.isnan(out["top2_contains_true_on_errors"]):
                agg["sum_top2_true_err"] += out["top2_contains_true_on_errors"] * out["num_errors"]
            for c in [0,1,2]:
                agg[f"err_true_{c}"] += out.get(f"err_true_{c}", 0)

        batches_done += 1
        if batches_done >= max_batches:
            break

    mean_acc = (agg["sum_acc_sites"] / agg["sites"]) if agg["sites"] > 0 else float("nan")
    mean_mse = agg["sum_mse"] / max(1, batches_done)
    mean_ce  = agg["sum_ce"]  / max(1, batches_done)

    if agg["errors"] > 0:
        mean_true_prob_err = agg["sum_true_prob_err"] / agg["errors"]
        mean_pred_prob_err = agg["sum_pred_prob_err"] / agg["errors"]
        mean_margin_err    = agg["sum_margin_err"] / agg["errors"]
        top2_rate_err      = agg["sum_top2_true_err"] / agg["errors"]
    else:
        mean_true_prob_err = float("nan"); mean_pred_prob_err = float("nan")
        mean_margin_err    = float("nan"); top2_rate_err      = float("nan")

    results = {
        "batches_evaluated": batches_done,
        "sites_evaluated": int(agg["sites"]),
        "accuracy": float(mean_acc),
        "dosage_mse": float(mean_mse),
        "cross_entropy": float(mean_ce),
        "num_error_sites": int(agg["errors"]),
        "mean_true_prob_on_errors": float(mean_true_prob_err),
        "mean_pred_prob_on_errors": float(mean_pred_prob_err),
        "mean_margin_pred_minus_true_on_errors": float(mean_margin_err),
        "top2_contains_true_on_errors": float(top2_rate_err),
        "confusion_errors": {
            "true_0": int(agg["err_true_0"]),
            "true_1": int(agg["err_true_1"]),
            "true_2": int(agg["err_true_2"]),
        },
    }

    if verbose:
        print(
            "[ValAgg] batches=%d | sites=%d | acc=%.4f | mse=%.6f | ce=%.6f | "
            "err_sites=%d | trueProb(err)=%.4f | predProb(err)=%.4f | margin(err)=%.4f | top2_true(err)=%.4f",
            results["batches_evaluated"], results["sites_evaluated"], results["accuracy"],
            results["dosage_mse"], results["cross_entropy"], results["num_error_sites"],
            results["mean_true_prob_on_errors"], results["mean_pred_prob_on_errors"],
            results["mean_margin_pred_minus_true_on_errors"], results["top2_contains_true_on_errors"]
        )
        print("[ValAgg] confusion on errors: %s", results["confusion_errors"])
    return results


def save_run(out_dir: str, model: GenotypeAutoencoder, cfg: VAEConfig, train_res: Dict[str, Any], val_metrics: Dict[str, Any], timings: Dict[str, Any], sysinfo: Dict[str, Any]) -> None:
    os.makedirs(out_dir, exist_ok=True)
    meta = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config": asdict(cfg),
        "derived": {
            "L": model.input_length,
            "K1_size": model.K1_size, "K2_size": model.K2_size, "C_channels": model.C,
            "K1_exp": model.K1, "K2_exp": model.K2, "C_exp": model.C_exp,
            "c": model.c, "L1": model.L1, "target_len": model.target_len
        },
        "train_best_meta": train_res.get("best_meta", {}),
        "train_last_meta": train_res.get("last_meta", {}),
        "val_metrics": val_metrics,
        "timings": timings,
        "system": sysinfo,
    }
    with open(os.path.join(out_dir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Final (last) checkpoint
    torch.save(
        {"model_state": model.state_dict(), "config": asdict(cfg), "meta": meta},
        os.path.join(out_dir, "last.pt"),
    )
    # Best snapshot if present
    if train_res.get("best_state_dict") is not None:
        torch.save(
            {"model_state": train_res["best_state_dict"], "config": asdict(cfg), "meta": meta},
            os.path.join(out_dir, "best.pt"),
        )


# ---- Measure eval runtime + peak GPU memory (training peak needs a reset before training) ----
# If you also want *training* peak memory, place this before train_vae_fn(...):
#   if device.type == "cuda": torch.cuda.reset_peak_memory_stats()
# and wrap the train call with a timer, then record torch.cuda.max_memory_allocated().

if device.type == "cuda":
    # This captures peak since the last reset (or since start of process)
    torch.cuda.synchronize()
    eval_start = time.perf_counter()
    val_metrics = eval_on_loader(model, val_loader, device=device, max_batches=5, verbose=True)
    torch.cuda.synchronize()
    eval_runtime_s = time.perf_counter() - eval_start
    eval_peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
    eval_peak_reserved_mb = torch.cuda.max_memory_reserved() / (1024**2)
else:
    eval_start = time.perf_counter()
    val_metrics = eval_on_loader(model, val_loader, device=device, max_batches=5, verbose=True)
    eval_runtime_s = time.perf_counter() - eval_start
    eval_peak_mb = None
    eval_peak_reserved_mb = None

timings = {
    "train_runtime_sec": float(train_time_s),
    "eval_runtime_sec": float(eval_runtime_s),
}
sysinfo = {
    "device": str(device),
    "train_cuda_peak_allocated_MB": float(train_peak_mb) if train_peak_mb is not None else None,
    "train_cuda_peak_reserved_MB": float(train_peak_reserved_mb) if train_peak_reserved_mb is not None else None,
    "eval_cuda_peak_allocated_MB": float(eval_peak_mb) if eval_peak_mb is not None else None,
    "eval_cuda_peak_reserved_MB": float(eval_peak_reserved_mb) if eval_peak_reserved_mb is not None else None,
    "torch_version": torch.__version__,
    "cuda_is_available": torch.cuda.is_available(),
}

# ---- Persist everything ----
run_tag = f"chr{chrom}_K1-{K1}_K2-{K2}_C-{C}_L-{L}_bs{bs}_e{epochs}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
out_dir = os.path.join("/n/home03/ahmadazim/WORKING/genGen/UKBVQVAE/hyperparam/ae_runs", run_tag)
save_run(out_dir, model, cfg, res, val_metrics, timings, sysinfo)

print(f"[Saved] Directory: {out_dir}")
print("         - last.pt, best.pt (if available), run_meta.json")
print(f"[Eval]   acc={val_metrics['accuracy']:.4f} | mse={val_metrics['dosage_mse']:.6f} | ce={val_metrics['cross_entropy']:.6f} | "
      f"errors={val_metrics['num_error_sites']} | "
      f"train_time={timings['train_runtime_sec']:.2f}s (peak={sysinfo['train_cuda_peak_allocated_MB']}) | "
      f"eval_time={timings['eval_runtime_sec']:.2f}s (peak={sysinfo['eval_cuda_peak_allocated_MB']})")