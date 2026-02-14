#!/usr/bin/env python3
"""
Apply saved PCA Procrustes alignment to generated genotype calls (per chromosome).
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from sklearn.decomposition import PCA

import sys
this_dir = os.path.dirname(__file__)
src_root = os.path.abspath(os.path.join(this_dir, "..", "..", ".."))
if src_root not in sys.path:
    sys.path.insert(0, src_root)

from DiffuGene.postGen.pcaProcrustes_fit import _resolve_chromosomes, load_generated_calls


def _load_meta(models_dir: str, chr_no: int) -> Dict[str, np.ndarray]:
    meta_path = os.path.join(models_dir, f"chr{chr_no}_pca_meta.pt")
    meta = torch.load(meta_path, map_location="cpu")
    def _np(x: torch.Tensor) -> np.ndarray:
        return x.cpu().numpy()
    out = {
        "mean_o": _np(meta["mean_o"]).astype(np.float64, copy=False),
        "std_o": _np(meta["std_o"]).astype(np.float64, copy=False),
        "pca_o_components": _np(meta["pca_o_components"]).astype(np.float64, copy=False),
        "pca_o_mean": _np(meta["pca_o_mean"]).astype(np.float64, copy=False),
        "pca_g_components": _np(meta["pca_g_components"]).astype(np.float64, copy=False),
        "pca_g_mean": _np(meta["pca_g_mean"]).astype(np.float64, copy=False),
        "k_pcs": int(meta.get("k_pcs", meta["pca_o_components"].shape[0])),
    }
    return out

def _load_R(models_dir: str, chr_no: int) -> np.ndarray:
    R_path = os.path.join(models_dir, f"chr{chr_no}_R.pt")
    R = torch.load(R_path, map_location="cpu")
    if torch.is_tensor(R):
        R = R.cpu().numpy()
    R = np.asarray(R, dtype=np.float64)
    if R.ndim != 2 or R.shape[0] != R.shape[1]:
        raise ValueError(f"R must be square; got {R.shape} in {R_path}")
    return R

def _pca_transform(X: np.ndarray, components: np.ndarray, mean: np.ndarray) -> np.ndarray:
    return (X - mean) @ components.T

def _pca_inverse_transform(Z: np.ndarray, components: np.ndarray, mean: np.ndarray) -> np.ndarray:
    return Z @ components + mean

def apply_alignment(
    X_g: np.ndarray,
    mean_o: np.ndarray,
    std_o: np.ndarray,
    pca_o_components: np.ndarray,
    pca_o_mean: np.ndarray,
    pca_g_components: np.ndarray,
    pca_g_mean: np.ndarray,
    R: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    X_g = np.asarray(X_g, dtype=np.float64)
    mean_o_ = mean_o if mean_o.ndim == 2 else mean_o.reshape(1, -1)
    std_o_ = std_o if std_o.ndim == 2 else std_o.reshape(1, -1)
    X_g_norm = (X_g - mean_o_) / std_o_
    Z_g = _pca_transform(X_g_norm, pca_g_components, pca_g_mean)  # [N,K]
    Z_g_corr = Z_g @ R.T
    X_g_norm_corr = _pca_inverse_transform(Z_g_corr, pca_o_components, pca_o_mean)  # [N,L]
    X_g_corr = X_g_norm_corr * std_o_ + mean_o_
    X_g_round = np.clip(np.rint(X_g_corr), 0, 2).astype(np.int8)
    return X_g_corr.astype(np.float32), X_g_round


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Apply PCA-Procrustes alignment to generated calls.")
    p.add_argument("--generated-dir", required=True, help="Directory containing generated calls.")
    p.add_argument("--chromosomes", nargs="*", default=["all"], help="Chromosomes to process, or 'all'.")
    p.add_argument("--alignment-dir", required=True, help="Directory containing PCA Procrustes alignment artifacts.")
    p.add_argument("--output-dir", required=True, help="Directory to write aligned outputs.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    chrs = _resolve_chromosomes(args.chromosomes)

    for chr_no in chrs:
        print(f"[chr{chr_no}] loading generated calls")
        X_g = load_generated_calls(args.generated_dir, chr_no)  # [N,L] int8

        print(f"[chr{chr_no}] loading alignment artifacts")
        meta = _load_meta(args.alignment_dir, chr_no)
        R = _load_R(args.alignment_dir, chr_no)

        # Basic shape checks
        L = int(X_g.shape[1])
        K = int(meta["pca_o_components"].shape[0])
        if meta["pca_o_components"].shape[1] != L or meta["pca_g_components"].shape[1] != L:
            raise ValueError(f"[chr{chr_no}] feature mismatch: calls L={L}, components L_o={meta['pca_o_components'].shape[1]}, L_g={meta['pca_g_components'].shape[1]}")
        if R.shape != (K, K):
            raise ValueError(f"[chr{chr_no}] R shape {R.shape} != ({K},{K})")

        print(f"[chr{chr_no}] applying alignment (N={X_g.shape[0]}, L={L}, K={K})")
        X_float, X_calls = apply_alignment(
            X_g=X_g,
            mean_o=meta["mean_o"],
            std_o=meta["std_o"],
            pca_o_components=meta["pca_o_components"],
            pca_o_mean=meta["pca_o_mean"],
            pca_g_components=meta["pca_g_components"],
            pca_g_mean=meta["pca_g_mean"],
            R=R,
        )

        out_float = os.path.join(args.output_dir, f"chr{chr_no}_aligned_float.pt")
        out_calls = os.path.join(args.output_dir, f"chr{chr_no}_aligned_calls.pt")
        torch.save(torch.from_numpy(X_float), out_float)
        torch.save(torch.from_numpy(X_calls), out_calls)
        print(f"[chr{chr_no}] saved: {out_float}")
        print(f"[chr{chr_no}] saved: {out_calls}")


if __name__ == "__main__":
    main()