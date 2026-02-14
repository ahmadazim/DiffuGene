#!/usr/bin/env python
"""
Post-generation PCA alignment via orthogonal Procrustes (per chromosome).

Procedure (matches smoothAE_eval.ipynb 1-53):
  1) Read N original and N generated samples (dosage calls 0/1/2).
  2) Normalize BOTH datasets using ORIGINAL mean/std only.
  3) Fit PCA separately on original and generated normalized data (retain K PCs).
  4) Align PC bases via orthogonal Procrustes:
        M = W_g @ W_o.T
        U, _, Vt = svd(M)
        R = U @ Vt
     then rotate generated scores: Z_g_corr = Z_g @ R.T
  5) Reconstruct generated data in original PCA space and undo normalization:
        X_g_norm_corr = pca_o.inverse_transform(Z_g_corr)
        X_g_corr = X_g_norm_corr * std_o + mean_o
     Save both continuous aligned dosages and clipped/rounded calls.

Inputs:
  - Original data: cached H5 batches: <orig_h5_root>/chr{c}/batch*.h5 (dataset "X")
  - Generated data: decoded calls from DiffuGene.generate.decode:
      directory with chr{c}_calls.pt OR a .pt payload with calls_by_chr/hard_calls_by_chr.
"""

import argparse
import glob
import os
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
from sklearn.decomposition import PCA


def _resolve_chromosomes(spec: List[str]) -> List[int]:
    if len(spec) == 1 and spec[0].lower() == "all":
        return list(range(1, 23))
    out: List[int] = []
    for token in spec:
        for piece in token.split(","):
            piece = piece.strip()
            if not piece:
                continue
            out.append(int(piece))
    return out


def _list_chr_h5_batches(orig_h5_root: str, chr_no: int) -> List[str]:
    chr_dir = os.path.join(orig_h5_root, f"chr{chr_no}")
    return sorted(glob.glob(os.path.join(chr_dir, "batch*.h5")))


def _load_one_h5_batch_calls(orig_h5_root: str, chr_no: int) -> np.ndarray:
    files = _list_chr_h5_batches(orig_h5_root, chr_no)
    if not files:
        raise FileNotFoundError(f"No H5 batch files found for chr{chr_no} under {orig_h5_root}")
    h5_path = files[0]
    with h5py.File(h5_path, "r") as f:
        X = f["X"][:]  # (N,L)
    return np.asarray(X, dtype=np.int8)


def load_generated_calls(generated_path: str, chr_no: int) -> np.ndarray:
    if os.path.isdir(generated_path):
        pth = os.path.join(generated_path, f"chr{chr_no}_calls.pt")
        t = torch.load(pth, map_location="cpu")
        if torch.is_tensor(t):
            return t.cpu().numpy().astype(np.int8, copy=False)
        if isinstance(t, dict) and "calls" in t and torch.is_tensor(t["calls"]):
            return t["calls"].cpu().numpy().astype(np.int8, copy=False)
        raise ValueError(f"Unexpected tensor payload in {pth}")

    payload = torch.load(generated_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported generated payload type: {type(payload)}")
    for key in ("calls_by_chr", "hard_calls_by_chr"):
        if key in payload and isinstance(payload[key], dict) and chr_no in payload[key]:
            t = payload[key][chr_no]
            if not torch.is_tensor(t):
                raise ValueError(f"{key}[{chr_no}] is not a tensor")
            return t.cpu().numpy().astype(np.int8, copy=False)
    raise KeyError(f"Could not find chr{chr_no} calls inside {generated_path}")


def _fit_alignment(
    X_o: np.ndarray,
    X_g: np.ndarray,
    k_pcs: int,
    seed: int,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], PCA, PCA]:
    """
    Fit PCA_o, PCA_g on normalized data and compute Procrustes rotation R.
    Returns:
      R: (K,K)
      norm: {"mean_o": (1,L), "std_o": (1,L)}
      pca_o, pca_g: fitted sklearn PCA objects on normalized data
    """
    # normalize using ORIGINAL stats only
    mean_o = X_o.mean(axis=0, keepdims=True)
    std_o = X_o.std(axis=0, keepdims=True)
    std_o[std_o == 0] = 1.0
    X_o_norm = (X_o - mean_o) / std_o
    X_g_norm = (X_g - mean_o) / std_o

    # PCA separately (same feature scaling)
    pca_o = PCA(n_components=k_pcs, svd_solver="randomized", random_state=seed)
    pca_g = PCA(n_components=k_pcs, svd_solver="randomized", random_state=seed)
    pca_o.fit(X_o_norm)
    pca_g.fit(X_g_norm)

    Z_g = pca_g.transform(X_g_norm)  # [N,K]
    W_o = pca_o.components_          # [K,L]
    W_g = pca_g.components_          # [K,L]

    # Procrustes rotation: R @ W_g ≈ W_o
    M = W_g @ W_o.T                  # [K,K]
    U, _, Vt = np.linalg.svd(M, full_matrices=False)
    R = U @ Vt                       # [K,K]

    norm = {"mean_o": mean_o, "std_o": std_o}
    return R.astype(np.float64), norm, pca_o, pca_g


def _apply_alignment_full(
    X_g_full: np.ndarray,
    R: np.ndarray,
    norm: Dict[str, np.ndarray],
    pca_o: PCA,
    pca_g: PCA,
) -> Tuple[np.ndarray, np.ndarray]:
    mean_o = norm["mean_o"]
    std_o = norm["std_o"]

    X_g_full = X_g_full.astype(np.float64, copy=False)
    X_g_norm = (X_g_full - mean_o) / std_o
    Z_g = pca_g.transform(X_g_norm)
    Z_g_corr = Z_g @ R.T
    X_g_norm_corr = pca_o.inverse_transform(Z_g_corr)
    X_g_corr = X_g_norm_corr * std_o + mean_o

    X_g_round = np.clip(np.round(X_g_corr), 0, 2).astype(np.int8)
    return X_g_corr.astype(np.float32), X_g_round


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Post-generation PCA alignment (Procrustes).")
    p.add_argument("--orig-h5-root", required=True, help="Root of per-chromosome H5 cache directories.")
    p.add_argument("--generated", required=True, help="Generated decoded calls: dir with chr{c}_calls.pt or .pt payload.")
    p.add_argument("--chromosomes", nargs="*", default=["all"], help="Chromosomes to process, or 'all'.")
    p.add_argument("--n-fit", type=int, default=10000, help="N samples used to fit PCA + Procrustes rotation.")
    p.add_argument("--k-pcs", type=int, default=3000, help="Number of PCs (K) to retain.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--models-dir", required=True, help="Directory to save rotation matrices (and metadata).")
    p.add_argument("--out-dir", required=True, help="Directory to save aligned generated outputs.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.models_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    chromosomes = _resolve_chromosomes(args.chromosomes)

    for chr_no in chromosomes:
        print(f"Processing chromosome {chr_no}")
        calls_gen = load_generated_calls(args.generated, chr_no)
        calls_orig_full = _load_one_h5_batch_calls(args.orig_h5_root, chr_no)

        print(f"Loaded {calls_gen.shape[0]} generated calls and {calls_orig_full.shape[0]} original calls")

        n_gen = int(calls_gen.shape[0])
        calls_orig = calls_orig_full[:n_gen, :]
        calls_gen = calls_gen[: calls_orig.shape[0], :]

        n_fit = min(int(args.n_fit), calls_orig.shape[0], calls_gen.shape[0])
        print(f"Using first {n_fit} samples for PCA fitting")

        X_o = calls_orig[:n_fit].astype(float)
        X_g = calls_gen[:n_fit].astype(float)

        R, norm, pca_o, pca_g = _fit_alignment(
            X_o=X_o,
            X_g=X_g,
            k_pcs=int(args.k_pcs),
            seed=int(args.seed),
        )

        # Save rotation matrix (KxK) as requested
        R_path = os.path.join(args.models_dir, f"chr{chr_no}_R.pt")
        torch.save(torch.from_numpy(R), R_path)
        print(f"Saved rotation matrix for chromosome {chr_no}")

        # Save minimal metadata to reproduce application
        meta_path = os.path.join(args.models_dir, f"chr{chr_no}_pca_meta.pt")
        meta = {
            "chr": int(chr_no),
            "n_fit": int(n_fit),
            "k_pcs": int(args.k_pcs),
            "seed": int(args.seed),
            "mean_o": torch.from_numpy(norm["mean_o"].astype(np.float32)),
            "std_o": torch.from_numpy(norm["std_o"].astype(np.float32)),
            "pca_o_components": torch.from_numpy(pca_o.components_.astype(np.float32)),
            "pca_o_mean": torch.from_numpy(pca_o.mean_.astype(np.float32)),
            "pca_g_components": torch.from_numpy(pca_g.components_.astype(np.float32)),
            "pca_g_mean": torch.from_numpy(pca_g.mean_.astype(np.float32)),
        }
        torch.save(meta, meta_path)

        # Apply alignment to FULL generated set for this chromosome
        X_g_corr, X_g_round = _apply_alignment_full(
            X_g_full=calls_gen.astype(float),
            R=R,
            norm=norm,
            pca_o=pca_o,
            pca_g=pca_g,
        )

        out_float = os.path.join(args.out_dir, f"chr{chr_no}_aligned_float.pt")
        out_calls = os.path.join(args.out_dir, f"chr{chr_no}_aligned_calls.pt")
        torch.save(torch.from_numpy(X_g_corr), out_float)   # (N,L) float32
        torch.save(torch.from_numpy(X_g_round), out_calls)  # (N,L) int8


if __name__ == "__main__":
    main()
