#!/usr/bin/env python
"""
Quality visualization for decoded diffusion samples (per chromosome).

Reads:
  - Original genotype dosage data from cached H5 batches:
      <orig_h5_root>/chr{chr}/batch*.h5 with dataset "X" (N,L_chr) in {0,1,2}
    For each chromosome, loads ONE batch file and subsets rows to match N_gen.

  - Generated decoded samples produced by `DiffuGene.generate.decode` (unaligned) OR
    aligned outputs produced by `DiffuGene.postGen.pca_align`:
      Either:
        (A) a directory containing per-chromosome outputs: chr{chr}_calls.pt
        (B) a single .pt payload containing calls_by_chr or hard_calls_by_chr
      For aligned evaluation, pass a directory containing:
        - chr{chr}_aligned_float.pt  (float dosages)
        - chr{chr}_aligned_calls.pt  (rounded/clipped calls)

Produces one combined figure per chromosome containing:
  1) MAF scatter plot (variant-wise)
  2) PCA overlays (unnormalized + normalized) for original vs generated
  3) LD block heatmap correspondence (up to K blocks) for original vs generated

Saves:
  - <out_dir>/chr{chr}_quality.png
  - <out_dir>/chr{chr}_stats.json
"""

import argparse
import glob
import json
import os
from typing import Dict, List, Optional

import h5py
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

import torch

import sys
this_dir = os.path.dirname(__file__)
src_root = os.path.abspath(os.path.join(this_dir, "..", "..", ".."))
if src_root not in sys.path:
    sys.path.insert(0, src_root)

from DiffuGene.utils.file_utils import read_bim_file


def _load_bim_snp_index(bim_path: str) -> Dict[int, Dict[str, int]]:
    """
    Load a single all-chromosome BIM once and build per-chromosome SNP->index maps.

    Index is 0-based within chromosome order in the BIM.
    """
    bim_df = pd.read_csv(
        bim_path,
        sep=r"\s+",
        header=None,
        names=["CHR", "SNP", "CM", "BP", "A1", "A2"],
        engine="python",
    )
    # normalize chromosome column to int (handles "chr1" style)
    chr_series = bim_df["CHR"].astype(str).str.replace(r"^chr", "", regex=True)
    bim_df["CHR"] = chr_series.astype(int)
    bim_df["SNP"] = bim_df["SNP"].astype(str)

    chr_to_map: Dict[int, Dict[str, int]] = {}
    # preserve per-chrom ordering by using groupby + enumerate
    for chr_no, sub in bim_df.groupby("CHR", sort=False):
        snps = sub["SNP"].tolist()
        chr_to_map[int(chr_no)] = {snp: i for i, snp in enumerate(snps)}
    return chr_to_map


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
    """
    Loads generated calls for a chromosome.

    Supported formats:
      - Directory: <generated_path>/chr{chr}_calls.pt containing a tensor (N,L)
      - File: a .pt payload containing either:
          payload["calls_by_chr"][chr]  or payload["hard_calls_by_chr"][chr]
    """
    if os.path.isdir(generated_path):
        pth = os.path.join(generated_path, f"chr{chr_no}_calls.pt")
        t = torch.load(pth, map_location="cpu")
        if torch.is_tensor(t):
            arr = t
        elif isinstance(t, dict) and "calls" in t and torch.is_tensor(t["calls"]):
            arr = t["calls"]
        else:
            raise ValueError(f"Unexpected tensor payload in {pth}")
        return arr.cpu().numpy().astype(np.int8, copy=False)

    payload = torch.load(generated_path, map_location="cpu")
    if torch.is_tensor(payload):
        raise ValueError("Generated .pt is a bare tensor; expected per-chromosome calls.")
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported generated payload type: {type(payload)}")
    for key in ("calls_by_chr", "hard_calls_by_chr"):
        if key in payload and isinstance(payload[key], dict) and chr_no in payload[key]:
            t = payload[key][chr_no]
            if not torch.is_tensor(t):
                raise ValueError(f"{key}[{chr_no}] is not a tensor")
            return t.cpu().numpy().astype(np.int8, copy=False)
    raise KeyError(f"Could not find chr{chr_no} calls inside {generated_path}")


def load_generated_matrix(generated_path: str, chr_no: int, mode: str) -> np.ndarray:
    """
    Unified loader for generated data.

    mode:
      - "unaligned": uses `load_generated_calls()` (int8 calls)
      - "aligned_float": loads chr{c}_aligned_float.pt (float32 dosages) from a directory
      - "aligned_rounded": loads chr{c}_aligned_calls.pt (int8 calls) from a directory
    """
    mode = str(mode)
    if mode == "unaligned":
        return load_generated_calls(generated_path, chr_no)

    if not os.path.isdir(generated_path):
        raise ValueError(f"Aligned mode '{mode}' requires --generated to be a directory.")

    if mode == "aligned_float":
        pth = os.path.join(generated_path, f"chr{chr_no}_aligned_float.pt")
        t = torch.load(pth, map_location="cpu")
        if not torch.is_tensor(t):
            raise ValueError(f"Expected a tensor in {pth}")
        return t.cpu().numpy().astype(np.float32, copy=False)

    if mode == "aligned_rounded":
        pth = os.path.join(generated_path, f"chr{chr_no}_aligned_calls.pt")
        t = torch.load(pth, map_location="cpu")
        if not torch.is_tensor(t):
            raise ValueError(f"Expected a tensor in {pth}")
        return t.cpu().numpy().astype(np.int8, copy=False)

    raise ValueError(f"Unknown generated mode: {mode}")


def _maf_from_calls(calls: np.ndarray) -> np.ndarray:
    # dosage in {0,1,2}, so alt allele freq = mean/2
    return calls.mean(axis=0) / 2.0


def _parse_ld_blocks(det_file: str, min_snps: int = 20, max_blocks: int = 5) -> List[List[str]]:
    blocks: List[List[str]] = []
    with open(det_file, "r") as fh:
        _ = fh.readline()
        for line in fh:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            try:
                nsnps = int(parts[4])
            except ValueError:
                continue
            snp_ids = parts[5].split("|")
            if nsnps >= min_snps and len(snp_ids) == nsnps:
                blocks.append(snp_ids)
            if len(blocks) >= max_blocks:
                break
    return blocks


def _corr_from_calls(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    x = x - x.mean(0, keepdims=True)
    cov = (x.T @ x) / max(1, x.shape[0] - 1)
    std = np.sqrt(np.clip(np.diag(cov), 1e-6, None))
    corr = cov / (std[:, None] * std[None, :])
    return corr


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize decoded generation quality per chromosome.")
    p.add_argument("--orig-h5-root", required=True, help="Root of per-chromosome H5 cache directories.")
    p.add_argument(
        "--generated",
        required=True,
        help="Generated decoded calls: either a directory with chr{c}_calls.pt files, or a .pt payload.",
    )
    p.add_argument(
        "--generated-mode",
        type=str,
        default="unaligned",
        choices=["unaligned", "aligned_float", "aligned_rounded"],
        help="Which generated data to evaluate (unaligned decoded calls, or PCA-aligned float/rounded outputs).",
    )
    p.add_argument("--out-dir", required=True, help="Output directory for per-chromosome figures/stats.")
    p.add_argument("--chromosomes", nargs="*", default=["all"], help="Chromosomes to process, or 'all'.")
    p.add_argument("--bim-path", required=True, help="Path to BIM file for SNP ID mapping (for LD blocks).")
    p.add_argument("--haploblock-dir", required=True, help="Directory containing LD block .det files.")
    p.add_argument("--min-snps-per-block", type=int, default=50, help="Minimum SNPs per LD block.")
    p.add_argument("--max-blocks", type=int, default=5, help="Max LD blocks to plot per chromosome.")
    p.add_argument("--ld-max-rows", type=int, default=2000, help="Max rows (individuals) used for LD heatmaps.")
    p.add_argument("--pca-samples-cap", type=int, default=20000, help="Cap rows used for PCA (speed).")
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    chromosomes = _resolve_chromosomes(args.chromosomes)

    # BIM is one file across all chromosomes; load once for fast SNP ID -> index mapping.
    chr_to_id_to_index = _load_bim_snp_index(args.bim_path)

    for chr_no in chromosomes:
        # --- Load generated and original calls
        calls_gen = load_generated_matrix(args.generated, chr_no, args.generated_mode)
        n_gen = int(calls_gen.shape[0])

        calls_orig_full = _load_one_h5_batch_calls(args.orig_h5_root, chr_no)
        calls_orig = calls_orig_full[:n_gen, :]
        calls_gen = calls_gen[: calls_orig.shape[0], :]

        # --- MAF scatter (per variant)
        maf_gen = _maf_from_calls(calls_gen)
        maf_orig = _maf_from_calls(calls_orig)

        # --- PCA overlay
        X_orig = calls_orig.astype(float)
        X_gen = calls_gen.astype(float)

        if X_orig.shape[0] > args.pca_samples_cap:
            X_orig = X_orig[: args.pca_samples_cap, :]
        if X_gen.shape[0] > args.pca_samples_cap:
            X_gen = X_gen[: args.pca_samples_cap, :]

        aligned_mode = str(args.generated_mode) in ("aligned_float", "aligned_rounded")
        if aligned_mode:
            # For aligned evaluation: normalize using ORIGINAL stats only, fit PCA on ORIGINAL only,
            # then project both original and generated into that PC space.
            mean_orig = X_orig.mean(axis=0, keepdims=True)
            std_orig = X_orig.std(axis=0, keepdims=True)
            std_orig[std_orig == 0] = 1.0
            X_orig_norm = (X_orig - mean_orig) / std_orig
            X_gen_norm = (X_gen - mean_orig) / std_orig

            pca_n = PCA(n_components=2)
            pca_n.fit(X_orig_norm)
            Z_orig_norm = pca_n.transform(X_orig_norm)
            Z_gen_norm = pca_n.transform(X_gen_norm)
        else:
            # Unaligned evaluation: keep the original viz behavior (notebook-inspired):
            # - PCA on combined (unnormalized)
            # - PCA on combined after per-dataset normalization
            X_all = np.concatenate([X_orig, X_gen], axis=0)
            pca = PCA(n_components=2)
            pca.fit(X_all)
            Z_all = pca.transform(X_all)

            mean_orig = X_orig.mean(axis=0, keepdims=True)
            std_orig = X_orig.std(axis=0, keepdims=True)
            std_orig[std_orig == 0] = 1.0
            X_orig_norm = (X_orig - mean_orig) / std_orig

            mean_gen = X_gen.mean(axis=0, keepdims=True)
            std_gen = X_gen.std(axis=0, keepdims=True)
            std_gen[std_gen == 0] = 1.0
            X_gen_norm = (X_gen - mean_gen) / std_gen

            X_all_norm = np.concatenate([X_orig_norm, X_gen_norm], axis=0)
            pca_n = PCA(n_components=2)
            pca_n.fit(X_all_norm)
            Z_all_norm = pca_n.transform(X_all_norm)

        # --- LD blocks + mapping to indices
        det_files = glob.glob(os.path.join(args.haploblock_dir, f"*chr{chr_no}_blocks.blocks.det"))
        block_indices: List[List[int]] = []
        det_file_used: Optional[str] = None
        if det_files:
            det_file_used = sorted(det_files)[0]
            blocks = _parse_ld_blocks(
                det_file_used,
                min_snps=args.min_snps_per_block,
                max_blocks=args.max_blocks,
            )
            id_to_index = chr_to_id_to_index.get(int(chr_no), {})

            for snp_list in blocks:
                idxs = [id_to_index[s] for s in snp_list if s in id_to_index]
                if len(idxs) >= 2:
                    block_indices.append(idxs)

        max_rows = min(args.ld_max_rows, calls_orig.shape[0], calls_gen.shape[0])
        block_true_rows = [calls_orig[:max_rows, idxs] for idxs in block_indices]
        block_gen_rows = [calls_gen[:max_rows, idxs] for idxs in block_indices]

        # --- Combined figure (one file per chromosome)
        nb = max(1, len(block_indices))
        fig = plt.figure(figsize=(15, 8))
        gs = fig.add_gridspec(nrows=3, ncols=max(3, nb), height_ratios=[1.0, 1.2, 1.2])

        ax_maf = fig.add_subplot(gs[0, 0])
        ax_pca_u = fig.add_subplot(gs[0, 1])
        ax_pca_n = fig.add_subplot(gs[0, 2])

        # MAF scatter (based on smoothAE_eval.ipynb)
        ax_maf.scatter(maf_orig, maf_gen, alpha=0.25, s=10)
        ax_maf.plot(
            np.linspace(0, 0.5, 100),
            np.linspace(0, 0.5, 100),
            color="darkred",
            linewidth=2,
            linestyle="--",
        )
        ax_maf.set_xlabel("Original MAF")
        ax_maf.set_ylabel("Generated MAF")
        ax_maf.set_title("MAF Scatter Plot")

        # PCA overlays
        n0 = X_orig.shape[0]
        if aligned_mode:
            # Only normalized PCA; fit on original.
            ax_pca_u.scatter(Z_orig_norm[:, 0], Z_orig_norm[:, 1], label="Original", alpha=0.5, s=10)
            ax_pca_u.scatter(Z_gen_norm[:, 0], Z_gen_norm[:, 1], label="Generated", alpha=0.5, s=10)
            ax_pca_u.set_title("PCA Scatter Plot (normalized; fit on original)")
            ax_pca_u.set_xlabel("PC1")
            ax_pca_u.set_ylabel("PC2")
            ax_pca_u.legend()
            ax_pca_n.axis("off")
        else:
            ax_pca_u.scatter(Z_all[:n0, 0], Z_all[:n0, 1], label="Original", alpha=0.5, s=10)
            ax_pca_u.scatter(Z_all[n0:, 0], Z_all[n0:, 1], label="Generated", alpha=0.5, s=10)
            ax_pca_u.set_title("PCA Scatter Plot (unnormalized)")
            ax_pca_u.set_xlabel("PC1")
            ax_pca_u.set_ylabel("PC2")
            ax_pca_u.legend()

            ax_pca_n.scatter(Z_all_norm[:n0, 0], Z_all_norm[:n0, 1], label="Original", alpha=0.5, s=10)
            ax_pca_n.scatter(Z_all_norm[n0:, 0], Z_all_norm[n0:, 1], label="Generated", alpha=0.5, s=10)
            ax_pca_n.set_title("PCA Scatter Plot (normalized)")
            ax_pca_n.set_xlabel("PC1")
            ax_pca_n.set_ylabel("PC2")
            ax_pca_n.legend()

        # LD heatmaps: 2 rows (orig/gen) x nb blocks
        if len(block_true_rows) > 0:
            vmin, vmax = -1.0, 1.0
            for j, (Xt, Xg) in enumerate(zip(block_true_rows, block_gen_rows)):
                ax_t = fig.add_subplot(gs[1, j])
                ax_g = fig.add_subplot(gs[2, j])
                ct = _corr_from_calls(Xt)
                cg = _corr_from_calls(Xg)
                ax_t.imshow(ct, vmin=vmin, vmax=vmax, cmap="coolwarm")
                ax_g.imshow(cg, vmin=vmin, vmax=vmax, cmap="coolwarm")
                ax_t.set_title(f"LD Block {j + 1} Original")
                ax_g.set_title(f"LD Block {j + 1} Generated")
                ax_t.set_xticks([]); ax_t.set_yticks([])
                ax_g.set_xticks([]); ax_g.set_yticks([])
            fig.suptitle(
                f"Chr{chr_no} quality (LD rows={max_rows}, blocks={len(block_true_rows)}, det={os.path.basename(det_file_used) if det_file_used else 'NA'})"
            )
        else:
            ax_empty = fig.add_subplot(gs[1:, :])
            ax_empty.axis("off")
            ax_empty.text(
                0.5,
                0.5,
                "No LD blocks found / mapped for this chromosome.",
                ha="center",
                va="center",
            )
            fig.suptitle(f"Chr{chr_no} quality")

        fig.tight_layout()
        tag = str(args.generated_mode)
        out_png = os.path.join(args.out_dir, f"chr{chr_no}_quality_{tag}.png")
        fig.savefig(out_png, dpi=args.dpi)
        plt.close(fig)

        # --- Stats (simple, high-signal)
        maf_corr = float(np.corrcoef(maf_orig, maf_gen)[0, 1]) if maf_orig.size > 1 else float("nan")
        maf_mae = float(np.mean(np.abs(maf_orig - maf_gen))) if maf_orig.size else float("nan")

        ld_block_mae: List[float] = []
        for Xt, Xg in zip(block_true_rows, block_gen_rows):
            ct = _corr_from_calls(Xt)
            cg = _corr_from_calls(Xg)
            ld_block_mae.append(float(np.mean(np.abs(ct - cg))))

        stats = {
            "chr": int(chr_no),
            "generated_mode": str(args.generated_mode),
            "n_gen": int(calls_gen.shape[0]),
            "l_chr": int(calls_gen.shape[1]),
            "maf_corr": maf_corr,
            "maf_mae": maf_mae,
            "ld_blocks_used": int(len(ld_block_mae)),
            "ld_block_mae": ld_block_mae,
            "ld_block_mae_mean": float(np.mean(ld_block_mae)) if ld_block_mae else None,
        }
        out_json = os.path.join(args.out_dir, f"chr{chr_no}_stats_{tag}.json")
        with open(out_json, "w") as f:
            json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main()


# python -u /n/home03/ahmadazim/WORKING/genGen/DiffuGene/src/DiffuGene/generate/viz_decoded_quality.py \
#     --orig-h5-root /n/home03/ahmadazim/WORKING/genGen/UKBVQVAE/genomic_data/vqvae_h5_cache \
#     --generated /n/home03/ahmadazim/WORKING/genGen/UKBVQVAE/genomic_data/generated_samples/generated_decoded_unrelWhite_allchr_AE128z_2500_genBatch1/ \
#     --out-dir /n/home03/ahmadazim/WORKING/genGen/UKBVQVAE/genomic_data/generated_samples/generated_decoded_unrelWhite_allchr_AE128z_2500_genBatch1/quality_viz \
#     --bim-path /n/home03/ahmadazim/WORKING/genGen/UKBVQVAE/genomic_data/geneticBinary/ukb_allchr_unrel_britishWhite.bim \
#     --haploblock-dir /n/home03/ahmadazim/WORKING/genGen/UKB6PC/genomic_data/haploblocks/ \
#     --min-snps-per-block 50 \
#     --max-blocks 5 \
#     --ld-max-rows 2500 \
#     --pca-samples-cap 2500 