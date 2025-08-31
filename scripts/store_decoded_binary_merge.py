#!/usr/bin/env python
"""
Convert decoded batch .pt files to PLINK bed/bim/fam, then merge them.

Inputs:
- batch directory containing files like <basename>_batch<k>_decoded.pt
- a BIM file corresponding to all variants across blocks
- a FAM file containing all individuals across all batches (in order)
- an output prefix for the final merged PLINK dataset

Assumptions:
- Each decoded .pt file is a dict with keys: 'reconstructed_snps' (list of tensors) and 'n_samples' (int)
- Batches are in order; the first batch's individuals correspond to the first N rows in the provided FAM file, and so on.
- No fallbacks; strict validation is enforced.
"""

import argparse
import os
import re
import subprocess
import shutil
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import xarray as xr
from pandas_plink import write_plink1_bin


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert decoded batch .pt files to PLINK and merge them",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--batch-dir", required=True, type=str, help="Directory with *_batch*_decoded.pt files")
    parser.add_argument("--bim-file", required=True, type=str, help="BIM file matching all variants")
    parser.add_argument("--fam-file", required=True, type=str, help="FAM file containing all individuals across batches")
    parser.add_argument("--output-prefix", required=True, type=str, help="Output prefix for merged PLINK dataset")
    parser.add_argument("--plink", default="plink", type=str, help="Path to PLINK executable")
    return parser.parse_args()


def discover_batches(batch_dir: str) -> List[Tuple[int, str]]:
    files = [f for f in os.listdir(batch_dir) if f.endswith("_decoded.pt")]
    batch_entries = []
    for fname in files:
        m = re.search(r"_batch(\d+)_decoded\.pt$", fname)
        if not m:
            continue
        batch_no = int(m.group(1))
        batch_entries.append((batch_no, os.path.join(batch_dir, fname)))
    if not batch_entries:
        raise FileNotFoundError("No *_batch*_decoded.pt files found in batch directory")
    batch_entries.sort(key=lambda x: x[0])
    return batch_entries


def load_decoded_batch(pt_file: str) -> Tuple[np.ndarray, int]:
    data = torch.load(pt_file, map_location="cpu")
    if not isinstance(data, dict) or "reconstructed_snps" not in data or "n_samples" not in data:
        raise ValueError(f"Decoded batch file has unexpected format: {pt_file}")
    blocks = data["reconstructed_snps"]
    n_samples = int(data["n_samples"])  # type: ignore
    # Concatenate blocks along variant dimension
    geno = np.concatenate([b.numpy() for b in blocks], axis=1)
    # Round to nearest genotype and clip to [0,2]
    geno = np.rint(geno)
    geno = np.clip(geno, 0, 2).astype(np.float32)
    if geno.shape[0] != n_samples:
        raise ValueError(f"n_samples mismatch in {pt_file}: geno rows {geno.shape[0]} vs n_samples {n_samples}")
    return geno, n_samples


def read_bim(bim_path: str) -> pd.DataFrame:
    if not os.path.exists(bim_path):
        raise FileNotFoundError(f"BIM file not found: {bim_path}")
    bim = pd.read_csv(
        bim_path,
        sep=r"\s+",
        header=None,
        names=["chrom", "snp", "cm", "pos", "a1", "a2"],
    )
    return bim


def read_fam_all(fam_path: str) -> pd.DataFrame:
    if not os.path.exists(fam_path):
        raise FileNotFoundError(f"FAM file not found: {fam_path}")
    fam = pd.read_csv(
        fam_path,
        sep=r"\s+",
        header=None,
        names=["fid", "iid", "pid", "mid", "sex", "phen"],
    )
    return fam


def write_batch_plink(
    batch_idx: int,
    geno: np.ndarray,
    bim: pd.DataFrame,
    fam_slice: pd.DataFrame,
    out_prefix: str,
    bim_path_src: str,
) -> str:
    # Validate shapes
    if geno.shape[1] != len(bim):
        raise ValueError(
            f"Variant count mismatch for batch {batch_idx}: geno has {geno.shape[1]} variants, BIM has {len(bim)}"
        )
    if geno.shape[0] != len(fam_slice):
        raise ValueError(
            f"Sample count mismatch for batch {batch_idx}: geno has {geno.shape[0]} samples, FAM slice has {len(fam_slice)}"
        )

    # Skip creation if full PLINK trio already exists
    bed_path = f"{out_prefix}.bed"
    bim_path_out = f"{out_prefix}.bim"
    fam_path_out = f"{out_prefix}.fam"
    if os.path.exists(bed_path) and os.path.exists(bim_path_out) and os.path.exists(fam_path_out):
        return out_prefix

    # Persist per-batch FAM slice
    fam_path_tmp = fam_path_out
    # Ensure strict column order and types for PLINK .fam
    fam_to_write = fam_slice[["fid","iid","pid","mid","sex","phen"]].copy()
    # Coerce types minimally (fid/iid string; sex/phen numeric as-is)
    fam_to_write["fid"] = fam_to_write["fid"].astype(str)
    fam_to_write["iid"] = fam_to_write["iid"].astype(str)
    fam_to_write.to_csv(fam_path_tmp, sep=" ", header=False, index=False)

    # Build xarray DataArray with coordinates matching fam/bim
    G = xr.DataArray(
        geno,
        dims=("sample", "variant"),
        coords={
            "fid": ("sample", fam_slice["fid"].values),
            "iid": ("sample", fam_slice["iid"].values),
            "snp": ("variant", bim["snp"].values),
            "chrom": ("variant", bim["chrom"].values),
            "pos": ("variant", bim["pos"].values),
            "a1": ("variant", bim["a1"].values),
            "a2": ("variant", bim["a2"].values),
        },
    )

    # Write PLINK1 bed/bim/fam trio using provided BIM path and per-batch FAM
    write_plink1_bin(G, f"{out_prefix}.bed", bim=bim_path_src, fam=fam_path_tmp, verbose=False)
    # Copy the provided BIM exactly; do not alter contents or formatting
    dst_bim = f"{out_prefix}.bim"
    if not os.path.exists(dst_bim):
        shutil.copy2(bim_path_src, dst_bim)
    return out_prefix


def run_plink_merge(plink: str, base_prefix: str, merge_prefix: str, out_prefix: str):
    cmd = [
        plink,
        "--bfile", base_prefix,
        "--bmerge", f"{merge_prefix}.bed", f"{merge_prefix}.bim", f"{merge_prefix}.fam",
        "--allow-no-sex",
        "--make-bed",
        "--out", out_prefix,
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"PLINK merge failed: {' '.join(cmd)}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")


def main():
    args = parse_args()

    batch_dir = os.path.abspath(args.batch_dir)
    out_prefix_final = os.path.abspath(args.output_prefix)
    global bim_path
    bim_path = os.path.abspath(args.bim_file)
    fam_all_path = os.path.abspath(args.fam_file)
    plink_bin = args.plink

    batches = discover_batches(batch_dir)
    bim = read_bim(bim_path)
    fam_all = read_fam_all(fam_all_path)

    # Generate per-batch PLINK trios
    per_batch_prefixes: List[str] = []
    fam_row_cursor = 0
    for batch_no, pt_path in batches:
        geno, n_samples = load_decoded_batch(pt_path)
        fam_slice = fam_all.iloc[fam_row_cursor : fam_row_cursor + n_samples]
        if len(fam_slice) != n_samples:
            raise ValueError(
                f"Insufficient FAM rows for batch {batch_no}: needed {n_samples}, have {len(fam_slice)}"
            )
        fam_row_cursor += n_samples

        batch_prefix = os.path.join(batch_dir, f"decoded_batch{batch_no}")
        write_batch_plink(batch_no, geno, bim, fam_slice, batch_prefix, bim_path)
        per_batch_prefixes.append(batch_prefix)

    # Merge batches iteratively
    current_prefix = per_batch_prefixes[0]
    for idx in range(1, len(per_batch_prefixes)):
        next_prefix = per_batch_prefixes[idx]
        merge_out = os.path.join(batch_dir, f".tmp_merge_{idx}")
        run_plink_merge(plink_bin, current_prefix, next_prefix, merge_out)
        current_prefix = merge_out

    # Move final merged to desired output prefix
    for ext in (".bed", ".bim", ".fam", ".log", ".nosex"):
        src = f"{current_prefix}{ext}"
        if os.path.exists(src):
            dst = f"{out_prefix_final}{ext}"
            os.replace(src, dst)

    # Cleanup per-batch PLINK trios and temp merges
    for prefix in per_batch_prefixes:
        for ext in (".bed", ".bim", ".fam", ".log", ".nosex"):
            p = f"{prefix}{ext}"
            if os.path.exists(p):
                os.remove(p)
    for idx in range(1, len(per_batch_prefixes)):
        prefix = os.path.join(batch_dir, f".tmp_merge_{idx}")
        for ext in (".bed", ".bim", ".fam", ".log", ".nosex"):
            p = f"{prefix}{ext}"
            if os.path.exists(p):
                os.remove(p)


if __name__ == "__main__":
    main()
