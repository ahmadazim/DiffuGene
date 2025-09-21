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
import shlex
from typing import List, Tuple

import numpy as np
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
    # Round to nearest genotype and clip to [0,2]; ensure no NaNs remain
    geno = np.rint(geno)
    geno = np.clip(geno, 0, 2).astype(np.float32)
    if geno.shape[0] != n_samples:
        raise ValueError(f"n_samples mismatch in {pt_file}: geno rows {geno.shape[0]} vs n_samples {n_samples}")
    return geno, n_samples


def write_batch_plink(
    batch_idx: int,
    geno: np.ndarray,
    n_samples: int,
    fam_start_row_zero_indexed: int,
    out_prefix: str,
    bim_path_src: str,
    fam_all_path: str,
) -> str:
    # Validate sample dimension only (BIM/FAM handled via shell)
    if geno.shape[0] != n_samples:
        raise ValueError(
            f"Sample count mismatch for batch {batch_idx}: geno has {geno.shape[0]} samples, expected {n_samples}"
        )

    # Skip creation if full PLINK trio already exists
    bed_path = f"{out_prefix}.bed"
    bim_path_out = f"{out_prefix}.bim"
    fam_path_out = f"{out_prefix}.fam"
    if os.path.exists(bed_path) and os.path.exists(bim_path_out) and os.path.exists(fam_path_out):
        return out_prefix

    # 1) Copy BIM via shell
    cp_res = subprocess.run(["cp", "--", bim_path_src, bim_path_out], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if cp_res.returncode != 0:
        raise RuntimeError(
            f"Failed to copy BIM for batch {batch_idx}: {bim_path_src} -> {bim_path_out}\nSTDOUT:\n{cp_res.stdout}\nSTDERR:\n{cp_res.stderr}"
        )

    # 2) Slice FAM via shell (tail/head piping)
    #    fam files are 1-indexed for tail's +N; start line is zero_indexed + 1
    start_line = fam_start_row_zero_indexed + 1
    slice_cmd = (
        f"tail -n +{start_line} {shlex.quote(fam_all_path)} | head -n {n_samples} > {shlex.quote(fam_path_out)}"
    )
    fam_res = subprocess.run(["/bin/bash", "-lc", slice_cmd], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if fam_res.returncode != 0:
        raise RuntimeError(
            f"Failed to slice FAM for batch {batch_idx}: lines {start_line}..{start_line + n_samples - 1}\nSTDOUT:\n{fam_res.stdout}\nSTDERR:\n{fam_res.stderr}"
        )

    # 3) Write only the BED using pandas_plink, referencing the shell-prepared BIM/FAM
    geno = (2.0 - geno).astype(np.float32)
    G = xr.DataArray(
        geno,
        dims=("sample", "variant"),
    )
    write_plink1_bin(G, f"{out_prefix}.bed", bim=bim_path_out, fam=fam_path_out, verbose=False)
    
    # 4) Restore BIM and FAM via shell since writer overwrites them with placeholders
    cp_res_after = subprocess.run(["cp", "--", bim_path_src, bim_path_out], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if cp_res_after.returncode != 0:
        raise RuntimeError(
            f"Failed to re-copy BIM after bed write for batch {batch_idx}: {bim_path_src} -> {bim_path_out}\nSTDOUT:\n{cp_res_after.stdout}\nSTDERR:\n{cp_res_after.stderr}"
        )
    fam_res_after = subprocess.run(["/bin/bash", "-lc", slice_cmd], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if fam_res_after.returncode != 0:
        raise RuntimeError(
            f"Failed to re-slice FAM after bed write for batch {batch_idx}: lines {start_line}..{start_line + n_samples - 1}\nSTDOUT:\n{fam_res_after.stdout}\nSTDERR:\n{fam_res_after.stderr}"
        )
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

    print(f"Found {len(batches)} batches")

    # Generate per-batch PLINK trios
    per_batch_prefixes: List[str] = []
    fam_row_cursor = 0
    for batch_no, pt_path in batches:
        print(f"Loading batch {batch_no} from {pt_path}")
        geno, n_samples = load_decoded_batch(pt_path)
        print(f"Loaded batch {batch_no} with {n_samples} samples")
        # Prepare per-batch outputs without reading/writing FAM/BIM in Python
        start_row = fam_row_cursor
        fam_row_cursor += n_samples

        batch_prefix = os.path.join(batch_dir, f"decoded_batch{batch_no}")
        write_batch_plink(batch_no, geno, n_samples, start_row, batch_prefix, bim_path, fam_all_path)
        print(f"Wrote batch {batch_no} to {batch_prefix}")
        per_batch_prefixes.append(batch_prefix)

    # Merge batches iteratively
    current_prefix = per_batch_prefixes[0]
    for idx in range(1, len(per_batch_prefixes)):
        next_prefix = per_batch_prefixes[idx]
        print(f"Merging batch {idx} from {current_prefix} to {next_prefix}")
        merge_out = os.path.join(batch_dir, f".tmp_merge_{idx}")
        run_plink_merge(plink_bin, current_prefix, next_prefix, merge_out)
        print(f"Merged batch {idx} to {merge_out}")
        current_prefix = merge_out

    # Move final merged to desired output prefix
    for ext in (".bed", ".bim", ".fam", ".log", ".nosex"):
        src = f"{current_prefix}{ext}"
        if os.path.exists(src):
            print(f"Moving final merged to {out_prefix_final}{ext}")
            dst = f"{out_prefix_final}{ext}"
            os.replace(src, dst)

    # Cleanup per-batch PLINK trios and temp merges
    print("Cleaning up per-batch PLINK trios and temp merges")
    for prefix in per_batch_prefixes:
        for ext in (".bed", ".bim", ".fam", ".log", ".nosex"):
            p = f"{prefix}{ext}"
            if os.path.exists(p):
                os.remove(p)
    print("Cleaned up per-batch PLINK trios")
    for idx in range(1, len(per_batch_prefixes)):
        prefix = os.path.join(batch_dir, f".tmp_merge_{idx}")
        for ext in (".bed", ".bim", ".fam", ".log", ".nosex"):
            p = f"{prefix}{ext}"
            if os.path.exists(p):
                os.remove(p)
    print("Cleaned up temp merges")

if __name__ == "__main__":
    main()
