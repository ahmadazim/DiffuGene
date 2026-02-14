#!/usr/bin/env python3
"""
Write a single PLINK1 binary dataset (bed/bim/fam) from per-chromosome .pt call tensors.

Inputs:
  --pt-dir      directory containing chr{1..22}_calls.pt (ignores chr*_logits.pt)
  --bim-all     BIM for ALL variants across all chromosomes, in the SAME order as concatenation of chr calls
  --fam         FAM for individuals (N rows), in the SAME order as tensors
  --out-prefix  output prefix for .bed/.bim/.fam

Assumptions (STRICT):
  - Each chr{c}_calls.pt or chr{c}_aligned_calls.pt is a tensor or dict with key "calls", shape [N, p_c], genotypes in {0,1,2}
  - All chromosomes have identical N and the intended variant order matches the provided --bim-all.
  - Concatenating chr1..chr22 along variants gives total variants == number of lines in --bim-all.

This script:
  - loads chr*_calls.pt
  - concatenates into one [N, P] matrix
  - writes PLINK bed using pandas_plink.write_plink1_bin
  - then FORCE-copies the provided BIM/FAM into place (writer may overwrite placeholders)
"""

from __future__ import annotations

import argparse
import os
import subprocess
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
import xarray as xr
import re
import pandas as pd
from pandas_plink import write_plink1_bin


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert per-chr .pt calls to a single PLINK bed/bim/fam.")
    p.add_argument("--pt-dir", required=True, help="Directory with chr*_calls.pt files.")
    p.add_argument("--bim", required=True, help="BIM for ALL variants (all chromosomes) in correct order, to be copied as is.")
    p.add_argument("--out-prefix", required=True, help="Output prefix for PLINK files.")
    p.add_argument("--chromosomes", default="all", help="Chromosomes to include, e.g. 'all', '1-22', '1,2,3,22'.")
    p.add_argument("--covar-csv", required=True, help="Master covariate CSV containing sex for all individuals across batches.")
    return p.parse_args()


def parse_chr_spec(spec: str) -> List[int]:
    spec = spec.strip().lower()
    if spec == "all":
        return list(range(1, 23))
    out: List[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            a_i, b_i = int(a), int(b)
            out.extend(list(range(a_i, b_i + 1)))
        else:
            out.append(int(part))
    out = sorted(set(out))
    return out


def count_lines(path: str) -> int:
    n = 0
    with open(path, "r") as f:
        for _ in f:
            n += 1
    return n


def _load_calls_pt(path: str) -> np.ndarray:
    obj = torch.load(path, map_location="cpu")
    if torch.is_tensor(obj):
        t = obj
    elif isinstance(obj, dict):
        if "calls" in obj and torch.is_tensor(obj["calls"]):
            t = obj["calls"]
        else:
            raise ValueError(f"Unsupported dict format in {path} (expected tensor or dict with key 'calls').")
    else:
        raise ValueError(f"Unsupported payload type in {path}: {type(obj)}")

    X = t.detach().cpu().numpy()
    if X.ndim != 2:
        raise ValueError(f"{path}: expected 2D [N,p] calls tensor, got {X.shape}")
    if not np.issubdtype(X.dtype, np.integer):
        X = np.rint(X)
    X = np.clip(X, 0, 2).astype(np.float32, copy=False)
    # hard validation: values must be in {0,1,2}
    u = np.unique(X)
    if u.size > 3 or np.any((u < 0) | (u > 2)) or not np.all(np.isin(u, [0.0, 1.0, 2.0])):
        raise ValueError(f"{path}: calls have unexpected values {u[:10]}")
    return X

def _infer_prefix_digit(name: str) -> int:
    """
    FID/IID first digit convention:
      - contains 'synObs_decoded_aligned'         -> 1
      - contains 'synObs_decoded' (no _aligned)   -> 2
      - contains 'generated_decoded_aligned'      -> 3
      - contains 'generated_decoded' (no _aligned)-> 4
    """
    s = name
    if "synObs_decoded_aligned" in s:
        return 1
    if "synObs_decoded" in s and "_aligned" not in s:
        return 2
    if "generated_decoded_aligned" in s:
        return 3
    if "generated_decoded" in s and "_aligned" not in s:
        return 4
    raise ValueError(
        f"Could not infer FID/IID prefix digit from name='{name}'. "
        "Expected one of: synObs_decoded(_aligned) or generated_decoded(_aligned)."
    )

def _infer_genbatch(name: str) -> int:
    """
    Parse genBatch{n} from directory/prefix name. Defaults to 1 if absent.
    """
    m = re.search(r"genBatch(\d+)", name)
    return int(m.group(1)) if m else 1

def _to_plink_sex(v) -> int:
    """
    Convert common encodings to PLINK sex codes: 1=male, 2=female, 0=unknown.
    Accepts:
      - already {0,1,2}
      - {0,1} (assume 1=male, 0=female)  [common in UKB fields]
      - strings starting with 'M'/'F'
    """
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return 0
    if isinstance(v, (np.integer, int)):
        iv = int(v)
        if iv in (0, 1, 2):
            return iv
        # fallthrough
    if isinstance(v, (np.floating, float)):
        iv = int(v)
        if iv in (0, 1, 2):
            return iv
        # fallthrough
    if isinstance(v, str):
        s = v.strip().lower()
        if not s:
            return 0
        if s.startswith("m"):
            return 1
        if s.startswith("f"):
            return 2
        # fallthrough
    # heuristic for {0,1} coded as floats/ints but not caught above
    try:
        iv = int(v)
        if iv in (0, 1):
            return 1 if iv == 1 else 2
    except Exception:
        pass
    return 0


def _write_fam_from_master(
    fam_path_out: str,
    covar_csv: str,
    prefix_digit: int,
    genbatch: int,
    n_samples: int,
) -> None:
    """
    Writes a PLINK .fam file with FID/IID as 7-digit numbers:
      FID=IID = prefix_digit*1_000_000 + offset
      offset runs from n_samples*(genbatch-1)+1 ... n_samples*genbatch

    Sex is taken from covar CSV by selecting rows:
      start_row = n_samples*(genbatch-1)
      end_row   = n_samples*genbatch - 1
    """
    df = pd.read_csv(covar_csv)
    sex_c = "SEX_MALE"

    start = n_samples * (genbatch - 1)
    end_excl = n_samples * genbatch
    if start < 0 or end_excl > len(df):
        raise ValueError(
            f"covar CSV too short for genbatch={genbatch}, n_samples={n_samples}. "
            f"Need rows [{start}, {end_excl}) but covar has {len(df)} rows."
        )
    sex_vals = df.iloc[start:end_excl][sex_c].tolist()
    if len(sex_vals) != n_samples:
        raise RuntimeError("Internal error: sex slice length mismatch.")

    # IDs: 7-digit integer with first digit fixed by prefix_digit.
    # Example: prefix_digit=4, offset=1 -> 4000001
    offsets = np.arange(start=1, stop=n_samples + 1, dtype=np.int64) + n_samples * (genbatch - 1)
    ids = (prefix_digit * 1_000_000 + offsets).astype(np.int64)

    # Write FAM: FID IID PID MID SEX PHENO
    # PID/MID unknown -> 0; PHENO missing -> -9
    with open(fam_path_out, "w") as f:
        for i in range(n_samples):
            sid = str(int(ids[i]))
            sex_plink = _to_plink_sex(sex_vals[i])
            f.write(f"{sid} {sid} 0 0 {sex_plink} -9\n")


def main() -> None:
    args = parse_args()
    pt_dir = os.path.abspath(args.pt_dir)
    out_prefix = os.path.abspath(args.out_prefix)
    covar_csv = os.path.abspath(args.covar_csv)
    bim_src = os.path.abspath(args.bim)

    # Write BED using pandas_plink
    bed_path = f"{out_prefix}.bed"
    bim_path = f"{out_prefix}.bim"
    fam_path = f"{out_prefix}.fam"
    fam_tmp_path = fam_path + ".tmp"

    chrs = parse_chr_spec(args.chromosomes)

    # Determine naming-based conventions from the pt-dir basename (most stable signal)
    pt_name = os.path.basename(os.path.normpath(pt_dir))
    prefix_digit = _infer_prefix_digit(pt_name)
    genbatch = _infer_genbatch(pt_name)

    # Load per-chr calls and concatenate
    X_parts: List[np.ndarray] = []
    N_ref: int = -1
    total_p = 0

    for c in chrs:
        pth = os.path.join(pt_dir, f"chr{c}_calls.pt")
        if not os.path.exists(pth):
            pth = os.path.join(pt_dir, f"chr{c}_aligned_calls.pt")
            if not os.path.exists(pth):
                msg = f"Missing {pth}"
                raise FileNotFoundError(msg)
                # print(f"[warn] {msg}; skipping chr{c}")
                # continue

        Xc = _load_calls_pt(pth)  # float32 [N,p_c] in {0,1,2}
        if N_ref < 0:
            N_ref = int(Xc.shape[0])
        if int(Xc.shape[0]) != N_ref:
            raise ValueError(f"Sample mismatch: chr{c} has N={Xc.shape[0]} vs expected N={N_ref}")

        X_parts.append(Xc)
        total_p += int(Xc.shape[1])
        print(f"[chr{c}] loaded calls: N={Xc.shape[0]}, p={Xc.shape[1]}")

    if not X_parts:
        raise FileNotFoundError(f"No chr*_calls.pt loaded from {pt_dir} with spec {args.chromosomes}")

    X = np.concatenate(X_parts, axis=1).astype(np.float32, copy=False)  # [N, P]
    N, P = int(X.shape[0]), int(X.shape[1])

    # Validate BIM/FAM sizes
    n_bim = count_lines(bim_src)
    if n_bim != P:
        raise ValueError(f"BIM variant count mismatch: BIM has {n_bim} lines, but concatenated calls have P={P}")
    
    # pandas_plink expects allele counts for A1? Many pipelines invert 0/1/2; keep consistent with earlier scripts
    X_plink = (2.0 - X).astype(np.float32, copy=False)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)

    # 1) Create output FAM that pandas_plink will use while writing
    _write_fam_from_master(
        fam_path_out=fam_tmp_path,
        covar_csv=covar_csv,
        prefix_digit=prefix_digit,
        genbatch=genbatch,
        n_samples=N,
    )

    # 2) Ensure output BIM is present at the right location before writing.
    cp_bim = subprocess.run(["cp", "--", bim_src, bim_path], capture_output=True, text=True)
    if cp_bim.returncode != 0:
        raise RuntimeError(f"Failed to copy BIM -> {bim_path}\nSTDERR:\n{cp_bim.stderr}")

    # 3) Write PLINK trio at out_prefix (we control exact bim/fam paths)
    G = xr.DataArray(X_plink, dims=("sample", "variant"))
    write_plink1_bin(G, bed_path, bim=bim_path, fam=fam_path, verbose=False)
    
    # 4) Overwrite final BIM and FAM with our convention via mv/replace (avoid rewriting twice)
    os.replace(fam_tmp_path, fam_path)
    cp_bim = subprocess.run(["cp", "--", bim_src, bim_path], capture_output=True, text=True)
    if cp_bim.returncode != 0:
        raise RuntimeError(f"Failed to copy BIM -> {bim_path}\nSTDERR:\n{cp_bim.stderr}")

    print(f"[done] Wrote PLINK files: {out_prefix}.bed/.bim/.fam")
    print(f"       N={N}, P={P}, chromosomes={','.join(map(str, chrs))}")
    print(f"       FID/IID prefix_digit={prefix_digit}, genbatch={genbatch} (from pt-dir name='{pt_name}')")


if __name__ == "__main__":
    main()