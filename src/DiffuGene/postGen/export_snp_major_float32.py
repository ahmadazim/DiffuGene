#!/usr/bin/env python3
"""
Export per-chromosome aligned dosage floats into raw SNP-major float32 matrices.

Input expectation:
  - A directory containing per-batch per-chromosome aligned float files from
    `pcaProcrustes_align.py`, typically named like:
        <prefix>batch3chr10_aligned_float.pt
    but `.npy` is also supported.

Output per chromosome:
  - <output-prefix>chr{c}_snp_major.float32.bin
      Raw float32 bytes storing a matrix with logical shape (n_snps, n_indiv)
      in C row-major order. Each row is one SNP, each column one individual.
  - <output-prefix>chr{c}_snp_major.metadata.json
      Sidecar metadata describing the matrix and variant IDs.

Batch order is STRICT:
  - batches are concatenated in the exact order supplied by --batch-numbers
  - if --batch-numbers=all, the exporter discovers the common batch numbers
    across chromosomes and uses them in ascending numeric order
"""

from __future__ import annotations

import argparse
import fnmatch
import glob
import json
import os
import re
import sys
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

try:
    from ..utils.file_utils import read_bim_file
except ImportError:
    this_dir = os.path.dirname(__file__)
    src_root = os.path.abspath(os.path.join(this_dir, "..", "..", ".."))
    if src_root not in sys.path:
        sys.path.insert(0, src_root)
    from DiffuGene.utils.file_utils import read_bim_file


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export aligned dosage floats to raw SNP-major float32 matrices.")
    p.add_argument("--input-dir", required=True, help="Directory containing aligned float files (.pt or .npy).")
    p.add_argument("--bim", required=True, help="PLINK BIM file used to recover per-chromosome variant IDs.")
    p.add_argument("--output-dir", required=True, help="Directory to write per-chromosome raw matrices and metadata.")
    p.add_argument(
        "--output-prefix",
        default="",
        help="Optional prefix for output filenames. Example: run1_ -> run1_chr22_snp_major.float32.bin",
    )
    p.add_argument(
        "--file-prefix-pattern",
        default=None,
        help=(
            "Optional basename prefix or glob-style pattern used to select one run when the input "
            "directory contains multiple matching files, e.g. 'generated_decoded_aligned_USiT_H_wNC_H_*'."
        ),
    )
    p.add_argument("--chromosomes", default="all", help="Chromosomes to include, e.g. 'all', '1-22', '1,2,3,22'.")
    p.add_argument(
        "--batch-numbers",
        default="all",
        help="Batches to merge in order, e.g. 'all', '1-5', '1,2,5'. Batch 1 is written first, then batch 2, etc.",
    )
    return p.parse_args()


def parse_int_spec(spec: str, default_all: Sequence[int]) -> List[int]:
    spec = str(spec).strip().lower()
    if spec == "all":
        return list(default_all)
    out: List[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            a_i, b_i = int(a), int(b)
            step = 1 if b_i >= a_i else -1
            out.extend(list(range(a_i, b_i + step, step)))
        else:
            out.append(int(part))
    return sorted(set(out))


def parse_chr_spec(spec: str) -> List[int]:
    return parse_int_spec(spec, default_all=list(range(1, 23)))


def _matches_prefix_pattern(name: str, pattern: str) -> bool:
    if any(ch in pattern for ch in "*?[]"):
        return fnmatch.fnmatch(name, pattern)
    return name.startswith(pattern)


def _extract_batch_number(path: str) -> Optional[int]:
    name = os.path.basename(path)
    m = re.search(r"batch0*(\d+)", name, flags=re.IGNORECASE)
    if m is not None:
        return int(m.group(1))
    return None


def _candidate_float_files(input_dir: str, chr_no: int, file_prefix_pattern: Optional[str]) -> List[str]:
    patterns = [
        os.path.join(input_dir, f"*chr{chr_no}_aligned_float.pt"),
        os.path.join(input_dir, f"*chr{chr_no}_aligned_float.npy"),
    ]
    candidates: List[str] = []
    for pattern in patterns:
        candidates.extend(sorted(glob.glob(pattern)))

    # Also accept direct non-prefixed names if present.
    direct_pt = os.path.join(input_dir, f"chr{chr_no}_aligned_float.pt")
    direct_npy = os.path.join(input_dir, f"chr{chr_no}_aligned_float.npy")
    for direct in (direct_pt, direct_npy):
        if os.path.exists(direct):
            candidates.append(direct)

    # deduplicate
    candidates = sorted(set(candidates))

    if file_prefix_pattern:
        candidates = [
            path for path in candidates
            if _matches_prefix_pattern(os.path.basename(path), file_prefix_pattern)
        ]
    return candidates


def _discover_batches_for_chr(input_dir: str, chr_no: int, file_prefix_pattern: Optional[str]) -> Dict[int, str]:
    candidates = _candidate_float_files(input_dir, chr_no, file_prefix_pattern)
    if not candidates:
        pattern_msg = f" with prefix pattern '{file_prefix_pattern}'" if file_prefix_pattern else ""
        raise FileNotFoundError(
            f"No aligned float files found for chr{chr_no} in {input_dir}{pattern_msg}. "
            "Expected files like '*chr{chr}_aligned_float.pt' or '.npy'."
        )

    mapping: Dict[int, str] = {}
    unbatched: List[str] = []
    for path in candidates:
        batch_no = _extract_batch_number(path)
        if batch_no is None:
            unbatched.append(path)
            continue
        if batch_no in mapping:
            raise ValueError(
                f"Multiple files found for chr{chr_no}, batch {batch_no}: "
                f"{os.path.basename(mapping[batch_no])}, {os.path.basename(path)}"
            )
        mapping[batch_no] = path

    if unbatched:
        if mapping:
            raise ValueError(
                f"Found a mix of batched and unbatched files for chr{chr_no}: "
                f"{[os.path.basename(p) for p in unbatched[:5]]}"
            )
        if len(unbatched) == 1:
            mapping[1] = unbatched[0]
        else:
            raise ValueError(
                f"Multiple unbatched aligned float files found for chr{chr_no}: "
                f"{[os.path.basename(p) for p in unbatched[:10]]}. "
                "Add batch numbers to filenames or use --file-prefix-pattern to isolate one run."
            )

    return mapping


def _load_float_matrix(path: str) -> np.ndarray:
    suffix = os.path.splitext(path)[1].lower()
    if suffix == ".npy":
        X = np.load(path)
    elif suffix == ".pt":
        obj = torch.load(path, map_location="cpu")
        if torch.is_tensor(obj):
            X = obj.detach().cpu().numpy()
        elif isinstance(obj, dict):
            tensor_keys = ("aligned_float", "dosage", "data", "values", "float")
            found = None
            for key in tensor_keys:
                value = obj.get(key)
                if torch.is_tensor(value):
                    found = value.detach().cpu().numpy()
                    break
                if isinstance(value, np.ndarray):
                    found = value
                    break
            if found is None:
                raise ValueError(
                    f"Unsupported dict payload in {path}. Expected a tensor or one of keys {tensor_keys}."
                )
            X = found
        else:
            raise ValueError(f"Unsupported .pt payload in {path}: {type(obj)}")
    else:
        raise ValueError(f"Unsupported file extension for {path}; expected .pt or .npy")

    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"{path}: expected 2D array [N, p_chr], got shape {X.shape}")
    return X


def _discover_common_batches(
    input_dir: str,
    chromosomes: Sequence[int],
    file_prefix_pattern: Optional[str],
) -> Dict[int, Dict[int, str]]:
    per_chr: Dict[int, Dict[int, str]] = {}
    common_batches: Optional[set[int]] = None
    for chr_no in chromosomes:
        mapping = _discover_batches_for_chr(input_dir, chr_no, file_prefix_pattern)
        per_chr[int(chr_no)] = mapping
        batch_set = set(mapping.keys())
        common_batches = batch_set if common_batches is None else (common_batches & batch_set)

    if not common_batches:
        raise ValueError("No common batch numbers were found across the selected chromosomes.")

    return {
        chr_no: {batch_no: path for batch_no, path in mapping.items() if batch_no in common_batches}
        for chr_no, mapping in per_chr.items()
    }


def _resolve_batch_order(
    discovered: Dict[int, Dict[int, str]],
    chromosomes: Sequence[int],
    batch_spec: str,
) -> List[int]:
    common_batches = sorted(set.intersection(*(set(discovered[chr_no].keys()) for chr_no in chromosomes)))
    requested = parse_int_spec(batch_spec, default_all=common_batches)
    missing: Dict[int, List[int]] = {}
    for chr_no in chromosomes:
        avail = set(discovered[chr_no].keys())
        miss = [b for b in requested if b not in avail]
        if miss:
            missing[int(chr_no)] = miss
    if missing:
        raise ValueError(f"Requested batch numbers missing for some chromosomes: {missing}")
    return requested


def export_chromosome(
    chr_no: int,
    batch_files: Dict[int, str],
    batch_order: Sequence[int],
    bim_file: str,
    output_dir: str,
    output_prefix: str,
) -> None:
    bim_chr = read_bim_file(bim_file, int(chr_no))
    variant_ids = bim_chr["SNP"].astype(str).tolist()
    n_snps = len(variant_ids)
    if n_snps == 0:
        raise ValueError(f"No variants found in BIM for chr{chr_no}")

    n_total_indiv = 0
    n_batch_rows: Dict[int, int] = {}
    source_files: Dict[int, str] = {}
    for batch_no in batch_order:
        path = batch_files[int(batch_no)]
        X = _load_float_matrix(path)
        if int(X.shape[1]) != n_snps:
            raise ValueError(
                f"chr{chr_no} batch {batch_no}: file has p={X.shape[1]} SNPs, but BIM has {n_snps} variants"
            )
        n_batch_rows[int(batch_no)] = int(X.shape[0])
        n_total_indiv += int(X.shape[0])
        source_files[int(batch_no)] = os.path.basename(path)

    raw_path = os.path.join(output_dir, f"{output_prefix}chr{chr_no}_snp_major.float32.bin")
    meta_path = os.path.join(output_dir, f"{output_prefix}chr{chr_no}_snp_major.metadata.json")

    # Raw float32 bytes, C-order, logical shape (n_snps, n_indiv).
    dst = np.memmap(raw_path, dtype="float32", mode="w+", shape=(n_snps, n_total_indiv), order="C")

    col_start = 0
    for batch_no in batch_order:
        path = batch_files[int(batch_no)]
        X = _load_float_matrix(path)  # [N, p_chr]
        n_rows = int(X.shape[0])
        col_end = col_start + n_rows
        # Input is individual-major [N, p]. Output is SNP-major [p, N].
        dst[:, col_start:col_end] = np.asarray(X.T, dtype=np.float32, order="C")
        col_start = col_end
        print(
            f"[chr{chr_no}] wrote batch {batch_no} from {os.path.basename(path)} "
            f"into columns [{col_start - n_rows}, {col_end})"
        )

    dst.flush()
    del dst

    metadata = {
        "chromosome": int(chr_no),
        "n_snps": int(n_snps),
        "n_indiv": int(n_total_indiv),
        "dtype": "float32",
        "shape": [int(n_snps), int(n_total_indiv)],
        "storage_order": "C row-major raw binary",
        "logical_layout": "SNP-major: rows are SNPs, columns are individuals",
        "individual_order": "batch-major; batches concatenated in listed order, preserving within-batch row order",
        "batch_order": [int(b) for b in batch_order],
        "batch_sample_counts": {str(k): int(v) for k, v in n_batch_rows.items()},
        "source_files": {str(k): v for k, v in source_files.items()},
        "variant_ids": variant_ids,
        "raw_file": os.path.basename(raw_path),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[chr{chr_no}] exported raw float32 matrix -> {raw_path}")
    print(f"[chr{chr_no}] metadata -> {meta_path}")


def main() -> None:
    args = parse_args()
    input_dir = os.path.abspath(os.path.expanduser(args.input_dir))
    output_dir = os.path.abspath(os.path.expanduser(args.output_dir))
    bim_file = os.path.abspath(os.path.expanduser(args.bim))

    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"--input-dir does not exist or is not a directory: {input_dir}")
    if not os.path.exists(bim_file):
        raise FileNotFoundError(f"--bim does not exist: {bim_file}")
    os.makedirs(output_dir, exist_ok=True)

    chromosomes = parse_chr_spec(args.chromosomes)
    discovered = _discover_common_batches(input_dir, chromosomes, args.file_prefix_pattern)
    batch_order = _resolve_batch_order(discovered, chromosomes, args.batch_numbers)

    print(f"[INFO] input_dir={input_dir}")
    print(f"[INFO] output_dir={output_dir}")
    print(f"[INFO] bim={bim_file}")
    print(f"[INFO] chromosomes={chromosomes}")
    print(f"[INFO] batch_order={batch_order}")
    print(f"[INFO] file_prefix_pattern={args.file_prefix_pattern}")
    print(f"[INFO] output_prefix={args.output_prefix}")

    for chr_no in chromosomes:
        export_chromosome(
            chr_no=int(chr_no),
            batch_files=discovered[int(chr_no)],
            batch_order=batch_order,
            bim_file=bim_file,
            output_dir=output_dir,
            output_prefix=str(args.output_prefix),
        )

    print("[DONE] Exported SNP-major float32 matrices for all requested chromosomes.")


if __name__ == "__main__":
    main()
