#!/usr/bin/env python

import os
import sys
import argparse
import subprocess
from typing import List, Dict

import numpy as np
import pandas as pd
import h5py

from ..utils import ensure_dir_exists, get_logger
from ..utils.file_utils import read_bim_file


logger = get_logger(__name__)


def chunk_iids(fam_path: str, batch_size: int) -> List[pd.DataFrame]:
    fam_cols = ["FID", "IID", "PAT", "MAT", "SEX", "PHENOTYPE"]
    fam = pd.read_csv(fam_path, sep=r"\s+", header=None, names=fam_cols)
    chunks = []
    for s in range(0, len(fam), batch_size):
        e = min(s + batch_size, len(fam))
        chunks.append(fam.iloc[s:e, :2].copy())
    return chunks


def run_plink_recode_batch(bfile_prefix: str, chromosomes: List[int] | None, keep_tsv: str, out_prefix: str) -> str:
    # Build command ensuring option-argument pairs stay together
    cmd = ["plink", "--bfile", bfile_prefix]
    if chromosomes is not None and len(chromosomes) > 0:
        # Pass each chromosome as a separate arg after --chr (preferred by PLINK)
        cmd += ["--chr"] + [str(c) for c in chromosomes]
    cmd += [
        "--keep", keep_tsv,
        "--recode", "A",
        "--out", out_prefix,
    ]
    logger.info(f"Running: {' '.join(cmd)}")
    env = os.environ.copy()
    env.setdefault("LC_ALL", "C")
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, env=env)
    raw_path = f"{out_prefix}.raw"
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"PLINK .raw not found: {raw_path}")
    return raw_path


def _read_header(raw_path: str) -> List[str]:
    with open(raw_path, 'r') as f:
        header = f.readline().strip().split()
    return header


def write_raw_to_h5_fast(raw_path: str,
                         h5_path: str,
                         expected_rows: int,
                         bp: np.ndarray,
                         snp_ids: List[str],
                         chunk_rows: int = 10000) -> None:
    """
    Load the .raw file in row chunks using pandas and write to H5 efficiently on CPU.
    Replaces string 'NA' with 0 and casts to int8.
    """
    header = _read_header(raw_path)
    L_raw = len(header) - 6
    if L_raw <= 0:
        raise ValueError(f"Unexpected .raw format: no SNP columns in {raw_path}")
    L = min(L_raw, int(bp.shape[0]))
    if L_raw != bp.shape[0]:
        logger.warning(f"SNP count mismatch .raw={L_raw} vs BIM {bp.shape[0]}; using min={L}")
    ensure_dir_exists(os.path.dirname(h5_path))
    with h5py.File(h5_path, "w") as f:
        dset_X = f.create_dataset("X", shape=(expected_rows, L), dtype='i1', compression="gzip", compression_opts=4)
        dset_iid = f.create_dataset("iid", shape=(expected_rows,), dtype=h5py.string_dtype(encoding='utf-8'))
        f.create_dataset("bp", data=bp[:L])
        try:
            f.create_dataset("snp_ids", data=np.array(snp_ids[:L], dtype=object), dtype=h5py.string_dtype(encoding='utf-8'))
        except Exception:
            pass
        offset = 0
        usecols = list(range(0, 2)) + list(range(6, 6 + L))
        for df in pd.read_csv(raw_path, sep=r"\s+", header=0, usecols=usecols, chunksize=chunk_rows, na_values=['NA']):
            iids = df.iloc[:, 1].astype(str).to_numpy()
            snp_df = df.iloc[:, 2:]
            # Replace NaN with 0 and cast to int8 in a vectorized way
            X_chunk = snp_df.fillna(0).to_numpy(dtype=np.int16, copy=False)
            X_chunk = np.asarray(X_chunk, dtype=np.int8)
            n = X_chunk.shape[0]
            dset_X[offset:offset+n, :L] = X_chunk
            dset_iid[offset:offset+n] = iids
            offset += n
        if offset != expected_rows:
            logger.warning(f"Row count mismatch for {raw_path}: expected {expected_rows}, wrote {offset}")


def h5_is_complete(h5_path: str, expected_rows: int, expected_cols: int) -> bool:
    try:
        if not os.path.exists(h5_path):
            return False
        with h5py.File(h5_path, 'r') as f:
            if 'X' not in f or 'iid' not in f or 'bp' not in f:
                return False
            shape_ok = f['X'].shape == (expected_rows, expected_cols)
            iid_ok = f['iid'].shape == (expected_rows,)
            bp_ok = f['bp'].shape == (expected_cols,)
            return shape_ok and iid_ok and bp_ok
    except Exception:
        return False


def temp_files_exist(out_prefix: str) -> bool:
    for ext in ('.raw', '.log', '.nosex'):
        if os.path.exists(out_prefix + ext):
            return True
    return False

def build_snp_info(bim_file: str, chromosomes: List[int]) -> tuple:
    """Return SNP ids, BP vector, and per-chromosome column offsets for concatenation."""
    all_ids: List[str] = []
    all_bp = []
    chr_offsets = {}
    col_start = 0
    for chr_no in chromosomes:
        bim_chr = read_bim_file(bim_file, chr_no)
        snp_ids_chr = bim_chr["SNP"].astype(str).tolist()
        bp_chr = bim_chr["BP"].astype(np.int64).values
        all_ids.extend(snp_ids_chr)
        all_bp.append(bp_chr)
        chr_len = len(snp_ids_chr)
        chr_offsets[chr_no] = (col_start, col_start + chr_len)
        col_start += chr_len
    bp = np.concatenate(all_bp, axis=0) if all_bp else np.array([], dtype=np.int64)
    return all_ids, bp, chr_offsets


def save_h5(h5_path: str, X: np.ndarray, iids: np.ndarray, bp: np.ndarray) -> None:
    ensure_dir_exists(os.path.dirname(h5_path))
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("X", data=X, compression="gzip", compression_opts=4)
        # variable length utf-8 strings for IIDs
        dt = h5py.string_dtype(encoding='utf-8')
        f.create_dataset("iid", data=iids.astype(str), dtype=dt)
        f.create_dataset("bp", data=bp)
    logger.info(f"Wrote {h5_path} | X={X.shape} bp={bp.shape}")


def main():
    p = argparse.ArgumentParser(description="Prepare VQ-VAE HDF5 caches from PLINK bfile")
    p.add_argument("--bfile", required=True, help="PLINK bfile prefix or .bed path")
    p.add_argument("--fam", required=True, help="FAM file path for sample order")
    p.add_argument("--bim", required=True, help="BIM file path for SNP positions")
    p.add_argument("--out-dir", required=True, help="Output directory for HDF5 caches")
    p.add_argument("--batch-size", type=int, default=10000, help="Samples per batch")
    p.add_argument("--chromosomes", nargs="+", default=["all"], help="Chromosomes to process; use 'all' for 1-22")
    # SNP chunking removed; we process all SNPs at once per batch
    args = p.parse_args()

    bfile_prefix = args.bfile[:-4] if args.bfile.endswith(".bed") else args.bfile
    ensure_dir_exists(args.out_dir)

    iid_chunks = chunk_iids(args.fam, args.batch_size)
    logger.info(f"Found {len(iid_chunks)} sample batches of size up to {args.batch_size}")

    # Determine chromosome set once
    if len(args.chromosomes) == 1 and str(args.chromosomes[0]).lower() == "all":
        chromosomes = list(range(1, 23))
    else:
        chromosomes = [int(c) for c in args.chromosomes]
    # Build SNP info per chromosome
    snp_info: Dict[int, Dict[str, any]] = {}
    for chr_no in chromosomes:
        ids_chr, bp_chr, _ = build_snp_info(args.bim, [chr_no])
        snp_info[chr_no] = {"snp_ids": ids_chr, "bp": bp_chr}

    for bi, chunk in enumerate(iid_chunks, start=1):
        for chr_no in chromosomes:
            chr_dir = os.path.join(args.out_dir, f"chr{chr_no}")
            ensure_dir_exists(chr_dir)
            keep_tsv = os.path.join(chr_dir, f"keep_batch{bi:05d}.tsv")
            chunk.to_csv(keep_tsv, sep="\t", header=False, index=False)
            out_prefix = os.path.join(chr_dir, f"tmp_batch{bi:05d}")
            # Per-chromosome PLINK recode
            cmd = [
                "plink", "--bfile", bfile_prefix,
                "--chr", str(chr_no),
                "--keep", keep_tsv,
                "--recode", "A",
                "--out", out_prefix,
            ]
            env = os.environ.copy(); env.setdefault("LC_ALL", "C")
            logger.info(f"Running: {' '.join(cmd)}")
            h5_path = os.path.join(chr_dir, f"batch{bi:05d}.h5")
            expected_cols = int(snp_info[chr_no]["bp"].shape[0])
            # Skip if H5 complete and no temp files remain
            if h5_is_complete(h5_path, expected_rows=len(chunk), expected_cols=expected_cols) and not temp_files_exist(out_prefix):
                logger.info(f"Found complete cache, skipping: {h5_path}")
                # cleanup keep file
                try:
                    os.remove(keep_tsv)
                except Exception:
                    pass
                continue
            # If partial/incomplete, remove stale H5 before re-creating
            if os.path.exists(h5_path) and not h5_is_complete(h5_path, expected_rows=len(chunk), expected_cols=expected_cols):
                try:
                    os.remove(h5_path)
                    logger.warning(f"Removed incomplete cache: {h5_path}")
                except Exception:
                    pass
            # Run PLINK and build H5
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, env=env)
            raw_chr = f"{out_prefix}.raw"
            # Stream into per-chromosome H5 for this batch
            try:
                write_raw_to_h5_fast(
                    raw_path=raw_chr,
                    h5_path=h5_path,
                    expected_rows=len(chunk),
                    bp=snp_info[chr_no]["bp"],
                    snp_ids=snp_info[chr_no]["snp_ids"],
                    chunk_rows=10000,
                )
            finally:
                # cleanup tmp and keep
                try:
                    if os.path.exists(raw_chr):
                        os.remove(raw_chr)
                    for ext in [".log", ".nosex"]:
                        pth = out_prefix + ext
                        if os.path.exists(pth):
                            os.remove(pth)
                    os.remove(keep_tsv)
                except Exception:
                    pass

    # Write simple manifests per chromosome
    try:
        import json
        for chr_no in chromosomes:
            chr_dir = os.path.join(args.out_dir, f"chr{chr_no}")
            batch_files = sorted(glob.glob(os.path.join(chr_dir, "batch*.h5")))
            manifest = {
                "chromosome": chr_no,
                "num_batches": len(batch_files),
                "batches": [os.path.basename(p) for p in batch_files],
                "snp_count": int(len(snp_info[chr_no]["bp"]))
            }
            with open(os.path.join(chr_dir, "manifest.json"), 'w') as mf:
                json.dump(manifest, mf)
    except Exception:
        pass


if __name__ == "__main__":
    main()


