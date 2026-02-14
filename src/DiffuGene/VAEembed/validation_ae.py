import os
import sys
import argparse
import glob
import json
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import h5py

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors as mpl_colors

this_dir = os.path.dirname(__file__)
src_root = os.path.abspath(os.path.join(this_dir, "..", "..", ".."))
if src_root not in sys.path:
    sys.path.insert(0, src_root)
from DiffuGene.VAEembed.ae import GenotypeAutoencoder, VAEConfig
from DiffuGene.VAEembed.sharedEmbed import FiLM2D
from DiffuGene.utils import ensure_dir_exists
from DiffuGene.utils.file_utils import read_bim_file


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

def get_chromosomes(spec: List[str]) -> List[int]:
    """Parse a list of chromosome identifiers into a list of integers.

    If the specification contains a single element equal to "all" (case
    insensitive), chromosomes 1–22 are returned. Otherwise, each element
    is cast to int.
    """
    if len(spec) == 1 and str(spec[0]).lower() == 'all':
        return list(range(1, 23))
    return [int(x) for x in spec]


def run_plink_recode_all(bfile_prefix: str, fam_keep: str, chr_no: int, out_prefix: str) -> str:
    """Run PLINK to recode a binary PLINK file to allele counts for a single chromosome.

    Parameters
    ----------
    bfile_prefix: str
        Prefix of the PLINK binary dataset (without extension).
    fam_keep: str
        Path to a two-column keep file specifying FID and IID to retain.
    chr_no: int
        Chromosome number to extract.
    out_prefix: str
        Output prefix. A `.raw` file will be produced with allele counts.

    Returns
    -------
    str
        Path to the generated `.raw` file.
    """
    cmd = [
        'plink', '--bfile', bfile_prefix,
        '--chr', str(chr_no),
        '--keep', fam_keep,
        '--recode', 'A',
        '--out', out_prefix,
    ]
    print(f"Running: {' '.join(cmd)}")
    env = os.environ.copy(); env.setdefault('LC_ALL', 'C')
    import subprocess
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, env=env)
    raw_path = f"{out_prefix}.raw"
    if not os.path.exists(raw_path):
        raise FileNotFoundError(raw_path)
    return raw_path


def prepare_validation_h5(bfile: str,
                          fam: str,
                          bim: str,
                          out_dir: str,
                          chromosomes: List[int],
                          warn_batch_size: int = 10000) -> Dict[int, Dict[str, any]]:
    """Prepare per-chromosome H5 caches for validation.

    This routine reads the entire `.fam` file to determine all individuals
    participating in validation (no batching on individuals). For each
    chromosome, if a batch file (`batch00001.h5`) already exists in
    `out_dir/chr{chr_no}`, the recoding is skipped. Otherwise, PLINK is
    invoked to extract allele counts and the resulting `.raw` file is
    converted to H5 for efficient loading.
    """
    ensure_dir_exists(out_dir)
    print(f"[VAL] Preparing validation H5 caches in {out_dir} for chromosomes={chromosomes}")
    # Read the FAM file; extract FID and IID columns
    import pandas as pd
    fam_cols = ["FID", "IID", "PAT", "MAT", "SEX", "PHENOTYPE"]
    fam_df = pd.read_csv(fam, sep=r"\s+", header=None, names=fam_cols)
    if len(fam_df) > warn_batch_size:
        print(f"[VAL] Number of validation individuals ({len(fam_df)}) exceeds warn_batch_size={warn_batch_size}; proceeding.")
    keep_df = fam_df.iloc[:, :2].copy()
    # Build SNP info per chromosome
    snp_info: Dict[int, Dict[str, any]] = {}
    for chr_no in chromosomes:
        ids_chr, bp_chr, _ = build_snp_info(bim, [chr_no])
        snp_info[chr_no] = {"snp_ids": ids_chr, "bp": bp_chr}
    # Recode once per chromosome
    bfile_prefix = bfile[:-4] if bfile.endswith('.bed') else bfile
    for chr_no in chromosomes:
        print(f"[VAL] Chromosome {chr_no}: starting cache prep")
        chr_dir = os.path.join(out_dir, f"chr{chr_no}")
        ensure_dir_exists(chr_dir)
        keep_tsv = os.path.join(chr_dir, f"keep_val.tsv")
        keep_df.to_csv(keep_tsv, sep='\t', header=False, index=False)
        out_prefix = os.path.join(chr_dir, f"tmp_val_all")
        h5_path = os.path.join(chr_dir, f"batch00001.h5")
        # If the batch already exists, skip recreation entirely
        if os.path.exists(h5_path):
            print(f"Found existing validation cache, skipping: {h5_path}")
            try:
                os.remove(keep_tsv)
            except Exception:
                pass
            continue
        raw_chr = run_plink_recode_all(bfile_prefix, keep_tsv, chr_no, out_prefix)
        try:
            print(f"[VAL] Writing H5 {h5_path}")
            write_raw_to_h5_fast(
                raw_path=raw_chr,
                h5_path=h5_path,
                expected_rows=len(keep_df),
                bp=snp_info[chr_no]["bp"],
                snp_ids=snp_info[chr_no]["snp_ids"],
                chunk_rows=20000,
            )
            print(f"[VAL] Wrote H5 {h5_path} (rows={len(keep_df)}, cols={int(snp_info[chr_no]['bp'].shape[0])})")
        finally:
            for ext in [".raw", ".log", ".nosex"]:
                pth = out_prefix + ext
                if os.path.exists(pth):
                    try:
                        os.remove(pth)
                    except Exception:
                        pass
            try:
                os.remove(keep_tsv)
            except Exception:
                pass
        print(f"[VAL] Chromosome {chr_no}: cache prep done")
    return snp_info


def _extract_prefixed_state(state_dict: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    plen = len(prefix)
    return {k[plen:]: v for k, v in state_dict.items() if k.startswith(prefix)}


def load_ae_from_checkpoint(ae_ckpt_path: str, device: torch.device) -> Tuple[GenotypeAutoencoder, VAEConfig, Dict[str, any]]:
    """
    Load a base AE checkpoint saved by train.py and return the model in eval mode.
    Returns (ae_model, cfg, raw_payload_meta_dict).
    """
    payload = torch.load(ae_ckpt_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"AE checkpoint must be a dict payload: {ae_ckpt_path}")
    cfg_dict = payload.get("config")
    state = payload.get("model_state") or payload.get("last_state_dict") or payload.get("best_state_dict")
    if cfg_dict is None or state is None:
        raise KeyError(f"AE checkpoint missing 'config' or a valid state dict: {ae_ckpt_path}")
    cfg = VAEConfig(**cfg_dict)
    ae = GenotypeAutoencoder(
        input_length=cfg.input_length,
        K1=cfg.K1,
        K2=cfg.K2,
        C=cfg.C,
        embed_dim=cfg.embed_dim,
    )
    incompat = ae.load_state_dict(state, strict=True)
    if getattr(incompat, "missing_keys", None):
        print(f"[AE] Missing keys (ok for buffers/new heads): {incompat.missing_keys}")
    if getattr(incompat, "unexpected_keys", None):
        print(f"[AE] Unexpected keys in checkpoint: {incompat.unexpected_keys}")
    ae.to(device).eval()
    for p in ae.parameters():
        p.requires_grad = False
    meta = payload.get("meta", {})
    return ae, cfg, meta


def load_homogenized_heads_and_ae(model_path: str, device: torch.device) -> Tuple[GenotypeAutoencoder, FiLM2D, FiLM2D, VAEConfig, Dict[str, any]]:
    """
    Load a homogenized stage-2 checkpoint saved by train_stage2.py:
      - Rebuild AE backbone from meta['config']
      - Instantiate FiLM2D encode/decode heads and load their states
    Returns (ae, encode_head, decode_head, cfg, meta)
    """
    payload = torch.load(model_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint {model_path} is not a dict payload; got {type(payload)}")
    state_dict = payload.get("model_state") or payload
    meta = payload.get("meta", {})
    cfg_dict = meta.get("config") if isinstance(meta, dict) else None
    if cfg_dict is None:
        raise KeyError(f"Checkpoint {model_path} missing meta['config'] for AE reconstruction")
    cfg = VAEConfig(**cfg_dict)
    ae = GenotypeAutoencoder(
        input_length=cfg.input_length,
        K1=cfg.K1,
        K2=cfg.K2,
        C=cfg.C,
        embed_dim=cfg.embed_dim,
    ).to(device)
    # AE params are prefixed under HomogenizedAE as aes.0.
    ae_state = _extract_prefixed_state(state_dict, 'aes.0.')
    incompat = ae.load_state_dict(ae_state, strict=True)
    if getattr(incompat, 'missing_keys', None) or getattr(incompat, 'unexpected_keys', None):
        print(f"[STAGE2] AE state load info: {incompat}")
    ae.eval()
    for p in ae.parameters():
        p.requires_grad = False

    encode_head = FiLM2D(ae.latent_channels).to(device).eval()
    decode_head = FiLM2D(ae.latent_channels).to(device).eval()
    enc_state = _extract_prefixed_state(state_dict, 'encode_head.')
    dec_state = _extract_prefixed_state(state_dict, 'decode_head.')
    enc_missing = encode_head.load_state_dict(enc_state, strict=True)
    dec_missing = decode_head.load_state_dict(dec_state, strict=True)
    if getattr(enc_missing, 'missing_keys', None) or getattr(enc_missing, 'unexpected_keys', None):
        print(f"[STAGE2] Encode head state load info: {enc_missing}")
    if getattr(dec_missing, 'missing_keys', None) or getattr(dec_missing, 'unexpected_keys', None):
        print(f"[STAGE2] Decode head state load info: {dec_missing}")
    return ae, encode_head, decode_head, cfg, meta


def compute_confusion_counts(pred: torch.Tensor, true: torch.Tensor) -> np.ndarray:
    """Compute a 3x3 confusion matrix between predicted and true genotype calls.

    Both `pred` and `true` should have shape [B, L] and contain values in
    {0, 1, 2}. The output counts the number of occurrences of each (true,
    predicted) pair.
    """
    cm_mat = np.zeros((3, 3), dtype=np.int64)
    t = true.view(-1).cpu().numpy()
    p = pred.view(-1).cpu().numpy()
    for a in (0, 1, 2):
        for b in (0, 1, 2):
            cm_mat[a, b] = int(np.sum((t == a) & (p == b)))
    return cm_mat


def plot_confusion_matrix(cm_mat: np.ndarray, out_path: str, title: str) -> None:
    """
    Plot a 3x3 confusion matrix. Accepts either counts or proportions.

    If `cm_mat` contains floating point values, they are formatted with
    three decimal places; otherwise, the raw integer counts are displayed.
    The matrix is visualised using a blue colormap.
    """
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm_mat, cmap=cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_xticks([0, 1, 2]); ax.set_yticks([0, 1, 2])
    # Determine if the matrix contains float values
    is_float = np.issubdtype(cm_mat.dtype, np.floating)
    for i in range(3):
        for j in range(3):
            val = cm_mat[i, j]
            if is_float:
                txt = f"{val:.3f}"
            else:
                txt = f"{val}"
            ax.text(j, i, txt, ha='center', va='center', color='black', fontsize=9)
    ax.set_title(title)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def plot_af_scatter(orig: np.ndarray, recon: np.ndarray, out_path: str, title: str) -> None:
    """Scatter plot of original versus reconstructed allele frequencies."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(orig, recon, s=5, alpha=0.5)
    ax.plot([0, 1], [0, 1], 'r--', linewidth=1)
    ax.set_xlabel('Original AF'); ax.set_ylabel('Reconstructed AF')
    ax.set_title(title)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def plot_entropy_vs_acc_per_snp(mean_entropy: np.ndarray, acc_per_snp: np.ndarray,
                                out_path: str, title: str) -> None:
    """Plot per-SNP mean predictive entropy versus per-SNP accuracy."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(mean_entropy, acc_per_snp, s=6, alpha=0.6)
    ax.set_xlabel('Mean predictive entropy per SNP')
    ax.set_ylabel('Per-SNP accuracy')
    ax.set_title(title)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def plot_confidence_vs_acc_per_snp(mean_conf: np.ndarray, acc_per_snp: np.ndarray,
                                   out_path: str, title: str) -> None:
    """Plot per-SNP mean predictive confidence versus per-SNP accuracy.

    Each point corresponds to a single SNP, with the x-axis representing
    the average confidence (maximum predicted probability) and the y-axis
    representing the accuracy of that SNP. The scatter is coloured by the
    accuracy value to highlight poorly performing SNPs.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    sc = ax.scatter(mean_conf, acc_per_snp, c=acc_per_snp, cmap='viridis', s=6, alpha=0.6)
    ax.set_xlabel('Mean predictive confidence per SNP')
    ax.set_ylabel('Per-SNP accuracy')
    ax.set_title(title)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label('Accuracy')
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def plot_ld_heatmaps(x_true: np.ndarray, x_hat: np.ndarray, out_path: str, title: str, max_snps: int = 256) -> None:
    """Plot LD heatmaps for original and reconstructed genotypes using subsampled SNPs."""
    xt = x_true.astype(np.float32)
    xr = x_hat.astype(np.float32)
    # Normalize per SNP
    xt = (xt - xt.mean(0, keepdims=True))
    xr = (xr - xr.mean(0, keepdims=True))
    cov_t = (xt.T @ xt) / max(1, xt.shape[0] - 1)
    cov_r = (xr.T @ xr) / max(1, xr.shape[0] - 1)
    std_t = np.sqrt(np.clip(np.diag(cov_t), 1e-6, None))
    std_r = np.sqrt(np.clip(np.diag(cov_r), 1e-6, None))
    corr_t = cov_t / (std_t[:, None] * std_t[None, :])
    corr_r = cov_r / (std_r[:, None] * std_r[None, :])
    vmin, vmax = -1.0, 1.0
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    im0 = axs[0].imshow(corr_t, vmin=vmin, vmax=vmax, cmap='coolwarm'); axs[0].set_title('Original LD')
    im1 = axs[1].imshow(corr_r, vmin=vmin, vmax=vmax, cmap='coolwarm'); axs[1].set_title('Reconstructed LD')
    for ax in axs:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(title)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def plot_ld_blocks_heatmaps(block_true_rows: List[np.ndarray],
                            block_recon_rows: List[np.ndarray],
                            block_snps: List[List[str]],
                            out_path: str,
                            title: str) -> None:
    """
    Plot LD heatmaps for a list of LD blocks. Each block will generate a
    pair of heatmaps (original and reconstructed) in a single figure. The
    number of blocks should be modest (e.g. 5) so the resulting figure
    remains legible.

    Parameters
    ----------
    block_true_rows: list of 2D arrays
        Each entry has shape [n_rows, n_snps] for the true genotypes of a block.
    block_recon_rows: list of 2D arrays
        Each entry has shape [n_rows, n_snps] for the reconstructed genotypes.
    block_snps: list of lists of SNP IDs corresponding to each block (used
        only for potential labelling; not displayed currently).
    out_path: str
        Where to save the resulting figure.
    title: str
        Overall figure title.
    """
    import math
    num_blocks = len(block_true_rows)
    if num_blocks == 0:
        return
    fig, axes = plt.subplots(num_blocks, 2, figsize=(10, 4 * num_blocks))
    # Ensure 2D indexing
    if num_blocks == 1:
        axes = np.array([axes])
    vmin, vmax = -1.0, 1.0
    for idx, (Xt, Xr) in enumerate(zip(block_true_rows, block_recon_rows)):
        xt = Xt.astype(np.float32)
        xr = Xr.astype(np.float32)
        xt = xt - xt.mean(0, keepdims=True)
        xr = xr - xr.mean(0, keepdims=True)
        cov_t = (xt.T @ xt) / max(1, xt.shape[0] - 1)
        cov_r = (xr.T @ xr) / max(1, xr.shape[0] - 1)
        std_t = np.sqrt(np.clip(np.diag(cov_t), 1e-6, None))
        std_r = np.sqrt(np.clip(np.diag(cov_r), 1e-6, None))
        corr_t = cov_t / (std_t[:, None] * std_t[None, :])
        corr_r = cov_r / (std_r[:, None] * std_r[None, :])
        ax_t = axes[idx, 0]
        ax_r = axes[idx, 1]
        im_t = ax_t.imshow(corr_t, vmin=vmin, vmax=vmax, cmap='coolwarm')
        im_r = ax_r.imshow(corr_r, vmin=vmin, vmax=vmax, cmap='coolwarm')
        ax_t.set_title(f'Block {idx + 1} Original')
        ax_r.set_title(f'Block {idx + 1} Recon')
        for ax in (ax_t, ax_r):
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(title)
    # Colourbar across figure
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    norm = mpl_colors.Normalize(vmin=vmin, vmax=vmax)
    cb = matplotlib.colorbar.ColorbarBase(cbar_ax, norm=norm, cmap='coolwarm')
    cb.set_label('LD correlation')
    fig.tight_layout(rect=[0, 0, 0.9, 1])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def parse_ld_blocks(det_file: str, min_snps: int = 20, max_blocks: int = 5) -> List[List[str]]:
    """
    Parse a PLINK .blocks.det file to extract LD blocks.

    Each block is represented as a list of SNP identifiers. Blocks with
    fewer than `min_snps` SNPs are ignored. At most `max_blocks` blocks
    are returned (the earliest ones in the file meeting the criterion).
    """
    blocks = []
    try:
        with open(det_file, 'r') as fh:
            header = fh.readline()  # skip header
            for line in fh:
                parts = line.strip().split()
                if len(parts) < 6:
                    continue
                try:
                    nsnps = int(parts[4])
                except ValueError:
                    continue
                snps_field = parts[5]
                snp_ids = snps_field.split('|')
                if nsnps >= min_snps and len(snp_ids) == nsnps:
                    blocks.append(snp_ids)
                if len(blocks) >= max_blocks:
                    break
    except Exception as e:
        print(f"[VAL] Error reading LD blocks from {det_file}: {e}")
        return []
    return blocks


def evaluate_chromosome(ae: GenotypeAutoencoder,
                        chr_h5_dir: str,
                        device: torch.device,
                        results_dir: str,
                        chr_no: int,
                        batch_size: int = 512,
                        ld_max_snps: int = 256,
                        ld_max_rows: int = 1024,
                        ld_block_dir: Optional[str] = None,
                        bim_path: Optional[str] = None,
                        *,
                        encode_head: Optional[FiLM2D] = None,
                        decode_head: Optional[FiLM2D] = None) -> Dict[str, any]:
    """
    Evaluate a single chromosome for reconstruction quality.

    Parameters
    ----------
    model: SNPVQVAE
        Loaded VQ-VAE model.
    chr_h5_dir: str
        Directory containing per-chromosome H5 files (batch00001.h5, etc.).
    device: torch.device
        Torch device to use for evaluation.
    results_dir: str
        Directory where plots and metrics will be saved.
    chr_no: int
        Chromosome number being evaluated.
    batch_size: int
        Batch size for forward passes through the model.
    ld_max_snps: int
        Number of SNPs to subsample for simple LD heatmaps when no block
        directory is provided.
    ld_max_rows: int
        Maximum number of individuals (rows) to sample for LD heatmaps.
    ld_block_dir: Optional[str]
        Directory containing `.blocks.det` files. If provided, LD heatmaps
        will be drawn on a handful of pre-computed LD blocks rather than
        uniformly subsampled SNPs. The file matching `*chr{chr_no}_blocks.blocks.det`
        will be parsed.
    bim_path: Optional[str]
        Path to the BIM file, required for resolving SNP identifiers to
        indices when using LD blocks.

    Returns
    -------
    Dict[str, any]
        Dictionary of summary metrics for this chromosome.
    """
    ensure_dir_exists(results_dir)
    # Collect H5 files
    h5_files = sorted(glob.glob(os.path.join(chr_h5_dir, 'batch*.h5')))
    if not h5_files:
        raise FileNotFoundError(f"No H5 files in {chr_h5_dir}")
    print(f"[VAL] chr{chr_no}: evaluating {len(h5_files)} H5 batch file(s) from {chr_h5_dir}")
    total_acc = []
    total_mse = []
    cm_sum = np.zeros((3, 3), dtype=np.int64)
    af_true_all = []
    af_recon_all = []
    # Per-SNP running aggregates for entropy, accuracy and confidence
    sum_entropy = None   # shape [L]
    sum_correct = None   # shape [L], integer counts
    sum_confidence = None  # shape [L], accumulates predicted confidence
    n_individuals = 0
    # For simple LD heatmaps: choose columns once; collect limited rows only
    ld_cols = None
    ld_rows_collected = 0
    ld_true_rows = []
    ld_recon_rows = []
    # For block-based LD heatmaps
    block_indices: List[List[int]] = []
    block_rows_collected: List[int] = []
    block_true_rows: List[List[np.ndarray]] = []
    block_recon_rows: List[List[np.ndarray]] = []
    if ld_block_dir and bim_path:
        # Attempt to locate the .blocks.det file corresponding to this chromosome
        pattern = os.path.join(ld_block_dir, f"*chr{chr_no}_blocks.blocks.det")
        det_files = glob.glob(pattern)
        if det_files:
            det_file = det_files[0]
            ld_blocks = parse_ld_blocks(det_file, min_snps=20, max_blocks=5)
            if ld_blocks:
                # Resolve BIM file to map SNP names to indices
                snp_ids, _, _ = build_snp_info(bim_path, [chr_no])
                id_to_index = {snp: idx for idx, snp in enumerate(snp_ids)}
                for block_snps in ld_blocks:
                    indices = [id_to_index[s] for s in block_snps if s in id_to_index]
                    # Only include blocks where we can resolve all SNPs (or at least >1)
                    if len(indices) >= 2:
                        block_indices.append(indices)
                if block_indices:
                    block_rows_collected = [0] * len(block_indices)
                    block_true_rows = [[] for _ in block_indices]
                    block_recon_rows = [[] for _ in block_indices]
        else:
            print(f"[VAL] No LD block file found for chr{chr_no} in {ld_block_dir}")
    total_samples = 0
    with torch.no_grad():
        for idx_h5, h5p in enumerate(h5_files, start=1):
            with h5py.File(h5p, 'r') as f:
                X = f['X'][:].astype('int64')  # [N, L]
            N, L = X.shape
            print(f"[VAL] chr{chr_no}: processing {os.path.basename(h5p)} with N={N}, L={L}")
            # Initialise per-SNP aggregators when L is known
            if sum_entropy is None:
                sum_entropy = np.zeros(L, dtype=np.float64)
                sum_correct = np.zeros(L, dtype=np.int64)
                sum_confidence = np.zeros(L, dtype=np.float64)
            # Choose LD columns once (deterministic stride sampling)
            if ld_cols is None:
                sel = min(L, ld_max_snps)
                if sel < L:
                    ld_cols = np.linspace(0, L - 1, num=sel, dtype=int)
                else:
                    ld_cols = np.arange(L, dtype=int)
            for s in range(0, N, batch_size):
                e = min(N, s + batch_size)
                xb = torch.from_numpy(X[s:e])
                xb = xb.to(device)
                # Forward: base AE or stage-2 homogenized (encode+decode heads)
                if encode_head is not None and decode_head is not None:
                    chrom_idx = int(chr_no) - 1
                    _, z = ae(xb)
                    z_hom = encode_head(z, chrom_idx)
                    z_dec = decode_head(z_hom, chrom_idx)
                    logits = ae.decode(z_dec)
                else:
                    logits, _ = ae(xb)
                # metrics (acc, mse)
                probs = torch.softmax(logits, dim=1)
                pred_dev = probs.argmax(dim=1)  # on device
                # confusion on CPU
                pred_cpu = pred_dev.detach().to('cpu')
                xb_cpu = xb.detach().to('cpu')
                cm_sum += compute_confusion_counts(pred_cpu, xb_cpu)
                # AFs: expected dosage on device, then move to CPU for logging
                class_vals = torch.tensor([0.0, 1.0, 2.0], device=probs.device).view(1, 3, 1)
                x_hat_dev = (probs * class_vals).sum(dim=1)  # (B, L) on device
                x_hat = x_hat_dev.detach().to('cpu').numpy()
                af_true_all.append((xb_cpu.numpy().mean(axis=0) / 2.0).astype(np.float32))
                af_recon_all.append((x_hat.mean(axis=0) / 2.0).astype(np.float32))
                # batch accuracy/mse on device
                acc_b = float((pred_dev.to(dtype=torch.int64) == xb.to(dtype=torch.int64)).float().mean().item())
                mse_b = float(((x_hat_dev - xb.float()) ** 2).mean().item())
                total_acc.append(acc_b)
                total_mse.append(mse_b)
                # Entropy and correctness (aggregate per SNP)
                p = probs.clamp_min(1e-9)
                ent = -(p * p.log()).sum(dim=1).cpu().numpy()  # [B, L]
                sum_entropy += ent.sum(axis=0)
                # correctness per SNP
                sum_correct += (pred_cpu.numpy() == xb_cpu.numpy()).sum(axis=0)
                # confidence per SNP: max probability per call
                conf_vals = probs.max(dim=1).values.cpu().numpy()  # shape [B, L]
                sum_confidence += conf_vals.sum(axis=0)
                n_individuals += (e - s)
                # Collect subset for simple LD
                if ld_rows_collected < ld_max_rows:
                    take = min((e - s), ld_max_rows - ld_rows_collected, 256)
                    if take > 0:
                        ld_true_rows.append(xb[:take, :][:, ld_cols].cpu().numpy())
                        ld_recon_rows.append(x_hat[:take, :][:, ld_cols])
                        ld_rows_collected += take
                # Collect subset for block-based LD
                if block_indices:
                    for bi, indices in enumerate(block_indices):
                        if block_rows_collected[bi] < ld_max_rows:
                            take_b = min((e - s), ld_max_rows - block_rows_collected[bi], 256)
                            if take_b > 0:
                                # Extract block columns
                                block_true_rows[bi].append(xb[:take_b, :][:, indices].cpu().numpy())
                                block_recon_rows[bi].append(x_hat[:take_b, :][:, indices])
                                block_rows_collected[bi] += take_b
                total_samples += (e - s)
            print(f"[VAL] chr{chr_no}: finished {idx_h5}/{len(h5_files)} H5 files; processed samples so far: {total_samples}")
    # Aggregate accuracy and MSE
    acc = float(np.mean(total_acc)) if total_acc else 0.0
    mse = float(np.mean(total_mse)) if total_mse else 0.0
    af_true = np.mean(np.stack(af_true_all, axis=0), axis=0)
    af_recon = np.mean(np.stack(af_recon_all, axis=0), axis=0)
    # Finalise per-SNP entropy, accuracy and confidence
    if n_individuals > 0:
        mean_entropy = (sum_entropy / float(n_individuals)).astype(np.float32)
        acc_per_snp = (sum_correct / float(n_individuals)).astype(np.float32)
        mean_conf = (sum_confidence / float(n_individuals)).astype(np.float32)
    else:
        mean_entropy = np.array([])
        acc_per_snp = np.array([])
        mean_conf = np.array([])
    # Normalise confusion matrix to proportions by true class
    cm_prop = cm_sum.astype(np.float64)
    row_sums = cm_prop.sum(axis=1, keepdims=True)
    # Avoid division by zero
    row_sums[row_sums == 0] = 1
    cm_prop = cm_prop / row_sums
    # Visuals
    plot_confusion_matrix(cm_prop, os.path.join(results_dir, f"chr{chr_no}_confusion.png"), f"Chr{chr_no} Confusion (Proportion)")
    # AF scatter
    if af_true.size and af_recon.size:
        max_snps_plot = 5000
        if af_true.shape[0] > max_snps_plot:
            idx = np.linspace(0, af_true.shape[0] - 1, num=max_snps_plot, dtype=int)
            af_true_plot = af_true[idx]
            af_recon_plot = af_recon[idx]
        else:
            af_true_plot = af_true
            af_recon_plot = af_recon
        plot_af_scatter(af_true_plot, af_recon_plot, os.path.join(results_dir, f"chr{chr_no}_af_scatter.png"), f"Chr{chr_no} AF: orig vs recon")
    # Entropy vs accuracy
    if mean_entropy.size and acc_per_snp.size:
        plot_entropy_vs_acc_per_snp(mean_entropy, acc_per_snp,
                                    os.path.join(results_dir, f"chr{chr_no}_entropy_vs_acc_per_snp.png"),
                                    f"Chr{chr_no}: per-SNP entropy vs accuracy")
    # Confidence vs accuracy
    if mean_conf.size and acc_per_snp.size:
        plot_confidence_vs_acc_per_snp(mean_conf, acc_per_snp,
                                       os.path.join(results_dir, f"chr{chr_no}_confidence_vs_acc_per_snp.png"),
                                       f"Chr{chr_no}: per-SNP confidence vs accuracy")
    # LD heatmaps
    if block_indices and block_true_rows:
        # Concatenate rows for each block
        bt_rows = [np.concatenate(lst, axis=0) for lst in block_true_rows]
        br_rows = [np.concatenate(lst, axis=0) for lst in block_recon_rows]
        plot_ld_blocks_heatmaps(bt_rows, br_rows, [None] * len(bt_rows),
                                os.path.join(results_dir, f"chr{chr_no}_ld_blocks_heatmap.png"),
                                f"Chr{chr_no} LD blocks")
    elif ld_true_rows:
        Xt = np.concatenate(ld_true_rows, axis=0)
        Xr = np.concatenate(ld_recon_rows, axis=0)
        plot_ld_heatmaps(Xt, Xr, os.path.join(results_dir, f"chr{chr_no}_ld_heatmap.png"), f"Chr{chr_no} LD", max_snps=ld_max_snps)
    # Save metrics json
    metrics = {
        'chromosome': chr_no,
        'accuracy': acc,
        'mse': mse,
        'confusion': cm_sum.tolist(),
        'af_true_mean': float(np.mean(af_true)),
        'af_recon_mean': float(np.mean(af_recon)),
        'num_positions': int(af_true.shape[0])
    }
    with open(os.path.join(results_dir, f"chr{chr_no}_metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"[VAL] chr{chr_no}: accuracy={acc:.4f}, mse={mse:.6f}, positions={metrics['num_positions']}")
    return metrics


def main():
    p = argparse.ArgumentParser(description="Validate VQ-VAE embeddings on a held-out test set (per chromosome and genome-wide)")
    p.add_argument('--bfile', required=True, help='PLINK bfile prefix or .bed path for validation')
    p.add_argument('--bim', required=True, help='BIM file path')
    p.add_argument('--fam', required=True, help='FAM file path containing validation individuals (~14k)')
    p.add_argument('--val-h5-dir', required=True, help='Output directory for validation H5 caches')
    p.add_argument('--models-dir', required=True, help='Directory containing per-chromosome AE checkpoints')
    p.add_argument('--model-pattern', default='ae_chr{chr}.pt', help='Filename pattern for per-chromosome models (ae_chr{chr}.pt or ae_chr{chr}_homog.pt)')
    p.add_argument('--homogenized', action='store_true', help='If set, treat checkpoints as stage-2 homogenized heads + AE')
    p.add_argument('--results-dir', required=True, help='Directory to save metrics and visualisations')
    p.add_argument('--chromosomes', nargs='+', default=['all'])
    p.add_argument('--device', default='cuda')
    p.add_argument('--batch-size', type=int, default=512)
    p.add_argument('--ld-block-dir', default=None, help='Path to directory containing PLINK .blocks.det files for LD blocks')
    args = p.parse_args()
    ensure_dir_exists(args.results_dir)
    chromosomes = get_chromosomes(args.chromosomes)
    print(f"[VAL] Starting AE validation for chromosomes={chromosomes} | homogenized={args.homogenized}")
    # Step 1: Prepare validation H5 caches (no batching by size; just warn if large)
    prepare_validation_h5(
        bfile=args.bfile,
        fam=args.fam,
        bim=args.bim,
        out_dir=args.val_h5_dir,
        chromosomes=chromosomes,
        warn_batch_size=max(args.batch_size, 10000)
    )
    print(f"[VAL] Validation H5 preparation complete")
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith('cuda') else 'cpu')
    # Step 2: Per-chromosome evaluation
    chr_metrics: List[Dict[str, any]] = []
    for chr_no in chromosomes:
        model_path = os.path.join(args.models_dir, args.model_pattern.format(chr=chr_no))
        if not os.path.exists(model_path):
            print(f"[VAL] Model for chr{chr_no} not found: {model_path}; skipping")
            continue
        print(f"[VAL] Loading AE for chr{chr_no}: {model_path}")
        if args.homogenized:
            ae, enc_head, dec_head, cfg, meta = load_homogenized_heads_and_ae(model_path, device)
        else:
            ae, cfg, meta = load_ae_from_checkpoint(model_path, device)
            enc_head = None
            dec_head = None
        chr_dir = os.path.join(args.val_h5_dir, f"chr{chr_no}")
        out_dir = os.path.join(args.results_dir, f"chr{chr_no}")
        ensure_dir_exists(out_dir)
        m = evaluate_chromosome(ae, chr_dir, device, out_dir, chr_no,
                                batch_size=int(args.batch_size),
                                ld_max_snps=256, ld_max_rows=1024,
                                ld_block_dir=args.ld_block_dir,
                                bim_path=args.bim,
                                encode_head=enc_head,
                                decode_head=dec_head)
        chr_metrics.append(m)
        del ae
        if device.type == "cuda":
            torch.cuda.empty_cache()
    # Step 3: Genome-wide aggregation
    if chr_metrics:
        total_pos = sum(m['num_positions'] for m in chr_metrics)
        acc = sum(m['accuracy'] * m['num_positions'] for m in chr_metrics) / max(1, total_pos)
        mse = sum(m['mse'] * m['num_positions'] for m in chr_metrics) / max(1, total_pos)
        gw = {
            'accuracy': float(acc),
            'mse': float(mse),
            'total_positions': int(total_pos),
            'num_chromosomes': int(len(chr_metrics))
        }
        with open(os.path.join(args.results_dir, 'genome_wide_metrics.json'), 'w') as f:
            json.dump(gw, f, indent=2)
        print(f"[VAL] Genome-wide: accuracy={acc:.4f}, mse={mse:.6f}, positions={total_pos}, chrs={len(chr_metrics)}")
        # Simple bar chart per chromosome accuracy
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar([m['chromosome'] for m in chr_metrics], [m['accuracy'] for m in chr_metrics])
        ax.set_xlabel('Chromosome'); ax.set_ylabel('Accuracy'); ax.set_title('Per-chromosome reconstruction accuracy')
        fig.tight_layout(); fig.savefig(os.path.join(args.results_dir, 'per_chrom_accuracy.png'), dpi=150); plt.close(fig)


if __name__ == '__main__':
    main()


# AE_DIR=/n/home03/ahmadazim/WORKING/genGen/UKBVQVAE/
# python /n/home03/ahmadazim/WORKING/genGen/DiffuGene/src/DiffuGene/VAEembed/validation_ae.py \
#   --bfile ${AE_DIR}/genomic_data/geneticBinary/ukb_allchr_unrel_britishWhite \
#   --bim ${AE_DIR}/genomic_data/geneticBinary/ukb_allchr_unrel_britishWhite.bim \
#   --fam ${AE_DIR}/genomic_data/geneticBinary/ukb_allchr_unrel_britishWhite_genome_encoder_test.fam \
#   --val-h5-dir ${AE_DIR}/genomic_data/val_h5_cache/ \
#   --models-dir ${AE_DIR}/models/AE/ \
#   --model-pattern ae_chr{chr}.pt \
#   --results-dir ${AE_DIR}/encoderEvals/val_res128_AE/ \
#   --chromosomes all \
#   --device cuda \
#   --batch-size 512 \
#   --ld-block-dir ${AE_DIR}/../UKB6PC/genomic_data/haploblocks/ 