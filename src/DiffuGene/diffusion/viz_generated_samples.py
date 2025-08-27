#!/usr/bin/env python

import argparse
import os
import random
from typing import List, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd


def ensure_dir_exists(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def load_latents(path: str) -> torch.Tensor:
    """Load latents from a .pt file or a memmap .npy with a sibling _shape.txt.

    Returns CPU float32 tensor shaped (N, C, H, W).
    """
    if path.endswith('.pt'):
        data = torch.load(path, map_location='cpu')
        if isinstance(data, dict) and 'latents' in data:
            data = data['latents']
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32)
        return data.float()
    elif path.endswith('.npy'):
        shape_file = path.replace('.npy', '_shape.txt')
        if not os.path.exists(shape_file):
            raise FileNotFoundError(f"Shape file not found for memmap: {shape_file}")
        with open(shape_file, 'r') as f:
            shape = tuple(map(int, f.read().strip().split(',')))
        arr = np.memmap(path, dtype='float32', mode='r', shape=shape)
        return torch.from_numpy(np.array(arr))
    else:
        raise ValueError(f"Unsupported latents format: {path}")


def load_decoded_recon(path: str) -> List[torch.Tensor]:
    """Load decoded SNP reconstructions saved by decode_vae_latents.py.

    Returns list of tensors, one per block, each shaped (N_samples, N_snps_block).
    """
    data = torch.load(path, map_location='cpu')
    if isinstance(data, dict) and 'reconstructed_snps' in data:
        blocks = data['reconstructed_snps']
        # Ensure tensors
        out = []
        for b in blocks:
            if isinstance(b, torch.Tensor):
                out.append(b.float())
            else:
                out.append(torch.tensor(b, dtype=torch.float32))
        return out
    else:
        # Accept a list directly as a fallback
        if isinstance(data, list):
            return [b if isinstance(b, torch.Tensor) else torch.tensor(b, dtype=torch.float32) for b in data]
        raise ValueError(f"Decoded file format not recognized: keys={list(data.keys()) if isinstance(data, dict) else type(data)}")


def build_block_map_from_spans(spans_file: str) -> List[tuple]:
    """Return list of (chr_num:int, block_no:int, basename:str) per block in spans order."""
    df = pd.read_csv(spans_file)
    mapping = []
    for _, row in df.iterrows():
        chr_num = int(row['chr'])
        # Extract basename and block number from block_file
        fname = os.path.basename(str(row['block_file']))
        # expected: <basename>_chr{chr}_block{blockNo}_embeddings.pt
        try:
            parts = fname.split('_chr')
            base = parts[0]
            rest = parts[1]
            block_part = rest.split('_block')[1]
            block_no = int(block_part.split('_')[0])
        except Exception:
            # Fallback regex
            import re
            m = re.search(r'^(.*)_chr(\d+)_block(\d+)_embeddings\.pt$', fname)
            if not m:
                raise ValueError(f"Cannot parse block_file name: {fname}")
            base = m.group(1)
            block_no = int(m.group(3))
        mapping.append((chr_num, block_no, base))
    return mapping


def load_raw_block(recoded_dir: str, chr_num: int, block_no: int, basename: str, max_samples: int = 5000) -> torch.Tensor:
    """Load PLINK recodeA raw for a block and return float32 tensor (imputing missing).
    Caps the number of rows to max_samples to limit memory.
    """
    raw_path = os.path.join(recoded_dir, f"chr{chr_num}", f"{basename}_chr{chr_num}_block{block_no}_recodeA.raw")
    if not os.path.exists(raw_path):
        # Try without chr subdir
        raw_path = os.path.join(recoded_dir, f"{basename}_chr{chr_num}_block{block_no}_recodeA.raw")
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw file not found for chr{chr_num} block{block_no}: {raw_path}")
    # PLINK .raw is space-delimited; first 6 columns are IDs/phenos
    df = pd.read_csv(raw_path, delim_whitespace=True)
    keep_cols = [c for c in df.columns if c.upper() not in {'FID','IID','PAT','MAT','SEX','PHENOTYPE'}]
    X = df[keep_cols].to_numpy(dtype=np.float32)
    # Cap rows for memory control
    if X.shape[0] > max_samples:
        X = X[:max_samples, :]
    # Impute missing with column means
    if np.isnan(X).any():
        col_means = np.nanmean(X, axis=0)
        idx = np.where(np.isnan(X))
        X[idx] = np.take(col_means, idx[1])
    return torch.from_numpy(X)


def plot_latent_histograms(gen_latents: torch.Tensor,
                           orig_latents: torch.Tensor,
                           dims: List[int],
                           num_samples: int,
                           output_path: str) -> None:
    """Plot histograms for selected flattened dims using a sampled subset only.
    Avoids flattening the entire tensor to reduce peak memory.
    """
    # Shapes: (N, C, H, W)
    n_gen_total = gen_latents.shape[0]
    n_org_total = orig_latents.shape[0]
    n_gen = min(num_samples, n_gen_total)
    n_org = min(num_samples, n_org_total)

    rng = np.random.default_rng(100)
    gen_idx = rng.choice(n_gen_total, size=n_gen, replace=False)
    org_idx = rng.choice(n_org_total, size=n_org, replace=False)

    # Slice first, then flatten only the subset
    gen_sel = gen_latents[gen_idx].reshape(n_gen, -1).cpu().numpy()
    org_sel = orig_latents[org_idx].reshape(n_org, -1).cpu().numpy()

    n_cols = len(dims)
    fig, axes = plt.subplots(1, n_cols, figsize=(3.0 * n_cols + 6, 3.5))
    if n_cols == 1:
        axes = [axes]
    for i, dim in enumerate(dims):
        ax = axes[i]
        # Compute joint bounds
        vals_org = org_sel[:, dim]
        vals_gen = gen_sel[:, dim]
        vmin = np.minimum(vals_org.min(), vals_gen.min())
        vmax = np.maximum(vals_org.max(), vals_gen.max())
        ax.hist(vals_org, bins=50, alpha=0.6, density=True, color='tab:blue', label='Original')
        ax.hist(vals_gen, bins=50, alpha=0.6, density=True, color='tab:orange', label='Generated')
        ax.set_xlim(vmin, vmax)
        ax.set_title(f"Dim {dim}")
        if i == 0:
            ax.legend(loc='upper right', frameon=True, framealpha=0.3)
    fig.suptitle("Latent value distributions (flattened dims)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def pick_blocks_with_min_snps(blocks: List[torch.Tensor], min_snps: int, max_blocks: int) -> List[int]:
    candidates = [i for i, b in enumerate(blocks) if b.shape[1] >= min_snps]
    # Stable selection: first max_blocks
    return candidates[:max_blocks]


def plot_ld_heatmaps(orig_blocks: List[torch.Tensor],
                     gen_blocks: List[torch.Tensor],
                     block_indices: List[int],
                     output_path: str) -> None:
    """Plot lower-triangle heatmaps of LD (pearson correlation) for selected blocks."""
    n = len(block_indices)
    if n == 0:
        return
    rows = 2
    cols = n
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols + 3, 2.5 * rows), constrained_layout=True)
    if rows == 1:
        axes = np.array([axes])

    def lower_tri_heatmap(ld: np.ndarray, ax, title: str):
        mask = np.tril(np.ones_like(ld, dtype=bool), k=-1)
        ld_tri = np.ma.array(ld, mask=mask)
        im = ax.imshow(ld_tri, cmap='RdBu_r', origin='lower', vmin=-1, vmax=1, interpolation='nearest')
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        return im

    last_im = None
    for j, bi in enumerate(block_indices):
        # Original
        Xo = orig_blocks[bi].cpu().numpy()
        ld_o = np.corrcoef(Xo.T)
        last_im = lower_tri_heatmap(ld_o, axes[0, j], f"Orig block {bi}")
        # Generated
        Xg = gen_blocks[bi].cpu().numpy()
        ld_g = np.corrcoef(Xg.T)
        lower_tri_heatmap(ld_g, axes[1, j], f"Gen block {bi}")

    # Shared colorbar
    cbar = fig.colorbar(last_im, ax=axes.ravel().tolist(), orientation='horizontal', fraction=0.05, pad=0.07)
    cbar.set_label('LD (r)')
    fig.suptitle("LD (correlation) lower-triangle: Original vs Generated")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_af_and_variance(orig_blocks: List[torch.Tensor],
                         gen_blocks: List[torch.Tensor],
                         block_indices: List[int],
                         output_path: str) -> None:
    """For each block, overlay AF (mean/2) and variance per SNP for original, generated, and rounded generated."""
    if not block_indices:
        return
    fig, axes = plt.subplots(len(block_indices), 2, figsize=(12, 2.8 * len(block_indices)), constrained_layout=True)
    if len(block_indices) == 1:
        axes = np.array([axes])

    for row, bi in enumerate(block_indices):
        orig = orig_blocks[bi].cpu().numpy()
        gen = gen_blocks[bi].cpu().numpy()
        gen_rounded = np.rint(np.clip(gen, 0, 2))

        # AF
        af_orig = orig.mean(axis=0) / 2.0
        af_gen = gen.mean(axis=0) / 2.0
        af_round = gen_rounded.mean(axis=0) / 2.0
        ax_af = axes[row, 0]
        ax_af.plot(af_orig, label='Original', color='black', linewidth=2, linestyle='--', alpha=0.7)
        ax_af.plot(af_gen, label='Generated', color='tab:orange')
        ax_af.plot(af_round, label='Generated (Rounded)', color='tab:green')
        ax_af.set_title(f"Block {bi} AF")
        ax_af.legend(loc='upper right')

        # Variance
        var_orig = orig.var(axis=0)
        var_gen = gen.var(axis=0)
        var_round = gen_rounded.var(axis=0)
        ax_var = axes[row, 1]
        ax_var.plot(var_orig, label='Original', color='black', linewidth=2, linestyle='--', alpha=0.7)
        ax_var.plot(var_gen, label='Generated', color='tab:orange')
        ax_var.plot(var_round, label='Generated (Rounded)', color='tab:green')
        ax_var.set_title(f"Block {bi} Variance")
        ax_var.legend(loc='upper right')

    fig.suptitle("AF and Variance across SNPs in selected blocks")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_af_scatter(orig_blocks: List[torch.Tensor],
                    gen_blocks: List[torch.Tensor],
                    block_indices: List[int],
                    output_path: str,
                    max_blocks: int = 1000) -> None:
    """Scatter AF(original) vs AF(generated) with 45-degree line and best-fit line.
    Uses up to max_blocks blocks (default 1000). Also writes a rounded version.
    """
    if orig_blocks is None or gen_blocks is None or len(orig_blocks) == 0 or len(gen_blocks) == 0:
        return
    # Build selection up to max_blocks
    n_total = min(max_blocks, len(orig_blocks), len(gen_blocks))
    sel = []
    if block_indices:
        # de-duplicate and clip
        seen = set()
        for bi in block_indices:
            if bi not in seen and bi < n_total:
                sel.append(bi)
                seen.add(bi)
            if len(sel) >= n_total:
                break
        # extend with leading indices if we still need more
        if len(sel) < n_total:
            for i in range(n_total):
                if i not in seen:
                    sel.append(i)
                if len(sel) >= n_total:
                    break
    else:
        sel = list(range(n_total))

    # Unrounded scatter
    af_orig_all = []
    af_gen_all = []
    for bi in sel:
        orig = orig_blocks[bi].cpu().numpy()
        gen = gen_blocks[bi].cpu().numpy()
        af_orig_all.append(orig.mean(axis=0) / 2.0)
        af_gen_all.append(gen.mean(axis=0) / 2.0)
    af_orig_all = np.concatenate(af_orig_all)
    af_gen_all = np.concatenate(af_gen_all)

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    ax.scatter(af_orig_all, af_gen_all, alpha=0.4, s=8)
    ax.set_xlabel('Original AF')
    ax.set_ylabel('Generated AF')
    ax.set_title('Allele Frequency: Original vs Generated')
    lims = [0, 1]
    ax.plot(lims, lims, color='red', linestyle='--', linewidth=1.5, label='y = x')
    x_unique = np.unique(af_orig_all)
    coef = np.polyfit(af_orig_all, af_gen_all, 1)
    ax.plot(x_unique, np.poly1d(coef)(x_unique), color='blue', label='Best fit')
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    # Rounded scatter
    af_gen_round_all = []
    for bi in sel:
        gen = gen_blocks[bi].cpu().numpy()
        gen_rounded = np.rint(np.clip(gen, 0, 2))
        af_gen_round_all.append(gen_rounded.mean(axis=0) / 2.0)
    af_gen_round_all = np.concatenate(af_gen_round_all)

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    ax.scatter(af_orig_all, af_gen_round_all, alpha=0.4, s=8)
    ax.set_xlabel('Original AF')
    ax.set_ylabel('Generated AF (rounded)')
    ax.set_title('Allele Frequency: Original vs Generated (Rounded)')
    ax.plot(lims, lims, color='red', linestyle='--', linewidth=1.5, label='y = x')
    x_unique = np.unique(af_orig_all)
    coef = np.polyfit(af_orig_all, af_gen_round_all, 1)
    ax.plot(x_unique, np.poly1d(coef)(x_unique), color='blue', label='Best fit')
    ax.legend()
    fig.tight_layout()
    base, ext = os.path.splitext(output_path)
    fig.savefig(f"{base}_rounded{ext}", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize generated latents and decoded SNPs")
    # Latent visualization
    parser.add_argument('--generated-latents', type=str, required=True, help='Path to generated latents (.pt or .npy)')
    parser.add_argument('--original-latents', type=str, required=True, help='Path to original/train latents (.pt or .npy)')
    parser.add_argument('--latent-dims', type=int, nargs='+', default=[0,1,2,3,4,5,6], help='Flattened dims to visualize')
    parser.add_argument('--latent-samples', type=int, default=512, help='Samples from each set for histograms')

    # Decoded comparisons
    parser.add_argument('--decoded-generated', type=str, required=True, help='Path to decoded generated SNPs (.pt from decode_vae_latents.py)')
    # Prefer raw originals: provide recoded_dir and spans_file
    parser.add_argument('--recoded-dir', type=str, required=True, help='Path to haploblocks_recoded directory with recodeA.raw files')
    parser.add_argument('--spans-file', type=str, required=True, help='Spans CSV used for VAE/decoder; defines block order and basenames')
    parser.add_argument('--min-snps', type=int, default=80, help='Minimum SNPs per block to consider')
    parser.add_argument('--num-blocks', type=int, default=5, help='Number of blocks to visualize')

    parser.add_argument('--output-dir', type=str, default=None, help='Directory to save figures (default: alongside generated latents)')

    args = parser.parse_args()

    # Resolve output dir
    out_dir = args.output_dir or os.path.join(os.path.dirname(args.generated_latents), '')
    ensure_dir_exists(out_dir)

    # 1) Latent histograms
    gen_latents = load_latents(args.generated_latents)
    orig_latents = load_latents(args.original_latents)
    # Ensure latents are shaped (N,C,H,W); tolerate (N,64,64,64)
    if gen_latents.dim() == 4 and gen_latents.shape[1] not in (64,):
        pass
    # Plot
    plot_latent_histograms(
        gen_latents=gen_latents,
        orig_latents=orig_latents,
        dims=args.latent_dims,
        num_samples=args.latent_samples,
        output_path=os.path.join(out_dir, 'latent_histograms.png')
    )

    # 2-4) Decoded comparisons
    gen_blocks = load_decoded_recon(args.decoded_generated)
    # Build mapping and load originals for selected blocks from raw
    block_map = build_block_map_from_spans(args.spans_file)

    # Determine candidate indices based on generated blocks (decode order matches spans)
    n_blocks = len(gen_blocks)
    block_indices = pick_blocks_with_min_snps(gen_blocks, min_snps=args.min_snps, max_blocks=args.num_blocks)
    # Load raw originals for selected block indices
    orig_sel: List[torch.Tensor] = []
    gen_sel: List[torch.Tensor] = []
    for bi in block_indices:
        if bi >= len(block_map) or bi >= len(gen_blocks):
            continue
        chr_num, block_no, base = block_map[bi]
        try:
            orig_tensor = load_raw_block(args.recoded_dir, chr_num, block_no, base, max_samples=5000)
            # Slice generated samples to match orig rows
            gen_block = gen_blocks[bi]
            n = min(orig_tensor.shape[0], gen_block.shape[0])
            orig_sel.append(orig_tensor[:n])
            gen_sel.append(gen_block[:n])
        except Exception:
            continue
    # Fallback: if no selection loaded, try first few
    if not orig_sel:
        for bi in range(min(args.num_blocks, len(block_map), len(gen_blocks))):
            chr_num, block_no, base = block_map[bi]
            try:
                orig_tensor = load_raw_block(args.recoded_dir, chr_num, block_no, base, max_samples=5000)
                gen_block = gen_blocks[bi]
                n = min(orig_tensor.shape[0], gen_block.shape[0])
                orig_sel.append(orig_tensor[:n])
                gen_sel.append(gen_block[:n])
            except Exception:
                continue

    plot_ld_heatmaps(
        orig_blocks=orig_sel,
        gen_blocks=gen_sel,
        block_indices=list(range(len(orig_sel))),
        output_path=os.path.join(out_dir, 'ld_heatmaps.png')
    )

    plot_af_and_variance(
        orig_blocks=orig_sel,
        gen_blocks=gen_sel,
        block_indices=list(range(len(orig_sel))),
        output_path=os.path.join(out_dir, 'af_and_variance.png')
    )

    plot_af_scatter(
        orig_blocks=orig_sel,
        gen_blocks=gen_sel,
        block_indices=list(range(len(orig_sel))),
        output_path=os.path.join(out_dir, 'af_scatter.png')
    )


if __name__ == '__main__':
    main()


