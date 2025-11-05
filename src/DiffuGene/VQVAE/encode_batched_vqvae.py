#!/usr/bin/env python
import os
import sys
import argparse
import glob
import json
from typing import List, Dict

import numpy as np
import torch
import h5py

this_dir = os.path.dirname(__file__)
src_root = os.path.abspath(os.path.join(this_dir, "..", "..", ".."))
if src_root not in sys.path:
    sys.path.insert(0, src_root)

from DiffuGene.VAEembed.vae import SNPVQVAE, VQVAEConfig
from DiffuGene.utils import ensure_dir_exists
from DiffuGene.VAEembed.prepare_vqvae_data import write_raw_to_h5_fast
from contextlib import nullcontext
try:
    from torch.cuda.amp import autocast as cuda_autocast
except Exception:  # CPU-only environments
    cuda_autocast = None


def get_chromosomes(spec: List[str]) -> List[int]:
    if len(spec) == 1 and str(spec[0]).lower() == 'all':
        return list(range(1, 23))
    return [int(x) for x in spec]


def run_plink_recode_batch(bfile_prefix: str, chromosomes: List[int], keep_tsv: str, out_prefix: str) -> str:
    import subprocess
    cmd = ["plink", "--bfile", bfile_prefix]
    if chromosomes:
        cmd += ["--chr"] + [str(c) for c in chromosomes]
    cmd += ["--keep", keep_tsv, "--recode", "A", "--out", out_prefix]
    print(f"[ENC] Running: {' '.join(cmd)}")
    env = os.environ.copy(); env.setdefault("LC_ALL", "C")
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, env=env)
    raw_path = f"{out_prefix}.raw"
    if not os.path.exists(raw_path):
        raise FileNotFoundError(raw_path)
    return raw_path


def create_batched_h5(bfile: str, fam: str, bim: str, out_dir: str, chromosomes: List[int], batch_size: int) -> Dict[int, int]:
    import pandas as pd
    ensure_dir_exists(out_dir)
    # Load all IIDs
    fam_cols = ["FID", "IID", "PAT", "MAT", "SEX", "PHENOTYPE"]
    fam_df = pd.read_csv(fam, sep=r"\s+", header=None, names=fam_cols)
    iids = fam_df.iloc[:, :2].copy()

    # For each chromosome, create batched H5 files under out_dir/chr{c}
    per_chr_batches: Dict[int, int] = {}
    bfile_prefix = bfile[:-4] if bfile.endswith('.bed') else bfile
    for chr_no in chromosomes:
        chr_dir = os.path.join(out_dir, f"chr{chr_no}")
        ensure_dir_exists(chr_dir)
        # Determine SNP ids and bp for this chromosome
        # Reuse prepare_vqvae_data.build_snp_info via a minimal call
        from DiffuGene.VAEembed.prepare_vqvae_data import build_snp_info
        ids_chr, bp_chr, _ = build_snp_info(bim, [chr_no])
        # Batch individuals by batch_size
        num_batches = int(np.ceil(len(iids) / float(batch_size)))
        for bi in range(1, num_batches + 1):
            s = (bi - 1) * batch_size
            e = min(len(iids), bi * batch_size)
            keep_tsv = os.path.join(chr_dir, f"keep_batch{bi:05d}.tsv")
            iids.iloc[s:e, :].to_csv(keep_tsv, sep='\t', header=False, index=False)
            out_prefix = os.path.join(chr_dir, f"tmp_batch{bi:05d}")
            h5_path = os.path.join(chr_dir, f"batch{bi:05d}.h5")
            if os.path.exists(h5_path):
                print(f"[ENC] Found existing cache, skipping: {h5_path}")
                try:
                    os.remove(keep_tsv)
                except Exception:
                    pass
                continue
            # Recode and write H5
            raw_path = run_plink_recode_batch(bfile_prefix, [chr_no], keep_tsv, out_prefix)
            try:
                write_raw_to_h5_fast(
                    raw_path=raw_path,
                    h5_path=h5_path,
                    expected_rows=(e - s),
                    bp=np.array(bp_chr, dtype=np.int64),
                    snp_ids=list(ids_chr),
                    chunk_rows=10000,
                )
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
        per_chr_batches[chr_no] = num_batches
    return per_chr_batches


def load_model(model_path: str, device: torch.device) -> SNPVQVAE:
    ckpt = torch.load(model_path, map_location='cpu')
    cfg = VQVAEConfig(**ckpt['config'])
    model = SNPVQVAE(cfg)
    model.load_state_dict(ckpt['model_state'])
    model.to(device)
    model.eval()
    return model


def encode_per_chr_batches(models_dir: str,
                           model_pattern: str,
                           h5_root: str,
                           out_root: str,
                           chromosomes: List[int],
                           device: torch.device,
                           encode_batch_size: int = 128) -> Dict[int, int]:
    ensure_dir_exists(out_root)
    per_chr_batches: Dict[int, int] = {}
    for chr_no in chromosomes:
        model_path = os.path.join(models_dir, model_pattern.format(chr=chr_no))
        if not os.path.exists(model_path):
            print(f"[ENC] Missing model for chr{chr_no}: {model_path}; skipping")
            continue
        model = load_model(model_path, device)
        chr_h5_dir = os.path.join(h5_root, f"chr{chr_no}")
        chr_out_dir = os.path.join(out_root, f"chr{chr_no}")
        ensure_dir_exists(chr_out_dir)
        h5_files = sorted(glob.glob(os.path.join(chr_h5_dir, 'batch*.h5')))
        per_chr_batches[chr_no] = len(h5_files)
        with torch.no_grad():
            for h5p in h5_files:
                bn = os.path.splitext(os.path.basename(h5p))[0]
                out_pt = os.path.join(chr_out_dir, f"{bn}_latents.pt")
                if os.path.exists(out_pt):
                    print(f"[ENC] Found existing latents, skipping: {out_pt}")
                    continue
                with h5py.File(h5p, 'r') as f:
                    X = f['X'][:].astype('int64')
                # Keep data on CPU; only transfer slices to GPU for forward pass
                X_cpu = torch.from_numpy(X)
                if device.type == 'cuda':
                    X_cpu = X_cpu.pin_memory()
                grid = model.cfg.latent_grid_dim
                n_samples = X_cpu.size(0)
                latent_dim = model.cfg.latent_dim
                latents_chunks: List[torch.Tensor] = []
                amp_ctx = (cuda_autocast if (device.type == 'cuda' and cuda_autocast is not None) else nullcontext)
                bs = int(max(1, encode_batch_size))
                for s in range(0, n_samples, bs):
                    e = min(n_samples, s + bs)
                    with amp_ctx():
                        X_dev = X_cpu[s:e].to(device, non_blocking=(device.type == 'cuda'))
                        logits3, z_e_seq, commit_loss, indices_list, stats_list, mask_tokens = model(X_dev)
                    lat_s = z_e_seq.detach().to('cpu').float().view(e - s, latent_dim, grid, grid)
                    latents_chunks.append(lat_s)
                    del X_dev, logits3, z_e_seq, commit_loss, indices_list, stats_list, mask_tokens
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                latents = torch.cat(latents_chunks, dim=0)
                del latents_chunks, X_cpu, X
                torch.save(latents, out_pt)
        # free
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    return per_chr_batches


def unify_batches(layout_json: str,
                  latents_root: str,
                  out_unified_root: str,
                  chromosomes: List[int],
                  embed_dtype: str = 'float32') -> None:
    ensure_dir_exists(out_unified_root)
    with open(layout_json, 'r') as jf:
        layout = json.load(jf)
    grid_M = int(layout.get('grid_M', 512))
    layout_map = {int(rec['chromosome']): rec for rec in layout['layout']}
    # Determine number of batches from one chromosome dir
    any_chr = chromosomes[0]
    lat_dir = os.path.join(latents_root, f"chr{any_chr}")
    batch_pts = sorted(glob.glob(os.path.join(lat_dir, 'batch*_latents.pt')))
    num_batches = len(batch_pts)
    print(f"[ENC] Unifying {num_batches} batches into {grid_M}x{grid_M} grid")
    for bi in range(1, num_batches + 1):
        # Prepare unified tensor shape: (N, C, M, M)
        # Load N and channel from any chr batch
        first_pt = os.path.join(latents_root, f"chr{any_chr}", f"batch{bi:05d}_latents.pt")
        L0 = torch.load(first_pt, map_location='cpu')  # (N, C, s, s)
        N, C, _, _ = L0.shape
        unified = torch.zeros((N, C, grid_M, grid_M), dtype=getattr(torch, embed_dtype))
        # Place each chromosome tile
        for chr_no in chromosomes:
            rec = layout_map.get(int(chr_no))
            if rec is None:
                continue
            tile = torch.load(os.path.join(latents_root, f"chr{chr_no}", f"batch{bi:05d}_latents.pt"), map_location='cpu')
            s = int(rec['tile_side'])
            x0 = int(rec['x0']); y0 = int(rec['y0'])
            unified[:, :, y0:y0+s, x0:x0+s] = tile.to(unified.dtype)
        out_pt = os.path.join(out_unified_root, f"batch{bi:05d}_unified.pt")
        torch.save(unified, out_pt)
        # free per-batch tensors
        del unified, L0
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def cleanup_temp(paths: List[str]) -> None:
    import shutil
    for p in paths:
        try:
            if os.path.isdir(p):
                shutil.rmtree(p)
            elif os.path.exists(p):
                os.remove(p)
        except Exception:
            pass


def main():
    p = argparse.ArgumentParser(description="Encode batched VQ-VAE latents per chromosome and unify via MILP layout")
    p.add_argument('--bfile', required=True)
    p.add_argument('--bim', required=True)
    p.add_argument('--fam', required=True)
    p.add_argument('--chromosomes', nargs='+', default=['all'])
    p.add_argument('--batch-size', type=int, default=12000, help='Individuals per batch when creating H5 caches')
    p.add_argument('--h5-out-root', required=True, help='Root dir for temporary batched H5 caches (genomic_data/unet_prep)')
    p.add_argument('--models-dir', required=True, help='Directory with per-chromosome VQ-VAE ckpts')
    p.add_argument('--model-pattern', default='vqvae_chr{chr}.pt')
    p.add_argument('--latents-out-root', required=True, help='Root dir for temporary per-chromosome latents (genomic_data/VQVAE_embeddings)')
    p.add_argument('--layout-json', required=True, help='Path to vqvae_milp_layout.json saved by orchestrator')
    p.add_argument('--unified-out-root', required=True, help='Root dir for unified batched embeddings')
    p.add_argument('--device', default='cuda')
    p.add_argument('--encode-batch-size', type=int, default=128, help='Micro-batch size for GPU forward pass during encoding')
    p.add_argument('--cleanup', action='store_true', help='Delete temporary H5 caches and per-chr latents after unifying')
    args = p.parse_args()

    chromosomes = get_chromosomes(args.chromosomes)
    ensure_dir_exists(args.h5_out_root)
    ensure_dir_exists(args.latents_out_root)
    ensure_dir_exists(args.unified_out_root)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith('cuda') else 'cpu')

    # 1) Create batched H5 caches per chromosome
    per_chr_batches = create_batched_h5(
        bfile=args.bfile,
        fam=args.fam,
        bim=args.bim,
        out_dir=args.h5_out_root,
        chromosomes=chromosomes,
        batch_size=int(args.batch_size)
    )
    print(f"[ENC] H5 caches created per chromosome: {per_chr_batches}")

    # 2) Encode per-chromosome batches into latents (.pt)
    per_chr_batches = encode_per_chr_batches(
        models_dir=args.models_dir,
        model_pattern=args.model_pattern,
        h5_root=args.h5_out_root,
        out_root=args.latents_out_root,
        chromosomes=chromosomes,
        device=device,
        encode_batch_size=int(args.encode_batch_size),
    )
    print(f"[ENC] Encoded latents per chromosome: {per_chr_batches}")

    # 3) Unify batched embeddings using saved MILP layout
    unify_batches(
        layout_json=args.layout_json,
        latents_root=args.latents_out_root,
        out_unified_root=args.unified_out_root,
        chromosomes=chromosomes,
    )
    print("[ENC] Unification complete")

    # 4) Cleanup temporary files
    # if args.cleanup:
    #     print("[ENC] Cleaning up temporary caches and per-chromosome latents")
    #     cleanup_temp([args.h5_out_root, args.latents_out_root])


if __name__ == '__main__':
    main()


# python encode_batched_vqvae.py \
#   --bfile ${VQVAE_DIR}/genomic_data/geneticBinary/ukb_allchr_unrel_britishWhite \
#   --bim ${VQVAE_DIR}/genomic_data/geneticBinary/ukb_allchr_unrel_britishWhite.bim \
#   --fam ${VQVAE_DIR}/genomic_data/geneticBinary/ukb_allchr_unrel_britishWhite_genome_encoder_test.fam \
#   --chromosomes all \
#   --batch-size 12000 \
#   --h5-out-root /n/home03/ahmadazim/WORKING/genGen/UKBVQVAE/genomic_data/unet_prep \
#   --models-dir /n/home03/ahmadazim/WORKING/genGen/UKBVQVAE/models/vqvae512 \
#   --model-pattern vqvae_chr{chr}.pt \
#   --latents-out-root /n/home03/ahmadazim/WORKING/genGen/UKBVQVAE/genomic_data/VQVAE_embeddings \
#   --layout-json /n/home03/ahmadazim/WORKING/genGen/UKBVQVAE/models/vqvae512/vqvae_milp_layout.json \
#   --unified-out-root /n/home03/ahmadazim/WORKING/genGen/UKBVQVAE/VQVAE_embeddings/final_embed \
#   --device cuda 