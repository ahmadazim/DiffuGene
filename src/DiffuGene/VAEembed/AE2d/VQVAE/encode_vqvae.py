#!/usr/bin/env python

import os
import argparse
import glob
from typing import List

import torch
import h5py
import numpy as np
from tqdm import tqdm

from .vae import SNPVQVAE, VQVAEConfig
from ..utils import ensure_dir_exists, get_logger


logger = get_logger(__name__)


def load_model(model_path: str, device: torch.device) -> SNPVQVAE:
    ckpt = torch.load(model_path, map_location='cpu')
    cfg_dict = ckpt.get('config')
    if cfg_dict is None:
        raise ValueError("Checkpoint missing 'config'")
    cfg = VQVAEConfig(**cfg_dict)
    model = SNPVQVAE(cfg)
    model.load_state_dict(ckpt['model_state'])
    model.to(device)
    model.eval()
    return model


def enumerate_batches(h5_dir: str, chromosomes: List[int]) -> List[str]:
    all_files = glob.glob(os.path.join(h5_dir, "batch*_chr*.h5"))
    batch_ids = sorted({os.path.basename(p).split("_")[0] for p in all_files})
    # validate completeness
    for b in batch_ids:
        for c in chromosomes:
            p = os.path.join(h5_dir, f"{b}_chr{c}.h5")
            if not os.path.exists(p):
                raise FileNotFoundError(f"Missing {p}")
    return batch_ids


def load_batch_concat(h5_dir: str, batch_id: str, chromosomes: List[int]) -> torch.Tensor:
    arrays = []
    for c in chromosomes:
        p = os.path.join(h5_dir, f"{batch_id}_chr{c}.h5")
        with h5py.File(p, 'r') as f:
            arrays.append(torch.from_numpy(f['X'][:].astype('int64')))
    X = torch.cat(arrays, dim=1)
    return X  # (B, L_total)


def main():
    p = argparse.ArgumentParser(description="Encode genotypes to VQ-VAE latents and export UNet-ready arrays")
    p.add_argument("--model", required=True)
    p.add_argument("--h5-dir", required=True)
    p.add_argument("--chromosomes", nargs='+', type=int, default=list(range(1,23)))
    p.add_argument("--out-dir", required=True, help="Directory for encoded latents (.pt batches) and optional memmap")
    p.add_argument("--write-memmap", action='store_true', help="Also write consolidated memmap .npy for UNet")
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    ensure_dir_exists(args.out_dir)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith('cuda') else 'cpu')
    model = load_model(args.model, device)

    batches = enumerate_batches(args.h5_dir, args.chromosomes)
    latent_dim = model.cfg.latent_dim
    grid = model.cfg.latent_grid_dim
    sample_shape = (latent_dim, grid, grid)

    batch_paths = []
    for b in tqdm(batches, desc="Encoding batches"):
        X = load_batch_concat(args.h5_dir, b, args.chromosomes).to(device)
        with torch.no_grad():
            logits3, z_q_seq, _, _, _, _ = model(X)
        # reshape to (B, C, H, W)
        B = z_q_seq.size(0)
        latents = z_q_seq.view(B, latent_dim, grid, grid).cpu().float()
        out_path = os.path.join(args.out_dir, f"latents_{b}.pt")
        torch.save(latents, out_path)
        batch_paths.append(out_path)
        # free
        del X, logits3, z_q_seq, latents
        if device.type == 'cuda':
            torch.cuda.synchronize(); torch.cuda.empty_cache()

    # Optionally write a memmap for UNet training
    if args.write_memmap:
        import numpy as np
        memmap_path = os.path.join(args.out_dir, "unet_latents_memmap.npy")
        # First pass: sizes
        total = 0
        for pth in batch_paths:
            t = torch.load(pth, weights_only=False)
            total += t.shape[0]
        full_shape = (total,) + sample_shape
        arr = np.memmap(memmap_path, dtype='float32', mode='w+', shape=full_shape)
        off = 0
        for pth in tqdm(batch_paths, desc="Writing memmap"):
            t = torch.load(pth, weights_only=False).numpy()
            n = t.shape[0]
            arr[off:off+n] = t
            off += n
        del arr
        with open(memmap_path.replace('.npy', '_shape.txt'), 'w') as f:
            f.write(','.join(map(str, full_shape)))
        logger.info(f"Wrote memmap {memmap_path} with shape {full_shape}")


if __name__ == "__main__":
    main()


