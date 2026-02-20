#!/usr/bin/env python

import os
import argparse
import torch
from typing import Optional

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


def main():
    p = argparse.ArgumentParser(description="Decode VQ-VAE latents back to genotype logits")
    p.add_argument("--model", required=True)
    p.add_argument("--latents-file", required=True, help="Tensor file (B,C,H,W)")
    p.add_argument("--out-file", required=True, help="Output logits tensor file (B,3,L)")
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith('cuda') else 'cpu')
    model = load_model(args.model, device)

    z = torch.load(args.latents_file, weights_only=False).to(device)
    if z.dim() != 4:
        raise ValueError(f"Expected latents of shape (B,C,H,W), got {tuple(z.shape)}")
    B, C, H, W = z.shape
    if C != model.cfg.latent_dim or H != model.cfg.latent_grid_dim or W != model.cfg.latent_grid_dim:
        raise ValueError("Latent shape does not match model config")

    with torch.no_grad():
        z_seq = z.view(B, C, H*W)
        logits3 = model.decode_logits(z_seq)

    ensure_dir_exists(os.path.dirname(args.out_file))
    torch.save(logits3.cpu(), args.out_file)
    logger.info(f"Wrote decoded logits to {args.out_file} with shape {tuple(logits3.shape)}")


if __name__ == "__main__":
    main()


