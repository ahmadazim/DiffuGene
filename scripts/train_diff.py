import os
import argparse
import glob
from typing import List, Optional, Tuple, Dict, Any
import bisect
from dataclasses import asdict

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import h5py
import torch.nn.functional as F

import sys
sys.path.insert(0, '/n/home03/ahmadazim/WORKING/genGen/DiffuGene')

from src.DiffuGene.VAEembed.ae import (
    GenotypeAutoencoder as SrcGenotypeAutoencoder,
    VAEConfig,
    build_vae,
    train_vae as train_vae_fn,
    find_best_ck,
)
from src.DiffuGene.utils import setup_logging, get_logger

from importlib import reload
import notebooks.compare_SiT_utils as compare_utils
reload(compare_utils)

# read command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model_type", type=str, required=True, choices=["unet", "dit", "sit", 'udit', 'usit'])
parser.add_argument("--normalize_latents", action="store_true", default=False)
args = parser.parse_args()
model_type = args.model_type
normalize_latents = args.normalize_latents

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

chr_no = 22
home = '/n/home03/ahmadazim/WORKING/genGen/UKBVQVAE/'
compare_dir = '/n/home03/ahmadazim/WORKING/genGen/UKB_compare_chr22'
model_dir = os.path.join(home, 'models')
max_h5_batches = 6

h5_batch_path = f"/n/home03/ahmadazim/WORKING/genGen/UKBVQVAE/genomic_data/ae_h5/chr{chr_no}/batch00001.h5"
h5_paths = compare_utils.enumerate_h5_batch_paths(h5_batch_path, max_batches=max_h5_batches)

L = 9661
latent_length = 64
latent_dim = 256
embed_dim = 8
num_epochs = 5

if model_type == "unet":
    def load_ae_from_checkpoint(ae_ckpt_path: str, device: torch.device) -> Tuple[SrcGenotypeAutoencoder, VAEConfig]:
        """
        Load a trained AE checkpoint that includes a 'config' and a 'model_state'.
        """
        payload = torch.load(ae_ckpt_path, map_location="cpu")
        if not isinstance(payload, dict):
            raise ValueError(f"AE checkpoint must be a dict payload: {ae_ckpt_path}")
        cfg_dict = payload.get("config")
        state = payload.get("model_state")
        if cfg_dict is None or state is None:
            raise KeyError(
                f"AE checkpoint missing 'config' or a valid state dict "
                f"('model_state'): {ae_ckpt_path}"
            )
        cfg = VAEConfig(**cfg_dict)
        ae = SrcGenotypeAutoencoder(
            input_length=cfg.input_length,
            K1=cfg.K1,
            K2=cfg.K2,
            C=cfg.C,
            embed_dim=cfg.embed_dim,
        )
        incompat = ae.load_state_dict(state, strict=True)
        if getattr(incompat, "missing_keys", None):
            print(f"[AE] Missing keys (expected for new heads or buffers): {incompat.missing_keys}")
        if getattr(incompat, "unexpected_keys", None):
            print(f"[AE] Unexpected keys present in checkpoint: {incompat.unexpected_keys}")
        ae.to(device).eval()
        for p in ae.parameters():
            p.requires_grad = False
        return ae, cfg
    ae_dir = os.path.join(model_dir, 'AE')
    ae_model_path = os.path.join(ae_dir, f'ae_chr{chr_no}.pt')  # will use non-homogenized model
    ae, cfg = load_ae_from_checkpoint(ae_model_path, device=device)
    print(f"[AE] Loaded AE with input_length={cfg.input_length}, latent_channels={ae.latent_channels}, M2D={ae.M2D}")
    
    # Avoid H5 lock contention when multiple jobs run concurrently.
    # Use a job-specific latent file for UNet training.
    job_id = os.environ.get("SLURM_JOB_ID", str(os.getpid()))
    latents_h5 = os.path.join(compare_dir, f"latents_16x16x64_job{job_id}.h5")
    latent_meta = compare_utils.build_latents_h5(
        ae=ae,
        h5_paths=h5_paths,
        out_latents_h5=latents_h5,
        device=device,
        batch_size=256,
        latent_key="Z",
        compression="gzip",
        compression_opts=4,
        chunk_rows=1024,
        dtype="float16",
        limit_total_examples=None
    )

    unet, unet_ema = compare_utils.train_unet_on_latents_h5(
        latents_h5=latents_h5,
        latent_key="Z",
        num_epochs=num_epochs,
        batch_size=64,
        lr=1e-4,
        num_train_timesteps=1000,
        device=device,
        num_workers=0,
        pin_memory=True,
        amp=True,
        log_every=50,
        ema_decay=0.99,
        use_ema=True,
    )

    unet_ckpt = os.path.join(compare_dir, f"unet_chr{chr_no}_vpred_job{job_id}.pt")
    torch.save(
        {
            "model_state": unet.state_dict(),
            "latent_meta": latent_meta,
            "latents_h5": latents_h5,
        },
        unet_ckpt,
    )
    print(f"[UNET] Saved checkpoint: {unet_ckpt}")

else: 
    ae1d_ckpt = os.path.join(compare_dir, f"ae1d_chr{chr_no}_L{latent_length}_D{latent_dim}.pt")
    ae1d_dict = torch.load(ae1d_ckpt, weights_only=False)
    ae = compare_utils.GenotypeAutoencoder(
        input_length=ae1d_dict["config"]["input_length"],
        latent_length=ae1d_dict["config"]["latent_length"],
        latent_dim=ae1d_dict["config"]["latent_dim"],
        embed_dim=ae1d_dict["config"]["embed_dim"],
    ).to(device)
    ae.load_state_dict(ae1d_dict["model_state"])

    if model_type == "dit":
        dit_ckpt = os.path.join(compare_dir, f"dit_chr{chr_no}_L{latent_length}_D{latent_dim}.pt")
        dit, dit_ema, latent_norm_stats = compare_utils.train_dit_latent_diffusion_h5(
            ae=ae,
            h5_paths=h5_paths,
            num_epochs=num_epochs,
            batch_size=64,
            lr=2e-4,
            device=device,
            num_workers=2,
            pin_memory=True,
            amp=True,
            log_every=50,
            num_train_timesteps=1000,
            num_layers=9,
            num_heads=8,
            mlp_ratio=4,
            ema_decay=0.999,
            use_ema=True,
            use_udit=False,
            use_latent_normalization=normalize_latents,
        )
        torch.save(
            {
                "dit_state": dit.state_dict(),
                "dit_ema_state": (dit_ema.state_dict() if dit_ema is not None else None),
                "latent_length": latent_length,
                "latent_dim": latent_dim,
                "chr_no": chr_no,
            },
            dit_ckpt,
        )
        print(f"[DiT] Saved checkpoint: {dit_ckpt}")
    
    if model_type == "udit":
        udit_ckpt = os.path.join(compare_dir, f"udit_chr{chr_no}_L{latent_length}_D{latent_dim}.pt")
        udit, udit_ema, latent_norm_stats = compare_utils.train_dit_latent_diffusion_h5(
            ae=ae,
            h5_paths=h5_paths,
            num_epochs=num_epochs,
            batch_size=64,
            lr=2e-4,
            device=device,
            num_workers=2,
            pin_memory=True,
            amp=True,
            log_every=50,
            num_train_timesteps=1000,
            num_layers=9,
            num_heads=8,
            mlp_ratio=4,
            ema_decay=0.999,
            use_ema=True,
            use_udit=True,
            use_latent_normalization=normalize_latents,
        )
        torch.save(
            {
                "udit_state": udit.state_dict(),
                "udit_ema_state": (udit_ema.state_dict() if udit_ema is not None else None),
                "latent_length": latent_length,
                "latent_dim": latent_dim,
                "chr_no": chr_no,
            },
            udit_ckpt,
        )
        print(f"[UDiT] Saved checkpoint: {udit_ckpt}")
    
    if model_type == "sit":
        sit_ckpt = os.path.join(compare_dir, f"sit_chr{chr_no}_L{latent_length}_D{latent_dim}.pt")
        sit, sit_ema, latent_norm_stats = compare_utils.train_sit_flow_matching_h5(
            ae=ae,
            h5_paths=h5_paths,
            num_epochs=num_epochs,
            batch_size=64,
            lr=2e-4,
            device=device,
            num_workers=2,
            pin_memory=True,
            amp=True,
            log_every=50,
            num_layers=9,
            num_heads=8,
            mlp_ratio=4,
            ema_decay=0.999,
            use_ema=True,
            use_udit=False,
            use_latent_normalization=normalize_latents,
        )
        torch.save(
            {
                "sit_state": sit.state_dict(),
                "sit_ema_state": (sit_ema.state_dict() if sit_ema is not None else None),
                "latent_length": latent_length,
                "latent_dim": latent_dim,
                "chr_no": chr_no,
            },
            sit_ckpt,
        )
        print(f"[SiT] Saved checkpoint: {sit_ckpt}")
    
    if model_type == "usit":
        if normalize_latents:
            usit_ckpt = os.path.join(compare_dir, f"usit_norm_chr{chr_no}_L{latent_length}_D{latent_dim}.pt")
        else:
            usit_ckpt = os.path.join(compare_dir, f"usit_chr{chr_no}_L{latent_length}_D{latent_dim}.pt")
        usit, usit_ema, latent_norm_stats = compare_utils.train_sit_flow_matching_h5(
            ae=ae,
            h5_paths=h5_paths,
            num_epochs=num_epochs,
            batch_size=64,
            lr=2e-4,
            device=device,
            num_workers=2,
            pin_memory=True,
            amp=True,
            log_every=50,
            num_layers=9,
            num_heads=8,
            mlp_ratio=4,
            ema_decay=0.999,
            use_ema=True,
            use_udit=True,
            use_latent_normalization=normalize_latents,
        )
        torch.save(
            {
                "usit_state": usit.state_dict(),
                "usit_ema_state": (usit_ema.state_dict() if usit_ema is not None else None),
                "latent_length": latent_length,
                "latent_dim": latent_dim,
                "chr_no": chr_no,
            },
            usit_ckpt,
        )
        print(f"[USiT] Saved checkpoint: {usit_ckpt}")
        