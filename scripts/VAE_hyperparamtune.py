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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# add the path to the DiffuGene package
import sys
sys.path.append('/n/home03/ahmadazim/WORKING/genGen/DiffuGene/src')
from DiffuGene.VAEembed.ae import GenotypeAutoencoder, VAEConfig
from DiffuGene.VAEembed.train import H5ChromosomeDataset
from DiffuGene.VAEembed.sharedEmbed import HomogenizedAE, FiLM2D
from DiffuGene.VAEembed.train_stage2 import load_ae_model
from DiffuGene.VAEembed.vae import GenotypeVAE, load_vae_from_ae_checkpoint, train_vae_variational

# Script arguments (only expose what's requested)
parser = argparse.ArgumentParser(description="Train VAE heads on top of AE backbone")
parser.add_argument("--ld-lambda", type=float, default=0.01, help="LD penalty weight")
parser.add_argument("--beta", type=float, default=1e-5, help="Final beta (KL weight) after warmup")
parser.add_argument("--bottleneck-channels", type=int, default=16, help="Number of bottleneck channels")
args = parser.parse_args() if __name__ == "__main__" else argparse.Namespace(ld_lambda=0.01, beta=1e-5, bottleneck_channels=16)
print(f"LD lambda: {args.ld_lambda}, Beta: {args.beta}")

# python -u VAE_hyperparamtune.py --ld-lambda 0.001 --beta 1e-6 --bottleneck-channels 16


# 1. Load AE checkpoint as VAE
vae_model, cfg_dict = load_vae_from_ae_checkpoint("/n/home03/ahmadazim/WORKING/genGen/UKBVQVAE/models/AE/ae_chr22.pt", device, bottleneck_channels=args.bottleneck_channels)

# 2. Build optimizer (same as build_vae)
# trainable_params = [p for p in vae_model.parameters() if p.requires_grad]
# print("Num trainable params:", sum(p.numel() for p in trainable_params))

optimizer = torch.optim.AdamW(
    vae_model.parameters(),
    lr=2e-4,
    betas=(0.9, 0.999),
    weight_decay=0.0,
)

# 3. Train with KL warmup
dataset = H5ChromosomeDataset('/n/home03/ahmadazim/WORKING/genGen/UKBVQVAE/genomic_data/vqvae_h5_cache', '22', 1)
train_dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=0, pin_memory=True)
val_dataset = H5ChromosomeDataset('/n/home03/ahmadazim/WORKING/genGen/UKBVQVAE/genomic_data/val_h5_cache', '22', 1)
val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)

result = train_vae_variational(
    vae_model,
    train_dataloader,
    optimizer,
    device=device,
    num_epochs=50,
    val_dataloader=val_dataloader,
    maf_lambda=1e-3,
    ld_lambda=args.ld_lambda,
    ld_window=128,
    beta_max=args.beta,
    beta_warmup_epochs=10,
)

# save the result
torch.save(result, f'/n/home03/ahmadazim/WORKING/genGen/UKBVQVAE/models/VAE/vae_chr22_warmup10_bc{args.bottleneck_channels}_beta{args.beta}_ld{args.ld_lambda}.pt')