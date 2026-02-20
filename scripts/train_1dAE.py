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
    GenotypeAutoencoder,
    VAEConfig,
    build_vae,
    train_vae as train_vae_fn,
    find_best_ck,
)
from src.DiffuGene.utils import setup_logging, get_logger

from importlib import reload
import notebooks.compare_SiT_utils
reload(notebooks.compare_SiT_utils)
from notebooks.compare_SiT_utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

chr_no = 22
home = '/n/home03/ahmadazim/WORKING/genGen/UKBVQVAE/'
compare_dir = '/n/home03/ahmadazim/WORKING/genGen/UKB_compare_chr22'
model_dir = os.path.join(home, 'models')
max_h5_batches = 6

h5_batch_path = f"/n/home03/ahmadazim/WORKING/genGen/UKBVQVAE/genomic_data/ae_h5/chr{chr_no}/batch00001.h5"
h5_paths = enumerate_h5_batch_paths(h5_batch_path, max_batches=max_h5_batches)

L = 9661
latent_length = 32
latent_dim = 256
embed_dim = 8

ae1d_ckpt = os.path.join(compare_dir, f"ae1d_chr{chr_no}_L{latent_length}_D{latent_dim}.pt")
ae1d = train_1d_autoencoder_h5(
    h5_paths=h5_paths,
    input_length=L,
    latent_length=latent_length,
    latent_dim=latent_dim,
    embed_dim=embed_dim,
    num_epochs=50,
    batch_size=128,
    lr=2e-4,
    device=device,
    num_workers=2,
    pin_memory=True,
    amp=True,
    lam_maf=1e-3,
    lam_ld=1e-3,
    lam_tv=0.0,
    ld_window=128,
    log_every=50,
    ckpt_path=ae1d_ckpt,
)

torch.save({"model_state": ae1d.state_dict(), "config": ae1d.config}, ae1d_ckpt)