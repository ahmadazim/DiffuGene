#!/usr/bin/env python

import argparse
import os
import h5py
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

from ..utils import get_logger

logger = get_logger(__name__)

def precompute_embeddings(spans_file: str, h5_path: str, n_train: int, pc_dim: int = None):
    """
    Read individual block embedding .pt files and create a single HDF5 file.
    
    Processes blocks in the exact order specified by spans_file to maintain correct block indexing for multi-chromosome data.
    
    Args:
        spans_file: CSV file with columns [block_file, chr, start, end]
        h5_path: Output HDF5 file path
        n_train: Number of training samples
        pc_dim: PC dimension (if None, auto-detect from first block)
    """
    logger.info(f"Reading spans file: {spans_file}")
    df = pd.read_csv(spans_file)
    N_blocks = len(df)
    
    # Auto-detect PC dimension from first block if not provided
    if pc_dim is None:
        first_emb = torch.load(df.iloc[0].block_file, weights_only=False)
        if not isinstance(first_emb, torch.Tensor):
            first_emb = torch.tensor(first_emb, dtype=torch.float32)
        pc_dim = first_emb.shape[-1]
        logger.info(f"Auto-detected PC dimension: {pc_dim}")
    
    logger.info(f"Creating HDF5 file: {h5_path}")
    logger.info(f"Shape: ({n_train}, {N_blocks}, {pc_dim})")
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(h5_path), exist_ok=True)
    
    # Create HDF5 file with contiguous storage for random access
    with h5py.File(h5_path, "w") as h5f:
        # No chunking - store contiguously for optimal random sample access
        # Training accesses dataset[random_idx] -> (N_blocks, pc_dim)
        dset = h5f.create_dataset(
            "pc", 
            shape=(n_train, N_blocks, pc_dim),
            dtype="f4",
            compression=None,
            shuffle=False
        )
        
        # Process each block in spans file order
        logger.info("Loading all block embeddings into memory...")  #TODO: remove this, sort of defeats the purpose of this memory efficient dataset
        all_embeddings = np.zeros((n_train, N_blocks, pc_dim), dtype=np.float32)
        
        for j, row in enumerate(tqdm(df.itertuples(), total=N_blocks, desc="Loading blocks")):
            # Load embedding file
            emb = torch.load(row.block_file, weights_only=False)
            if not isinstance(emb, torch.Tensor):
                emb = torch.tensor(emb, dtype=torch.float32)
            else:
                emb = emb.float()
            
            # Validate shape
            if emb.shape[0] != n_train:
                raise ValueError(f"Block {j} (chr{row.chr}_block) has {emb.shape[0]} samples, expected {n_train}")
            
            # Handle dimension mismatch
            if emb.shape[-1] < pc_dim:
                # Pad with zeros
                pad_size = pc_dim - emb.shape[-1]
                if emb.dim() == 2:
                    padding = torch.zeros(emb.shape[0], pad_size, dtype=torch.float32)
                    emb = torch.cat([emb, padding], dim=1)
                logger.debug(f"Padded chr{row.chr}_block{j} from {emb.shape[-1] - pad_size} to {emb.shape[-1]} dimensions")
            elif emb.shape[-1] > pc_dim:
                # Truncate (shouldn't happen if auto-detected correctly)
                emb = emb[:, :pc_dim]
                logger.warning(f"Truncated chr{row.chr}_block{j} from {emb.shape[-1] + (emb.shape[-1] - pc_dim)} to {emb.shape[-1]} dimensions")
            
            all_embeddings[:, j, :] = emb.numpy()
        
        # Write all data at once (much faster for contiguous storage)
        logger.info("Writing all data to HDF5...")
        dset[:] = all_embeddings
        
        # Store metadata
        h5f.attrs['n_train'] = n_train
        h5f.attrs['n_blocks'] = N_blocks
        h5f.attrs['pc_dim'] = pc_dim
        h5f.attrs['spans_file'] = spans_file
        
        logger.info(f"Successfully created HDF5 file: {h5_path}")
        logger.info(f"File size: {os.path.getsize(h5_path) / (1024**3):.2f} GB")
        logger.info(f"Block order preserved from spans file for multi-chromosome compatibility")


def main():
    parser = argparse.ArgumentParser(description="Precompute block embeddings into HDF5 format")
    parser.add_argument("--spans_file", required=True, help="Spans CSV file")
    parser.add_argument("--h5_path", required=True, help="Output HDF5 file path")
    parser.add_argument("--n_train", type=int, required=True, help="Number of training samples")
    parser.add_argument("--pc_dim", type=int, help="PC dimension (auto-detect if not provided)")
    
    args = parser.parse_args()
    
    precompute_embeddings(args.spans_file, args.h5_path, args.n_train, args.pc_dim)


if __name__ == "__main__":
    main() 