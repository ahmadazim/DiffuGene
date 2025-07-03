#!/usr/bin/env python

import os
import re
import h5py
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from typing import Optional, List, Tuple, Dict

from ..utils import read_raw, get_logger
from ..block_embed import PCA_Block

logger = get_logger(__name__)

# Chromosome lengths (GRCh37/hg19 reference)
CHROMOSOME_LENGTHS = {
    1: 249250621,
    2: 243199373,
    3: 198022430,
    4: 191154276,
    5: 180915260,
    6: 171115067,
    7: 159138663,
    8: 146364022,
    9: 141213431,
    10: 135534747,
    11: 135006516,
    12: 133851895,
    13: 115169878,
    14: 107349540,
    15: 102531392,
    16: 90338345,
    17: 81195210,
    18: 78077248,
    19: 59128983,
    20: 63025520,
    21: 48129895,
    22: 51304566,
}

# TODO: update load_pca_block to allow for one block loading, instead of creating a new function
def load_single_pca_block(embeddings_dir: str, block_info: tuple) -> PCA_Block:
    """Load a single PCA block on-demand.
    
    Args:
        embeddings_dir: Directory containing PCA models
        block_info: Tuple of (chr, block_num, block_base) for file matching
        
    Returns:
        PCA_Block instance
    """
    chr_num, block_num, block_base = block_info
    
    load_dir = os.path.join(embeddings_dir, "loadings")
    mean_dir = os.path.join(embeddings_dir, "means")
    metadata_dir = os.path.join(embeddings_dir, "metadata")
    
    # Find the loading file
    load_file = os.path.join(load_dir, f"{block_base}_pca_loadings.pt")
    if not os.path.exists(load_file):
        # Try pattern matching for flexible filenames
        load_files = [f for f in os.listdir(load_dir) 
                     if f"chr{chr_num}_block{block_num}" in f and f.endswith("_pca_loadings.pt")]
        if not load_files:
            raise FileNotFoundError(f"No PCA loading file found for chr{chr_num}_block{block_num}")
        load_file = os.path.join(load_dir, load_files[0])
        # Update block_base for consistent file matching
        block_base = load_files[0].replace("_pca_loadings.pt", "")
    
    # Find corresponding means and metadata files
    means_file = os.path.join(mean_dir, f"{block_base}_pca_means.pt")
    metadata_file = os.path.join(metadata_dir, f"{block_base}_pca_metadata.pt")
    
    if not os.path.exists(means_file):
        # Try pattern matching
        means_files = [f for f in os.listdir(mean_dir) 
                      if f"chr{chr_num}_block{block_num}" in f and f.endswith("_pca_means.pt")]
        if not means_files:
            raise FileNotFoundError(f"No PCA means file found for chr{chr_num}_block{block_num}")
        means_file = os.path.join(mean_dir, means_files[0])
    
    if not os.path.exists(metadata_file):
        # Try pattern matching
        metadata_files = [f for f in os.listdir(metadata_dir) 
                         if f"chr{chr_num}_block{block_num}" in f and f.endswith("_pca_metadata.pt")]
        if metadata_files:
            metadata_file = os.path.join(metadata_dir, metadata_files[0])
    
    # Load the files
    loadings = torch.load(load_file, map_location="cuda", weights_only=False)
    means = torch.load(means_file, map_location="cuda", weights_only=False)
    
    # Load metadata if available
    if os.path.exists(metadata_file):
        metadata = torch.load(metadata_file, map_location="cpu", weights_only=False)
        k = metadata['k']
        actual_k = metadata['actual_k']
    else:
        # Fallback for old format
        if hasattr(loadings, 'size'):
            k = loadings.size(1)
        elif hasattr(loadings, 'shape'):
            k = loadings.shape[1]
        else:
            raise ValueError(f"Unsupported loadings type: {type(loadings)}")
        actual_k = k
    
    # Convert to PyTorch tensors
    if not isinstance(loadings, torch.Tensor):
        loadings = torch.from_numpy(loadings).float().cuda()
    else:
        loadings = loadings.float()
        
    if not isinstance(means, torch.Tensor):
        means = torch.from_numpy(means).float().cuda()
    else:
        means = means.float()
    
    pca_block = PCA_Block(k=k)
    pca_block.actual_k = actual_k
    pca_block.components_ = loadings.T if hasattr(loadings, 'T') else loadings.transpose()
    pca_block.means = means
    
    return pca_block

class BlockPCDataset(Dataset):
    """
    Memory-efficient dataset that reads PC embeddings from HDF5.
    SNPs are loaded separately on-demand via MemoryEfficientSNPLoader.
    """
    
    def __init__(self, 
                 emb_h5_path: str, 
                 spans_file: str, 
                 recoded_dir: str, 
                 embeddings_dir: str,
                 scale_pc_embeddings: bool = False):
        """
        Args:
            emb_h5_path: Path to HDF5 file with PC embeddings
            spans_file: CSV file with block information
            recoded_dir: Directory with recoded SNP files (for SNP loader setup)
            embeddings_dir: Directory with PCA loadings/means (for SNP loader setup)
            scale_pc_embeddings: Whether to apply PC standardization
        """
        self.emb_h5_path = emb_h5_path
        self.scale_pc_embeddings = scale_pc_embeddings
        self.recoded_dir = recoded_dir
        self.embeddings_dir = embeddings_dir
        
        # Read spans file and build block info
        logger.info(f"Loading spans file: {spans_file}")
        self.df = pd.read_csv(spans_file)
        self.N_blocks = len(self.df)
        
        # Build list of raw SNP file paths and block info for on-demand PCA loading
        self.block_raws = []
        self.block_info = []  # For on-demand PCA loading
        scaled_spans = []
        
        for _, row in self.df.iterrows():
            # Build raw file path
            filename = os.path.basename(row.block_file)
            block_match = re.search(r'block(\d+)', filename)
            if not block_match:
                raise ValueError(f"Could not extract block number from {filename}")
            block_no = block_match.group(1)
            basename = filename.split("_chr")[0]
            raw_fn = f"{basename}_chr{row.chr}_block{block_no}_recodeA.raw"
            raw_path = os.path.join(recoded_dir, raw_fn)
            self.block_raws.append(raw_path)
            
            # Store block info for on-demand PCA loading
            block_base = os.path.basename(row.block_file).replace(".pt", "")
            self.block_info.append((int(row.chr), int(block_no), block_base))
            
            # Build chromosome-aware spans: (chr_idx, start_norm, length_norm)
            chr_idx = int(row.chr)  # Keep as integer index (1-22)
            
            # Get chromosome-specific length
            if chr_idx not in CHROMOSOME_LENGTHS:
                raise ValueError(f"Unknown chromosome: {chr_idx}. Supported: {list(CHROMOSOME_LENGTHS.keys())}")
            chrom_length = CHROMOSOME_LENGTHS[chr_idx]
            
            # Normalize start position by chromosome length
            start_norm = row.start / chrom_length
            
            # Normalize block length by this chromosome's maximum length
            block_length = row.end - row.start
            length_norm = block_length / chrom_length
            
            scaled_spans.append([chr_idx, start_norm, length_norm])
        
        self.spans = torch.tensor(scaled_spans, dtype=torch.float32)
        logger.info(f"Spans shape: {self.spans.shape} (chr_idx, start_norm, length_norm)")
        
        # Get dataset dimensions from HDF5 metadata
        with h5py.File(emb_h5_path, "r") as h5f:
            self.N_train = h5f.attrs['n_train']
            self.pc_dim = h5f.attrs['pc_dim']
            logger.info(f"Dataset: {self.N_train} samples, {self.N_blocks} blocks, {self.pc_dim} PCs")
        
        # PC scaling setup
        self.pc_means = None
        self.pc_scales = None
        if scale_pc_embeddings:
            self._compute_pc_scaling()

    def _compute_pc_scaling(self):
        """Compute PC scaling statistics from the HDF5 data."""
        logger.info("Computing PC scaling statistics...")
        logger.info(f"Dataset dimensions: {self.N_train} samples × {self.N_blocks} blocks × {self.pc_dim} PCs")
        
        # Calculate total data size
        total_elements = self.N_train * self.N_blocks * self.pc_dim
        total_size_gb = total_elements * 4 / (1024**3)  # 4 bytes per float32
        logger.info(f"Total data size: {total_elements:,} elements = {total_size_gb:.2f} GB")
        
        # Read all data to compute statistics (this is done once at initialization)
        with h5py.File(self.emb_h5_path, "r") as h5f:
            # Read in chunks to avoid memory issues
            chunk_size = min(10000, self.N_train)
            means_sum = np.zeros(self.pc_dim, dtype=np.float64)
            vars_sum = np.zeros(self.pc_dim, dtype=np.float64)
            total_elements = 0
            
            for start_idx in range(0, self.N_train, chunk_size):
                end_idx = min(start_idx + chunk_size, self.N_train)
                chunk = h5f["pc"][start_idx:end_idx]  # (chunk_size, N_blocks, pc_dim)
                
                # Flatten to compute global statistics
                chunk_flat = chunk.reshape(-1, self.pc_dim)  # (chunk_size * N_blocks, pc_dim)
                
                means_sum += chunk_flat.sum(axis=0)
                vars_sum += (chunk_flat ** 2).sum(axis=0)
                total_elements += chunk_flat.shape[0]
        
        # Compute global mean and std
        global_mean = means_sum / total_elements
        global_var = (vars_sum / total_elements) - (global_mean ** 2)
        global_std = np.sqrt(global_var)
        
        # Avoid division by zero for padded dimensions
        global_std[global_std == 0] = 1.0
        
        self.pc_means = torch.from_numpy(global_mean).float()
        self.pc_scales = torch.from_numpy(global_std).float()
        
        logger.info(f"PC scaling - Mean range: [{global_mean.min():.4f}, {global_mean.max():.4f}], "
                   f"Std range: [{global_std.min():.4f}, {global_std.max():.4f}]")

    def __len__(self):
        return self.N_train

    def __getitem__(self, idx):
        # Lazy-open HDF5 in each worker process
        if not hasattr(self, "pc_dset"):
            self._h5f = h5py.File(self.emb_h5_path, "r", libver="latest", swmr=True)
            self.pc_dset = self._h5f["pc"]

        # Read PC embedding for this sample: (N_blocks, pc_dim)
        emb = torch.from_numpy(self.pc_dset[idx]).float()
        
        # Apply PC scaling if requested
        if self.scale_pc_embeddings and self.pc_means is not None:
            emb = (emb - self.pc_means.unsqueeze(0)) / self.pc_scales.unsqueeze(0)
        
        # Always return embeddings, spans, and sample index
        # SNP loading is handled separately by MemoryEfficientSNPLoader when needed
        return emb, self.spans, idx

    def __del__(self):
        # Clean up HDF5 file handle
        if hasattr(self, '_h5f'):
            self._h5f.close()


class MemoryEfficientSNPLoader:
    """
    Helper class for loading SNPs and PCA blocks on-demand during training/evaluation.
    Handles random block sampling and efficient SNP/PCA loading with caching.
    This is only used when SNPs are actually needed (SNP loss or evaluation).
    """
    
    def __init__(self, dataset: BlockPCDataset, snp_blocks_per_batch: int = 64, pca_cache_size: int = 128, pca_metadata: Optional[List[Dict]] = None):
        """
        Args:
            dataset: BlockPCDataset instance
            snp_blocks_per_batch: Number of blocks to sample per batch
            pca_cache_size: Number of PCA blocks to keep in memory (LRU cache)
            pca_metadata: Optional metadata for filtering blocks with insufficient SNPs
        """
        self.dataset = dataset
        self.snp_blocks_per_batch = snp_blocks_per_batch
        self.block_raws = dataset.block_raws
        self.block_info = dataset.block_info
        self.embeddings_dir = dataset.embeddings_dir
        self.N_blocks = dataset.N_blocks
        
        # PCA block cache (LRU-style)
        self.pca_cache: Dict[int, PCA_Block] = {}
        self.pca_cache_size = pca_cache_size
        self.pca_access_order = []  # For LRU eviction
        
        # Filter blocks with sufficient SNPs for meaningful reconstruction
        self.valid_block_indices = list(range(self.N_blocks))  # Default: all blocks
        if pca_metadata is not None:
            pc_dim = dataset.pc_dim
            valid_blocks = []
            filtered_count = 0
            
            for i, meta in enumerate(pca_metadata):
                n_snps = meta.get('n_snps', 0)
                # Only include blocks with more SNPs than PCs (e.g., >4 SNPs for 4 PCs)
                if n_snps > pc_dim:
                    valid_blocks.append(i)
                else:
                    if n_snps > 0:  # Only log if we have SNP count info
                        filtered_count += 1
            
            self.valid_block_indices = valid_blocks
            if filtered_count > 0:
                logger.info(f"Filtered out {filtered_count} blocks with ≤{pc_dim} SNPs from sampling "
                           f"({len(self.valid_block_indices)}/{self.N_blocks} blocks remain)")
        
        if len(self.valid_block_indices) == 0:
            raise ValueError("No blocks have sufficient SNPs for reconstruction!")
        
        logger.info(f"SNP loader: {snp_blocks_per_batch} blocks per batch from {len(self.valid_block_indices)} valid blocks")
        logger.info(f"PCA cache size: {pca_cache_size} blocks")

    def _get_pca_block(self, block_idx: int) -> PCA_Block:
        """Get PCA block with LRU caching."""
        if block_idx in self.pca_cache:
            # Move to end of access order (most recently used)
            self.pca_access_order.remove(block_idx)
            self.pca_access_order.append(block_idx)
            return self.pca_cache[block_idx]
        
        # Load PCA block on-demand
        block_info = self.block_info[block_idx]
        pca_block = load_single_pca_block(self.embeddings_dir, block_info)
        
        # Add to cache
        self.pca_cache[block_idx] = pca_block
        self.pca_access_order.append(block_idx)
        
        # Evict oldest if cache is full
        if len(self.pca_cache) > self.pca_cache_size:
            oldest_idx = self.pca_access_order.pop(0)
            del self.pca_cache[oldest_idx]
        
        return pca_block

    def load_snps_for_batch(self, sample_indices: torch.Tensor, device: torch.device) -> Tuple[List[int], List[torch.Tensor]]:
        """
        Load SNPs for a random subset of blocks for the given samples.
        Only samples from blocks with sufficient SNPs for meaningful reconstruction.
        
        Args:
            sample_indices: Tensor of sample indices in the batch
            device: Target device for tensors
            
        Returns:
            block_indices: List of selected block indices
            true_snps: List of tensors with true SNPs for each selected block
        """
        # Randomly sample blocks from valid blocks only
        n_blocks_to_sample = min(self.snp_blocks_per_batch, len(self.valid_block_indices))
        sampled_positions = torch.randperm(len(self.valid_block_indices))[:n_blocks_to_sample]
        block_indices = [self.valid_block_indices[pos] for pos in sampled_positions]
        
        true_snps = []
        for block_idx in block_indices:
            # Debug: Log block info
            block_info = self.block_info[block_idx]
            chr_num, block_num, block_base = block_info
            
            # Load raw SNP data for this block
            raw_path = self.block_raws[block_idx]
            if not os.path.exists(raw_path):
                raise FileNotFoundError(f"SNP file not found: {raw_path}")
            
            # Read and process SNP data
            rec = read_raw(raw_path)
            X = rec.impute().get_variants()  # (N_train, n_snps)
            
            # Extract SNPs for the current batch samples
            batch_snps = torch.from_numpy(X[sample_indices.cpu().numpy()]).float().to(device)
            
            # Debug: Log true SNP dimensions (first few blocks only to avoid spam)
            if block_idx < 5:  # Only log first few blocks to avoid spam
                logger.info(f"LOAD Block {block_idx} (chr{chr_num}_block{block_num}): "
                           f"True SNPs shape: {batch_snps.shape}, Raw file: {os.path.basename(raw_path)}")
            
            true_snps.append(batch_snps)
        
        return block_indices, true_snps

    def decode_predictions(self, recon_emb: torch.Tensor, block_indices: List[int], device: torch.device) -> List[torch.Tensor]:
        """
        Decode PC embeddings back to SNP space for selected blocks using on-demand PCA loading.
        
        Args:
            recon_emb: Reconstructed embeddings (batch_size, N_blocks, pc_dim)
            block_indices: List of block indices to decode
            device: Target device for tensors
            
        Returns:
            pred_snps: List of predicted SNP tensors for each block
        """
        pred_snps = []
        for block_idx in block_indices:
            # Extract reconstructed PCs for this block
            block_recon = recon_emb[:, block_idx, :]  # (batch_size, pc_dim)
            
            # Get PCA block on-demand (with caching)
            pca_block = self._get_pca_block(block_idx)
            
            # Debug: Log PCA block dimensions
            block_info = self.block_info[block_idx]
            chr_num, block_num, block_base = block_info
            
            # Decode using PCA
            pred_snp = pca_block.decode(block_recon)
            
            # Debug: Log predicted SNP dimensions (first few blocks only to avoid spam)
            if block_idx < 5:  # Only log first few blocks to avoid spam
                logger.info(f"DECODE Block {block_idx} (chr{chr_num}_block{block_num}): "
                           f"Predicted SNPs shape: {pred_snp.shape}")
            
            pred_snps.append(pred_snp)
        
        return pred_snps


def unscale_embeddings(embeddings: torch.Tensor, pc_means: Optional[torch.Tensor], pc_scales: Optional[torch.Tensor]) -> torch.Tensor:
    """
    Reverse PC scaling applied during training.
    
    Args:
        embeddings: Scaled embeddings (batch_size, n_blocks, pc_dim)
        pc_means: Means used for scaling (pc_dim,)
        pc_scales: Standard deviations used for scaling (pc_dim,)
    
    Returns:
        Unscaled embeddings in original PC space
    """
    if pc_means is None or pc_scales is None:
        return embeddings
    
    device = embeddings.device
    pc_means = pc_means.to(device)
    pc_scales = pc_scales.to(device)
    
    # Reverse standardization: x_orig = x_scaled * std + mean
    return embeddings * pc_scales.unsqueeze(0).unsqueeze(0) + pc_means.unsqueeze(0).unsqueeze(0) 