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
    """
    Load a single PCA block on-demand.
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
    
    load_file = os.path.join(load_dir, f"{block_base}_pca_loadings.pt")
    if not os.path.exists(load_file):
        raise FileNotFoundError(f"No PCA loading file found for chr{chr_num}_block{block_num}: {load_file}")
    
    means_file = os.path.join(mean_dir, f"{block_base}_pca_means.pt")
    metadata_file = os.path.join(metadata_dir, f"{block_base}_pca_metadata.pt")
    if not os.path.exists(means_file):
        raise FileNotFoundError(f"No PCA means file found for chr{chr_num}_block{block_num}: {means_file}")
    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"No PCA metadata file found for chr{chr_num}_block{block_num}: {metadata_file}")
        
    loadings = torch.load(load_file, map_location="cuda", weights_only=False)
    means = torch.load(means_file, map_location="cuda", weights_only=False)
    
    metadata = torch.load(metadata_file, map_location="cpu", weights_only=False)
    k = metadata['k']
    actual_k = metadata['actual_k']
    
    # Convert to PyTorch tensors
    loadings = loadings.float() if isinstance(loadings, torch.Tensor) else torch.from_numpy(loadings).float().cuda()
    means = means.float() if isinstance(means, torch.Tensor) else torch.from_numpy(means).float().cuda()
    
    pca_block = PCA_Block(k=k)
    pca_block.actual_k = actual_k
    pca_block.components_ = loadings.T if hasattr(loadings, 'T') else loadings.transpose()
    pca_block.means = means
    
    return pca_block

class BlockPCDataset(Dataset):
    """
    Memory-efficient dataset that reads PC embeddings from HDF5.
    SNPs are loaded separately on-demand via SNPLoader.
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
        self.spans_df = pd.read_csv(spans_file)
        self.N_blocks = len(self.spans_df)
        
        # Build list of raw SNP file paths and block info for on-demand PCA loading
        self.block_raws = []
        self.block_info = []
        scaled_spans = []
        
        for _, row in self.spans_df.iterrows():
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
            block_base = os.path.basename(row.block_file).replace("_embeddings.pt", "")
            self.block_info.append((int(row.chr), int(block_no), block_base))
            
            # Build scaled spans
            chr_idx = int(row.chr)
            if chr_idx not in CHROMOSOME_LENGTHS:
                raise ValueError(f"Unknown chromosome: {chr_idx}. Supported: {list(CHROMOSOME_LENGTHS.keys())}")
            chrom_length = CHROMOSOME_LENGTHS[chr_idx]
            start_norm = row.start / chrom_length
            end_norm = row.end / chrom_length
            length_norm = end_norm - start_norm
            scaled_spans.append([chr_idx, start_norm, length_norm])
        
        self.spans = torch.tensor(scaled_spans, dtype=torch.float32)
        logger.info(f"Spans shape: {self.spans.shape} (chr_idx, start_norm, length_norm)")
        
        # Open HDF5 file and keep it open for the lifetime of the dataset
        self._h5f = h5py.File(emb_h5_path, "r")
        self.N_train = self._h5f.attrs['n_train']
        self.pc_dim = self._h5f.attrs['pc_dim']
        logger.info(f"Dataset: {self.N_train} samples, {self.N_blocks} blocks, {self.pc_dim} PCs")
        
        self.pc_means = None
        self.pc_scales = None
        if scale_pc_embeddings:
            self._compute_pc_scaling()

    def _compute_pc_scaling(self):
        """Compute PC scaling statistics from the HDF5 data."""
        logger.info("Computing PC scaling statistics...")
        logger.info(f"Dataset dimensions: {self.N_train} samples × {self.N_blocks} blocks × {self.pc_dim} PCs")
        
        total_elements = self.N_train * self.N_blocks * self.pc_dim
        total_size_gb = total_elements * 4 / (1024**3)  # 4 bytes per float32
        logger.info(f"Total data size: {total_elements:,} elements = {total_size_gb:.2f} GB")
        
        # Read min(10000, N_train) individuals to scale
        num_indivs_to_scale = min(10000, self.N_train)
        chunk = self._h5f["pc"][0:num_indivs_to_scale]  # (num_indivs_to_scale, N_blocks, pc_dim)
        chunk_flat = chunk.reshape(-1, self.pc_dim)  # (num_indivs_to_scale * N_blocks, pc_dim)
        
        approx_means = chunk_flat.mean(axis=0)
        approx_stds = chunk_flat.std(axis=0)
        self.pc_means = torch.from_numpy(approx_means).float()
        self.pc_scales = torch.from_numpy(approx_stds).float()
        
        logger.info(f"PC scaling - Mean range: [{approx_means.min():.4f}, {approx_means.max():.4f}], "
                   f"Std range: [{approx_stds.min():.4f}, {approx_stds.max():.4f}]")

    def __len__(self):
        return self.N_train

    def __getitem__(self, idx):
        # Handle multiprocessing: reopen file if needed in worker processes
        if not hasattr(self, '_h5f') or self._h5f is None:
            self._h5f = h5py.File(self.emb_h5_path, "r")
        emb = torch.from_numpy(self._h5f["pc"][idx]).float()
        if self.scale_pc_embeddings:
            emb = (emb - self.pc_means.unsqueeze(0)) / self.pc_scales.unsqueeze(0)
        return emb, self.spans, idx

    def close(self):
        """(Soft) close the HDF5 file."""
        if hasattr(self, '_h5f') and self._h5f is not None:
            try:
                self._h5f.close()
                self._h5f = None
            except:
                pass

    def __del__(self):
        self.close()


class SNPLoader:
    """
    Helper class for loading SNPs and PCA blocks on-demand during training/evaluation.
    Handles random block sampling and efficient SNP/PCA loading.
    Only used when SNPs are actually needed (SNP loss or evaluation).
    """
    
    def __init__(self, dataset: BlockPCDataset, snp_blocks_per_batch: int = 64, pca_metadata: Optional[List[Dict]] = None):
        """
        Args:
            dataset: BlockPCDataset instance
            snp_blocks_per_batch: Number of blocks to sample per batch
            pca_metadata: Optional metadata for filtering blocks with insufficient SNPs
        """
        self.dataset = dataset
        self.snp_blocks_per_batch = snp_blocks_per_batch
        self.block_raws = dataset.block_raws
        self.block_info = dataset.block_info
        self.embeddings_dir = dataset.embeddings_dir
        self.N_blocks = dataset.N_blocks
        
        # Filter blocks with sufficient SNPs for meaningful reconstruction
        self.valid_block_indices = list(range(self.N_blocks))
        if pca_metadata is not None:
            self.valid_block_indices = [i for i,meta in enumerate(pca_metadata) if meta.get('n_snps', 0) > dataset.pc_dim]
        logger.info(f"SNP loader: {snp_blocks_per_batch} blocks per batch from {len(self.valid_block_indices)} valid blocks")

    def _get_pca_block(self, block_idx: int) -> PCA_Block:
        """Load PCA block on-demand."""
        block_info = self.block_info[block_idx]
        return load_single_pca_block(self.embeddings_dir, block_info)

    def load_snps_for_batch(self, sample_indices: torch.Tensor, device: torch.device) -> Tuple[List[int], List[torch.Tensor]]:
        """
        Load SNPs for a random subset of blocks for the given samples.
        """
        n_blocks_to_sample = min(self.snp_blocks_per_batch, len(self.valid_block_indices))
        sampled_positions = torch.randperm(len(self.valid_block_indices))[:n_blocks_to_sample]
        block_indices = [self.valid_block_indices[pos] for pos in sampled_positions]
        
        true_snps = []
        for block_idx in block_indices:
            block_info = self.block_info[block_idx]
            chr_num, block_num, block_base = block_info
            raw_path = self.block_raws[block_idx]
            if not os.path.exists(raw_path):
                raise FileNotFoundError(f"SNP file not found: {raw_path}")
            rec = read_raw(raw_path)
            X = rec.impute().get_variants()  # (N_train, n_snps)
            batch_snps = torch.from_numpy(X[sample_indices.cpu().numpy()]).float().to(device)
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
            block_recon = recon_emb[:, block_idx, :]  # (batch_size, pc_dim)
            pca_block = self._get_pca_block(block_idx)
            block_info = self.block_info[block_idx]
            chr_num, block_num, block_base = block_info
            pred_snp = pca_block.decode(block_recon)
            pred_snps.append(pred_snp)
        return pred_snps


def unscale_embeddings(embeddings: torch.Tensor, pc_means: Optional[torch.Tensor], pc_scales: Optional[torch.Tensor]) -> torch.Tensor:
    if pc_means is None or pc_scales is None:
        return embeddings
    device = embeddings.device
    pc_means = pc_means.to(device)
    pc_scales = pc_scales.to(device)
    return embeddings * pc_scales.unsqueeze(0).unsqueeze(0) + pc_means.unsqueeze(0).unsqueeze(0) 