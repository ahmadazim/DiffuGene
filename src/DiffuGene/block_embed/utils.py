#!/usr/bin/env python

import os
import glob
import re
import torch
from ..utils import get_logger
from .pca import PCA_Block

logger = get_logger(__name__)

def load_pca_blocks(embeddings_dir, chromosome=None):
    """Load PCA blocks for SNP reconstruction with dimension metadata.
    
    Args:
        embeddings_dir: Directory containing PCA models
        chromosome: Chromosome number (optional, for filtering files)
    """
    logger.info("Loading PCA blocks...")
    pca_blocks = []
    load_dir = os.path.join(embeddings_dir, "loadings")
    mean_dir = os.path.join(embeddings_dir, "means")
    metadata_dir = os.path.join(embeddings_dir, "metadata")
    
    # Get all PCA loading files and sort them numerically by block number
    if chromosome is not None:
        # More specific pattern if chromosome is provided
        load_files = glob.glob(os.path.join(load_dir, f"*_chr{chromosome}_block*_pca_loadings.pt"))
    else:
        load_files = glob.glob(os.path.join(load_dir, "*_pca_loadings.pt"))
    
    if not load_files:
        raise FileNotFoundError(f"No PCA loading files found in {load_dir}. Expected pattern: *_pca_loadings.pt")
    
    # Extract block numbers and sort numerically
    def extract_block_number(filepath):
        filename = os.path.basename(filepath)
        match = re.search(r'block(\d+)', filename)
        return int(match.group(1)) if match else 0
    
    load_files_sorted = sorted(load_files, key=extract_block_number)
    logger.info(f"Found {len(load_files_sorted)} PCA loading files")
    
    for load_file in load_files_sorted:
        block_base = os.path.basename(load_file).replace("_pca_loadings.pt", "")
        
        # Find corresponding means and metadata files using flexible patterns
        means_pattern = os.path.join(mean_dir, f"{block_base}_pca_means.pt")
        metadata_pattern = os.path.join(metadata_dir, f"{block_base}_pca_metadata.pt")
        
        # Check if exact files exist, otherwise use glob pattern
        if not os.path.exists(means_pattern):
            # Extract block info and find using pattern
            block_match = re.search(r'_chr(\d+)_block(\d+)', block_base)
            if block_match:
                chr_num, block_num = block_match.groups()
                means_files = glob.glob(os.path.join(mean_dir, f"*_chr{chr_num}_block{block_num}_pca_means.pt"))
                if means_files:
                    means_pattern = means_files[0]
                    logger.debug(f"Using means file: {means_pattern}")
                else:
                    raise FileNotFoundError(f"No means file found for block {block_num}")
        
        if not os.path.exists(metadata_pattern):
            # Extract block info and find using pattern  
            block_match = re.search(r'_chr(\d+)_block(\d+)', block_base)
            if block_match:
                chr_num, block_num = block_match.groups()
                metadata_files = glob.glob(os.path.join(metadata_dir, f"*_chr{chr_num}_block{block_num}_pca_metadata.pt"))
                if metadata_files:
                    metadata_pattern = metadata_files[0]
                    logger.debug(f"Using metadata file: {metadata_pattern}")
        
        # Load the files
        loadings = torch.load(load_file, map_location="cuda", weights_only=False)    # (n_snps, k)
        means = torch.load(means_pattern, map_location="cuda", weights_only=False)      # (n_snps,)
        
        # Load metadata if available (backwards compatibility)
        if os.path.exists(metadata_pattern):
            metadata = torch.load(metadata_pattern, map_location="cpu", weights_only=False)
            k = metadata['k']
            actual_k = metadata['actual_k']
        else:
            # Fallback for old format - assume all dimensions are used
            if hasattr(loadings, 'size') and callable(loadings.size):
                k = loadings.size(1)
            elif hasattr(loadings, 'shape'):
                k = loadings.shape[1]
            else:
                raise ValueError(f"Unsupported loadings type: {type(loadings)}")
            actual_k = k
            logger.warning(f"No metadata found for {block_base}, assuming actual_k = k = {k}")
        
        # Convert to PyTorch tensors and ensure float32
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
        pca_block.components_ = loadings.T if hasattr(loadings, 'T') else loadings.transpose()  # (actual_k, n_snps)
        pca_block.means = means
        pca_blocks.append(pca_block)
    
    logger.info(f"PCA blocks loaded. Found {len(pca_blocks)} blocks.")
    
    # Log dimension statistics
    total_blocks = len(pca_blocks)
    padded_blocks = sum(1 for pca in pca_blocks if pca.actual_k < pca.k)
    if padded_blocks > 0:
        logger.info(f"Dimension padding: {padded_blocks}/{total_blocks} blocks have actual_k < k")
        for i, pca in enumerate(pca_blocks):
            if pca.actual_k < pca.k:
                logger.info(f"  Block {i}: k={pca.k}, actual_k={pca.actual_k} ({pca.k - pca.actual_k} padded dims)")
    
    return pca_blocks 