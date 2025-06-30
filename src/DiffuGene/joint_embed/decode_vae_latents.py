#!/usr/bin/env python

import os
import glob
import argparse
import re
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from tqdm import tqdm

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))

from src.DiffuGene.joint_embed.vae import SNPVAE
from src.DiffuGene.block_embed import PCA_Block
from src.DiffuGene.utils import setup_logging, get_logger

# Setup logging
setup_logging()
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
        # NOTE: Training saves pca.components_.T, so loadings are (n_snps, actual_k)
        loadings = torch.load(load_file, map_location="cuda", weights_only=False)    # (n_snps, actual_k) 
        means = torch.load(means_pattern, map_location="cuda", weights_only=False)   # (n_snps,)
        
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
        # Transpose back: (n_snps, actual_k) -> (actual_k, n_snps) for PCA_Block format
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

def unscale_embeddings(embeddings, pc_means, pc_scales):
    """Reverse the PC scaling applied during training.
    
    Args:
        embeddings: (batch_size, n_blocks, k) - scaled embeddings
        pc_means: (k,) - means used for scaling  
        pc_scales: (k,) - standard deviations used for scaling
    
    Returns:
        unscaled_embeddings: (batch_size, n_blocks, k) - original scale embeddings
    """
    if pc_means is None or pc_scales is None:
        # logger.info("No PC scaling factors found - embeddings were not scaled during training")
        return embeddings  # No scaling was applied
    
    # Ensure scaling factors are on the same device as embeddings
    device = embeddings.device
    pc_means = pc_means.to(device)
    pc_scales = pc_scales.to(device)
    
    # Reverse standardization: x_orig = x_scaled * std + mean
    # logger.info("Applying PC unscaling to embeddings")
    return embeddings * pc_scales.unsqueeze(0).unsqueeze(0) + pc_means.unsqueeze(0).unsqueeze(0)

def load_spans_file(spans_file):
    """Load spans file to get the positioning information for VAE decoding."""
    logger.info(f"Loading spans file: {spans_file}")
    df = pd.read_csv(spans_file)
    
    # Create scaled spans (same as in training)
    MAX_COORD_CHR22 = 50_818_468
    scaled_spans = []
    for _, row in df.iterrows():
        chr_norm   = row.chr / 22
        start_norm = row.start / MAX_COORD_CHR22
        end_norm   = row.end / MAX_COORD_CHR22
        scaled_spans.append([chr_norm, start_norm, end_norm])
    
    spans_tensor = torch.tensor(scaled_spans, dtype=torch.float32)  # (N_blocks, 3)
    logger.info(f"Loaded spans for {len(scaled_spans)} blocks")
    return spans_tensor

def decode_latents(latents_file, model_file, embeddings_dir, spans_file, output_file, batch_size=32, chromosome=None):
    """
    Decode VAE latents back to original SNP space.
    
    Args:
        latents_file: Path to VAE latents (.pt file)
        model_file: Path to trained VAE model (.pt file) 
        embeddings_dir: Directory containing PCA models
        spans_file: CSV file with block positioning info
        output_file: Where to save the reconstructed SNPs
        batch_size: Batch size for processing
        chromosome: Chromosome number (optional, for better file matching)
    """
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # 1. Load VAE latents
    logger.info(f"Loading VAE latents from: {latents_file}")
    latents = torch.load(latents_file, map_location=device)
    
    if isinstance(latents, dict):
        if 'latents' in latents:
            latents = latents['latents']
        else:
            raise ValueError(f"VAE latents file format not recognized. Keys: {list(latents.keys())}")
    
    if not isinstance(latents, torch.Tensor):
        latents = torch.tensor(latents, dtype=torch.float32, device=device)
    
    latents = latents.float().to(device)
    logger.info(f"Loaded latents with shape: {latents.shape}")
    
    # 2. Load VAE model
    logger.info(f"Loading VAE model from: {model_file}")
    model_data = torch.load(model_file, map_location=device)
    
    # Extract configuration and scaling info
    if isinstance(model_data, dict) and 'config' in model_data:
        config = model_data['config']
        model_state = model_data['model_state_dict']
        
        # Extract PC scaling factors if available
        pc_means = None
        pc_scales = None
        if 'pc_scaling' in model_data:
            pc_means = model_data['pc_scaling']['pc_means'].to(device)
            pc_scales = model_data['pc_scaling']['pc_scales'].to(device)
            logger.info("Found PC scaling factors in model file")
    else:
        # Fallback for older model format
        raise ValueError("Model file must contain configuration. Please use a model saved with the current training script.")
    
    # Initialize VAE model
    model = SNPVAE(
        grid_size=(config['grid_h'], config['grid_w']),
        block_emb_dim=config['block_dim'],
        pos_emb_dim=config['pos_dim'],
        latent_channels=config['latent_channels']
    ).to(device)
    
    model.load_state_dict(model_state)
    model.eval()
    logger.info(f"VAE model loaded with config: {config}")
    
    # 3. Load PCA blocks
    pca_blocks = load_pca_blocks(embeddings_dir, chromosome=chromosome)
    
    # 4. Load spans for positioning
    spans = load_spans_file(spans_file)
    spans = spans.to(device)
    
    # Verify dimensions match
    n_samples = latents.shape[0]
    n_blocks = len(pca_blocks)
    if spans.shape[0] != n_blocks:
        raise ValueError(f"Spans file has {spans.shape[0]} blocks but PCA has {n_blocks} blocks")
    
    logger.info(f"Decoding {n_samples} samples across {n_blocks} blocks")
    
    # 5. Decode in batches
    all_reconstructed_snps = []
    
    with torch.no_grad():
        for start_idx in tqdm(range(0, n_samples, batch_size), desc="Decoding batches"):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_latents = latents[start_idx:end_idx]  # (batch_size, latent_channels, H, W)
            batch_size_actual = batch_latents.shape[0]
            
            # Expand spans to batch size: (batch_size, n_blocks, 3)
            batch_spans = spans.unsqueeze(0).expand(batch_size_actual, -1, -1)
            
            # VAE decode: latents → PC embeddings
            decoded_pc_embeddings = model.decode(batch_latents, batch_spans)  # (batch_size, n_blocks, block_dim)
            
            # Unscale PC embeddings if scaling was applied during training
            decoded_pc_embeddings = unscale_embeddings(decoded_pc_embeddings, pc_means, pc_scales)
            
            # PCA decode: PC embeddings → SNPs for each block
            batch_reconstructed_snps = []
            for block_idx in range(n_blocks):
                # Get PC embeddings for this block: (batch_size, block_dim)
                block_pc_embeddings = decoded_pc_embeddings[:, block_idx, :]
                
                # Decode using PCA block
                reconstructed_snps = pca_blocks[block_idx].decode(block_pc_embeddings)  # (batch_size, n_snps_in_block)
                batch_reconstructed_snps.append(reconstructed_snps)
            
            all_reconstructed_snps.append(batch_reconstructed_snps)
    
    # 6. Concatenate all batches
    logger.info("Concatenating results from all batches...")
    final_reconstructed_snps = []
    
    for block_idx in range(n_blocks):
        # Collect this block's reconstructions from all batches
        block_reconstructions = [batch_snps[block_idx] for batch_snps in all_reconstructed_snps]
        # Concatenate along batch dimension
        block_final = torch.cat(block_reconstructions, dim=0)  # (n_samples, n_snps_in_block)
        final_reconstructed_snps.append(block_final)
    
    # 7. Save results
    logger.info(f"Saving reconstructed SNPs to: {output_file}")
    
    # Create output dictionary with metadata
    output_data = {
        'reconstructed_snps': final_reconstructed_snps,  # List of tensors, one per block
        'n_samples': n_samples,
        'n_blocks': n_blocks,
        'block_snp_counts': [snps.shape[1] for snps in final_reconstructed_snps],
        'model_config': config,
        'latents_file': latents_file,
        'model_file': model_file,
        'embeddings_dir': embeddings_dir
    }
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    torch.save(output_data, output_file)
    
    # Log summary statistics
    total_snps = sum(snps.shape[1] for snps in final_reconstructed_snps)
    logger.info(f"Reconstruction complete!")
    logger.info(f"  Total samples: {n_samples}")
    logger.info(f"  Total blocks: {n_blocks}")
    logger.info(f"  Total SNPs: {total_snps}")
    logger.info(f"  SNPs per block range: {min(snps.shape[1] for snps in final_reconstructed_snps)} - {max(snps.shape[1] for snps in final_reconstructed_snps)}")
    
    # Compute some basic statistics
    all_values = torch.cat([snps.flatten() for snps in final_reconstructed_snps])
    logger.info(f"  Reconstructed values range: [{all_values.min():.3f}, {all_values.max():.3f}]")
    logger.info(f"  Reconstructed values mean: {all_values.mean():.3f} ± {all_values.std():.3f}")
    
    return output_data

def main():
    parser = argparse.ArgumentParser(description="Decode VAE latents back to original SNP space")
    parser.add_argument("--latents-file", type=str, required=True,
                        help="Path to VAE latents file (.pt)")
    parser.add_argument("--model-file", type=str, required=True,
                        help="Path to trained VAE model file (.pt)")
    parser.add_argument("--embeddings-dir", type=str, required=True,
                        help="Directory containing PCA models (with loadings/, means/, metadata/ subdirs)")
    parser.add_argument("--spans-file", type=str, required=True,
                        help="CSV file with block positioning information")
    parser.add_argument("--output-file", type=str, required=True,
                        help="Where to save the reconstructed SNPs (.pt)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for processing (default: 32)")
    parser.add_argument("--chromosome", type=int, default=None,
                        help="Chromosome number (helps with file matching, e.g., 22)")
    
    args = parser.parse_args()
    
    # Validate input files exist
    if not os.path.exists(args.latents_file):
        raise FileNotFoundError(f"Latents file not found: {args.latents_file}")
    if not os.path.exists(args.model_file):
        raise FileNotFoundError(f"Model file not found: {args.model_file}")
    if not os.path.exists(args.embeddings_dir):
        raise FileNotFoundError(f"Embeddings directory not found: {args.embeddings_dir}")
    if not os.path.exists(args.spans_file):
        raise FileNotFoundError(f"Spans file not found: {args.spans_file}")
    
    # Run decoding
    decode_latents(
        latents_file=args.latents_file,
        model_file=args.model_file,
        embeddings_dir=args.embeddings_dir,
        spans_file=args.spans_file,
        output_file=args.output_file,
        batch_size=args.batch_size,
        chromosome=args.chromosome
    )

if __name__ == "__main__":
    main() 



# for i in 3 4 5; do
# python src/DiffuGene/joint_embed/decode_vae_latents.py \
#     --latents-file data/VAE_embeddings/all_hm3_15k_chr22_VAE_latents_${i}PC.pt \
#     --model-file models/joint_embed/vae_model_${i}PC.pt \
#     --embeddings-dir data/haploblocks_embeddings_${i}PC \
#     --spans-file data/haploblocks/all_hm3_15k_chr22_blocks_${i}PC.csv \
#     --output-file data/VAE_embeddings/all_hm3_15k_chr22_VAE_decoded_${i}PC.pt \
#     --batch-size 256
# done

# for i in 3 4 5; do
# python src/DiffuGene/joint_embed/decode_vae_latents.py \
#     --latents-file data/VAE_embeddings/all_hm3_15k_chr22_VAE_latents_${i}PCscale.pt \
#     --model-file models/joint_embed/vae_model_${i}PCscale.pt \
#     --embeddings-dir data/haploblocks_embeddings_${i}PC \
#     --spans-file data/haploblocks/all_hm3_15k_chr22_blocks_${i}PC.csv \
#     --output-file data/VAE_embeddings/all_hm3_15k_chr22_VAE_decoded_${i}PCscale.pt \
#     --batch-size 256
# done

# python src/DiffuGene/joint_embed/decode_vae_latents.py \
#     --latents-file data/generated_samples/generated_latents_all_hm3_45k_4PCscale.pt \
#     --model-file models/joint_embed/vae_model_4PCscale.pt \
#     --embeddings-dir data/haploblocks_embeddings_4PC \
#     --spans-file data/work_encode/encoding_all_hm3_45k_chr22/all_hm3_45k_chr22_blocks_4PC_inference.csv \
#     --output-file data/generated_samples/generated_decoded_all_hm3_45k_4PCscale.pt \
#     --batch-size 256
