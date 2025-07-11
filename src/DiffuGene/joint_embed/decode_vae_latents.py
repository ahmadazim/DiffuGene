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
from src.DiffuGene.joint_embed.memory_efficient_dataset import load_single_pca_block
from src.DiffuGene.utils import setup_logging, get_logger

# Setup logging
setup_logging()
logger = get_logger(__name__)

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
    
    # Chromosome lengths (GRCh37/hg19 reference) - same as in memory_efficient_dataset.py
    CHROMOSOME_LENGTHS = {
        1: 249250621, 2: 243199373, 3: 198022430, 4: 191154276, 5: 180915260,
        6: 171115067, 7: 159138663, 8: 146364022, 9: 141213431, 10: 135534747,
        11: 135006516, 12: 133851895, 13: 115169878, 14: 107349540, 15: 102531392,
        16: 90338345, 17: 81195210, 18: 78077248, 19: 59128983, 20: 63025520,
        21: 48129895, 22: 51304566,
    }
    
    # Create chromosome-aware spans: (chr_idx, start_norm, end_norm)
    scaled_spans = []
    for _, row in df.iterrows():
        chr_idx = int(row.chr)  # Keep as integer index (1-22)
        
        # Get chromosome-specific length
        if chr_idx not in CHROMOSOME_LENGTHS:
            raise ValueError(f"Unknown chromosome: {chr_idx}. Supported: {list(CHROMOSOME_LENGTHS.keys())}")
        chrom_length = CHROMOSOME_LENGTHS[chr_idx]
        
        # Normalize start and end positions by chromosome length
        start_norm = row.start / chrom_length
        end_norm = row.end / chrom_length
        length_norm = end_norm - start_norm  # VAE expects length, not end position
        
        scaled_spans.append([chr_idx, start_norm, length_norm])
    
    spans_tensor = torch.tensor(scaled_spans, dtype=torch.float32)  # (N_blocks, 3)
    logger.info(f"Loaded spans for {len(scaled_spans)} blocks (chr_idx, start_norm, length_norm)")
    return spans_tensor

def prepare_block_info_for_decoding(spans_file, chromosome=None):
    """Prepare block information for on-demand PCA loading during decoding."""
    logger.info(f"Preparing block info from spans file: {spans_file}")
    df = pd.read_csv(spans_file)
    
    if chromosome is not None:
        # Filter by chromosome if specified
        df = df[df.chr == chromosome]
        logger.info(f"Filtered to chromosome {chromosome}: {len(df)} blocks")
    
    block_info = []
    for _, row in df.iterrows():
        # Extract block information
        filename = os.path.basename(row.block_file)
        block_match = re.search(r'block(\d+)', filename)
        if not block_match:
            raise ValueError(f"Could not extract block number from {filename}")
        block_no = int(block_match.group(1))
        chr_num = int(row.chr)
        block_base = os.path.basename(row.block_file).replace(".pt", "")
        
        block_info.append((chr_num, block_no, block_base))
    
    logger.info(f"Prepared block info for {len(block_info)} blocks")
    return block_info

def decode_latents(latents_file, model_file, embeddings_dir, spans_file, output_file, batch_size=32, chromosome=None):
    """
    Decode VAE latents back to original SNP space using on-demand PCA loading.
    
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
            
        # Get n_blocks from saved config if available
        saved_n_blocks = config.get('n_blocks', None)
    else:
        # Fallback for older model format
        raise ValueError("Model file must contain configuration. Please use a model saved with the current training script.")
    
    # 3. Prepare block info for on-demand PCA loading
    block_info = prepare_block_info_for_decoding(spans_file, chromosome=chromosome)
    n_blocks = len(block_info)
    
    # Initialize VAE model
    model = SNPVAE(
        n_blocks=n_blocks,
        grid_size=(config['grid_h'], config['grid_w']),
        block_emb_dim=config['block_dim'],
        pos_emb_dim=config['pos_dim'],
        latent_channels=config['latent_channels']
    ).to(device)
    
    model.load_state_dict(model_state)
    model.eval()
    logger.info(f"VAE model loaded with config: {config}")
    
    # 4. Load spans for positioning
    spans = load_spans_file(spans_file)
    spans = spans.to(device)
    
    # Verify dimensions match
    n_samples = latents.shape[0]
    if spans.shape[0] != n_blocks:
        raise ValueError(f"Spans file has {spans.shape[0]} blocks but block info has {n_blocks} blocks")
    
    # Validate against saved config if available
    if saved_n_blocks is not None and saved_n_blocks != n_blocks:
        logger.warning(f"Saved model n_blocks ({saved_n_blocks}) != current spans file n_blocks ({n_blocks}). Using current spans file.")
    elif saved_n_blocks is not None:
        logger.info(f"n_blocks matches saved model config: {n_blocks}")
    
    logger.info(f"Decoding {n_samples} samples across {n_blocks} blocks")
    
    # 5. Create PCA block cache for on-demand loading
    pca_cache = {}
    
    def get_pca_block(block_idx):
        """Get PCA block with caching."""
        if block_idx not in pca_cache:
            pca_cache[block_idx] = load_single_pca_block(embeddings_dir, block_info[block_idx])
        return pca_cache[block_idx]
    
    # 6. Decode in batches
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
            
            # PCA decode: PC embeddings → SNPs for each block (on-demand PCA loading)
            batch_reconstructed_snps = []
            for block_idx in range(n_blocks):
                # Get PC embeddings for this block: (batch_size, block_dim)
                block_pc_embeddings = decoded_pc_embeddings[:, block_idx, :]
                
                # Get PCA block on-demand
                pca_block = get_pca_block(block_idx)
                
                # Decode using PCA block
                reconstructed_snps = pca_block.decode(block_pc_embeddings)  # (batch_size, n_snps_in_block)
                batch_reconstructed_snps.append(reconstructed_snps)
            
            all_reconstructed_snps.append(batch_reconstructed_snps)
    
    # 7. Concatenate all batches
    logger.info("Concatenating results from all batches...")
    final_reconstructed_snps = []
    
    for block_idx in range(n_blocks):
        # Collect this block's reconstructions from all batches
        block_reconstructions = [batch_snps[block_idx] for batch_snps in all_reconstructed_snps]
        # Concatenate along batch dimension
        block_final = torch.cat(block_reconstructions, dim=0)  # (n_samples, n_snps_in_block)
        final_reconstructed_snps.append(block_final)
    
    # 8. Save results
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
    logger.info(f"  PCA blocks loaded on-demand: {len(pca_cache)}")
    
    # Compute some basic statistics
    all_values = torch.cat([snps.flatten() for snps in final_reconstructed_snps])
    logger.info(f"  Reconstructed values range: [{all_values.min():.3f}, {all_values.max():.3f}]")
    logger.info(f"  Reconstructed values mean: {all_values.mean():.3f} ± {all_values.std():.3f}")
    
    return output_data

def main():
    parser = argparse.ArgumentParser(description="Decode VAE latents back to original SNP space using memory-efficient approach")
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
