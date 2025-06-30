#!/usr/bin/env python
"""
Encode new genetic data using pre-trained DiffuGene encoders.

This script takes new PLINK files and encodes them using:
1. Pre-trained PCA loadings for block-wise embedding
2. Pre-trained VAE model for joint embedding to latent space

The script reuses existing LD block definitions from training and applies
pre-trained models in inference mode.
"""

import argparse
import os
import sys
import glob
import re
import subprocess
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from types import SimpleNamespace

# Add the src directory to the path to import DiffuGene modules
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from DiffuGene.utils import (
    setup_logging, get_logger, ensure_dir_exists, 
    load_blocks_for_chr, read_raw, create_snplist_files
)
from DiffuGene.block_embed.pca import PCA_Block
from DiffuGene.joint_embed.infer import inference as vae_inference

logger = get_logger(__name__)

def run_plink_recode_blocks(plink_basename, genetic_binary_folder, chromosome, 
                           block_file, output_dir, snplist_folder):
    """Recode genetic blocks using PLINK with EXACT same approach as training."""
    logger.info(f"Recoding blocks using exact same SNPs as training data...")
    
    # Load existing LD blocks (includes SNP IDs)
    LD_blocks = load_blocks_for_chr(block_file, chromosome)
    logger.info(f"Loaded {len(LD_blocks)} LD blocks from {block_file}")
    
    # Create SNP list files using the EXACT same function as training
    logger.info("Creating SNP list files...")
    ensure_dir_exists(snplist_folder)
    create_snplist_files(LD_blocks, snplist_folder, plink_basename, chromosome)
    
    # Find all created SNP list files
    snplist_pattern = os.path.join(snplist_folder, f"{plink_basename}_chr{chromosome}_block*.snplist")
    snplist_files = glob.glob(snplist_pattern)
    logger.info(f"Created {len(snplist_files)} SNP list files")
    
    # Recode each block using PLINK with --extract (EXACT same as training)
    ensure_dir_exists(output_dir)
    recoded_files = []
    
    for snpfile in tqdm(snplist_files, desc="Recoding blocks"):
        # Extract block number from filename
        match = re.search(r'block(\d+)', snpfile)
        if not match:
            logger.warning(f"Could not extract block number from {snpfile}")
            continue
            
        block_no = match.group(1)
        
        output_prefix = os.path.join(
            output_dir, 
            f"{plink_basename}_chr{chromosome}_block{block_no}_recodeA"
        )
        
        # Use EXACT same PLINK command as training
        cmd = [
            "plink",
            "--bfile", os.path.join(genetic_binary_folder, plink_basename),
            "--chr", str(chromosome),
            "--extract", snpfile,
            "--recodeA",
            "--out", output_prefix
        ]
        
        # Run PLINK with error output to debug issues
        result = subprocess.run(cmd, capture_output=True, text=True)
            
        if result.returncode != 0:
            logger.warning(f"PLINK failed for block {block_no}: {result.stderr}")
            continue
            
        raw_file = f"{output_prefix}.raw"
        if os.path.exists(raw_file):
            recoded_files.append(raw_file)
            # Log SNP count for debugging
            try:
                with open(raw_file, 'r') as f:
                    header = f.readline().strip().split()
                    snp_count = len([col for col in header if col.endswith('_A') or col.endswith('_T') or col.endswith('_G') or col.endswith('_C')])
                    # logger.info(f"Block {block_no}: Extracted {snp_count} SNPs (should match training)")
            except Exception as e:
                logger.debug(f"Block {block_no}: Could not count SNPs: {e}")
        else:
            logger.warning(f"Expected output file not found: {raw_file}")
    
    logger.info(f"Successfully recoded {len(recoded_files)} blocks using exact training SNPs")
    return recoded_files

def apply_pretrained_pca(raw_file, loadings_dir, means_dir, basename, chromosome, k):
    """Apply pre-trained PCA loadings to a recoded block file."""
    # Extract block number from filename
    match = re.search(r'block(\d+)', raw_file)
    if not match:
        raise ValueError(f"Could not extract block number from {raw_file}")
    
    block_no = match.group(1)
    
    # Load the genetic data
    rec = read_raw(raw_file)
    X = rec.impute().get_variants()  # (n_samples, n_snps)
    
    # Find pre-trained PCA components and means using glob pattern (basename may differ)
    loadings_pattern = os.path.join(loadings_dir, f"*_chr{chromosome}_block{block_no}_pca_loadings.pt")
    loadings_files = glob.glob(loadings_pattern)
    
    means_pattern = os.path.join(means_dir, f"*_chr{chromosome}_block{block_no}_pca_means.pt")
    means_files = glob.glob(means_pattern)
    
    if not loadings_files:
        raise FileNotFoundError(f"PCA loadings not found with pattern: {loadings_pattern}")
    if not means_files:
        raise FileNotFoundError(f"PCA means not found with pattern: {means_pattern}")
    
    if len(loadings_files) > 1:
        logger.warning(f"Multiple loadings files found for block {block_no}, using first: {loadings_files[0]}")
    if len(means_files) > 1:
        logger.warning(f"Multiple means files found for block {block_no}, using first: {means_files[0]}")
    
    loadings_file = loadings_files[0]
    means_file = means_files[0]
    
    # Load the pre-trained PCA parameters
    # CRITICAL: Training saves pca.components_.T, so we load (n_snps, actual_k) and need to transpose back!
    components_transposed = torch.load(loadings_file, map_location='cpu', weights_only=False)  # (n_snps, actual_k)
    means = torch.load(means_file, map_location='cpu', weights_only=False)  # (n_snps,)
    
    # Convert to numpy and transpose back to get the correct shape
    components_transposed_np = components_transposed.numpy() if isinstance(components_transposed, torch.Tensor) else components_transposed
    components = components_transposed_np.T  # Transpose: (n_snps, actual_k) -> (actual_k, n_snps)
    means_np = means.numpy() if isinstance(means, torch.Tensor) else means
    
    # Create PCA_Block instance and manually set parameters
    pca = PCA_Block(k)
    pca.actual_k = components.shape[0]  # Number of actual components
    pca.components_ = components  # numpy array (actual_k, n_snps) - correct shape!
    pca.means = means_np  # numpy array (n_snps,)
    
    # Check dimensional compatibility and debug matrix shapes
    logger.debug(f"Block {block_no}: X shape: {X.shape}, means shape: {means_np.shape}, components shape: {components.shape}")
    
    if X.shape[1] != len(means_np):
        logger.warning(f"Block {block_no}: SNP count mismatch. "
                      f"New data: {X.shape[1]}, trained: {len(means_np)}")
        # Handle by taking intersection or padding/truncating as needed
        min_snps = min(X.shape[1], len(means_np))
        X = X[:, :min_snps]
        pca.means = means_np[:min_snps]
        pca.components_ = components[:, :min_snps]
        logger.debug(f"Block {block_no}: After truncation - X: {X.shape}, means: {pca.means.shape}, components: {pca.components_.shape}")
    
    # Verify matrix multiplication compatibility before encoding
    logger.debug(f"Block {block_no}: Pre-encoding check - X: {X.shape}, components.T: {pca.components_.T.shape}")
    
    # Apply PCA transformation
    try:
        scores = pca.encode(X)  # (n_samples, k)
        logger.debug(f"Block {block_no}: Successfully encoded to shape: {scores.shape}")
    except Exception as e:
        logger.error(f"Block {block_no}: PCA encoding failed: {e}")
        logger.error(f"Block {block_no}: X: {X.shape}, components: {pca.components_.shape}, means: {pca.means.shape}")
        logger.error(f"Block {block_no}: Expected multiplication: {X.shape} @ {pca.components_.T.shape}")
        raise
    
    return torch.from_numpy(scores).float(), block_no

def create_spans_csv(block_file, embeddings_dir, basename, chromosome, pca_k):
    """Create spans CSV file for VAE inference."""
    logger.info("Creating spans CSV for VAE inference...")
    
    # Load LD blocks
    LD_blocks = load_blocks_for_chr(block_file, chromosome)
    
    # Create DataFrame with block information
    block_spans = []
    for i, block in enumerate(LD_blocks):
        block_spans.append([block.chr, block.bp1, block.bp2])
    
    df = pd.DataFrame(block_spans, columns=["chr", "start", "end"])
    
    # Add embedding file paths
    df["block_file"] = [
        os.path.join(
            embeddings_dir,
            f"{basename}_chr{row.chr}_block{idx+1}_embeddings.pt"
        )
        for idx, row in df.iterrows()
    ]
    df = df[["block_file", "chr", "start", "end"]]
    
    # Create CSV filename
    csv_path = os.path.join(
        os.path.dirname(embeddings_dir),
        f"{basename}_chr{chromosome}_blocks_{pca_k}PC_inference.csv"
    )
    df.to_csv(csv_path, index=False)
    logger.info(f"Spans CSV saved to {csv_path}")
    
    return csv_path

def encode_genetic_data(args):
    """Main function to encode genetic data using pre-trained models."""
    setup_logging()
    
    logger.info("Starting genetic data encoding with pre-trained DiffuGene models")
    logger.info(f"Input PLINK files: {args.genetic_binary_folder}/{args.basename}")
    logger.info(f"Chromosome: {args.chromosome}")
    logger.info(f"PCA loadings: {args.pca_loadings_dir}")
    logger.info(f"VAE model: {args.vae_model_path}")
    logger.info(f"Output path: {args.output_path}")
    
    # Create working directories
    work_dir = os.path.join(args.work_dir, f"encoding_{args.basename}_chr{args.chromosome}")
    recoded_dir = os.path.join(work_dir, "recoded_blocks")
    embeddings_dir = os.path.join(work_dir, "embeddings")
    snplist_dir = os.path.join(work_dir, "snplists")
    
    for dir_path in [work_dir, recoded_dir, embeddings_dir, snplist_dir]:
        ensure_dir_exists(dir_path)
    
    # Step 1: Recode genetic blocks using existing block definitions
    logger.info("Step 1: Recoding genetic blocks...")
    if not os.path.exists(args.block_file):
        raise FileNotFoundError(f"Block definition file not found: {args.block_file}")
    
    # Check if recoding already done
    existing_recoded_pattern = os.path.join(recoded_dir, f"{args.basename}_chr{args.chromosome}_block*_recodeA.raw")
    existing_recoded = glob.glob(existing_recoded_pattern)
    
    if existing_recoded:
        logger.info(f"Found {len(existing_recoded)} existing recoded files, skipping recoding step...")
        recoded_files = existing_recoded
    else:
        recoded_files = run_plink_recode_blocks(
            plink_basename=args.basename,
            genetic_binary_folder=args.genetic_binary_folder,
            chromosome=args.chromosome,
            block_file=args.block_file,
            output_dir=recoded_dir,
            snplist_folder=snplist_dir
        )
        
        if not recoded_files:
            raise RuntimeError("No blocks were successfully recoded")
    
    # Step 2: Apply pre-trained PCA to each block
    logger.info("Step 2: Applying pre-trained PCA to blocks...")
    
    # Check if embeddings already exist
    existing_embeddings_pattern = os.path.join(embeddings_dir, f"{args.basename}_chr{args.chromosome}_block*_embeddings.pt")
    existing_embeddings = glob.glob(existing_embeddings_pattern)
    
    if existing_embeddings:
        logger.info(f"Found {len(existing_embeddings)} existing embedding files, skipping PCA application step...")
        embeddings_created = existing_embeddings
    else:
        # Find means directory (should be alongside loadings)
        means_dir = os.path.join(os.path.dirname(args.pca_loadings_dir), "means")
        if not os.path.exists(means_dir):
            # Try alternative location
            means_dir = args.pca_loadings_dir.replace("loadings", "means")
            if not os.path.exists(means_dir):
                raise FileNotFoundError(f"PCA means directory not found. Expected: {means_dir}")
        
        embeddings_created = []
        for raw_file in tqdm(recoded_files, desc="Applying PCA"):
            try:
                # Apply PCA and get embeddings
                embeddings, block_no = apply_pretrained_pca(
                    raw_file=raw_file,
                    loadings_dir=args.pca_loadings_dir,
                    means_dir=means_dir,
                    basename=args.basename,
                    chromosome=args.chromosome,
                    k=args.pca_k
                )
                
                # Save embeddings
                embedding_file = os.path.join(
                    embeddings_dir,
                    f"{args.basename}_chr{args.chromosome}_block{block_no}_embeddings.pt"
                )
                torch.save(embeddings, embedding_file)
                embeddings_created.append(embedding_file)
                
            except Exception as e:
                logger.warning(f"Failed to process {raw_file}: {e}")
                continue
        
        logger.info(f"Successfully created {len(embeddings_created)} block embeddings")
    
    # Step 3: Create spans CSV for VAE inference
    logger.info("Step 3: Creating spans CSV...")
    
    # Check if spans CSV already exists
    spans_csv = os.path.join(
        os.path.dirname(embeddings_dir),
        f"{args.basename}_chr{args.chromosome}_blocks_{args.pca_k}PC_inference.csv"
    )
    
    if os.path.exists(spans_csv):
        logger.info(f"Spans CSV already exists at {spans_csv}, skipping creation...")
    else:
        spans_csv = create_spans_csv(
            block_file=args.block_file,
            embeddings_dir=embeddings_dir,
            basename=args.basename,
            chromosome=args.chromosome,
            pca_k=args.pca_k
        )
    
    # Step 4: Run VAE inference
    logger.info("Step 4: Running VAE inference...")
    
    # Check if VAE inference already done
    if os.path.exists(args.output_path):
        logger.info(f"VAE inference output already exists at {args.output_path}, skipping VAE inference step...")
    else:
        # Prepare arguments for VAE inference
        vae_args = SimpleNamespace()
        vae_args.model_path = args.vae_model_path
        vae_args.spans_file = spans_csv
        vae_args.recoded_dir = recoded_dir
        vae_args.output_path = args.output_path
        
        # VAE model parameters (these should match the trained model)
        vae_args.grid_h = args.grid_h
        vae_args.grid_w = args.grid_w
        vae_args.block_dim = args.block_dim
        vae_args.pos_dim = args.pos_dim
        vae_args.latent_channels = args.latent_channels
        
        # Run VAE inference
        vae_inference(vae_args)
    
    logger.info(f"Encoding completed successfully! Latents saved to: {args.output_path}")
    
    # Print summary
    logger.info("=== ENCODING SUMMARY ===")
    logger.info(f"Input data: {args.basename} (chr{args.chromosome})")
    logger.info(f"Output latents: {args.output_path}")
    logger.info(f"Working directory: {work_dir}")
    logger.info("Steps completed successfully!")
    
    # Cleanup temporary files if requested
    if args.cleanup:
        logger.info("Cleaning up temporary files...")
        import shutil
        shutil.rmtree(work_dir)
        logger.info("Cleanup completed")

def main():
    """Main entry point for the encoding script."""
    parser = argparse.ArgumentParser(
        description="Encode genetic data using pre-trained DiffuGene models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python encode_genetic_data.py \\
    --basename mydata \\
    --genetic-binary-folder /path/to/plink/files \\
    --chromosome 22 \\
    --block-file /path/to/training_blocks.blocks.det \\
    --pca-loadings-dir /path/to/pca/loadings \\
    --vae-model-path /path/to/trained_vae.pt \\
    --output-path /path/to/output_latents.pt
    
  # With custom VAE parameters
  python encode_genetic_data.py \\
    --basename mydata \\
    --genetic-binary-folder /path/to/plink/files \\
    --chromosome 22 \\
    --block-file /path/to/training_blocks.blocks.det \\
    --pca-loadings-dir /path/to/pca/loadings \\
    --vae-model-path /path/to/trained_vae.pt \\
    --output-path /path/to/output_latents.pt \\
    --grid-h 32 --grid-w 32 \\
    --block-dim 3 --pos-dim 16 \\
    --latent-channels 32
        """
    )
    
    # Input data arguments
    parser.add_argument("--basename", type=str, required=True,
                       help="Base name for PLINK files (without extensions)")
    parser.add_argument("--genetic-binary-folder", type=str, required=True,
                       help="Directory containing PLINK binary files (.bed, .bim, .fam)")
    parser.add_argument("--chromosome", type=int, required=True,
                       help="Chromosome number to process")
    
    # Pre-trained model arguments
    parser.add_argument("--block-file", type=str, required=True,
                       help="Path to LD block definition file (.blocks.det) from training")
    parser.add_argument("--pca-loadings-dir", type=str, required=True,
                       help="Directory containing pre-trained PCA loadings")
    parser.add_argument("--vae-model-path", type=str, required=True,
                       help="Path to trained VAE model")
    
    # Output arguments
    parser.add_argument("--output-path", type=str, required=True,
                       help="Output path for encoded latents (.pt file)")
    parser.add_argument("--work-dir", type=str, default="./work",
                       help="Working directory for temporary files")
    
    # PCA parameters
    parser.add_argument("--pca-k", type=int, default=3,
                       help="Number of PCA components (should match training)")
    
    # VAE model parameters (should match trained model)
    parser.add_argument("--grid-h", type=int, default=32,
                       help="VAE grid height")
    parser.add_argument("--grid-w", type=int, default=32,
                       help="VAE grid width")
    parser.add_argument("--block-dim", type=int, default=3,
                       help="Block embedding dimension")
    parser.add_argument("--pos-dim", type=int, default=16,
                       help="Position embedding dimension")
    parser.add_argument("--latent-channels", type=int, default=32,
                       help="Number of latent channels")
    
    # Other options
    parser.add_argument("--cleanup", action="store_true",
                       help="Clean up temporary files after completion")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.genetic_binary_folder):
        parser.error(f"Genetic binary folder not found: {args.genetic_binary_folder}")
    
    plink_bed = os.path.join(args.genetic_binary_folder, f"{args.basename}.bed")
    if not os.path.exists(plink_bed):
        parser.error(f"PLINK .bed file not found: {plink_bed}")
    
    if not os.path.exists(args.block_file):
        parser.error(f"Block definition file not found: {args.block_file}")
    
    if not os.path.exists(args.pca_loadings_dir):
        parser.error(f"PCA loadings directory not found: {args.pca_loadings_dir}")
    
    if not os.path.exists(args.vae_model_path):
        parser.error(f"VAE model not found: {args.vae_model_path}")
    
    # Ensure output directory exists
    ensure_dir_exists(os.path.dirname(args.output_path))
    
    # Run encoding
    encode_genetic_data(args)

if __name__ == "__main__":
    main() 