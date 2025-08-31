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
from DiffuGene.joint_embed.infer import inference
from DiffuGene.joint_embed.precompute_embeddings import precompute_embeddings

logger = get_logger(__name__)

def run_plink_recode_blocks(plink_basename, genetic_binary_folder, chromosomes, 
                           block_files, output_dir, snplist_folder, keep_fam_file=None, global_bfile=None,
                           training_snplist_dir=None, training_basename=None, training_bim_file=None):
    """Recode genetic blocks using PLINK with EXACT same approach as training."""
    logger.info(f"Recoding blocks for chromosomes {chromosomes} using exact same SNPs as training data...")
    
    # Determine the actual bfile to use
    if global_bfile:
        actual_bfile = global_bfile
        logger.info(f"Using specified global bfile: {actual_bfile}")
    else:
        actual_bfile = os.path.join(genetic_binary_folder, plink_basename)
        logger.info(f"Using default bfile: {actual_bfile}")
    
    # Check if the files exist
    if not os.path.exists(f"{actual_bfile}.bed"):
        logger.error(f"PLINK files not found with basename {actual_bfile}")
        raise FileNotFoundError(f"PLINK .bed file not found: {actual_bfile}.bed")
    
    if keep_fam_file:
        if not os.path.exists(keep_fam_file):
            raise FileNotFoundError(f"Keep fam file not found: {keep_fam_file}")
        logger.info(f"Will use --keep {keep_fam_file} to subset individuals")
    
    recoded_files = []
    
    for chromosome in chromosomes:
        block_file = block_files.get(chromosome)
        if not block_file:
            logger.warning(f"No block file specified for chromosome {chromosome}, skipping")
            continue
            
        # Load existing LD blocks (includes SNP IDs) and decide snplist source
        # Prefer EXACT training snplist files if available to avoid SNP mismatches
        if training_snplist_dir and training_basename:
            logger.info(f"Chr{chromosome}: Using training snplist files from {training_snplist_dir} with basename {training_basename}")
            # Training snplists are expected under training_snplist_dir[/chr{chromosome}]
            chr_snplist_folder = os.path.join(training_snplist_dir, f"chr{chromosome}")
            if not os.path.exists(chr_snplist_folder):
                chr_snplist_folder = training_snplist_dir
            snplist_basename = training_basename
        else:
            logger.info(f"Chr{chromosome}: Using original block file {block_file}")
            
            LD_blocks = load_blocks_for_chr(block_file, chromosome, training_snplist_dir, training_basename, training_bim_file)
            
            # Log SNP counts per block to verify we're getting the right data
            total_snps = sum(len(block.snps) for block in LD_blocks)
            logger.info(f"Chr{chromosome}: Loaded {len(LD_blocks)} LD blocks with {total_snps} total SNPs")
            
            # Create SNP list files using the EXACT same function as training
            chr_snplist_folder = os.path.join(snplist_folder, f"chr{chromosome}")
            ensure_dir_exists(chr_snplist_folder)
            create_snplist_files(LD_blocks, chr_snplist_folder, plink_basename, chromosome)
            snplist_basename = plink_basename
        
        # Find SNP list files
        snplist_pattern = os.path.join(chr_snplist_folder, f"{snplist_basename}_chr{chromosome}_block*.snplist")
        snplist_files = glob.glob(snplist_pattern)
        logger.info(f"Chr{chromosome}: Created {len(snplist_files)} SNP list files")
        
        # Recode each block using PLINK with --extract
        chr_output_dir = os.path.join(output_dir, f"chr{chromosome}")
        ensure_dir_exists(chr_output_dir)
        
        for snpfile in tqdm(snplist_files, desc=f"Recoding chr{chromosome} blocks"):
            # Extract block number from filename
            match = re.search(r'block(\d+)', snpfile)
            if not match:
                logger.warning(f"Could not extract block number from {snpfile}")
                continue
                
            block_no = match.group(1)
            
            output_prefix = os.path.join(
                chr_output_dir, 
                f"{plink_basename}_chr{chromosome}_block{block_no}_recodeA"
            )
            
            # Build PLINK command
            cmd = [
                "plink",
                "--bfile", actual_bfile,
                "--chr", str(chromosome),
                "--extract", snpfile,
                "--recodeA",
                "--out", output_prefix
            ]
            
            # Add --keep if using a subset fam file
            if keep_fam_file:
                cmd.extend(["--keep", keep_fam_file])
            
            # Run PLINK with error handling
            result = subprocess.run(cmd, capture_output=True, text=True)
                
            if result.returncode != 0:
                logger.warning(f"Chr{chromosome} block {block_no}: PLINK failed: {result.stderr}")
                continue
                
            raw_file = f"{output_prefix}.raw"
            if os.path.exists(raw_file):
                recoded_files.append(raw_file)
            else:
                logger.warning(f"Expected output file not found: {raw_file}")
    
    logger.info(f"Successfully recoded {len(recoded_files)} blocks across all chromosomes")
    return recoded_files

def apply_pretrained_pca(raw_file, loadings_dir, means_dir, metadata_dir, basename, k):
    """Apply pre-trained PCA loadings to a recoded block file."""
    # Extract chromosome and block number from filename
    chr_match = re.search(r'chr(\d+)', raw_file)
    block_match = re.search(r'block(\d+)', raw_file)
    
    if not chr_match or not block_match:
        raise ValueError(f"Could not extract chromosome and block number from {raw_file}")
    
    chromosome = chr_match.group(1)
    block_no = block_match.group(1)
    
    # Load the genetic data
    rec = read_raw(raw_file)
    X = rec.impute().get_variants()  # (n_samples, n_snps)
    
    # Find pre-trained PCA components, means, and metadata using glob pattern
    loadings_pattern = os.path.join(loadings_dir, f"*_chr{chromosome}_block{block_no}_pca_loadings.pt")
    means_pattern = os.path.join(means_dir, f"*_chr{chromosome}_block{block_no}_pca_means.pt")
    metadata_pattern = os.path.join(metadata_dir, f"*_chr{chromosome}_block{block_no}_pca_metadata.pt")
    
    loadings_files = glob.glob(loadings_pattern)
    means_files = glob.glob(means_pattern)
    metadata_files = glob.glob(metadata_pattern)
    
    if not loadings_files:
        raise FileNotFoundError(f"PCA loadings not found with pattern: {loadings_pattern}")
    if not means_files:
        raise FileNotFoundError(f"PCA means not found with pattern: {means_pattern}")
    
    if len(loadings_files) > 1:
        # logger.warning(f"Multiple loadings files found for chr{chromosome}_block{block_no}, using first: {loadings_files[0]}")
        raise ValueError(f"Multiple loadings files found for chr{chromosome}_block{block_no}, first: {loadings_files[0]}")
    if len(means_files) > 1:
        # logger.warning(f"Multiple means files found for chr{chromosome}_block{block_no}, using first: {means_files[0]}")
        raise ValueError(f"Multiple means files found for chr{chromosome}_block{block_no}, first: {means_files[0]}")
    
    loadings_file = loadings_files[0]
    means_file = means_files[0]
    
    # Load the pre-trained PCA parameters
    components_transposed = torch.load(loadings_file, map_location='cpu', weights_only=False)
    means = torch.load(means_file, map_location='cpu', weights_only=False)
    
    # Load metadata if available
    if metadata_files:
        metadata = torch.load(metadata_files[0], map_location='cpu', weights_only=False)
        actual_k = metadata.get('actual_k', k)
    else:
        actual_k = k
        logger.warning(f"No metadata found for chr{chromosome}_block{block_no}, using k={k}")
    
    # Convert to numpy and transpose back to get the correct shape
    components_transposed_np = components_transposed.numpy() if isinstance(components_transposed, torch.Tensor) else components_transposed
    components = components_transposed_np.T  # Transpose: (n_snps, actual_k) -> (actual_k, n_snps)
    means_np = means.numpy() if isinstance(means, torch.Tensor) else means
    
    # Create PCA_Block instance and manually set parameters
    pca = PCA_Block(k)
    pca.actual_k = actual_k
    pca.components_ = components
    pca.means = means_np
    
    # Check dimensional compatibility
    if X.shape[1] != len(means_np):
        logger.warning(f"Chr{chromosome} block {block_no}: SNP count mismatch. "
                      f"New data: {X.shape[1]}, trained: {len(means_np)}")
        # Handle by taking intersection
        min_snps = min(X.shape[1], len(means_np))
        X = X[:, :min_snps]
        pca.means = means_np[:min_snps]
        pca.components_ = components[:, :min_snps]
    
    # Apply PCA transformation
    try:
        scores = pca.encode(X)  # (n_samples, k)
        logger.debug(f"Chr{chromosome} block {block_no}: Successfully encoded to shape: {scores.shape}")
    except Exception as e:
        logger.error(f"Chr{chromosome} block {block_no}: PCA encoding failed: {e}")
        raise
    
    return torch.from_numpy(scores).float(), chromosome, block_no

def create_spans_csv_multi_chr(block_files, embeddings_base_dir, basename, pca_k):
    """Create spans CSV file for multi-chromosome VAE inference."""
    logger.info("Creating multi-chromosome spans CSV for VAE inference...")
    
    all_block_spans = []
    
    for chromosome, block_file in block_files.items():
        # Load LD blocks for this chromosome
        LD_blocks = load_blocks_for_chr(block_file, chromosome)
            
        chr_embeddings_dir = os.path.join(embeddings_base_dir, f"chr{chromosome}")
        
        for i, block in enumerate(LD_blocks):
            all_block_spans.append({
                "block_file": os.path.join(
                    chr_embeddings_dir,
                    f"{basename}_chr{chromosome}_block{i+1}_embeddings.pt"
                ),
                "chr": block.chr,
                "start": block.bp1,
                "end": block.bp2
            })
    
    df = pd.DataFrame(all_block_spans)
    
    # Create CSV filename
    csv_path = os.path.join(
        os.path.dirname(embeddings_base_dir),
        f"{basename}_blocks_{pca_k}PC_inference.csv"
    )
    df.to_csv(csv_path, index=False)
    logger.info(f"Multi-chromosome spans CSV saved to {csv_path} with {len(df)} blocks")
    
    return csv_path

def auto_discover_chromosomes_and_blocks(pca_embeddings_dir, block_files_arg=None):
    """Auto-discover available chromosomes from PCA embeddings directory and find corresponding block files."""
    loadings_dir = os.path.join(pca_embeddings_dir, "loadings")
    if not os.path.exists(loadings_dir):
        raise FileNotFoundError(f"PCA loadings directory not found: {loadings_dir}")
    
    # Find all chromosome numbers from PCA loadings files
    loadings_files = glob.glob(os.path.join(loadings_dir, "*_chr*_block*_pca_loadings.pt"))
    chromosomes = set()
    for file in loadings_files:
        match = re.search(r'chr(\d+)_block', os.path.basename(file))
        if match:
            chromosomes.add(int(match.group(1)))
    
    chromosomes = sorted(list(chromosomes))
    logger.info(f"Auto-discovered chromosomes: {chromosomes}")
    
    # Auto-discover block files if not provided
    block_files = {}
    if block_files_arg and block_files_arg.lower() != 'auto':
        # Parse provided block files
        if isinstance(block_files_arg, str):
            block_file_list = [x.strip() for x in block_files_arg.split(',')]
            if len(block_file_list) != len(chromosomes):
                raise ValueError(f"Number of block files ({len(block_file_list)}) must match number of discovered chromosomes ({len(chromosomes)})")
            block_files = dict(zip(chromosomes, block_file_list))
        else:
            block_files = block_files_arg
    else:
        # Auto-discover block files
        logger.info("Auto-discovering block files...")
        data_dir = os.path.dirname(pca_embeddings_dir)
        
        # Look for block files in common locations
        search_patterns = [
            os.path.join(data_dir, "haploblocks", f"*chr{chr}_blocks*.blocks.det"),
            os.path.join(data_dir, f"*chr{chr}_blocks*.blocks.det"),
            os.path.join(os.path.dirname(data_dir), "haploblocks", f"*chr{chr}_blocks*.blocks.det"),
            os.path.join(os.path.dirname(data_dir), f"*chr{chr}_blocks*.blocks.det")
        ]
        
        for chromosome in chromosomes:
            found = False
            for pattern in search_patterns:
                pattern_filled = pattern.replace(f"{chr}", str(chromosome))
                matches = glob.glob(pattern_filled)
                if matches:
                    block_files[chromosome] = matches[0]
                    logger.info(f"Chr{chromosome}: Found block file {matches[0]}")
                    found = True
                    break
            
            if not found:
                raise FileNotFoundError(f"Could not auto-discover block file for chromosome {chromosome}. "
                                      f"Please specify --block-files explicitly.")
    
    return chromosomes, block_files

def encode_genetic_data(args):
    """Main function to encode genetic data using pre-trained models."""
    setup_logging()
    
    logger.info("Starting genetic data encoding with pre-trained DiffuGene models")
    logger.info(f"Basename: {args.basename}")
    
    # Log multi-dataset mode information
    if args.global_bfile:
        logger.info(f"Global bfile: {args.global_bfile}")
        if hasattr(args, '_basename_derived') and args._basename_derived:
            logger.info(f"Basename derived from fam file: {args.basename}")
    else:
        logger.info(f"Input PLINK files: {args.genetic_binary_folder}/{args.basename}")
        
    if args.keep_fam_file:
        logger.info(f"Multi-dataset mode: subsetting with {args.keep_fam_file}")
    
    logger.info(f"Chromosomes: {args.chromosomes}")
    logger.info(f"PCA embeddings: {args.pca_embeddings_dir}")
    logger.info(f"VAE model: {args.vae_model_path}")
    logger.info(f"Output path: {args.output_path}")
    
    # Parse chromosomes and block files
    if args.chromosomes.lower() == 'all':
        logger.info("Auto-discovering chromosomes and block files...")
        chromosomes, block_files = auto_discover_chromosomes_and_blocks(
            args.pca_embeddings_dir, 
            getattr(args, 'block_files', 'auto')
        )
    else:
        # Parse chromosomes manually
        if isinstance(args.chromosomes, str):
            chromosomes = [int(x.strip()) for x in args.chromosomes.split(',')]
        else:
            chromosomes = args.chromosomes
        
        # Parse block files
        block_files = {}
        if isinstance(args.block_files, str):
            # Assume comma-separated list in same order as chromosomes
            block_file_list = [x.strip() for x in args.block_files.split(',')]
            if len(block_file_list) != len(chromosomes):
                raise ValueError(f"Number of block files ({len(block_file_list)}) must match number of chromosomes ({len(chromosomes)})")
            block_files = dict(zip(chromosomes, block_file_list))
        else:
            block_files = args.block_files
    
    # Create working directories
    work_dir = os.path.join(args.work_dir, f"encoding_{args.basename}")
    recoded_dir = os.path.join(work_dir, "recoded_blocks")
    embeddings_dir = os.path.join(work_dir, "embeddings")
    snplist_dir = os.path.join(work_dir, "snplists")
    
    for dir_path in [work_dir, recoded_dir, embeddings_dir, snplist_dir]:
        ensure_dir_exists(dir_path)
    
    # Step 1: Recode genetic blocks using existing block definitions
    logger.info("Step 1: Recoding genetic blocks...")
    
    recoded_files = run_plink_recode_blocks(
        plink_basename=args.basename,
        genetic_binary_folder=args.genetic_binary_folder,
        chromosomes=chromosomes,
        block_files=block_files,
        output_dir=recoded_dir,
        snplist_folder=snplist_dir,
        keep_fam_file=getattr(args, 'keep_fam_file', None),
        global_bfile=getattr(args, 'global_bfile', None),
        training_snplist_dir=getattr(args, 'training_snplist_dir', None),
        training_basename=getattr(args, 'training_basename', None),
        training_bim_file=getattr(args, 'training_bim_file', None)
    )
    
    if not recoded_files:
        raise RuntimeError("No blocks were successfully recoded")
    
    # Step 2: Apply pre-trained PCA to each block
    logger.info("Step 2: Applying pre-trained PCA to blocks...")
    
    # Find PCA directories
    loadings_dir = os.path.join(args.pca_embeddings_dir, "loadings")
    means_dir = os.path.join(args.pca_embeddings_dir, "means") 
    metadata_dir = os.path.join(args.pca_embeddings_dir, "metadata")
    
    if not os.path.exists(loadings_dir):
        raise FileNotFoundError(f"PCA loadings directory not found: {loadings_dir}")
    if not os.path.exists(means_dir):
        raise FileNotFoundError(f"PCA means directory not found: {means_dir}")
    
    embeddings_created = []
    for raw_file in tqdm(recoded_files, desc="Applying PCA"):
        try:
            # Apply PCA and get embeddings
            embeddings, chromosome, block_no = apply_pretrained_pca(
                raw_file=raw_file,
                loadings_dir=loadings_dir,
                means_dir=means_dir,
                metadata_dir=metadata_dir,
                basename=args.basename,
                k=args.pca_k
            )
            
            # Save embeddings
            chr_embeddings_dir = os.path.join(embeddings_dir, f"chr{chromosome}")
            ensure_dir_exists(chr_embeddings_dir)
            
            embedding_file = os.path.join(
                chr_embeddings_dir,
                f"{args.basename}_chr{chromosome}_block{block_no}_embeddings.pt"
            )
            torch.save(embeddings, embedding_file)
            embeddings_created.append(embedding_file)
            
        except Exception as e:
            logger.warning(f"Failed to process {raw_file}: {e}")
            continue
        
        logger.info(f"Successfully created {len(embeddings_created)} block embeddings")
    
    # Step 3: Create HDF5 file and spans CSV for VAE inference
    logger.info("Step 3: Creating HDF5 embeddings and spans CSV...")
    
    spans_csv = os.path.join(work_dir, f"{args.basename}_blocks_{args.pca_k}PC_inference.csv")
    h5_path = os.path.join(work_dir, f"{args.basename}_embeddings.h5")
    
    if not os.path.exists(spans_csv):
        spans_csv = create_spans_csv_multi_chr(
            block_files=block_files,
            embeddings_base_dir=embeddings_dir,
            basename=args.basename,
            pca_k=args.pca_k
        )
    
    # Create HDF5 file if not exists
    if not os.path.exists(h5_path):
        logger.info("Creating HDF5 embeddings file...")
        # Determine n_train from first embedding file
        first_embedding_file = glob.glob(os.path.join(embeddings_dir, "*/*_embeddings.pt"))[0]
        first_emb = torch.load(first_embedding_file, weights_only=False)
        n_train = first_emb.shape[0]
        
        precompute_embeddings(spans_csv, h5_path, n_train, args.pca_k)
    
    # Step 4: Run VAE inference
    logger.info("Step 4: Running VAE inference...")
    
    if os.path.exists(args.output_path):
        logger.info(f"VAE inference output already exists at {args.output_path}, skipping VAE inference step...")
    else:
        # Prepare arguments for VAE inference
        vae_args = SimpleNamespace()
        vae_args.model_path = args.vae_model_path
        vae_args.spans_file = spans_csv
        vae_args.recoded_dir = recoded_dir
        vae_args.embeddings_dir = args.pca_embeddings_dir
        vae_args.output_path = args.output_path
        
        # VAE model parameters (these should match the trained model)
        vae_args.grid_h = args.grid_h
        vae_args.grid_w = args.grid_w
        vae_args.block_dim = args.pca_k
        vae_args.pos_dim = args.pos_dim
        vae_args.latent_channels = args.latent_channels
        
        # Run VAE inference
        inference(vae_args)
    
    logger.info(f"Encoding completed successfully! Latents saved to: {args.output_path}")
    
    # Print summary
    logger.info("=== ENCODING SUMMARY ===")
    logger.info(f"Input data: {args.basename} (chromosomes: {chromosomes})")
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
  # Single chromosome
  python encode_genetic_data.py \\
    --basename mydata \\
    --genetic-binary-folder /path/to/plink/files \\
    --chromosomes "22" \\
    --block-files "/path/to/chr22.blocks.det" \\
    --pca-embeddings-dir /path/to/pca_embeddings \\
    --vae-model-path /path/to/trained_vae.pt \\
    --output-path /path/to/output_latents.pt
    
  # Multiple specific chromosomes 
  python encode_genetic_data.py \\
    --basename mydata \\
    --genetic-binary-folder /path/to/plink/files \\
    --chromosomes "20,21,22" \\
    --block-files "/path/to/chr20.blocks.det,/path/to/chr21.blocks.det,/path/to/chr22.blocks.det" \\
    --pca-embeddings-dir /path/to/pca_embeddings \\
    --vae-model-path /path/to/trained_vae.pt \\
    --output-path /path/to/output_latents.pt
    
  # All available chromosomes (auto-discovery) with training SNP lists
  python encode_genetic_data.py \\
    --basename mydata \\
    --genetic-binary-folder /path/to/plink/files \\
    --chromosomes "all" \\
    --training-snplist-dir /path/to/haploblocks_snps \\
    --training-basename training_basename \\
    --pca-embeddings-dir /path/to/pca_embeddings \\
    --vae-model-path /path/to/trained_vae.pt \\
    --output-path /path/to/output_latents.pt
    
  # Multi-dataset mode (basename derived from fam file)
  python encode_genetic_data.py \\
    --global-bfile /path/to/full/dataset \\
    --keep-fam-file /path/to/subset.fam \\
    --chromosomes "all" \\
    --pca-embeddings-dir /path/to/pca_embeddings \\
    --vae-model-path /path/to/trained_vae.pt \\
    --output-path /path/to/output_latents.pt
    
  # Multi-dataset mode with explicit basename and training SNP lists
  python encode_genetic_data.py \\
    --basename custom_name \\
    --global-bfile /path/to/full/dataset \\
    --keep-fam-file /path/to/subset.fam \\
    --chromosomes "all" \\
    --training-snplist-dir /path/to/haploblocks_snps \\
    --training-basename training_basename \\
    --pca-embeddings-dir /path/to/pca_embeddings \\
    --vae-model-path /path/to/trained_vae.pt \\
    --output-path /path/to/output_latents.pt
        """
    )
    
    # Input data arguments
    parser.add_argument("--basename", type=str, required=False,
                       help="Base name for PLINK files (without extensions). Required when using --genetic-binary-folder. Optional when using --global-bfile with --keep-fam-file (will be derived from fam file)")
    parser.add_argument("--genetic-binary-folder", type=str, required=False,
                       help="Directory containing PLINK binary files (.bed, .bim, .fam). Required if --global-bfile not provided")
    parser.add_argument("--chromosomes", type=str, required=True,
                       help="Comma-separated list of chromosome numbers to process (e.g., '20,21,22') or 'all' to auto-discover all available chromosomes")
    
    # Multi-dataset support  
    parser.add_argument("--global-bfile", type=str, help="Global PLINK bfile to use (overrides default genetic-binary-folder/basename)")
    parser.add_argument("--keep-fam-file", type=str, help="Fam file for --keep to subset individuals (enables multi-dataset mode)")
    
    # Pre-trained model arguments
    parser.add_argument("--block-files", type=str, required=False, default='auto',
                       help="Comma-separated list of LD block definition files (.blocks.det) from training, one per chromosome. Use 'auto' to auto-discover (default when chromosomes='all')")
    parser.add_argument("--training-snplist-dir", type=str, required=False,
                       help="Directory containing training snplist files (e.g., haploblocks_snps) to use exact SNPs from training. If not provided, uses original block files.")
    parser.add_argument("--training-basename", type=str, required=False,
                       help="Basename used during training for snplist files (required if --training-snplist-dir is used)")
    parser.add_argument("--training-bim-file", type=str, required=False,
                       help="BIM file to use for SNP position lookup when using training snplist files. If not provided, will try to auto-discover.")
    parser.add_argument("--pca-embeddings-dir", type=str, required=True,
                       help="Directory containing pre-trained PCA embeddings (with loadings/, means/, metadata/ subdirs)")
    parser.add_argument("--vae-model-path", type=str, required=True,
                       help="Path to trained VAE model")
    
    # Output arguments
    parser.add_argument("--output-path", type=str, required=True,
                       help="Output path for encoded latents (.pt file)")
    parser.add_argument("--work-dir", type=str, default="./work",
                       help="Working directory for temporary files")
    
    # PCA parameters
    parser.add_argument("--pca-k", type=int, default=4,
                       help="Number of PCA components (should match training)")
    
    # VAE model parameters (should match trained model)
    parser.add_argument("--grid-h", type=int, default=64,
                       help="VAE grid height")
    parser.add_argument("--grid-w", type=int, default=64,
                       help="VAE grid width")
    parser.add_argument("--pos-dim", type=int, default=16,
                       help="Position embedding dimension")
    parser.add_argument("--latent-channels", type=int, default=128,
                       help="Number of latent channels")
    
    # Other options
    parser.add_argument("--cleanup", action="store_true",
                       help="Clean up temporary files after completion")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.global_bfile and not args.genetic_binary_folder:
        parser.error("Either --global-bfile or --genetic-binary-folder must be provided.")
    
    if args.global_bfile:
        # Using global bfile - check that it exists
        plink_bed = f"{args.global_bfile}.bed"
        if not os.path.exists(plink_bed):
            parser.error(f"Global PLINK .bed file not found: {plink_bed}")
        # Derive basename from fam file if not provided
        if not args.basename:
            if args.keep_fam_file:
                # Derive from fam file
                fam_basename = os.path.basename(args.keep_fam_file)
                if fam_basename.endswith('.fam'):
                    args.basename = fam_basename[:-4]  # Remove .fam extension
                else:
                    args.basename = fam_basename
                args._basename_derived = True
                args._basename_source = "fam_file"
            else:
                parser.error("--basename must be provided when using --global-bfile without --keep-fam-file")
        else:
            args._basename_derived = False
            args._basename_source = "explicit"
    else:
        # Using default path - check genetic-binary-folder and basename
        if not args.basename:
            parser.error("--basename must be provided when using --genetic-binary-folder.")
        if not os.path.exists(args.genetic_binary_folder):
            parser.error(f"Genetic binary folder not found: {args.genetic_binary_folder}")
        plink_bed = os.path.join(args.genetic_binary_folder, f"{args.basename}.bed")
        if not os.path.exists(plink_bed):
            parser.error(f"PLINK .bed file not found: {plink_bed}")
        args._basename_derived = False  # Basename was explicitly provided
        args._basename_source = "explicit"
    
    # Check keep fam file if provided
    if args.keep_fam_file:
        if not os.path.exists(args.keep_fam_file):
            parser.error(f"Keep fam file not found: {args.keep_fam_file}")
    
    # Validate training snplist arguments
    if args.training_snplist_dir and not args.training_basename:
        parser.error("--training-basename is required when --training-snplist-dir is provided")
    if args.training_snplist_dir and not os.path.exists(args.training_snplist_dir):
        parser.error(f"Training snplist directory not found: {args.training_snplist_dir}")
    if args.training_bim_file and not os.path.exists(args.training_bim_file):
        parser.error(f"Training BIM file not found: {args.training_bim_file}")
    
    if not os.path.exists(args.pca_embeddings_dir):
        parser.error(f"PCA embeddings directory not found: {args.pca_embeddings_dir}")
    
    if not os.path.exists(args.vae_model_path):
        parser.error(f"VAE model not found: {args.vae_model_path}")
    
    # Ensure output directory exists
    ensure_dir_exists(os.path.dirname(args.output_path))
    
    # Run encoding
    encode_genetic_data(args)

if __name__ == "__main__":
    main() 