#!/usr/bin/env python
"""
DiffuGene End-to-End Pipeline

This script orchestrates the complete DiffuGene pipeline:
1. Data preparation and LD block inference
2. Block-wise PCA embedding
3. Joint VAE embedding
4. Diffusion model training
5. Sample generation
"""

import os
import sys
import argparse
import subprocess
import glob
from pathlib import Path
from typing import Dict, Any, List

# Suppress warnings before importing torch/other libraries
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
os.environ.setdefault('PYTHONWARNINGS', 'ignore')

import torch
import numpy as np
import warnings
import pandas as pd
warnings.filterwarnings('ignore')

from .config import load_config, get_default_config_path, expand_variables
from .utils import setup_logging, get_logger, ensure_dir_exists, load_blocks_for_chr

logger = get_logger(__name__)

def get_chromosome_list(chromosome_spec):
    """Get list of chromosomes to process based on chromosome specification.
    
    Args:
        chromosome_spec: Either "all" or specific chromosome number
        
    Returns:
        List of chromosome numbers to process
    """
    if str(chromosome_spec).lower() == "all":
        return list(range(1, 23))  # Chromosomes 1-22
    else:
        return [int(chromosome_spec)]

def create_embedding_spans_csv(config, embeddings_dir, pca_k):
    """Create CSV file listing all embedding files with their genomic coordinates."""
    logger.info("Creating embedding spans CSV...")
    
    chromosomes = get_chromosome_list(config['global']['chromosome'])
    all_spans = []
    
    for chr_num in chromosomes:
        # Load LD blocks from the original block definition file
        block_file = f"{config['data_prep']['block_folder']}/{config['global']['basename']}_chr{chr_num}_blocks.blocks.det"
        
        if not os.path.exists(block_file):
            logger.warning(f"Block definition file not found for chr {chr_num}: {block_file}")
            continue
        
        LD_blocks = load_blocks_for_chr(block_file, chr_num)
        
        # Create DataFrame with block information for this chromosome
        for i, block in enumerate(LD_blocks):
            block_spans = [block.chr, block.bp1, block.bp2]
            # Add embedding file paths (using actual block numbers from embeddings)
            block_file_path = os.path.join(
                embeddings_dir,
                f"{config['global']['basename']}_chr{block.chr}_block{i+1}_embeddings.pt"
            )
            all_spans.append([block_file_path, block.chr, block.bp1, block.bp2])
    
    if not all_spans:
        raise RuntimeError("No valid chromosome data found for embedding spans CSV")
    
    df = pd.DataFrame(all_spans, columns=["block_file", "chr", "start", "end"])
    
    # Create unique CSV filename with PC information
    chr_suffix = "all" if len(chromosomes) > 1 else str(chromosomes[0])
    csv_path = f"{config['data_prep']['block_folder']}/{config['global']['basename']}_chr{chr_suffix}_blocks_{pca_k}PC.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Embedding spans CSV saved to {csv_path} with {len(all_spans)} blocks from {len(chromosomes)} chromosomes")
    
    return csv_path

def create_multi_chromosome_spans_csv(config, embeddings_dir, pca_k, chromosomes):
    """Create a unified CSV file for multiple chromosomes."""
    logger.info(f"Creating multi-chromosome embedding spans CSV for chromosomes: {chromosomes}")
    
    all_spans = []
    total_blocks = 0
    
    for chr_num in chromosomes:
        # Load LD blocks from the original block definition file
        block_file = f"{config['data_prep']['block_folder']}/{config['global']['basename']}_chr{chr_num}_blocks.blocks.det"
        
        if not os.path.exists(block_file):
            logger.warning(f"Block definition file not found for chr {chr_num}: {block_file}")
            continue
        
        LD_blocks = load_blocks_for_chr(block_file, chr_num)
        chr_blocks = 0
        
        # Create DataFrame with block information for this chromosome
        for i, block in enumerate(LD_blocks):
            # Add embedding file paths (using actual block numbers from embeddings)
            block_file_path = os.path.join(
                embeddings_dir,
                f"{config['global']['basename']}_chr{block.chr}_block{i+1}_embeddings.pt"
            )
            # Verify the embedding file exists
            if os.path.exists(block_file_path):
                all_spans.append([block_file_path, block.chr, block.bp1, block.bp2])
                chr_blocks += 1
        
        total_blocks += chr_blocks
        logger.info(f"Added {chr_blocks} blocks from chromosome {chr_num}")
    
    if not all_spans:
        raise RuntimeError("No valid embedding files found for any chromosome")
    
    df = pd.DataFrame(all_spans, columns=["block_file", "chr", "start", "end"])
    
    # Create unique CSV filename with PC information for all chromosomes
    csv_path = f"{config['data_prep']['block_folder']}/{config['global']['basename']}_chrall_blocks_{pca_k}PC.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Multi-chromosome embedding spans CSV saved to {csv_path} with {total_blocks} total blocks from {len(chromosomes)} chromosomes")
    
    return csv_path

class DiffuGenePipeline:
    """Main pipeline orchestrator for DiffuGene."""
    
    def __init__(self, config_path: str = None):
        """Initialize pipeline with configuration."""
        if config_path is None:
            config_path = get_default_config_path()
        
        self.config = load_config(config_path)
        self.setup_logging()
        
        # Set random seed
        if self.config['global']['random_seed']:
            torch.manual_seed(int(self.config['global']['random_seed']))
    
    def setup_logging(self):
        """Setup logging based on config."""
        log_config = self.config.get('logging', {})
        setup_logging(
            level=log_config.get('level', 'INFO'),
            format_str="%(asctime)s [%(levelname)s] [Pipeline] %(message)s"
        )
        
        # Also log to file if specified
        if 'log_file' in log_config:
            ensure_dir_exists(os.path.dirname(log_config['log_file']))
    
    def check_step_outputs(self, step_name: str) -> bool:
        """Check if a step has already been completed by looking for expected outputs."""
        if self.config['pipeline'].get('force_rerun', False):
            return False
            
        chromosomes = get_chromosome_list(self.config['global']['chromosome'])
        
        if step_name == 'data_prep':
            # Check if recoded block files exist for all required chromosomes
            all_chr_exist = True
            for chr_num in chromosomes:
                pattern = os.path.join(
                    self.config['data_prep']['recoded_block_folder'],
                    f"{self.config['global']['basename']}_chr{chr_num}_block*_recodeA.raw"
                )
                if len(glob.glob(pattern)) == 0:
                    all_chr_exist = False
                    break
            return all_chr_exist
            
        elif step_name == 'block_embed':
            # Check if both embeddings and spans CSV exist for all chromosomes
            embeddings_dir = self.config['block_embed']['output_dirs']['embeddings']
            pca_k = int(self.config['block_embed']['pca_k'])
            
            # Check embeddings exist for all chromosomes
            all_emb_exist = True
            for chr_num in chromosomes:
                pattern = f"{embeddings_dir}/*_chr{chr_num}_*_embeddings.pt"
                if len(glob.glob(pattern)) == 0:
                    all_emb_exist = False
                    break
            
            # Check spans CSV exists
            chr_suffix = "all" if len(chromosomes) > 1 else str(chromosomes[0])
            spans_csv = f"{self.config['data_prep']['block_folder']}/{self.config['global']['basename']}_chr{chr_suffix}_blocks_{pca_k}PC.csv"
            
            return all_emb_exist and os.path.exists(spans_csv)
            
        elif step_name == 'joint_embed':
            model_path = self.config['joint_embed']['model_save_path']
            latents_path = self.config['joint_embed']['latents_output_path']
            return os.path.exists(model_path) and os.path.exists(latents_path)
            
        elif step_name == 'diffusion':
            model_path = self.config['diffusion']['model_output_path']
            return os.path.exists(model_path)
            
        elif step_name == 'generation':
            output_path = self.config['generation']['output_path']
            latents_exist = os.path.exists(output_path)
            
            # If decoding is enabled, also check for decoded samples
            gen_config = self.config['generation']
            if gen_config.get('decode_samples', False):
                # Ensure the decoded output directory path is properly expanded for checking
                temp_config = {
                    'global': self.config['global'],
                    'joint_embed': self.config['joint_embed'],
                    'block_embed': self.config['block_embed'],
                    'data_prep': self.config['data_prep'],
                    'generation': gen_config
                }
                expanded_config = expand_variables(temp_config)
                decoded_dir = expanded_config['generation']['decoded_output_dir']
                
                if os.path.exists(decoded_dir):
                    # Check for decoded samples across all chromosomes
                    chromosomes = get_chromosome_list(self.config['global']['chromosome'])
                    decoded_samples_exist = True
                    for chr_num in chromosomes:
                        pattern = os.path.join(decoded_dir, f"*chr{chr_num}_block_*_decoded.pt")
                        if len(glob.glob(pattern)) == 0:
                            decoded_samples_exist = False
                            break
                    return latents_exist and decoded_samples_exist
                else:
                    return False
            else:
                return latents_exist
            
        return False
    
    def run_data_prep(self):
        """Step 1: Data preparation and LD block inference."""
        logger.info("=" * 60)
        logger.info("STEP 1: Data Preparation & LD Block Inference")
        logger.info("=" * 60)
        
        if self.check_step_outputs('data_prep'):
            logger.info("Data prep outputs found, skipping...")
            return
        
        # Import and run the data prep module
        from .block_embed.run import main as data_prep_main
        from types import SimpleNamespace
        
        chromosomes = get_chromosome_list(self.config['global']['chromosome'])
        logger.info(f"Processing chromosomes: {chromosomes}")
        
        # Process each chromosome
        for chr_num in chromosomes:
            logger.info(f"Processing chromosome {chr_num}...")
            
            # Prepare arguments (ensure proper types)
            args = SimpleNamespace()
            args.basename = self.config['global']['basename']
            args.chrNo = chr_num
            args.genetic_binary_folder = self.config['data_prep']['genetic_binary_folder']
            args.block_folder = self.config['data_prep']['block_folder']
            args.recoded_block_folder = self.config['data_prep']['recoded_block_folder']
            args.snplist_folder = self.config['data_prep']['snplist_folder']
            args.embedding_folder = self.config['block_embed']['output_dirs']['embeddings']
            
            # Add PLINK parameters from config (ensure proper types)
            plink_config = self.config['data_prep']['plink']
            args.plink_max_kb = int(plink_config['max_kb'])
            args.plink_min_maf = float(plink_config['min_maf'])
            args.plink_strong_lowci = float(plink_config['strong_lowci'])
            args.plink_strong_highci = float(plink_config['strong_highci'])
            args.plink_recomb_highci = float(plink_config['recomb_highci'])
            args.plink_inform_frac = float(plink_config['inform_frac'])
            
            # Run data preparation for this chromosome
            try:
                data_prep_main(args)
                logger.info(f"Data preparation completed successfully for chromosome {chr_num}")
            except Exception as e:
                logger.error(f"Data preparation failed for chromosome {chr_num}: {e}")
                raise
        
        logger.info("Data preparation completed successfully for all chromosomes")
    
    def run_block_embed(self):
        """Step 2: Block-wise PCA embedding."""
        logger.info("=" * 60)
        logger.info("STEP 2: Block-wise PCA Embedding")
        logger.info("=" * 60)
        
        if self.check_step_outputs('block_embed'):
            logger.info("Block embedding outputs found, skipping...")
            return
        
        chromosomes = get_chromosome_list(self.config['global']['chromosome'])
        logger.info(f"Processing block embeddings for chromosomes: {chromosomes}")
        
        # Process each chromosome
        from .block_embed.fit_pca import main as fit_pca_main
        import re
        from tqdm import tqdm
        
        # Collect metrics for summary across all chromosomes
        all_block_metrics = []
        all_failed_blocks = []
        
        # Prepare config paths for fit_pca (ensure proper types)
        config_paths = {
            'recoded_dir': self.config['data_prep']['recoded_block_folder'],
            'output_dirs': self.config['block_embed']['output_dirs'],
            'basename': self.config['global']['basename'],
            'pca_k': int(self.config['block_embed']['pca_k'])
        }
        
        for chr_num in chromosomes:
            logger.info(f"Processing blocks for chromosome {chr_num}...")
            
            # Find all recoded block files for this chromosome
            pattern = os.path.join(
                self.config['data_prep']['recoded_block_folder'],
                f"{self.config['global']['basename']}_chr{chr_num}_block*_recodeA.raw"
            )
            raw_files = glob.glob(pattern)
            logger.info(f"Found {len(raw_files)} block files for chromosome {chr_num}")
            
            if not raw_files:
                logger.warning(f"No block files found for chromosome {chr_num}, skipping...")
                continue
            
            # Collect metrics for this chromosome
            chr_block_metrics = []
            chr_failed_blocks = []
            
            with tqdm(raw_files, desc=f"Chr {chr_num} blocks", unit="block") as pbar:
                for raw_file in pbar:
                    # Extract block number
                    match = re.search(r'block(\d+)', raw_file)
                    if not match:
                        logger.warning(f"Could not extract block number from {raw_file}")
                        continue
                    
                    block_no = match.group(1)
                    pbar.set_postfix(chr=chr_num, block=block_no)
                    
                    try:
                        result = fit_pca_main(
                            chrNo=str(chr_num),
                            blockNo=block_no,
                            config_paths=config_paths
                        )
                        if result:
                            chr_block_metrics.append(result)
                            all_block_metrics.append(result)
                    except Exception as e:
                        logger.warning(f"Error processing chr {chr_num} block {block_no}: {e}")
                        chr_failed_blocks.append(f"{chr_num}:{block_no}")
                        all_failed_blocks.append(f"{chr_num}:{block_no}")
                        continue
            
            # Report per-chromosome summary
            if chr_block_metrics:
                mse_values = [m['mse'] for m in chr_block_metrics]
                acc_values = [m['accuracy'] for m in chr_block_metrics]
                logger.info(f"Chr {chr_num} summary: {len(chr_block_metrics)} blocks, "
                           f"MSE={np.mean(mse_values):.4f}±{np.std(mse_values):.4f}, "
                           f"Acc={np.mean(acc_values):.4f}±{np.std(acc_values):.4f}")
        
        # Create embedding spans CSV after all embeddings are generated
        embeddings_dir = self.config['block_embed']['output_dirs']['embeddings']
        pca_k = int(self.config['block_embed']['pca_k'])
        
        if len(chromosomes) > 1:
            spans_csv_path = create_multi_chromosome_spans_csv(self.config, embeddings_dir, pca_k, chromosomes)
        else:
            spans_csv_path = create_embedding_spans_csv(self.config, embeddings_dir, pca_k)
        
        # Report overall summary statistics
        if all_block_metrics:
            mse_values = [m['mse'] for m in all_block_metrics]
            acc_values = [m['accuracy'] for m in all_block_metrics]
            
            logger.info(f"Overall block embedding summary:")
            logger.info(f"  Processed: {len(all_block_metrics)} blocks across {len(chromosomes)} chromosomes")
            logger.info(f"  Failed: {len(all_failed_blocks)} blocks")
            logger.info(f"  Mean MSE: {np.mean(mse_values):.6f} ± {np.std(mse_values):.6f}")
            logger.info(f"  Mean Accuracy: {np.mean(acc_values):.4f} ± {np.std(acc_values):.4f}")
            logger.info(f"  MSE range: [{np.min(mse_values):.6f}, {np.max(mse_values):.6f}]")
            logger.info(f"  Accuracy range: [{np.min(acc_values):.4f}, {np.max(acc_values):.4f}]")
        else:
            logger.error("No blocks were processed successfully for any chromosome")
        
        logger.info("Block embedding completed successfully for all chromosomes")
    
    def run_joint_embed(self):
        """Step 3: Joint VAE embedding."""
        logger.info("=" * 60)
        logger.info("STEP 3: Joint VAE Embedding")
        logger.info("=" * 60)
        
        if self.check_step_outputs('joint_embed'):
            logger.info("Joint embedding outputs found, skipping...")
            return
        
        # Import and run VAE training
        from .joint_embed.train import train as vae_train
        from types import SimpleNamespace
        
        # Use the PC-specific spans file created in block embedding step
        pca_k = int(self.config['block_embed']['pca_k'])
        chromosomes = get_chromosome_list(self.config['global']['chromosome'])
        chr_suffix = "all" if len(chromosomes) > 1 else str(chromosomes[0])
        spans_file = f"{self.config['data_prep']['block_folder']}/{self.config['global']['basename']}_chr{chr_suffix}_blocks_{pca_k}PC.csv"
        
        # Prepare arguments
        args = SimpleNamespace()
        args.spans_file = spans_file
        args.recoded_dir = self.config['joint_embed']['recoded_dir']
        args.embeddings_dir = self.config['joint_embed']['embeddings_dir']
        args.model_save_path = self.config['joint_embed']['model_save_path']
        args.checkpoint_path = self.config['joint_embed']['checkpoint_path']
        
        # Model parameters (ensure proper types)
        model_config = self.config['joint_embed']['model']
        args.grid_h = int(model_config['grid_h'])
        args.grid_w = int(model_config['grid_w'])
        args.block_dim = int(model_config['block_dim'])
        args.pos_dim = int(model_config['pos_dim'])
        args.latent_channels = int(model_config['latent_channels'])
        
        # Training parameters (ensure proper types)
        train_config = self.config['joint_embed']['training']
        args.epochs = int(train_config['epochs'])
        args.batch_size = int(train_config['batch_size'])
        args.lr = float(train_config['learning_rate'])
        args.kld_weight = float(train_config['kld_weight'])
        args.reconstruct_snps = bool(train_config['reconstruct_snps'])
        args.snp_start_epoch = int(train_config['snp_start_epoch'])
        args.decoded_mse_weight = float(train_config['decoded_mse_weight'])
        args.eval_frequency = int(train_config['eval_frequency'])
        args.scale_pc_embeddings = bool(train_config.get('scale_pc_embeddings', False))
        
        # Run VAE training
        vae_train(args)
        
        # Run VAE inference to generate latents
        logger.info("Generating VAE latents...")
        from .joint_embed.infer import inference as vae_inference
        
        infer_args = SimpleNamespace()
        infer_args.model_path = self.config['joint_embed']['model_save_path']  # Use final trained model
        infer_args.spans_file = spans_file
        infer_args.recoded_dir = self.config['joint_embed']['recoded_dir']
        infer_args.output_path = self.config['joint_embed']['latents_output_path']
        infer_args.grid_h = int(model_config['grid_h'])
        infer_args.grid_w = int(model_config['grid_w'])
        infer_args.block_dim = int(model_config['block_dim'])
        infer_args.pos_dim = int(model_config['pos_dim'])
        infer_args.latent_channels = int(model_config['latent_channels'])
        
        vae_inference(infer_args)
        logger.info("Joint embedding completed successfully")
    
    def run_diffusion(self):
        """Step 4: Diffusion model training."""
        logger.info("=" * 60)
        logger.info("STEP 4: Diffusion Model Training")
        logger.info("=" * 60)
        
        if self.check_step_outputs('diffusion'):
            logger.info("Diffusion model found, skipping...")
            return
        
        # Import and run diffusion training
        from .diffusion.train import train as diffusion_train
        
        # Prepare arguments
        model_config = self.config['diffusion']['model']
        train_config = self.config['diffusion']['training']
        
        diffusion_train(
            batch_size=int(train_config['batch_size']),
            num_time_steps=int(train_config['num_time_steps']),
            num_epochs=int(train_config['num_epochs']),
            seed=int(self.config['global']['random_seed']),
            ema_decay=float(train_config['ema_decay']),
            lr=float(train_config['learning_rate']),
            checkpoint_path=self.config['diffusion']['checkpoint_path'],
            model_output_path=self.config['diffusion']['model_output_path'],
            train_embed_dataset_path=self.config['diffusion']['train_embed_dataset_path']
        )
        
        logger.info("Diffusion training completed successfully")
    
    def run_generation(self):
        """Step 5: Sample generation."""
        logger.info("=" * 60)
        logger.info("STEP 5: Sample Generation")
        logger.info("=" * 60)
        
        gen_config = self.config['generation']
        output_path = gen_config['output_path']
        latents_exist = os.path.exists(output_path)
        
        # Check if decoding is needed and if decoded samples exist
        decode_enabled = gen_config.get('decode_samples', False)
        decoded_samples_exist = False
        
        if decode_enabled:
            # Ensure the decoded output directory path is properly expanded for checking
            temp_config = {
                'global': self.config['global'],
                'joint_embed': self.config['joint_embed'],
                'block_embed': self.config['block_embed'],
                'data_prep': self.config['data_prep'],
                'generation': gen_config
            }
            expanded_config = expand_variables(temp_config)
            decoded_dir = expanded_config['generation']['decoded_output_dir']
            
            if os.path.exists(decoded_dir):
                # Check for decoded samples across all chromosomes
                chromosomes = get_chromosome_list(self.config['global']['chromosome'])
                decoded_samples_exist = True
                for chr_num in chromosomes:
                    pattern = os.path.join(decoded_dir, f"*chr{chr_num}_block_*_decoded.pt")
                    if len(glob.glob(pattern)) == 0:
                        decoded_samples_exist = False
                        break
        
        # Determine what needs to be done
        if latents_exist and (not decode_enabled or decoded_samples_exist):
            logger.info("All required outputs found, skipping generation step...") 
            return
        elif latents_exist and decode_enabled and not decoded_samples_exist:
            logger.info("Latents exist but decoded samples missing - running decode-only...")
            self._run_decode_only(output_path, gen_config)
            return
        else:
            logger.info("Running full generation pipeline...")
            if not latents_exist:
                logger.info("Latents not found - will generate new samples")
            if decode_enabled and not decoded_samples_exist:
                logger.info("Decoded samples not found - will decode after generation")
        
        # Import and run sample generation
        from .diffusion.generate import generate
        from types import SimpleNamespace
        
        # Prepare arguments (ensure proper types)
        args = SimpleNamespace()
        args.model_path = gen_config['model_path']
        args.output_path = output_path
        args.num_samples = int(gen_config['num_samples'])
        args.batch_size = int(gen_config['batch_size'])
        args.num_time_steps = int(gen_config['num_time_steps'])
        args.num_inference_steps = int(gen_config['num_inference_steps'])
        
        # Add decoding parameters if enabled
        if decode_enabled:
            # Ensure paths are properly expanded
            temp_config = {
                'global': self.config['global'],
                'joint_embed': self.config['joint_embed'],
                'block_embed': self.config['block_embed'],
                'data_prep': self.config['data_prep'],
                'generation': gen_config
            }
            expanded_config = expand_variables(temp_config)
            expanded_gen_config = expanded_config['generation']
            
            args.decode_samples = True
            args.vae_model_path = expanded_gen_config['vae_model_path']
            args.pca_loadings_dir = expanded_gen_config['pca_loadings_dir']
            args.recoded_dir = expanded_gen_config['recoded_dir']
            args.decoded_output_dir = expanded_gen_config['decoded_output_dir']
            args.basename = self.config['global']['basename']
            args.chromosomes = get_chromosome_list(self.config['global']['chromosome'])
            
            logger.info("Decoding enabled - samples will be decoded to SNP space")
            logger.info(f"VAE model path: {args.vae_model_path}")
            logger.info(f"PCA loadings dir: {args.pca_loadings_dir}")
            logger.info(f"Decoded output dir: {args.decoded_output_dir}")
        else:
            args.decode_samples = False
            logger.info("Decoding disabled - only latent samples will be generated")
        
        generate(args)
        logger.info("Sample generation completed successfully")
    
    def _run_decode_only(self, latents_path, gen_config):
        """Run decoding only on existing latent samples."""
        logger.info("Loading existing latent samples for decoding...")
        
        # Import decoding functions
        from .diffusion.generate import load_vae_model, load_pca_models, load_spans_data, decode_samples, save_decoded_samples
        
        try:
            # Ensure all paths are properly expanded using the full config context
            # Create a temporary config dict with the generation section to expand variables
            temp_config = {
                'global': self.config['global'],
                'joint_embed': self.config['joint_embed'],
                'block_embed': self.config['block_embed'],
                'data_prep': self.config['data_prep'],
                'generation': gen_config
            }
            expanded_config = expand_variables(temp_config)
            expanded_gen_config = expanded_config['generation']
            
            # Load existing latents
            latents = torch.load(latents_path, map_location='cpu')
            logger.info(f"Loaded latents: {latents.shape}")
            
            # Load VAE model with expanded path
            vae_model_path = expanded_gen_config['vae_model_path']
            logger.info(f"Loading VAE model from {vae_model_path}")
            vae_model = load_vae_model(vae_model_path, device="cuda")
            
            # Load PCA models with expanded path
            pca_loadings_dir = expanded_gen_config['pca_loadings_dir']
            logger.info(f"Loading PCA models from {pca_loadings_dir}")
            pca_models, block_info = load_pca_models(pca_loadings_dir, int(self.config['global']['chromosome']))
            
            # Load spans data with expanded path
            recoded_dir = expanded_gen_config['recoded_dir']
            logger.info(f"Loading spans data from {recoded_dir}")
            spans = load_spans_data(recoded_dir, self.config['global']['basename'], int(self.config['global']['chromosome']))
            
            # Decode samples
            decoded_snps = decode_samples(latents, vae_model, spans, pca_models, device="cuda")
            
            # Save decoded samples with expanded path
            decoded_output_dir = expanded_gen_config['decoded_output_dir']
            logger.info(f"Saving decoded samples to {decoded_output_dir}")
            save_decoded_samples(decoded_snps, decoded_output_dir, self.config['global']['basename'], int(self.config['global']['chromosome']))
            
            logger.info("Decode-only operation completed successfully!")
            
        except Exception as e:
            logger.error(f"Error during decode-only operation: {e}")
            logger.info("You may need to regenerate the samples entirely.")
            raise
    
    def run_pipeline(self, steps: List[str] = None):
        """Run the complete pipeline or specified steps."""
        pipeline_config = self.config['pipeline']
        
        # Define all available steps
        all_steps = [
            ('data_prep', self.run_data_prep, pipeline_config.get('run_data_prep', True)),
            ('block_embed', self.run_block_embed, pipeline_config.get('run_block_embed', True)),
            ('joint_embed', self.run_joint_embed, pipeline_config.get('run_joint_embed', True)),
            ('diffusion', self.run_diffusion, pipeline_config.get('run_diffusion', True)),
            ('generation', self.run_generation, pipeline_config.get('run_generation', True))
        ]
        
        # Filter steps to run
        if steps:
            steps_to_run = [(name, func, True) for name, func, _ in all_steps if name in steps]
        else:
            steps_to_run = [(name, func, enabled) for name, func, enabled in all_steps if enabled]
        
        logger.info("Starting DiffuGene Pipeline")
        logger.info(f"Steps to run: {[name for name, _, _ in steps_to_run]}")
        
        # Create output directories
        for dir_path in [
            self.config['global']['model_root'],
            os.path.dirname(self.config['joint_embed']['model_save_path']),
            os.path.dirname(self.config['diffusion']['model_output_path']),
            os.path.dirname(self.config['generation']['output_path'])
        ]:
            ensure_dir_exists(dir_path)
        
        # Run steps
        try:
            for step_name, step_func, _ in steps_to_run:
                logger.info(f"\nExecuting step: {step_name}")
                step_func()
                
        except Exception as e:
            logger.error(f"Pipeline failed at step {step_name}: {e}")
            raise
        
        logger.info("\n" + "=" * 60)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)


def main():
    """Main entry point for the pipeline."""
    parser = argparse.ArgumentParser(
        description="DiffuGene End-to-End Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline with default config
  python -m DiffuGene.pipeline
  
  # Run with custom config
  python -m DiffuGene.pipeline --config my_config.yaml
  
  # Run specific steps only
  python -m DiffuGene.pipeline --steps data_prep block_embed
  
  # Force rerun all steps
  python -m DiffuGene.pipeline --force-rerun
  
  # Decode existing latent samples only
  python -m DiffuGene.pipeline --decode-only
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default=None,
        help='Path to configuration YAML file (default: use package default)'
    )
    
    parser.add_argument(
        '--steps',
        nargs='+',
        choices=['data_prep', 'block_embed', 'joint_embed', 'diffusion', 'generation'],
        help='Specific steps to run (default: run all enabled steps)'
    )
    
    parser.add_argument(
        '--force-rerun',
        action='store_true',
        help='Force rerun all steps even if outputs exist'
    )
    
    parser.add_argument(
        '--decode-only',
        action='store_true',
        help='Only decode existing latent samples (skips generation if latents exist)'
    )
    
    parser.add_argument(
        '--list-steps',
        action='store_true',
        help='List all available pipeline steps'
    )
    
    args = parser.parse_args()
    
    if args.list_steps:
        print("Available pipeline steps:")
        print("  1. data_prep    - Data preparation and LD block inference")
        print("  2. block_embed  - Block-wise PCA embedding")
        print("  3. joint_embed  - Joint VAE embedding")
        print("  4. diffusion    - Diffusion model training")
        print("  5. generation   - Sample generation")
        return
    
    # Initialize pipeline
    pipeline = DiffuGenePipeline(args.config)
    
    # Override force_rerun if specified
    if args.force_rerun:
        pipeline.config['pipeline']['force_rerun'] = True
    
    # Handle decode-only mode
    if args.decode_only:
        if args.steps and 'generation' not in args.steps:
            print("Warning: --decode-only specified but 'generation' not in --steps. Adding 'generation' step.")
            args.steps.append('generation')
        elif not args.steps:
            args.steps = ['generation']
        
        # Enable decoding in config if not already enabled
        if not pipeline.config['generation'].get('decode_samples', False):
            print("Warning: decode_samples is False in config, enabling it for --decode-only mode")
            pipeline.config['generation']['decode_samples'] = True
    
    # Run pipeline
    pipeline.run_pipeline(args.steps)


if __name__ == "__main__":
    main() 