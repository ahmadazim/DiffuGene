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
    
    def get_spans_file_path(self, step: str) -> str:
        """Get the spans file path for a given dataset step.
        
        Args:
            step: Pipeline step name ('pca', 'vae_test', 'diffusion', 'diffusion_val')
            
        Returns:
            Path to the spans file for this step's dataset
        """
        basename = self.get_dataset_basename(step)
        if basename is None:
            return None
            
        chromosomes = get_chromosome_list(self.config['global']['chromosome'])
        chr_suffix = "all" if len(chromosomes) > 1 else str(chromosomes[0])
        pca_k = int(self.config['block_embed']['pca_k'])
        
        # For VAE step, check if it uses the same dataset as PCA
        # If so, use the PCA spans file instead of creating a separate one
        if step == 'vae':
            pca_fam = self.get_dataset_fam_file('pca')
            vae_fam = self.get_dataset_fam_file('vae')
            if pca_fam == vae_fam:
                # VAE uses same dataset as PCA, so use PCA spans file
                step = 'pca'
                basename = self.get_dataset_basename('pca')
        
        # Map step to meaningful suffix
        step_suffix_map = {
            'pca': 'pca_train',
            'vae': 'vae_train', 
            'vae_test': 'vae_test',
            'diffusion': 'diffusion_train',
            'diffusion_val': 'diffusion_val'
        }
        
        step_suffix = step_suffix_map.get(step, step)
        spans_filename = f"{basename}_{step_suffix}_chr{chr_suffix}_blocks_{pca_k}PC.csv"
        
        return os.path.join(self.config['data_prep']['block_folder'], spans_filename)
    
    def get_dataset_fam_file(self, step: str) -> str:
        """Get the appropriate fam file for a given pipeline step.
        
        Args:
            step: Pipeline step name ('pca', 'vae', 'diffusion', 'vae_test', 'diffusion_val')
            
        Returns:
            Path to the fam file to use for this step
        """
        training_data = self.config.get('training_data', {})
        evaluation_data = self.config.get('evaluation_data', {})
        
        # Get the default fam file path
        default_fam = os.path.join(
            self.config['data_prep']['genetic_binary_folder'],
            f"{self.config['global']['basename']}.fam"
        )
        
        if step == 'pca':
            fam_file = training_data.get('pca_fam') or default_fam
        elif step == 'vae':
            # Fall back: vae_fam -> pca_fam -> default
            fam_file = (training_data.get('vae_fam') or 
                       training_data.get('pca_fam') or 
                       default_fam)
        elif step == 'diffusion':
            # Fall back: diffusion_fam -> vae_fam -> pca_fam -> default
            fam_file = (training_data.get('diffusion_fam') or
                       training_data.get('vae_fam') or
                       training_data.get('pca_fam') or
                       default_fam)
        elif step == 'vae_test':
            fam_file = evaluation_data.get('vae_test_fam')
        elif step == 'diffusion_val':
            fam_file = evaluation_data.get('diffusion_val_fam')
        else:
            raise ValueError(f"Unknown step: {step}")
        
        # Return None if evaluation data not specified or is null
        if step in ['vae_test', 'diffusion_val'] and not fam_file:
            return None
            
        # Use default if not specified for training steps
        if not fam_file:
            fam_file = default_fam
            
        return fam_file
    
    def get_dataset_basename(self, step: str) -> str:
        """Get a unique basename for a given pipeline step dataset.
        
        Args:
            step: Pipeline step name
            
        Returns:
            Basename to use for this step's data files
        """
        fam_file = self.get_dataset_fam_file(step)
        if fam_file is None:
            return None
            
        # Use the fam filename (without extension) as basename modifier
        fam_basename = os.path.splitext(os.path.basename(fam_file))[0]
        original_basename = self.config['global']['basename']
        
        # If using the default fam file, use original basename
        default_fam_basename = original_basename
        if fam_basename == default_fam_basename:
            return original_basename
        else:
            return f"{original_basename}_{fam_basename}"
    
    def should_run_evaluation_step(self, step: str) -> bool:
        """Check if an evaluation step should be run based on config.
        
        Args:
            step: Evaluation step name ('vae_evaluation', 'diffusion_val_encoding')
            
        Returns:
            True if the step should be run
        """
        if not self.config.get('pipeline', {}).get(f'run_{step}', False):
            return False
            
        if step == 'vae_evaluation':
            return self.get_dataset_fam_file('vae_test') is not None
        elif step == 'diffusion_val_encoding':
            return self.get_dataset_fam_file('diffusion_val') is not None
        elif step == 'diffusion_encoding':
            # Run if diffusion training data is different from VAE training data
            diffusion_fam = self.get_dataset_fam_file('diffusion')
            vae_fam = self.get_dataset_fam_file('vae')
            return diffusion_fam != vae_fam
            
        return False
    
    def run_data_encoding(self, step: str, output_suffix: str = None) -> tuple:
        """Encode genetic data using pre-trained models.
        
        Args:
            step: Dataset step name ('vae', 'vae_test', 'diffusion', 'diffusion_val')
            output_suffix: Optional suffix for output files
            
        Returns:
            Tuple of (latents_path, spans_file_path)
        """
        fam_file = self.get_dataset_fam_file(step)
        if fam_file is None:
            return None, None
            
        basename = self.get_dataset_basename(step)
        
        logger.info(f"Encoding {step} data: {fam_file} -> {basename}")
        
        # Setup for encoding with proper PLINK strategy
        genetic_binary_folder = self.config['data_prep']['genetic_binary_folder']
        original_basename = self.config['global']['basename']
        
        # Determine PLINK strategy for encoding
        if basename != original_basename:
            # Multi-dataset mode: create temporary files for the encoding process
            logger.info(f"Setting up encoding for different dataset: {basename}")
            
            # We'll create a temporary dataset with the proper files
            temp_work_dir = os.path.join(self.config['global']['data_root'], f"temp_dataset_{basename}")
            ensure_dir_exists(temp_work_dir)
            
            # Copy original bed/bim files to temp location with new basename
            original_bed = self.config['global'].get('bed_file', 
                os.path.join(genetic_binary_folder, f"{original_basename}.bed"))
            original_bim = self.config['global'].get('bim_file',
                os.path.join(genetic_binary_folder, f"{original_basename}.bim"))
            
            temp_bed = os.path.join(temp_work_dir, f"{basename}.bed")
            temp_bim = os.path.join(temp_work_dir, f"{basename}.bim")
            temp_fam = os.path.join(temp_work_dir, f"{basename}.fam")
            
            # Copy files (we'll clean up later)
            import shutil
            if not os.path.exists(temp_bed):
                shutil.copy2(original_bed, temp_bed)
            if not os.path.exists(temp_bim):
                shutil.copy2(original_bim, temp_bim)
            if not os.path.exists(temp_fam):
                shutil.copy2(fam_file, temp_fam)
            
            # Use temp directory for encoding
            encoding_genetic_folder = temp_work_dir
            encoding_basename = basename
            
            logger.info(f"Created temporary dataset files in {temp_work_dir}")
        else:
            # Single dataset mode: use original files
            encoding_genetic_folder = genetic_binary_folder
            encoding_basename = basename
        
        # Import the encoding script functionality
        import sys
        import subprocess
        from pathlib import Path
        
        # Prepare paths
        chromosomes = get_chromosome_list(self.config['global']['chromosome'])
        chr_suffix = "all" if len(chromosomes) > 1 else str(chromosomes[0])
        pca_basename = self.get_dataset_basename('pca')
        
        # PCA loadings directory
        pca_loadings_dir = self.config['block_embed']['output_dirs']['loadings']
        
        # VAE model path
        vae_model_path = self.config['joint_embed']['model_save_path']
        
        # Final output path
        if output_suffix:
            output_filename = f"{basename}_chr{chr_suffix}_VAE_latents_{output_suffix}.pt"
        else:
            output_filename = f"{basename}_chr{chr_suffix}_VAE_latents_{self.config['global']['unique_id']}.pt"
        final_output_path = os.path.join(
            os.path.dirname(self.config['joint_embed']['latents_output_path']),
            output_filename
        )
        
        # Work directory
        work_dir = os.path.join(self.config['global']['data_root'], f"work_encode_{step}")
        
        # Get model parameters from config
        model_config = self.config['joint_embed']['model']
        
        # Use the encode_genetic_data.py script functionality
        encode_script_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "scripts", "encode_genetic_data.py"
        )
        
        # For multiple chromosomes, we need to process all chromosomes together
        if len(chromosomes) > 1:
            logger.info(f"Processing all {len(chromosomes)} chromosomes together...")
            
            # Create unified work directory
            unified_work_dir = os.path.join(work_dir, "unified_encoding")
            unified_recoded_dir = os.path.join(unified_work_dir, "recoded_blocks")
            unified_embeddings_dir = os.path.join(unified_work_dir, "embeddings")
            unified_snplist_dir = os.path.join(unified_work_dir, "snplists")
            
            for dir_path in [unified_work_dir, unified_recoded_dir, unified_embeddings_dir, unified_snplist_dir]:
                ensure_dir_exists(dir_path)
            
            # Import necessary functions from the encoding script
            sys.path.insert(0, os.path.dirname(encode_script_path))
            try:
                from encode_genetic_data import run_plink_recode_blocks, apply_pretrained_pca
            except ImportError:
                # Import from the full path
                import importlib.util
                spec = importlib.util.spec_from_file_location("encode_genetic_data", encode_script_path)
                encode_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(encode_module)
                run_plink_recode_blocks = encode_module.run_plink_recode_blocks
                apply_pretrained_pca = encode_module.apply_pretrained_pca
            
            all_spans = []
            successful_chromosomes = []
            
            # Process each chromosome
            for chr_num in chromosomes:
                logger.info(f"Processing chromosome {chr_num}...")
                
                # Block file for this chromosome (from PCA training)
                block_file = f"{self.config['data_prep']['block_folder']}/{pca_basename}_chr{chr_num}_blocks.blocks.det"
                
                # Check if block file exists
                if not os.path.exists(block_file):
                    logger.warning(f"Block file not found for chromosome {chr_num}: {block_file}")
                    continue
                
                try:
                    # Step 1: Recode genetic blocks for this chromosome
                    recoded_files = run_plink_recode_blocks(
                        plink_basename=encoding_basename,
                        genetic_binary_folder=encoding_genetic_folder,
                        chromosome=chr_num,
                        block_file=block_file,
                        output_dir=unified_recoded_dir,
                        snplist_folder=unified_snplist_dir
                    )
                    
                    if not recoded_files:
                        logger.warning(f"No blocks recoded for chromosome {chr_num}")
                        continue
                    
                    # Step 2: Apply pre-trained PCA to each block
                    means_dir = os.path.join(os.path.dirname(pca_loadings_dir), "means")
                    if not os.path.exists(means_dir):
                        means_dir = pca_loadings_dir.replace("loadings", "means")
                    
                    for raw_file in recoded_files:
                        try:
                            embeddings, block_no = apply_pretrained_pca(
                                raw_file=raw_file,
                                loadings_dir=pca_loadings_dir,
                                means_dir=means_dir,
                                basename=encoding_basename,
                                chromosome=chr_num,
                                k=self.config['block_embed']['pca_k']
                            )
                            
                            # Save embeddings
                            embedding_file = os.path.join(
                                unified_embeddings_dir,
                                f"{basename}_chr{chr_num}_block{block_no}_embeddings.pt"
                            )
                            torch.save(embeddings, embedding_file)
                            
                            # Add to spans
                            LD_blocks = load_blocks_for_chr(block_file, chr_num)
                            block_idx = int(block_no) - 1  # Convert to 0-based index
                            if block_idx < len(LD_blocks):
                                block = LD_blocks[block_idx]
                                all_spans.append([embedding_file, block.chr, block.bp1, block.bp2])
                            
                        except Exception as e:
                            logger.warning(f"Failed to process {raw_file}: {e}")
                            continue
                    
                    successful_chromosomes.append(chr_num)
                    logger.info(f"Successfully processed chromosome {chr_num}")
                    
                except Exception as e:
                    logger.error(f"Failed to process chromosome {chr_num}: {e}")
                    continue
            
            if not successful_chromosomes:
                raise RuntimeError("No chromosomes were successfully processed")
            
            # Create unified spans CSV using consistent naming
            df = pd.DataFrame(all_spans, columns=["block_file", "chr", "start", "end"])
            unified_spans_file = self.get_spans_file_path(step)
            ensure_dir_exists(os.path.dirname(unified_spans_file))
            df.to_csv(unified_spans_file, index=False)
            logger.info(f"Created spans file with {len(all_spans)} blocks from {len(successful_chromosomes)} chromosomes: {unified_spans_file}")
            
            # Run VAE inference using unified data
            from .joint_embed.infer import inference as vae_inference
            from types import SimpleNamespace
            
            vae_args = SimpleNamespace()
            vae_args.model_path = vae_model_path
            vae_args.spans_file = unified_spans_file
            vae_args.recoded_dir = unified_recoded_dir
            vae_args.output_path = final_output_path
            vae_args.grid_h = int(model_config['grid_h'])
            vae_args.grid_w = int(model_config['grid_w'])
            vae_args.block_dim = int(model_config['block_dim'])
            vae_args.pos_dim = int(model_config['pos_dim'])
            vae_args.latent_channels = int(model_config['latent_channels'])
            
            logger.info("Running unified VAE inference...")
            vae_inference(vae_args)
            
            output_path = final_output_path
            spans_file_path = unified_spans_file
            logger.info(f"Multi-chromosome encoding completed: {output_path}")
        
        else:
            # Single chromosome processing (original logic)
            chr_num = chromosomes[0]
            logger.info(f"Processing single chromosome {chr_num}...")
            
            # Block file for this chromosome (from PCA training)
            block_file = f"{self.config['data_prep']['block_folder']}/{pca_basename}_chr{chr_num}_blocks.blocks.det"
            
            cmd = [
                sys.executable, encode_script_path,
                "--basename", encoding_basename,
                "--genetic-binary-folder", encoding_genetic_folder,
                "--chromosome", str(chr_num),
                "--block-file", block_file,
                "--pca-loadings-dir", pca_loadings_dir,
                "--vae-model-path", vae_model_path,
                "--output-path", final_output_path,
                "--work-dir", work_dir,
                "--pca-k", str(self.config['block_embed']['pca_k']),
                "--grid-h", str(model_config['grid_h']),
                "--grid-w", str(model_config['grid_w']),
                "--block-dim", str(model_config['block_dim']),
                "--pos-dim", str(model_config['pos_dim']),
                "--latent-channels", str(model_config['latent_channels'])
            ]
            
            logger.info(f"Running encoding command: {' '.join(cmd)}")
            
            try:
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                logger.info(f"Encoding completed successfully for {step}")
                output_path = final_output_path
                # For single chromosome, we need to create the spans file path
                spans_file_path = self.get_spans_file_path(step)
            except subprocess.CalledProcessError as e:
                logger.error(f"Encoding failed for {step}: {e}")
                logger.error(f"stdout: {e.stdout}")
                logger.error(f"stderr: {e.stderr}")
                raise
        
        # Cleanup temporary dataset files if created
        if basename != original_basename and 'temp_work_dir' in locals():
            try:
                import shutil
                shutil.rmtree(temp_work_dir)
                logger.info(f"Cleaned up temporary dataset directory: {temp_work_dir}")
            except Exception as e:
                logger.warning(f"Failed to cleanup temporary directory {temp_work_dir}: {e}")
        
        return output_path, spans_file_path
    
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
            pca_basename = self.get_dataset_basename('pca')
            all_chr_exist = True
            for chr_num in chromosomes:
                pattern = os.path.join(
                    self.config['data_prep']['recoded_block_folder'],
                    f"{pca_basename}_chr{chr_num}_block*_recodeA.raw"
                )
                if len(glob.glob(pattern)) == 0:
                    all_chr_exist = False
                    break
            return all_chr_exist
            
        elif step_name == 'block_embed':
            # Check if both embeddings and spans CSV exist for all chromosomes
            embeddings_dir = self.config['block_embed']['output_dirs']['embeddings']
            
            # Check embeddings exist for all chromosomes
            all_emb_exist = True
            for chr_num in chromosomes:
                pattern = f"{embeddings_dir}/*_chr{chr_num}_*_embeddings.pt"
                if len(glob.glob(pattern)) == 0:
                    all_emb_exist = False
                    break
            
            # Check automatically generated spans CSV exists
            spans_csv = self.get_spans_file_path('pca')
            
            return all_emb_exist and os.path.exists(spans_csv)
            
        elif step_name == 'joint_embed':
            model_path = self.config['joint_embed']['model_save_path']
            latents_path = self.config['joint_embed']['latents_output_path']
            return os.path.exists(model_path) and os.path.exists(latents_path)
            
        elif step_name == 'diffusion':
            model_path = self.config['diffusion']['model_output_path']
            return os.path.exists(model_path)
            
        elif step_name == 'vae_evaluation':
            if not self.should_run_evaluation_step('vae_evaluation'):
                return True  # Skip if not needed
            test_latents_path = self.config['joint_embed']['evaluation']['test_latents_path']
            test_reconstructions_path = self.config['joint_embed']['evaluation']['test_reconstructions_path']
            return os.path.exists(test_latents_path) and os.path.exists(test_reconstructions_path)
            
        elif step_name == 'diffusion_encoding':
            if not self.should_run_evaluation_step('diffusion_encoding'):
                return True  # Skip if not needed
            diffusion_train_latents_path = self.config['diffusion']['encoding']['diffusion_train_latents_path']
            return os.path.exists(diffusion_train_latents_path)
            
        elif step_name == 'diffusion_val_encoding':
            if not self.should_run_evaluation_step('diffusion_val_encoding'):
                return True  # Skip if not needed
            diffusion_val_latents_path = self.config['diffusion']['encoding']['diffusion_val_latents_path']
            return os.path.exists(diffusion_val_latents_path)
            
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
        
        # Get the dataset for PCA training
        pca_fam_file = self.get_dataset_fam_file('pca')
        pca_basename = self.get_dataset_basename('pca')
        
        logger.info(f"Using PCA training data: {pca_fam_file}")
        logger.info(f"PCA dataset basename: {pca_basename}")
        
        # No symlinks needed - we'll pass the actual file paths directly
        
        # Import and run the data prep module
        from .block_embed.run import main as data_prep_main
        from types import SimpleNamespace
        
        chromosomes = get_chromosome_list(self.config['global']['chromosome'])
        logger.info(f"Processing chromosomes: {chromosomes}")
        
        # Get the actual file paths for the genetic data
        genetic_binary_folder = self.config['data_prep']['genetic_binary_folder']
        original_basename = self.config['global']['basename']
        
        # Determine the PLINK strategy
        if pca_basename != original_basename:
            # Multi-dataset mode: use global bfile with --keep
            global_bfile_path = self.config['global'].get('bed_file', 
                os.path.join(genetic_binary_folder, f"{original_basename}"))
            # Remove .bed extension if present
            if global_bfile_path.endswith('.bed'):
                global_bfile_path = global_bfile_path[:-4]
            keep_fam_file = pca_fam_file
            use_keep_strategy = True
            logger.info(f"Multi-dataset mode: --bfile {global_bfile_path} --keep {keep_fam_file}")
        else:
            # Single dataset mode: use dataset-specific bfile
            global_bfile_path = os.path.join(genetic_binary_folder, pca_basename)
            keep_fam_file = None
            use_keep_strategy = False
            logger.info(f"Single dataset mode: --bfile {global_bfile_path}")
        
        # Process each chromosome
        failed_chromosomes = []
        successful_chromosomes = []
        
        for chr_num in chromosomes:
            logger.info(f"Processing chromosome {chr_num}...")
            
            # Prepare arguments (ensure proper types)
            args = SimpleNamespace()
            args.basename = pca_basename
            args.chrNo = chr_num
            args.genetic_binary_folder = genetic_binary_folder
            args.block_folder = self.config['data_prep']['block_folder']
            args.recoded_block_folder = self.config['data_prep']['recoded_block_folder']
            args.snplist_folder = self.config['data_prep']['snplist_folder']
            args.embedding_folder = self.config['block_embed']['output_dirs']['embeddings']
            
            # Add PLINK strategy arguments
            args.global_bfile = global_bfile_path
            if use_keep_strategy:
                args.keep_fam_file = keep_fam_file
            
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
                successful_chromosomes.append(chr_num)
            except Exception as e:
                logger.error(f"Data preparation failed for chromosome {chr_num}: {e}")
                failed_chromosomes.append(chr_num)
                
                # Continue with other chromosomes if this is not the only one
                if len(chromosomes) > 1:
                    logger.warning(f"Continuing with remaining chromosomes...")
                    continue
                else:
                    raise
        
        # Report summary
        if successful_chromosomes:
            logger.info(f"Data preparation completed successfully for {len(successful_chromosomes)} chromosomes: {successful_chromosomes}")
        if failed_chromosomes:
            logger.warning(f"Data preparation failed for {len(failed_chromosomes)} chromosomes: {failed_chromosomes}")
        
        if not successful_chromosomes:
            raise RuntimeError("Data preparation failed for all chromosomes")
        elif failed_chromosomes:
            logger.info(f"Continuing pipeline with {len(successful_chromosomes)} successful chromosomes")
        
        # No cleanup needed since we're not using symlinks
        
        logger.info("Data preparation phase completed")
    
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
        pca_basename = self.get_dataset_basename('pca')
        config_paths = {
            'recoded_dir': self.config['data_prep']['recoded_block_folder'],
            'output_dirs': self.config['block_embed']['output_dirs'],
            'basename': pca_basename,
            'pca_k': int(self.config['block_embed']['pca_k'])
        }
        
        for chr_num in chromosomes:
            logger.info(f"Processing blocks for chromosome {chr_num}...")
            
            # Find all recoded block files for this chromosome
            pattern = os.path.join(
                self.config['data_prep']['recoded_block_folder'],
                f"{pca_basename}_chr{chr_num}_block*_recodeA.raw"
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
        
        # Create embedding spans CSV using consistent naming
        embeddings_dir = self.config['block_embed']['output_dirs']['embeddings']
        pca_k = int(self.config['block_embed']['pca_k'])
        
        # Get consistent spans file path for PCA training data
        spans_csv_path = self.get_spans_file_path('pca')
        ensure_dir_exists(os.path.dirname(spans_csv_path))
        
        # Create the spans data
        all_spans = []
        for chr_num in chromosomes:
            # Load LD blocks from the original block definition file
            block_file = f"{self.config['data_prep']['block_folder']}/{pca_basename}_chr{chr_num}_blocks.blocks.det"
            
            if not os.path.exists(block_file):
                logger.warning(f"Block definition file not found for chr {chr_num}: {block_file}")
                continue
            
            LD_blocks = load_blocks_for_chr(block_file, chr_num)
            
            # Create DataFrame with block information for this chromosome
            for i, block in enumerate(LD_blocks):
                # Add embedding file paths (using actual block numbers from embeddings)
                block_file_path = os.path.join(
                    embeddings_dir,
                    f"{pca_basename}_chr{block.chr}_block{i+1}_embeddings.pt"
                )
                if os.path.exists(block_file_path):
                    all_spans.append([block_file_path, block.chr, block.bp1, block.bp2])
        
        # Save spans CSV
        if all_spans:
            df = pd.DataFrame(all_spans, columns=["block_file", "chr", "start", "end"])
            df.to_csv(spans_csv_path, index=False)
            logger.info(f"Created PCA training spans file: {spans_csv_path} with {len(all_spans)} blocks")
        else:
            logger.error("No valid embedding files found for spans CSV creation")
        
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
        
        # Get the datasets for VAE training
        vae_fam_file = self.get_dataset_fam_file('vae')
        vae_basename = self.get_dataset_basename('vae')
        pca_basename = self.get_dataset_basename('pca')  # For spans file
        
        logger.info(f"Using VAE training data: {vae_fam_file}")
        logger.info(f"VAE dataset basename: {vae_basename}")
        
        # Import and run VAE training
        from .joint_embed.train import train as vae_train
        from types import SimpleNamespace
        
        # Use the automatically generated spans file for VAE training data
        spans_file = self.get_spans_file_path('vae')
        
        # Check if VAE training data is different from PCA training data
        if vae_basename != pca_basename:
            logger.info("VAE training data differs from PCA training data - encoding VAE data first...")
            # For now, we'll use the PCA data paths and log a warning
            logger.warning("Multi-dataset VAE training not implemented - using PCA training data")
            recoded_dir = self.config['joint_embed']['recoded_dir']
            embeddings_dir = self.config['joint_embed']['embeddings_dir']
        else:
            # Use the original data paths
            recoded_dir = self.config['joint_embed']['recoded_dir']
            embeddings_dir = self.config['joint_embed']['embeddings_dir']
        
        # Prepare arguments
        args = SimpleNamespace()
        args.spans_file = spans_file
        args.recoded_dir = recoded_dir
        args.embeddings_dir = embeddings_dir
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
        infer_args.recoded_dir = recoded_dir
        infer_args.output_path = self.config['joint_embed']['latents_output_path']
        infer_args.grid_h = int(model_config['grid_h'])
        infer_args.grid_w = int(model_config['grid_w'])
        infer_args.block_dim = int(model_config['block_dim'])
        infer_args.pos_dim = int(model_config['pos_dim'])
        infer_args.latent_channels = int(model_config['latent_channels'])
        
        vae_inference(infer_args)
        logger.info("Joint embedding completed successfully")
    
    def run_vae_evaluation(self):
        """Step 3.5: VAE evaluation on test data."""
        logger.info("=" * 60)
        logger.info("STEP 3.5: VAE Evaluation")
        logger.info("=" * 60)
        
        if not self.should_run_evaluation_step('vae_evaluation'):
            logger.info("VAE evaluation not requested or test data not specified, skipping...")
            return
        
        # Check if evaluation outputs already exist
        test_latents_path = self.config['joint_embed']['evaluation']['test_latents_path']
        test_reconstructions_path = self.config['joint_embed']['evaluation']['test_reconstructions_path']
        
        if os.path.exists(test_latents_path) and os.path.exists(test_reconstructions_path):
            logger.info("VAE evaluation outputs found, skipping...")
            return
        
        logger.info("Running VAE evaluation on test data...")
        
        # Encode test data to get latents
        encoding_result = self.run_data_encoding('vae_test', 'test')
        
        if encoding_result is None or encoding_result[0] is None:
            logger.error("Failed to encode VAE test data")
            return
        
        test_latents_path_actual, test_spans_file = encoding_result
        
        # Decode the latents back to SNP space for evaluation
        logger.info("Decoding test latents for reconstruction evaluation...")
        
        try:
            # Import and run decoding
            import subprocess
            import sys
            
            # Get paths
            chromosomes = get_chromosome_list(self.config['global']['chromosome'])
            chr_suffix = "all" if len(chromosomes) > 1 else str(chromosomes[0])
            pca_basename = self.get_dataset_basename('pca')
            vae_test_basename = self.get_dataset_basename('vae_test')
            
            # VAE model and embeddings paths
            vae_model_path = self.config['joint_embed']['model_save_path']
            embeddings_dir = self.config['joint_embed']['embeddings_dir']
            
            # Use the spans file created during test data encoding
            spans_file = test_spans_file
            
            # Output path for reconstructions
            test_reconstructions_path_actual = test_reconstructions_path.replace(
                self.config['global']['unique_id'],
                f"{self.config['global']['unique_id']}_test"
            )
            
            # Run decoding using the decode_vae_latents.py script
            decode_script_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "joint_embed", "decode_vae_latents.py"
            )
            
            cmd = [
                sys.executable, decode_script_path,
                "--latents-file", test_latents_path_actual,
                "--model-file", vae_model_path,
                "--embeddings-dir", embeddings_dir,
                "--spans-file", spans_file,
                "--output-file", test_reconstructions_path_actual,
                "--batch-size", "256"
            ]
            
            if chr_suffix != "all":
                cmd.extend(["--chromosome", str(chr_suffix)])
            
            logger.info(f"Running decoding command: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info("VAE test data decoding completed successfully")
            
            # Copy outputs to expected paths
            if test_latents_path_actual != test_latents_path:
                import shutil
                ensure_dir_exists(os.path.dirname(test_latents_path))
                shutil.copy2(test_latents_path_actual, test_latents_path)
            
            if test_reconstructions_path_actual != test_reconstructions_path:
                ensure_dir_exists(os.path.dirname(test_reconstructions_path))
                shutil.copy2(test_reconstructions_path_actual, test_reconstructions_path)
            
            logger.info(f"VAE evaluation completed successfully!")
            logger.info(f"  Test latents: {test_latents_path}")
            logger.info(f"  Test reconstructions: {test_reconstructions_path}")
            
        except Exception as e:
            logger.error(f"VAE evaluation failed: {e}")
            raise
    
    def run_diffusion_encoding(self):
        """Step 3.7: Encode diffusion training data if different from VAE training data."""
        logger.info("=" * 60)
        logger.info("STEP 3.7: Diffusion Training Data Encoding")
        logger.info("=" * 60)
        
        if not self.should_run_evaluation_step('diffusion_encoding'):
            logger.info("Diffusion training data is same as VAE training data, skipping encoding...")
            return
        
        # Check if encoding output already exists
        diffusion_train_latents_path = self.config['diffusion']['encoding']['diffusion_train_latents_path']
        
        if os.path.exists(diffusion_train_latents_path):
            logger.info("Diffusion training latents found, skipping encoding...")
            return
        
        logger.info("Encoding diffusion training data...")
        
        # Encode diffusion training data
        encoding_result = self.run_data_encoding('diffusion', 'diffusion_train')
        
        if encoding_result is None or encoding_result[0] is None:
            logger.error("Failed to encode diffusion training data")
            return
        
        encoded_path, diffusion_spans_file = encoding_result
        
        # Copy to expected location if different
        if encoded_path != diffusion_train_latents_path:
            import shutil
            ensure_dir_exists(os.path.dirname(diffusion_train_latents_path))
            shutil.copy2(encoded_path, diffusion_train_latents_path)
        
        logger.info(f"Diffusion training data encoding completed: {diffusion_train_latents_path}")
    
    def run_diffusion_val_encoding(self):
        """Step 4.5: Encode diffusion validation data."""
        logger.info("=" * 60)
        logger.info("STEP 4.5: Diffusion Validation Data Encoding")
        logger.info("=" * 60)
        
        if not self.should_run_evaluation_step('diffusion_val_encoding'):
            logger.info("Diffusion validation data not specified, skipping...")
            return
        
        # Check if encoding output already exists
        diffusion_val_latents_path = self.config['diffusion']['encoding']['diffusion_val_latents_path']
        
        if os.path.exists(diffusion_val_latents_path):
            logger.info("Diffusion validation latents found, skipping encoding...")
            return
        
        logger.info("Encoding diffusion validation data...")
        
        # Encode diffusion validation data
        encoding_result = self.run_data_encoding('diffusion_val', 'diffusion_val')
        
        if encoding_result is None or encoding_result[0] is None:
            logger.error("Failed to encode diffusion validation data")
            return
        
        encoded_path, diffusion_val_spans_file = encoding_result
        
        # Copy to expected location if different
        if encoded_path != diffusion_val_latents_path:
            import shutil
            ensure_dir_exists(os.path.dirname(diffusion_val_latents_path))
            shutil.copy2(encoded_path, diffusion_val_latents_path)
        
        logger.info(f"Diffusion validation data encoding completed: {diffusion_val_latents_path}")
    
    def run_diffusion(self):
        """Step 4: Diffusion model training."""
        logger.info("=" * 60)
        logger.info("STEP 4: Diffusion Model Training")
        logger.info("=" * 60)
        
        if self.check_step_outputs('diffusion'):
            logger.info("Diffusion model found, skipping...")
            return
        
        # Determine which latents to use for diffusion training
        if self.should_run_evaluation_step('diffusion_encoding'):
            # Use separately encoded diffusion training latents
            train_embed_dataset_path = self.config['diffusion']['encoding']['diffusion_train_latents_path']
            logger.info(f"Using separate diffusion training latents: {train_embed_dataset_path}")
        else:
            # Use VAE training latents
            train_embed_dataset_path = self.config['diffusion']['train_embed_dataset_path']
            logger.info(f"Using VAE training latents: {train_embed_dataset_path}")
        
        # Import and run diffusion training
        from .diffusion.train import train as diffusion_train
        
        # Prepare arguments
        model_config = self.config['diffusion']['model']
        train_config = self.config['diffusion']['training']
        
        # Check if conditional generation is enabled
        conditional_config = self.config['diffusion'].get('conditional', {})
        is_conditional = conditional_config.get('enabled', False)
        
        # Basic training arguments
        train_args = {
            'batch_size': int(train_config['batch_size']),
            'num_time_steps': int(train_config['num_time_steps']),
            'num_epochs': int(train_config['num_epochs']),
            'seed': int(self.config['global']['random_seed']),
            'ema_decay': float(train_config['ema_decay']),
            'lr': float(train_config['learning_rate']),
            'checkpoint_path': self.config['diffusion']['checkpoint_path'],
            'model_output_path': self.config['diffusion']['model_output_path'],
            'train_embed_dataset_path': train_embed_dataset_path
        }
        
        # Add conditional arguments if enabled
        if is_conditional:
            logger.info("Training conditional diffusion model")
            
            # Determine which fam file to use for covariates
            if self.should_run_evaluation_step('diffusion_encoding'):
                # Use diffusion training fam file
                covariate_fam_file = self.get_dataset_fam_file('diffusion')
            else:
                # Use VAE training fam file
                covariate_fam_file = self.get_dataset_fam_file('vae')
            
            train_args.update({
                'conditional': True,
                'covariate_file': conditional_config['covariate_file'],
                'fam_file': covariate_fam_file,
                'cond_dim': model_config.get('cond_dim', 10),
                'binary_cols': conditional_config.get('binary_cols', []),
                'categorical_cols': conditional_config.get('categorical_cols', [])
            })
            
            logger.info(f"Conditional training parameters:")
            logger.info(f"  Covariate file: {train_args['covariate_file']}")
            logger.info(f"  Training fam file: {train_args['fam_file']}")
            logger.info(f"  Binary variables: {len(train_args['binary_cols'])}")
            logger.info(f"  Categorical variables: {len(train_args['categorical_cols'])}")
        else:
            logger.info("Training unconditional diffusion model")
            train_args['conditional'] = False
        
        diffusion_train(**train_args)
        
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
        
        # Add conditional generation settings
        diffusion_conditional_config = self.config['diffusion'].get('conditional', {})
        if diffusion_conditional_config.get('enabled', False):
            # For conditional generation, pass covariate file and appropriate fam file
            args.covariate_file = diffusion_conditional_config['covariate_file']
            
            # Use the same fam file as used for diffusion training
            if self.should_run_evaluation_step('diffusion_encoding'):
                args.fam_file = self.get_dataset_fam_file('diffusion')
            else:
                args.fam_file = self.get_dataset_fam_file('vae')
                
            args.random_seed = self.config['global']['random_seed']
            
            logger.info("Conditional generation enabled:")
            logger.info(f"  Covariate file: {args.covariate_file}")
            logger.info(f"  Fam file: {args.fam_file}")
            logger.info(f"  Random seed: {args.random_seed}")
        
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
            # Use the spans file from VAE training data (same as generation uses VAE training embeddings)
            args.spans_file = self.get_spans_file_path('vae')
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
            
            # Load spans data from VAE training data (used for generation)
            spans_file = self.get_spans_file_path('vae')
            logger.info(f"Loading spans data from {spans_file}")
            spans = load_spans_data(spans_file)
            
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
            ('vae_evaluation', self.run_vae_evaluation, pipeline_config.get('run_vae_evaluation', True)),
            ('diffusion_encoding', self.run_diffusion_encoding, pipeline_config.get('run_diffusion_encoding', True)),
            ('diffusion', self.run_diffusion, pipeline_config.get('run_diffusion', True)),
            ('diffusion_val_encoding', self.run_diffusion_val_encoding, pipeline_config.get('run_diffusion_val_encoding', True)),
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
        choices=['data_prep', 'block_embed', 'joint_embed', 'vae_evaluation', 'diffusion_encoding', 'diffusion', 'diffusion_val_encoding', 'generation'],
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
        print("  1. data_prep             - Data preparation and LD block inference")
        print("  2. block_embed           - Block-wise PCA embedding")
        print("  3. joint_embed           - Joint VAE embedding")
        print("  4. vae_evaluation        - VAE evaluation on test data")
        print("  5. diffusion_encoding    - Encode diffusion training data")
        print("  6. diffusion             - Diffusion model training")
        print("  7. diffusion_val_encoding - Encode diffusion validation data")
        print("  8. generation            - Sample generation")
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