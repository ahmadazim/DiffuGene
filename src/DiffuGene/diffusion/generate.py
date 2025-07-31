#!/usr/bin/env python

import argparse
import torch
import os
import glob
import re
from diffusers import DDIMScheduler #DDPMScheduler
from tqdm import tqdm
import numpy as np
import pandas as pd

from ..utils import setup_logging, get_logger, load_covariate_metadata, prepare_covariates_for_training, sample_training_covariates
from .unet import LatentUNET2D as ConditionalUNET
from .unet_unconditional import LatentUNET2D as UnconditionalUNET
from ..joint_embed.vae import SNPVAE
from ..block_embed.pca import PCA_Block

logger = get_logger(__name__)

def prepare_conditional_generation_data(model_dir, model_name, num_samples,
                                       covariate_file, fam_file, random_seed=None):
    """Prepare covariate data for conditional generation by sampling from training data.
    
    Args:
        model_dir: Directory containing covariate metadata
        model_name: Model name for metadata file
        num_samples: Number of samples to generate
        covariate_file: Path to covariate CSV file
        fam_file: Path to training fam file
        random_seed: Random seed for reproducible sampling
    
    Returns:
        Tuple of (covariates_for_model, original_covariate_profiles)
        - covariates_for_model: Tensor for model input (num_samples, cond_dim)
        - original_covariate_profiles: Original covariate values for saving
    """
    # Load covariate metadata to get variable type information
    metadata_path = os.path.join(model_dir, f"{model_name}_covariate_metadata.json")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Covariate metadata not found: {metadata_path}")
    
    covariate_names, norm_params = load_covariate_metadata(metadata_path)
    
    # Get variable type lists from metadata
    binary_cols = norm_params.get('binary_cols', [])
    categorical_cols = norm_params.get('categorical_cols', [])
    
    logger.info(f"Loaded covariate metadata: {len(covariate_names)} features")
    logger.info(f"Binary variables: {binary_cols}")
    logger.info(f"Categorical variables: {categorical_cols}")
    
    # Sample covariate profiles from actual training data
    covariates_for_model, original_profiles = sample_training_covariates(
        covariate_path=covariate_file,
        fam_path=fam_file,
        num_samples=num_samples,
        binary_cols=binary_cols,
        categorical_cols=categorical_cols,
        random_seed=random_seed
    )
    
    logger.info(f"Sampled covariate profiles from training data: {covariates_for_model.shape}")
    
    return covariates_for_model, original_profiles

def load_vae_model(vae_model_path, device="cuda"):
    """Load the trained VAE model."""
    logger.info(f"Loading VAE model from {vae_model_path}")
    
    # Load the checkpoint
    checkpoint = torch.load(vae_model_path, map_location=device)
    
    # Handle both checkpoint format and direct state_dict format
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # Checkpoint format
        logger.info("Loading from checkpoint format")
        model_state_dict = checkpoint['model_state_dict']
        model_config = checkpoint.get('config', {})
    elif isinstance(checkpoint, dict) and any(key.startswith('joint_') for key in checkpoint.keys()):
        # Direct state_dict format
        logger.info("Loading from direct state_dict format")
        model_state_dict = checkpoint
        model_config = {}
    else:
        # Try to treat the whole thing as a state_dict
        logger.info("Loading from raw state_dict format")
        model_state_dict = checkpoint
        model_config = {}
    
    # Infer model configuration from state_dict if config not available
    if not model_config:
        logger.info("Inferring model configuration from state_dict...")
        
        # Extract configuration from the model's state dict keys
        # Look for specific layer parameters to infer architecture
        try:
            # Get grid size from the spatial decoder layers
            if 'spatial_decoder.0.weight' in model_state_dict:
                # ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1)
                # Input channels = 16, so latent_channels = 32 (since VAE splits into 2*16)
                latent_channels = 32
            else:
                latent_channels = 32  # default
            
            # Infer block_dim from cell_unproj layer
            if 'cell_unproj.weight' in model_state_dict:
                # Linear(4, block_dim)
                block_dim = model_state_dict['cell_unproj.weight'].shape[0]
            else:
                block_dim = 3  # default
            
            # Infer pos_dim from pos_mlp layers
            if 'joint_embedder.pos_mlp.0.weight' in model_state_dict:
                # Linear(3, pos_dim)
                pos_dim = model_state_dict['joint_embedder.pos_mlp.0.weight'].shape[0]
            else:
                pos_dim = 16  # default
            
            model_config = {
                'grid_h': 32,  # Standard grid size
                'grid_w': 32,
                'block_dim': block_dim,
                'pos_dim': pos_dim,
                'latent_channels': latent_channels
            }
            
            logger.info(f"Inferred config: grid_size=(32,32), block_dim={block_dim}, pos_dim={pos_dim}, latent_channels={latent_channels}")
            
        except Exception as e:
            logger.warning(f"Could not infer model config from state_dict: {e}")
            # Use reasonable defaults
            model_config = {
                'grid_h': 32,
                'grid_w': 32,
                'block_dim': 3,
                'pos_dim': 16,
                'latent_channels': 32
            }
            logger.info("Using default model configuration")
    
    # Create VAE model with configuration
    vae_model = SNPVAE(
        grid_size=(model_config.get('grid_h', 32), model_config.get('grid_w', 32)),
        block_emb_dim=model_config.get('block_dim', 3), 
        pos_emb_dim=model_config.get('pos_dim', 16),
        latent_channels=model_config.get('latent_channels', 32)
    )
    
    # Load the model weights
    try:
        vae_model.load_state_dict(model_state_dict)
        logger.info("Successfully loaded model weights")
    except Exception as e:
        logger.error(f"Failed to load model weights: {e}")
        raise
    
    vae_model.to(device)
    vae_model.eval()
    
    logger.info("VAE model loaded successfully")
    return vae_model

def load_pca_models(pca_loadings_dir, chromosomes):
    """Load all PCA models for the given chromosome(s)."""
    logger.info(f"Loading PCA models from {pca_loadings_dir}")
    
    # Handle both single chromosome (int) and multiple chromosomes (list)
    if isinstance(chromosomes, int):
        chromosomes = [chromosomes]
    
    pca_models = {}
    block_info = {}
    
    for chromosome in chromosomes:
        logger.info(f"Loading PCA models for chromosome {chromosome}")
        
        # Find all PCA model files for this chromosome
        # The actual pattern is: *_chr{chromosome}_block*_pca_loadings.pt
        pattern = os.path.join(pca_loadings_dir, f"*chr{chromosome}_block*_pca_loadings.pt")
        pca_files = sorted(glob.glob(pattern))
        
        if not pca_files:
            logger.warning(f"No PCA model files found for chromosome {chromosome} in {pca_loadings_dir}")
            continue
        
        for pca_file in pca_files:
            # Extract block information from filename
            # Expected format: "all_hm3_15k_chr22_blockXXX_pca_loadings.pt"
            basename = os.path.basename(pca_file)
            
            # Extract block number using regex
            match = re.search(r'block(\d+)_pca_loadings\.pt$', basename)
            if not match:
                logger.warning(f"Could not extract block number from {basename}")
                continue
                
            block_num = match.group(1)
            # Create unique block ID that includes chromosome to avoid conflicts
            block_id = f"{chromosome}_{block_num}"
            
            # Load the PCA model from PyTorch format
            # We need to reconstruct the PCA_Block object from the saved loadings
            try:
                # Load the loadings (components) - may be numpy array or torch tensor
                loadings = torch.load(pca_file, map_location='cpu', weights_only=False)
                
                # Convert numpy array to torch tensor if needed
                if isinstance(loadings, np.ndarray):
                    loadings = torch.from_numpy(loadings).float()
                elif not isinstance(loadings, torch.Tensor):
                    loadings = torch.tensor(loadings).float()
                
                # Load corresponding means file - shape is (n_snps,)
                means_file = pca_file.replace('/loadings/', '/means/').replace('_pca_loadings.pt', '_pca_means.pt')
                if not os.path.exists(means_file):
                    logger.warning(f"Means file not found: {means_file}")
                    continue
                means = torch.load(means_file, map_location='cpu', weights_only=False)
                
                # Convert means to tensor if needed
                if isinstance(means, np.ndarray):
                    means = torch.from_numpy(means).float()
                elif not isinstance(means, torch.Tensor):
                    means = torch.tensor(means).float()
                
                # Load metadata if available
                metadata_file = pca_file.replace('_pca_loadings.pt', '_pca_metadata.pt').replace('/loadings/', '/metadata/')
                metadata = None
                if os.path.exists(metadata_file):
                    metadata = torch.load(metadata_file, map_location='cpu', weights_only=False)
                
                # Get dimensions from metadata or infer from loadings and means
                if metadata and isinstance(metadata, dict):
                    k = metadata.get('k', 3)
                    actual_k = metadata.get('actual_k', k)
                else:
                    # Infer from shapes - means tells us n_snps
                    n_snps = len(means)
                    
                    # Determine which dimension of loadings is actual_k
                    if loadings.shape[0] == n_snps:
                        # loadings is (n_snps, actual_k) - as expected from pca.components_.T
                        actual_k = loadings.shape[1]
                    elif loadings.shape[1] == n_snps:
                        # loadings is (actual_k, n_snps) - need to transpose
                        actual_k = loadings.shape[0]
                        loadings = loadings.T
                    else:
                        # Can't determine from shapes, assume smaller dimension is actual_k
                        if loadings.shape[0] <= loadings.shape[1]:
                            actual_k = loadings.shape[0]
                            loadings = loadings.T  # Make it (n_snps, actual_k)
                        else:
                            actual_k = loadings.shape[1]
                        logger.warning(f"Block {block_id}: Could not match loadings shape {loadings.shape} with means shape {means.shape}, assuming actual_k={actual_k}")
                    
                    k = actual_k  # Assume no padding for old format
                
                # Ensure loadings are in (n_snps, actual_k) format, then transpose to (actual_k, n_snps) for PCA_Block
                if loadings.shape[0] != len(means):
                    loadings = loadings.T
                components = loadings.T  # (actual_k, n_snps)
                
                # Create PCA_Block object
                pca_model = PCA_Block(k=k)
                pca_model.actual_k = actual_k
                pca_model.components_ = components  # (actual_k, n_snps)
                pca_model.means = means  # (n_snps,)
                
                pca_models[block_id] = pca_model
                
                # Store block info (useful for debugging)
                block_info[block_id] = {
                    'file': pca_file,
                    'chromosome': chromosome,
                    'block_num': block_num,
                    'k': k,
                    'actual_k': actual_k,
                    'n_snps': len(means) if means is not None else 0,
                    'loadings_shape': tuple(loadings.shape),
                    'components_shape': tuple(components.shape)
                }
                
            except Exception as e:
                logger.warning(f"Failed to load PCA model from {pca_file}: {e}")
                continue
    
    if not pca_models:
        raise FileNotFoundError(f"No PCA model files found for any of the chromosomes {chromosomes} in {pca_loadings_dir}")
    
    logger.info(f"Loaded {len(pca_models)} PCA models across {len(chromosomes)} chromosomes")
    return pca_models, block_info

def load_spans_data(spans_file_path):
    """Load the spans data needed for VAE decoding."""
    if not os.path.exists(spans_file_path):
        raise FileNotFoundError(f"Spans file not found: {spans_file_path}")
    
    logger.info(f"Loading spans data from: {spans_file_path}")
    
    # Load CSV and create spans tensor like in SNPBlocksDataset
    df = pd.read_csv(spans_file_path)
    
    MAX_COORD_CHR22 = 50_818_468        # TODO: update for all chromosomes
    scaled_spans = []
    
    for _, row in df.iterrows():
        chr_norm = row.chr / 22
        start_norm = row.start / MAX_COORD_CHR22  
        end_norm = row.end / MAX_COORD_CHR22
        scaled_spans.append([chr_norm, start_norm, end_norm])
    
    spans = torch.tensor(scaled_spans, dtype=torch.float32)  # (n_blocks, 3)
    logger.info(f"Loaded spans data: {spans.shape} blocks")
    return spans

def decode_samples(generated_latents, vae_model, spans, pca_models, device="cuda"):
    """
    Decode generated latent samples back to SNP space.
    
    Args:
        generated_latents: Tensor of shape (n_samples, 16, 16, 16)
        vae_model: Trained VAE model
        spans: Block span information for VAE decoding
        pca_models: Dictionary of PCA models for each block
        device: Computing device
    
    Returns:
        decoded_snps: Dictionary mapping block_id to decoded SNP arrays
    """
    logger.info("Decoding generated samples...")
    
    n_samples = generated_latents.shape[0]
    generated_latents = generated_latents.to(device)
    
    # Expand spans to match the number of samples
    if spans.dim() == 2:  # (n_blocks, 3)
        spans = spans.unsqueeze(0).expand(n_samples, -1, -1)  # (n_samples, n_blocks, 3)
    
    spans = spans.to(device)
    
    # Decode latents to block embeddings using VAE
    logger.info("Decoding latents to block embeddings using VAE...")
    decoded_snps = {}
    
    # Process in batches to avoid memory issues
    batch_size = 32
    all_block_embeddings = []
    
    with torch.no_grad():
        for i in tqdm(range(0, n_samples, batch_size), desc="VAE decoding"):
            end_idx = min(i + batch_size, n_samples)
            batch_latents = generated_latents[i:end_idx]
            batch_spans = spans[i:end_idx]
            
            # VAE decode: latents -> block embeddings
            batch_block_embs = vae_model.decode(batch_latents, batch_spans)  # (batch_size, n_blocks, 3)
            all_block_embeddings.append(batch_block_embs.cpu())
    
    # Concatenate all block embeddings
    all_block_embeddings = torch.cat(all_block_embeddings, dim=0)  # (n_samples, n_blocks, 3)
    logger.info(f"Decoded to block embeddings: {all_block_embeddings.shape}")
    
    # Decode each block using corresponding PCA model
    logger.info("Decoding block embeddings to SNPs using PCA...")
    n_blocks = all_block_embeddings.shape[1]
    
    # Get sorted block IDs for consistent ordering
    sorted_block_ids = sorted(pca_models.keys())
    
    if len(sorted_block_ids) != n_blocks:
        logger.warning(f"Mismatch: {len(sorted_block_ids)} PCA models but {n_blocks} blocks in embeddings")
        # Use the minimum to avoid index errors
        n_blocks_to_process = min(len(sorted_block_ids), n_blocks)
    else:
        n_blocks_to_process = n_blocks
    
    for block_idx in tqdm(range(n_blocks_to_process), desc="PCA decoding"):
        block_id = sorted_block_ids[block_idx]
        pca_model = pca_models[block_id]
        
        # Get block embeddings for this block across all samples
        block_embs = all_block_embeddings[:, block_idx, :]  # (n_samples, 3)
        
        # Decode using PCA
        decoded_snps_block = pca_model.decode(block_embs)  # (n_samples, n_snps_in_block)
        
        # Convert to numpy and store
        if isinstance(decoded_snps_block, torch.Tensor):
            decoded_snps_block = decoded_snps_block.cpu().numpy()
        
        decoded_snps[block_id] = decoded_snps_block
        
        logger.debug(f"Block {block_id}: decoded {decoded_snps_block.shape[1]} SNPs for {n_samples} samples")
    
    logger.info(f"Successfully decoded {len(decoded_snps)} blocks")
    return decoded_snps

def save_decoded_samples(decoded_snps, output_dir, basename, chromosomes):
    """Save decoded SNPs to files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Handle both single chromosome (int) and multiple chromosomes (list)
    if isinstance(chromosomes, int):
        chromosomes = [chromosomes]
    
    # Group blocks by chromosome for saving
    chr_blocks = {}
    for block_id, snps in decoded_snps.items():
        # Extract chromosome from block_id (format: "chr_blocknum")
        if '_' in block_id:
            chr_num, block_num = block_id.split('_', 1)
            chr_num = int(chr_num)
        else:
            # Fallback for old format
            chr_num = chromosomes[0] if len(chromosomes) == 1 else 22
            block_num = block_id
        
        if chr_num not in chr_blocks:
            chr_blocks[chr_num] = {}
        chr_blocks[chr_num][block_num] = snps
    
    # Save individual block files organized by chromosome
    total_blocks = 0
    for chr_num, blocks in chr_blocks.items():
        for block_num, snps in blocks.items():
            block_file = os.path.join(output_dir, f"{basename}_chr{chr_num}_block_{block_num}_decoded.pt")
            # Convert to tensor and save
            if isinstance(snps, torch.Tensor):
                torch.save(snps, block_file)
            else:
                torch.save(torch.from_numpy(snps), block_file)
        total_blocks += len(blocks)
        logger.info(f"Saved {len(blocks)} blocks for chromosome {chr_num}")
    
    # Save summary info
    chr_suffix = "all" if len(chromosomes) > 1 else str(chromosomes[0])
    summary_file = os.path.join(output_dir, f"{basename}_chr{chr_suffix}_decode_summary.txt")
    with open(summary_file, 'w') as f:
        f.write(f"Decoded SNPs Summary for {basename} chromosomes {chromosomes}\n")
        f.write(f"Number of chromosomes: {len(chr_blocks)}\n")
        f.write(f"Total blocks: {total_blocks}\n")
        f.write(f"Sample count: {next(iter(decoded_snps.values())).shape[0]}\n")
        f.write("\nChromosome details:\n")
        for chr_num, blocks in sorted(chr_blocks.items()):
            f.write(f"Chromosome {chr_num}: {len(blocks)} blocks\n")
            for block_num, snps in sorted(blocks.items(), key=lambda x: int(x[0])):
                f.write(f"  Block {block_num}: {snps.shape[1]} SNPs\n")
    
    logger.info(f"Saved decode summary to {summary_file}")

def generate(args):
    setup_logging()
    
    # Load checkpoint first to detect model type
    checkpoint = torch.load(args.model_path, map_location='cuda')
    
    # Detect if model is conditional or unconditional
    is_conditional = checkpoint.get('conditional', False)
    cond_dim = checkpoint.get('cond_dim', None)
    
    # Create appropriate model
    if is_conditional:
        if cond_dim is None:
            raise ValueError("Conditional model detected but cond_dim not found in checkpoint")
        logger.info(f"Loading conditional model with {cond_dim} covariate dimensions")
        model = ConditionalUNET(input_channels=64, output_channels=64, cond_dim=cond_dim).cuda()
    else:
        logger.info("Loading unconditional model")
        model = UnconditionalUNET(input_channels=64, output_channels=64).cuda()
    
    if 'ema' in checkpoint:
        logger.info("Loading EMA weights")
        try:
            from timm.utils import ModelEmaV3
            
            # Create temporary EMA wrapper to load the saved EMA state
            ema = ModelEmaV3(model, decay=0.9)
            ema.load_state_dict(checkpoint['ema'])
            
            # Handle different timm versions - try multiple methods to copy EMA weights
            if hasattr(ema, 'copy_to'):
                # Older timm versions
                logger.info("Using ema.copy_to() method")
                ema.copy_to(model)
            elif hasattr(ema, 'module'):
                # Newer timm versions - use the EMA module directly
                logger.info("Using ema.module.state_dict() method")
                model.load_state_dict(ema.module.state_dict())
            else:
                # Fallback - try alternative approaches
                logger.info("Trying manual EMA state dict extraction")
                success = False
                
                # Method 1: Try accessing shadow parameters
                if hasattr(ema, 'shadow'):
                    try:
                        model.load_state_dict(ema.shadow)
                        success = True
                        logger.info("Loaded EMA weights from shadow")
                    except:
                        pass
                
                # Method 2: Try getting state dict and removing prefixes
                if not success:
                    try:
                        ema_state = ema.state_dict()
                        model_state = {}
                        for k, v in ema_state.items():
                            # Handle various prefixes that might exist
                            key = k
                            for prefix in ['module.', 'shadow.', 'ema.']:
                                if key.startswith(prefix):
                                    key = key[len(prefix):]
                                    break
                            model_state[key] = v
                        model.load_state_dict(model_state)
                        success = True
                        logger.info("Loaded EMA weights via manual extraction")
                    except Exception as e:
                        logger.warning(f"Manual EMA extraction failed: {e}")
                
                if not success:
                    logger.warning("All EMA loading methods failed. Using non-EMA weights.")
                    model.load_state_dict(checkpoint['weights'])
                    
        except Exception as e:
            logger.warning(f"Failed to load EMA weights: {e}. Using non-EMA weights.")
            model.load_state_dict(checkpoint['weights'])
    else:
        logger.info("No EMA weights found, loading regular model weights")
        model.load_state_dict(checkpoint['weights'])
    
    model.eval()
    
    # Initialize scheduler
    scheduler = DDIMScheduler(
        num_train_timesteps=args.num_time_steps,
        beta_schedule="linear",
        beta_start=1e-4,
        beta_end=0.02, 
        clip_sample=False, 
        prediction_type="epsilon"
    )
    scheduler.set_timesteps(args.num_inference_steps, device="cuda")
    
    # Load normalization stats
    model_dir = os.path.dirname(args.model_path)
    # Extract model name from model path (without extension)
    model_name = os.path.splitext(os.path.basename(args.model_path))[0]
    # channel_means = torch.load(os.path.join(model_dir, f"train_{model_name}_channel_means.pt"), map_location='cuda', weights_only=False)
    # channel_stds = torch.load(os.path.join(model_dir, f"train_{model_name}_channel_stds.pt"), map_location='cuda', weights_only=False)
    sigma_hat = torch.load(os.path.join(model_dir, f"train_{model_name}_sigma.pt"), map_location='cuda', weights_only=False)
    
    # Prepare conditional data if needed
    covariates = None
    original_covariate_profiles = None
    if is_conditional:
        model_dir = os.path.dirname(args.model_path)
        model_name = os.path.splitext(os.path.basename(args.model_path))[0]
        
        # Get conditional generation arguments
        covariate_file = getattr(args, 'covariate_file', None)
        fam_file = getattr(args, 'fam_file', None)
        random_seed = getattr(args, 'random_seed', None)
        
        if not covariate_file or not fam_file:
            raise ValueError("Conditional generation requires --covariate-file and --fam-file arguments")
        
        covariates, original_covariate_profiles = prepare_conditional_generation_data(
            model_dir=model_dir,
            model_name=model_name,
            num_samples=args.num_samples,
            covariate_file=covariate_file,
            fam_file=fam_file,
            random_seed=random_seed
        )
        covariates = covariates.cuda()
    
    guidance_scale = args.guidance_scale

    # Generate samples
    logger.info(f"Generating {args.num_samples} samples...")
    all_samples = []
    
    for i in tqdm(range(0, args.num_samples, args.batch_size)):
        batch_size = min(args.batch_size, args.num_samples - i)
        
        # Start from random noise
        latents = torch.randn(batch_size, 64, 64, 64, device="cuda")
        
        # Get batch covariates if conditional
        batch_covariates = None
        if is_conditional:
            batch_covariates = covariates[i:i+batch_size]
        
        # Denoising loop
        with torch.no_grad():
            for t in tqdm(scheduler.timesteps, desc=f"Denoising batch {i//args.batch_size + 1}"):
                # Model prediction - conditional or unconditional
                # if is_conditional:
                #     noise_pred = model(latents, t.expand(batch_size), batch_covariates)
                # else:
                #     noise_pred = model(latents, t.expand(batch_size))
                if is_conditional:
                    B = latents.size(0)
                    e_cond = model.cond_emb(batch_covariates).unsqueeze(1)  # (B,1,256)
                    e_null = model.null_cond_emb.unsqueeze(0).expand(B,1,256)

                    # double batch
                    x_in   = torch.cat([latents,      latents],      dim=0)
                    t_in   = torch.cat([t.expand(B),  t.expand(B)],  dim=0)
                    emb_in = torch.cat([e_null,       e_cond],        dim=0)

                    h_all    = model.input_proj(x_in)
                    eps_all  = model.unet(h_all, t_in, encoder_hidden_states=emb_in).sample
                    eps0, eps1 = eps_all.chunk(2, dim=0)

                    # CFG mix
                    noise_pred = eps0 + guidance_scale * (eps1 - eps0)
                else:
                    noise_pred = model(latents, t.expand(batch_size))
                
                # Scheduler step
                latents = scheduler.step(
                    model_output=noise_pred,
                    timestep=t,
                    sample=latents,
                    eta=0.0
                ).prev_sample
        
        # Denormalize
        latents = latents * sigma_hat
        all_samples.append(latents.cpu())
    
    # Concatenate all samples
    all_samples = torch.cat(all_samples, dim=0)
    logger.info(f"Generated samples shape: {all_samples.shape}")
    
    # Save latent samples
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    torch.save(all_samples, args.output_path)
    logger.info(f"Latent samples saved to {args.output_path}")
    
    # Save covariate profiles if conditional generation
    if is_conditional and original_covariate_profiles is not None:
        # Save covariate profiles in the same directory as generated samples
        output_dir = os.path.dirname(args.output_path)
        covariate_output_path = os.path.join(output_dir, f"{model_name}_covariate_profiles.csv")
        
        # Load covariate metadata to get column names
        metadata_path = os.path.join(model_dir, f"{model_name}_covariate_metadata.json")
        covariate_names, _ = load_covariate_metadata(metadata_path)
        
        # Create DataFrame with original covariate profiles
        covariate_df = pd.DataFrame(original_covariate_profiles, columns=covariate_names)
        covariate_df.insert(0, 'sample_id', [f"sample_{i:06d}" for i in range(len(covariate_df))])
        
        # Save to CSV
        covariate_df.to_csv(covariate_output_path, index=False)
        logger.info(f"Covariate profiles saved to {covariate_output_path}")
        logger.info(f"Saved {len(covariate_df)} covariate profiles with {len(covariate_names)} features")
    
    # Decode samples if decoding arguments are provided
    if hasattr(args, 'decode_samples') and args.decode_samples:
        try:
            logger.info("Starting sample decoding...")
            
            # Load VAE model
            vae_model = load_vae_model(args.vae_model_path, device="cuda")
            
            # Handle chromosomes argument (can be list or single chromosome)
            chromosomes = getattr(args, 'chromosomes', getattr(args, 'chromosome', 22))
            
            # Load PCA models for all chromosomes
            pca_models, block_info = load_pca_models(args.pca_loadings_dir, chromosomes)
            
            # Load spans data from explicitly specified file
            spans = load_spans_data(args.spans_file)
            
            # Decode samples
            decoded_snps = decode_samples(all_samples, vae_model, spans, pca_models, device="cuda")
            
            # Save decoded samples
            save_decoded_samples(decoded_snps, args.decoded_output_dir, args.basename, chromosomes)
            
            logger.info("Sample decoding completed successfully!")
            
        except Exception as e:
            logger.error(f"Error during sample decoding: {e}")
            logger.info("Continuing without decoding...")

def main():
    """Main function for diffusion sample generation."""
    parser = argparse.ArgumentParser(description="Generate samples using trained diffusion model")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-time-steps", type=int, default=1000)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    
    # Conditional generation arguments (for conditional models only)
    parser.add_argument("--covariate-file", type=str, help="Path to covariate CSV file (for conditional models)")
    parser.add_argument("--fam-file", type=str, help="Path to training fam file (for conditional models)")
    parser.add_argument("--random-seed", type=int, help="Random seed for reproducible sampling")
    parser.add_argument("--guidance-scale", type=float, default=5, help="CFG scale (how strongly to apply covariates)")
    
    # Decoding arguments
    parser.add_argument("--decode-samples", action="store_true", help="Decode generated samples to SNP space")
    parser.add_argument("--vae-model-path", type=str, help="Path to trained VAE model")
    parser.add_argument("--pca-loadings-dir", type=str, help="Directory containing PCA model files")
    parser.add_argument("--spans-file", type=str, help="Path to spans CSV file")
    parser.add_argument("--decoded-output-dir", type=str, help="Directory to save decoded SNPs")
    parser.add_argument("--basename", type=str, help="Dataset basename")
    parser.add_argument("--chromosome", type=int, help="Chromosome number")
    
    args = parser.parse_args()
    generate(args)

if __name__ == "__main__":
    main()
