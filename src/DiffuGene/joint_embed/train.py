#!/usr/bin/env python

import os
import glob
import argparse
import re
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..utils import setup_logging, get_logger
from ..block_embed import PCA_Block, load_pca_blocks
from .vae import SNPVAE
from .memory_efficient_dataset import BlockPCDataset, MemoryEfficientSNPLoader, unscale_embeddings
from .precompute_embeddings import precompute_embeddings

logger = get_logger(__name__)

def load_pca_metadata_only(embeddings_dir, spans_file):
    """Load only PCA metadata for creating embedding masks without loading full PCA blocks.
    
    Uses the spans file to determine correct block order (critical for multi-chromosome data).
    """
    logger.info("Loading PCA metadata for embedding masks...")
    metadata_dir = os.path.join(embeddings_dir, "metadata")
    load_dir = os.path.join(embeddings_dir, "loadings")
    
    # Read spans file to get correct block order
    df = pd.read_csv(spans_file)
    
    pca_metadata = []
    for _, row in df.iterrows():
        # Extract the exact prefix from the spans file path
        # Spans file contains: {path}/{basename}_chr{chr}_block{block}_embeddings.pt
        # PCA files are saved as: {basename}_chr{chr}_block{block}_pca_metadata.pt
        block_file_path = row.block_file
        
        # Extract just the filename and remove _embeddings.pt suffix to get the prefix
        filename = os.path.basename(block_file_path)
        if filename.endswith('_embeddings.pt'):
            prefix = filename[:-14]  # Remove '_embeddings.pt'
        else:
            raise ValueError(f"Unexpected embeddings file format: {filename}")
        
        # Build metadata file path using exact prefix (no subdirectory)
        metadata_file = os.path.join(metadata_dir, f"{prefix}_pca_metadata.pt")
        
        if os.path.exists(metadata_file):
            metadata = torch.load(metadata_file, map_location="cpu", weights_only=False)
            k = metadata['k']
            actual_k = metadata['actual_k']
            n_snps = metadata.get('n_snps', 0)  # Include SNP count for filtering
        else:
            # Fallback: load from corresponding loading file using exact prefix
            load_file = os.path.join(load_dir, f"{prefix}_pca_loadings.pt")
            
            if os.path.exists(load_file):
                loadings = torch.load(load_file, map_location="cpu", weights_only=False)
                if hasattr(loadings, 'size') and callable(loadings.size):
                    k = loadings.size(1)
                elif hasattr(loadings, 'shape'):
                    k = loadings.shape[1] 
                else:
                    raise ValueError(f"Unsupported loadings type: {type(loadings)}")
                actual_k = k
                n_snps = 0  # Unknown - will not be filtered
                logger.warning(f"No metadata found for {prefix}, assuming actual_k = k = {k}, n_snps unknown")
            else:
                raise FileNotFoundError(f"No PCA files found for prefix: {prefix}")
        
        pca_metadata.append({'k': k, 'actual_k': actual_k, 'n_snps': n_snps})
    
    logger.info(f"Loaded metadata for {len(pca_metadata)} PCA blocks in spans file order")
    return pca_metadata

def create_embedding_masks_from_metadata(pca_metadata, max_dim, device="cuda"):
    """Create masks for valid (non-padded) dimensions using metadata only."""
    masks = []
    for meta in pca_metadata:
        mask = torch.zeros(max_dim, device=device)
        mask[:meta['actual_k']] = 1.0  # Mark valid dimensions as 1
        masks.append(mask)
    return torch.stack(masks, dim=0)  # (n_blocks, max_dim)

def masked_mse_loss(pred, target, mask, reduction='mean'):
    """Compute MSE loss with masking for padded dimensions."""
    # pred, target: (batch_size, n_blocks, k)
    # mask: (n_blocks, k)
    
    squared_diff = (pred - target) ** 2
    masked_diff = squared_diff * mask.unsqueeze(0)  # Broadcast mask to batch dimension
    
    if reduction == 'mean':
        # Only compute mean over valid (non-masked) elements
        n_valid = mask.sum()
        if n_valid == 0:
            return torch.tensor(0.0, device=pred.device)
        return masked_diff.sum() / (pred.size(0) * n_valid)  # Divide by batch_size * n_valid_dims
    elif reduction == 'sum':
        return masked_diff.sum()
    else:
        return masked_diff

def evaluate_model(model, dataset, batch_size, pca_metadata, eval_blocks=128):
    """Memory-efficient evaluation of model on SNP reconstruction."""
    model.eval()
    total_mse = 0.0
    total_acc = 0.0
    total_samples = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create evaluation SNP loader with more blocks for comprehensive evaluation
    eval_snp_loader = MemoryEfficientSNPLoader(dataset, eval_blocks, pca_metadata=pca_metadata)
    
    eval_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=1,
        pin_memory=True
    )
    
    with torch.no_grad():
        for emb, spans, sample_indices in tqdm(eval_loader, desc="Evaluating"):
            emb = emb.float().to(device)
            spans = spans.float().to(device)
            
            # Forward pass
            recon_emb, _ = model(emb, spans)
            
            # Unscale embeddings before PCA decoding if scaling was applied
            recon_emb_unscaled = unscale_embeddings(recon_emb, dataset.pc_means, dataset.pc_scales)
            
            # Load SNPs for evaluation blocks
            block_indices, true_snps = eval_snp_loader.load_snps_for_batch(sample_indices, device)
            
            # Decode predictions for evaluation blocks (now uses on-demand PCA loading)
            pred_snps = eval_snp_loader.decode_predictions(recon_emb_unscaled, block_indices, device)
            
            # Compute metrics
            mse_values = []
            acc_values = []
            for i, (pred, true) in enumerate(zip(pred_snps, true_snps)):
                block_idx = block_indices[i] if i < len(block_indices) else i
                try:
                    mse = F.mse_loss(pred, true)
                    acc = (pred.round() == true).float().mean()
                    mse_values.append(mse)
                    acc_values.append(acc)
                except RuntimeError as e:
                    logger.error(f"Evaluation SNP shape mismatch at block {block_idx}:")
                    logger.error(f"  Predicted SNPs shape: {pred.shape}")
                    logger.error(f"  True SNPs shape: {true.shape}")
                    logger.error(f"  Error: {e}")
                    raise
            
            batch_mse = sum(mse_values) / len(mse_values)
            batch_acc = sum(acc_values) / len(acc_values)
            
            bs = emb.size(0)
            total_mse += batch_mse.item() * bs
            total_acc += batch_acc.item() * bs
            total_samples += bs
    
    model.train()  # Return to training mode
    return {
        'mse': total_mse / total_samples,
        'accuracy': total_acc / total_samples
    }

def train(args):
    """
    Train VAE on block embeddings using memory-efficient approach.
    1. PC embeddings are stored in HDF5 format for efficient access
    2. SNPs are loaded on-demand during training (never all in memory)
    3. PCA blocks are loaded on-demand with LRU caching
    4. Random block sampling for SNP reconstruction (--snp-blocks-per-batch)
    5. Warmup phase with PC-only loss (--warmup-epochs)
    """
    setup_logging()
    
    # Initialize epoch variable to avoid UnboundLocalError when resuming from completed training
    epoch = 0
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    scale_pc = getattr(args, 'scale_pc_embeddings', False)
    
    # Generate HDF5 path automatically based on spans file
    spans_basename = os.path.splitext(os.path.basename(args.spans_file))[0]
    h5_dir = os.path.join(os.path.dirname(args.spans_file), "h5_cache")
    os.makedirs(h5_dir, exist_ok=True)
    emb_h5_path = os.path.join(h5_dir, f"{spans_basename}_embeddings.h5")
    
    logger.info("Using memory-efficient HDF5-based training")
    
    # Create HDF5 file if it doesn't exist
    if not os.path.exists(emb_h5_path):
        logger.info(f"Creating HDF5 embeddings file: {emb_h5_path}")
        # Need to determine n_train from one of the embedding files
        df_temp = pd.read_csv(args.spans_file)
        first_emb = torch.load(df_temp.iloc[0].block_file, weights_only=False)
        if not isinstance(first_emb, torch.Tensor):
            first_emb = torch.tensor(first_emb, dtype=torch.float32)
        n_train = first_emb.shape[0]
        
        precompute_embeddings(args.spans_file, emb_h5_path, n_train)
    
    # Create single memory-efficient dataset (always includes PC embeddings)
    dataset = BlockPCDataset(
        emb_h5_path=emb_h5_path,
        spans_file=args.spans_file,
        recoded_dir=args.recoded_dir,
        embeddings_dir=args.embeddings_dir,
        scale_pc_embeddings=scale_pc
    )
    
    # Get actual block dimension from HDF5 metadata
    actual_block_dim = dataset.pc_dim
    logger.info(f"Using actual block embedding dimension from HDF5: {actual_block_dim}")
    
    # Override args.block_dim if it doesn't match the actual embeddings
    if args.block_dim != actual_block_dim:
        logger.warning(f"Config block_dim ({args.block_dim}) != actual embeddings ({actual_block_dim}). Using actual dimension.")
        args.block_dim = actual_block_dim

    # Load PCA metadata for embedding masks (much more memory efficient)
    pca_metadata = load_pca_metadata_only(args.embeddings_dir, args.spans_file)
    n_blocks = len(pca_metadata)
    logger.info(f"Total number of blocks: {n_blocks}")

    # model
    model = SNPVAE(
        n_blocks        = n_blocks,
        grid_size       = (args.grid_h, args.grid_w),
        block_emb_dim   = args.block_dim,
        pos_emb_dim     = args.pos_dim,
        latent_channels = args.latent_channels
    ).to(device)
    
    # Create embedding dimension masks for padded dimensions
    embedding_mask = create_embedding_masks_from_metadata(pca_metadata, args.block_dim, device=device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Load initial checkpoint if specified (optional - for resuming training)
    start_epoch = 1
    if (hasattr(args, 'checkpoint_path') and 
        args.checkpoint_path and 
        args.checkpoint_path.strip() and 
        os.path.exists(args.checkpoint_path)):
        logger.info(f"Loading initial checkpoint from {args.checkpoint_path}")
        try:
            checkpoint = torch.load(args.checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            logger.info(f"Resumed training from epoch {start_epoch}")
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}. Starting from scratch.")
    else:
        if hasattr(args, 'checkpoint_path') and args.checkpoint_path and args.checkpoint_path.strip():
            logger.warning(f"Checkpoint path specified but file not found: {args.checkpoint_path}")
        logger.info("Starting training from scratch")
    
    # Training parameters
    kld_weight = args.kld_weight
    lambda_mse = args.decoded_mse_weight
    warmup_epochs = getattr(args, 'snp_start_epoch', 0)
    snp_blocks_per_batch = getattr(args, 'snp_blocks_per_batch', 512)
    
    # Create SNP loader only when needed (lazy initialization)
    snp_loader = None

    for epoch in range(start_epoch, args.epochs + 1):
        # Determine whether to use SNP loss for this epoch
        use_snp_loss = args.reconstruct_snps and (epoch > warmup_epochs)
        
        # Create SNP loader on-demand when first needed
        if use_snp_loss and snp_loader is None:
            logger.info(f"Creating SNP loader for on-demand SNP loading (blocks per batch: {snp_blocks_per_batch})")
            snp_loader = MemoryEfficientSNPLoader(dataset, snp_blocks_per_batch, pca_metadata=pca_metadata)
        
        # Create data loader (always uses same dataset)
        loader = DataLoader(
            dataset, 
            batch_size=args.batch_size, 
            shuffle=True,
            num_workers=1,
            pin_memory=True
        )
        
        model.train()
        total_recon = 0.0
        total_kld   = 0.0
        if use_snp_loss:
            total_snp_mse = 0.0
        total_samples = 0

        for batch_data in tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}"):
            emb, spans, sample_indices = batch_data
            # Move to GPU and ensure correct types
            emb = emb.float().to(device)
            spans = spans.float().to(device)
            
            # Forward pass
            recon_emb, dist = model(emb, spans)
            
            # Reconstruction loss (embedding space) with masking for padded dimensions
            recon_loss = masked_mse_loss(recon_emb, emb, embedding_mask)
            
            # KLD loss
            kld_loss = dist.kl().mean()
            
            # Start with basic VAE loss
            loss = recon_loss + kld_weight * kld_loss
            
            # Add SNP reconstruction loss if in SNP phase
            if use_snp_loss:
                # Unscale embeddings before PCA decoding if scaling was applied
                recon_emb_unscaled = unscale_embeddings(recon_emb, dataset.pc_means, dataset.pc_scales)
                
                # Load SNPs on-demand for random subset of blocks
                block_indices, true_snps = snp_loader.load_snps_for_batch(sample_indices, device)
                
                # Decode predictions for selected blocks only (now uses on-demand PCA loading)
                pred_snps = snp_loader.decode_predictions(recon_emb_unscaled, block_indices, device)
                
                # Compute SNP loss for selected blocks
                decoded_mse_values = []
                for i, (pred, true) in enumerate(zip(pred_snps, true_snps)):
                    block_idx = block_indices[i]
                    try:
                        mse = F.mse_loss(pred, true)
                        decoded_mse_values.append(mse)
                    except RuntimeError as e:
                        logger.error(f"SNP shape mismatch at block {block_idx}:")
                        logger.error(f"  Predicted SNPs shape: {pred.shape}")
                        logger.error(f"  True SNPs shape: {true.shape}")
                        logger.error(f"  Error: {e}")
                        # Get block info for more context
                        if hasattr(snp_loader, 'block_info') and block_idx < len(snp_loader.block_info):
                            chr_num, block_num, block_base = snp_loader.block_info[block_idx]
                            logger.error(f"  Block info: chr{chr_num}_block{block_num} ({block_base})")
                        raise
                
                decoded_mse = sum(decoded_mse_values) / len(decoded_mse_values)
                
                # Warm up SNP loss weight over first 50 epochs after warmup
                if epoch <= warmup_epochs + 50:
                    lambda_snp = lambda_mse * (epoch - warmup_epochs) / 50
                else:
                    lambda_snp = lambda_mse
                
                loss += lambda_snp * decoded_mse
                total_snp_mse += decoded_mse.item()
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Accumulate metrics
            bs = emb.size(0)
            total_recon += recon_loss.item() * bs
            total_kld += kld_loss.item() * bs
            total_samples += bs
        
        # Compute averages
        avg_recon = total_recon / total_samples
        avg_kld = total_kld / total_samples
        
        # Phase-aware logging
        phase = "SNP Loss" if use_snp_loss else "PC Only"
        if use_snp_loss:
            avg_snp_mse = total_snp_mse / len(loader)
            if epoch <= warmup_epochs + 50:
                lambda_snp = lambda_mse * (epoch - warmup_epochs) / 50
            else:
                lambda_snp = lambda_mse
            logger.info(f"Epoch {epoch} ({phase}): Recon={avg_recon:.4f} | KLD={avg_kld:.4f} | SNP_MSE={avg_snp_mse:.4f} | λ_SNP={lambda_snp:.2e} | KLD_weight={kld_weight:.2e} | LR={args.lr:.2e}")
        else:
            logger.info(f"Epoch {epoch} ({phase}): Recon={avg_recon:.4f} | KLD={avg_kld:.4f} | KLD_weight={kld_weight:.2e} | LR={args.lr:.2e}")
            
        # Log transition to SNP phase
        if epoch == warmup_epochs + 1:
            logger.info(f">>> TRANSITION: Starting SNP loss phase at epoch {epoch} <<<")
        
        # Comprehensive evaluation at specified frequency (or at end)
        eval_freq = getattr(args, 'eval_frequency', 50)  # Default to 50 if not specified
        if epoch % eval_freq == 0 or epoch == args.epochs:
            # Create evaluation SNP loader if not exists (for comprehensive evaluation)
            if snp_loader is None:
                logger.info("Creating SNP loader for evaluation")
                snp_loader = MemoryEfficientSNPLoader(dataset, snp_blocks_per_batch, pca_metadata=pca_metadata)
            
            eval_metrics = evaluate_model(model, dataset, args.batch_size, pca_metadata, eval_blocks=128)
            logger.info(f">>> Eval @ epoch {epoch}: SNP_MSE={eval_metrics['mse']:.4f} | SNP_Acc={eval_metrics['accuracy']:.4f}")
            
            # Save checkpoint that replaces the previous one (not accumulating)
            model_dir = os.path.dirname(args.model_save_path)
            model_name = os.path.splitext(os.path.basename(args.model_save_path))[0]
            
            # Single checkpoint file that gets overwritten
            checkpoint_path = os.path.join(model_dir, f"{model_name}_checkpoint.pt")
            
            # Remove previous checkpoint if it exists
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)
                logger.info(f"Replaced previous checkpoint")
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'eval_mse': eval_metrics['mse'],
                'eval_accuracy': eval_metrics['accuracy']
            }
            os.makedirs(model_dir, exist_ok=True)
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Checkpoint saved: {checkpoint_path} (epoch {epoch})")

    # Save model
    os.makedirs(os.path.dirname(args.model_save_path), exist_ok=True)
    
    # Save model with configuration for future compatibility
    final_model_data = {
        'model_state_dict': model.state_dict(),
        'config': {
            'n_blocks': n_blocks,
            'grid_h': args.grid_h,
            'grid_w': args.grid_w,
            'block_dim': args.block_dim,
            'pos_dim': args.pos_dim,
            'latent_channels': args.latent_channels,
            'scale_pc_embeddings': scale_pc
        },
        'training_info': {
            'epochs': args.epochs,
            'final_epoch': epoch,
            'lr': args.lr,
            'kld_weight': args.kld_weight
        }
    }
    
    # Save PC scaling factors if they were used
    if scale_pc and dataset.pc_means is not None:
        final_model_data['pc_scaling'] = {
            'pc_means': dataset.pc_means,
            'pc_scales': dataset.pc_scales
        }
        logger.info("Saved PC scaling factors with model")
    torch.save(final_model_data, args.model_save_path)
    logger.info(f"Model with config saved to {args.model_save_path}")

def main():
    """Main function for VAE training."""
    parser = argparse.ArgumentParser(description="Train VAE on block embeddings using memory-efficient approach")
    parser.add_argument("--spans-file", type=str, required=True)
    parser.add_argument("--recoded-dir", type=str, required=True)
    parser.add_argument("--embeddings-dir", type=str, required=True)
    parser.add_argument("--model-save-path", type=str, required=True)
    parser.add_argument("--checkpoint-path", type=str, default=None, help="Path to checkpoint for resuming training")
    
    # Memory-efficient training options
    parser.add_argument("--snp-blocks-per-batch", type=int, default=64,
                       help="Number of blocks to sample for SNP reconstruction per batch")
    parser.add_argument("--warmup-epochs", type=int, default=200,
                       help="Number of epochs to train with PC-only loss before starting SNP reconstruction")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    
    # Model architecture
    parser.add_argument("--grid-h", type=int, default=64)
    parser.add_argument("--grid-w", type=int, default=64)
    parser.add_argument("--block-dim", type=int, default=4)
    parser.add_argument("--pos-dim", type=int, default=16)
    parser.add_argument("--latent-channels", type=int, default=128)
    
    # Loss weights
    parser.add_argument("--kld-weight", type=float, default=1e-5)
    parser.add_argument("--reconstruct-snps", action="store_true")
    parser.add_argument("--decoded-mse-weight", type=float, default=1.0)
    
    # Evaluation and scaling
    parser.add_argument("--eval-frequency", type=int, default=50, help="Evaluation and checkpoint frequency (epochs)")
    parser.add_argument("--scale-pc-embeddings", action="store_true", help="Standardize PC embeddings across samples/blocks before VAE training")
    
    args = parser.parse_args()
    train(args)

if __name__ == "__main__":
    main()
