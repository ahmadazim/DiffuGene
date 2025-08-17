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
from torch.optim.lr_scheduler import CosineAnnealingLR

from ..utils import setup_logging, get_logger
from ..block_embed import PCA_Block, load_pca_blocks
from .vae import SNPVAE
from .memory_efficient_dataset import BlockPCDataset, SNPLoader, unscale_embeddings
from .precompute_embeddings import precompute_embeddings

logger = get_logger(__name__)

def load_pca_metadata_only(embeddings_dir, spans_file):
    """
    Load only PCA metadata for creating embedding masks without loading full PCA blocks.
    Follows spans file ordering.
    """
    logger.info("Loading PCA metadata for embedding masks...")
    metadata_dir = os.path.join(embeddings_dir, "metadata")
    load_dir = os.path.join(embeddings_dir, "loadings")
    df = pd.read_csv(spans_file)
    
    pca_metadata = []
    for _, row in df.iterrows():
        block_file_path = row.block_file
        filename = os.path.basename(block_file_path)
        if filename.endswith('_embeddings.pt'):
            prefix = filename[:-14]
        else:
            raise ValueError(f"Unexpected embeddings file format: {filename}")
        metadata_file = os.path.join(metadata_dir, f"{prefix}_pca_metadata.pt")
        if os.path.exists(metadata_file):
            metadata = torch.load(metadata_file, map_location="cpu", weights_only=False)
            k = metadata['k']
            actual_k = metadata['actual_k']
            n_snps = metadata.get('n_snps', 0)  # Include SNP count for filtering
        else:
            raise FileNotFoundError(f"No PCA metadata found for {prefix}")
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

def evaluate_model(model, dataset, batch_size, snp_loader):
    """Memory-efficient evaluation of model on SNP reconstruction."""
    model.eval()
    total_mse = 0.0
    total_acc = 0.0
    total_samples = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    eval_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=16,
        pin_memory=True,
        prefetch_factor=2
    )
    
    with torch.no_grad():
        for emb, spans, sample_indices in tqdm(eval_loader, desc="Evaluating"):
            emb = emb.float().to(device)
            spans = spans.float().to(device)
            
            # Forward pass
            recon_emb, _ = model(emb, spans)
            recon_emb_unscaled = unscale_embeddings(recon_emb, dataset.pc_means, dataset.pc_scales)
            block_indices, true_snps = snp_loader.load_snps_for_batch(sample_indices, device)
            pred_snps = snp_loader.decode_predictions(recon_emb_unscaled, block_indices, device)
            
            # Compute metrics
            mse_values = []
            acc_values = []
            for i, (pred, true) in enumerate(zip(pred_snps, true_snps)):
                block_idx = block_indices[i]
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
    model.train()
    return {
        'mse': total_mse / total_samples,
        'accuracy': total_acc / total_samples
    }

def train(args):
    """
    Train VAE on block embeddings using memory-efficient approach.
    1. PC embeddings are stored in HDF5 format for efficient access
    2. SNPs are loaded on-demand during training (never all in memory)
    3. PCA blocks are loaded on-demand 
    4. Random block sampling for SNP reconstruction (--snp-blocks-per-batch)
    5. Warmup phase with PC-only loss (--warmup-epochs)
    """
    setup_logging()
    
    # Initialize to avoid UnboundLocalError when resuming from completed training
    epoch = 0
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    scale_pc = getattr(args, 'scale_pc_embeddings', False)
    
    # Generate HDF5 path automatically based on spans file
    spans_basename = os.path.splitext(os.path.basename(args.spans_file))[0]
    h5_dir = os.path.join(args.embeddings_dir, "h5_cache")
    os.makedirs(h5_dir, exist_ok=True)
    emb_h5_path = os.path.join(h5_dir, f"{spans_basename}_embeddings.h5")
    logger.info("Using memory-efficient HDF5-based training")
    
    # Create HDF5 file if it doesn't exist
    if not os.path.exists(emb_h5_path):
        logger.info(f"Creating HDF5 embeddings file: {emb_h5_path}")
        # determine n_train from one of the embedding files
        df_temp = pd.read_csv(args.spans_file)
        first_emb = torch.load(df_temp.iloc[0].block_file, weights_only=False)
        if not isinstance(first_emb, torch.Tensor):
            first_emb = torch.tensor(first_emb, dtype=torch.float32)
        n_train = first_emb.shape[0]
        precompute_embeddings(args.spans_file, emb_h5_path, n_train, args.block_dim)
    
    dataset = BlockPCDataset(
        emb_h5_path=emb_h5_path,
        spans_file=args.spans_file,
        recoded_dir=args.recoded_dir,
        embeddings_dir=args.embeddings_dir,
        scale_pc_embeddings=scale_pc
    )
    if dataset.pc_dim != args.block_dim: 
        raise ValueError(f"Block dimension mismatch: {dataset.pc_dim} != {args.block_dim}")

    # Load PCA metadata for embedding masks
    pca_metadata = load_pca_metadata_only(args.embeddings_dir, args.spans_file)
    n_blocks = len(pca_metadata)
    logger.info(f"Total number of blocks: {n_blocks}")

    model = SNPVAE(
        n_blocks        = n_blocks,
        grid_size       = (args.grid_h, args.grid_w),
        block_emb_dim   = args.block_dim,
        pos_emb_dim     = args.pos_dim,
        latent_channels = args.latent_channels
    ).to(device)

    embedding_mask = create_embedding_masks_from_metadata(pca_metadata, args.block_dim, device=device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    scaler = torch.GradScaler('cuda')
    
    # Load initial checkpoint if specified
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
            last_checkpoint_path = args.checkpoint_path  # Set for NaN recovery
            logger.info(f"Resumed training from epoch {start_epoch}")
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}. Starting from scratch.")
    else:
        if hasattr(args, 'checkpoint_path') and args.checkpoint_path and args.checkpoint_path.strip():
            logger.warning(f"Checkpoint path specified but file not found: {args.checkpoint_path}")
        logger.info("Starting training from scratch")
        last_checkpoint_path = None
    
    # Training parameters
    kld_weight = args.kld_weight
    lambda_mse = args.decoded_mse_weight
    warmup_epochs = getattr(args, 'snp_start_epoch', 0)
    snp_blocks_per_batch = getattr(args, 'snp_blocks_per_batch', 64)
    snp_loader = None

    for epoch in range(start_epoch, args.epochs + 1):
        use_snp_loss = args.reconstruct_snps and (epoch > warmup_epochs)
        if use_snp_loss and snp_loader is None:
            logger.info(f"Creating SNP loader for on-demand SNP loading (blocks per batch: {snp_blocks_per_batch})")
            snp_loader = SNPLoader(dataset, snp_blocks_per_batch, pca_metadata=pca_metadata)
        
        loader = DataLoader(
            dataset, 
            batch_size=args.batch_size, 
            shuffle=True,
            num_workers=16,
            pin_memory=True, 
            prefetch_factor=2
        )
        
        eval_freq = getattr(args, 'eval_frequency', 50)
        flush_every = 100 # flush cache every 100 batches
        
        model.train()
        total_recon = 0.0
        total_kld   = 0.0
        if use_snp_loss:
            total_snp_mse = 0.0
        total_samples = 0
        batch_idx = 0

        for batch_data in tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}"):
            emb, spans, sample_indices = batch_data
            emb = emb.float().to(device)
            spans = spans.float().to(device)
            
            # # Forward pass
            # recon_emb, dist = model(emb, spans)
            
            # # Losses
            # recon_loss = masked_mse_loss(recon_emb, emb, embedding_mask)
            # kld_loss = dist.kl().mean()
            # loss = recon_loss + kld_weight * kld_loss
            
            # FORWARD & LOSS IN AUTOCAST
            with torch.autocast('cuda'):
                recon_emb, dist = model(emb, spans)
                recon_loss = masked_mse_loss(recon_emb, emb, embedding_mask)
                kld_loss = dist.kl().mean()
                loss = recon_loss + kld_weight * kld_loss
                
                # Check for NaN in reconstruction loss and revert to last checkpoint if found
                if torch.isnan(recon_loss):
                    logger.warning(f"NaN detected in reconstruction loss at epoch {epoch}, batch {batch_idx}")
                    if last_checkpoint_path and os.path.exists(last_checkpoint_path):
                        logger.info(f"Reverting to last checkpoint: {last_checkpoint_path}")
                        checkpoint = torch.load(last_checkpoint_path, map_location=device)
                        model.load_state_dict(checkpoint['model_state_dict'])
                        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                        logger.info("Model and optimizer restored from checkpoint")
                        torch.cuda.empty_cache()
                        continue  # Skip this batch and continue with next
                    else:
                        logger.error("No checkpoint available for recovery. Stopping training.")
                        raise RuntimeError("NaN detected in loss but no checkpoint available for recovery")
            
                if use_snp_loss:
                    recon_emb_unscaled = unscale_embeddings(recon_emb, dataset.pc_means, dataset.pc_scales)
                    block_indices, true_snps = snp_loader.load_snps_for_batch(sample_indices, device)
                    pred_snps = snp_loader.decode_predictions(recon_emb_unscaled, block_indices, device)
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
                            if hasattr(snp_loader, 'block_info') and block_idx < len(snp_loader.block_info):
                                chr_num, block_num, block_base = snp_loader.block_info[block_idx]
                                logger.error(f"  Block info: chr{chr_num}_block{block_num} ({block_base})")
                            raise
                    decoded_mse = sum(decoded_mse_values) / len(decoded_mse_values)
                    lambda_snp = lambda_mse
                    if epoch <= warmup_epochs + 50:
                        lambda_snp *= (epoch - warmup_epochs) / 50
                    loss += lambda_snp * decoded_mse
                    total_snp_mse += decoded_mse.item()
            
            # Backward pass
            optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if batch_idx % flush_every == 0:
                torch.cuda.empty_cache()
            batch_idx += 1
            
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
            lambda_snp = lambda_mse * (epoch - warmup_epochs) / 50 if epoch <= warmup_epochs + 50 else lambda_mse
            logger.info(f"Epoch {epoch} ({phase}): Recon={avg_recon:.4f} | KLD={avg_kld:.4f} | SNP_MSE={avg_snp_mse:.4f} | λ_SNP={lambda_snp:.2e} | KLD_weight={kld_weight:.2e} | LR={scheduler.get_last_lr()[0]:.2e}")
        else:
            logger.info(f"Epoch {epoch} ({phase}): Recon={avg_recon:.4f} | KLD={avg_kld:.4f} | KLD_weight={kld_weight:.2e} | LR={scheduler.get_last_lr()[0]:.2e}")
            
        if epoch == warmup_epochs + 1:
            logger.info(f">>> TRANSITION: Starting SNP loss phase at epoch {epoch} <<<")
        
        torch.cuda.empty_cache()
        scheduler.step()

        # Evaluate and save checkpoint
        if (epoch % eval_freq == 0 or epoch == args.epochs):
            if snp_loader is None:
                logger.info("Creating SNP loader for evaluation")
                snp_loader = SNPLoader(dataset, snp_blocks_per_batch, pca_metadata=pca_metadata)
            eval_metrics = evaluate_model(model, dataset, args.batch_size, snp_loader)
            logger.info(f">>> Eval @ epoch {epoch}: SNP_MSE={eval_metrics['mse']:.4f} | SNP_Acc={eval_metrics['accuracy']:.4f}")
            
            # Save checkpoint 
            model_dir = os.path.dirname(args.model_save_path)
            model_name = os.path.splitext(os.path.basename(args.model_save_path))[0]
            checkpoint_path = os.path.join(model_dir, f"{model_name}_checkpoint_{epoch}.pt")
            # if os.path.exists(checkpoint_path):
            #     os.remove(checkpoint_path)
            #     logger.info(f"Replaced previous checkpoint")
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'eval_mse': eval_metrics['mse'],
                'eval_accuracy': eval_metrics['accuracy']
            }
            os.makedirs(model_dir, exist_ok=True)
            torch.save(checkpoint, checkpoint_path)
            last_checkpoint_path = checkpoint_path  # Update for NaN recovery
            logger.info(f"Checkpoint saved: {checkpoint_path} at epoch {epoch}")

    # Save final model with config
    os.makedirs(os.path.dirname(args.model_save_path), exist_ok=True)
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
