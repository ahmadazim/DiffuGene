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
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from ..utils import read_raw, setup_logging, get_logger
from ..block_embed import PCA_Block  
from .vae import SNPVAE

MAX_COORD_CHR22 = 50_818_468

logger = get_logger(__name__)

class SNPBlocksDataset(Dataset):
    """
    Loads one chromosome's LD-block embeddings + spans as a single sample.
    Expects:
      - spans_file:   CSV with columns [block_file, chr, start, end]
    """
    def __init__(self, spans_file: str, recoded_dir: str, load_snps: bool = True):
        if load_snps:
            logger.info("Loading dataset...")
        self.df = pd.read_csv(spans_file)
        # load all embeddings into memory
        embs = []
        scaled_spans = []
        true_snps_blocks = []
        
        # First pass: find the maximum embedding dimension
        max_dim = 0
        for _, row in self.df.iterrows():
            emb = torch.load(row.block_file, weights_only=False)
            if not isinstance(emb, torch.Tensor):
                emb = torch.tensor(emb, dtype=torch.float32)
            else:
                emb = emb.float()
            max_dim = max(max_dim, emb.shape[-1])  # Last dimension is the embedding dimension
        
        logger.info(f"Found maximum embedding dimension: {max_dim}")
        
        # Second pass: load and pad embeddings to max_dim
        for _, row in self.df.iterrows():
            emb = torch.load(row.block_file, weights_only=False)
            if not isinstance(emb, torch.Tensor):
                emb = torch.tensor(emb, dtype=torch.float32)
            else:
                emb = emb.float()  # Ensure float32
            
            # Pad embedding to max_dim if necessary
            if emb.shape[-1] < max_dim:
                # Pad with zeros to reach max_dim
                pad_size = max_dim - emb.shape[-1]
                if emb.dim() == 2:  # (n_samples, emb_dim)
                    padding = torch.zeros(emb.shape[0], pad_size, dtype=torch.float32)
                    emb = torch.cat([emb, padding], dim=1)
                elif emb.dim() == 1:  # Single embedding vector
                    padding = torch.zeros(pad_size, dtype=torch.float32)
                    emb = torch.cat([emb, padding], dim=0)
                logger.debug(f"Padded embedding from {emb.shape[-1] - pad_size} to {emb.shape[-1]} dimensions")
            
            embs.append(emb)
            chr_norm   = row.chr / 22
            start_norm = row.start / MAX_COORD_CHR22
            end_norm   = row.end / MAX_COORD_CHR22
            scaled_spans.append([chr_norm, start_norm, end_norm])

            if load_snps:
                # get the true snps
                filename = os.path.basename(row.block_file)
                block_no = re.search(r'block(\d+)', filename).group(1)
                basename = os.path.basename(filename).split("_chr")[0]
                p = os.path.join(
                    recoded_dir, 
                    f"{basename}_chr{row.chr}_block{block_no}_recodeA.raw"
                ) 
                rec = read_raw(p)
                X = rec.impute().get_variants()
                true_snps_blocks.append(torch.from_numpy(X).long())

        block_embs = torch.stack(embs, dim=0).float()                 # (N_blocks, N_train, max_dim) - ensure float32
        self.block_embs = block_embs.permute(1, 0, 2)         # (N_train, N_blocks, max_dim)
        self.spans = torch.tensor(scaled_spans, dtype=torch.float32)       # (N_blocks, 3) - ensure float32
        
        # transpose true_snps_blocks → per‐sample lists
        N_blocks = len(true_snps_blocks)
        N_train  = self.block_embs.size(0)
        self.true_snps: list[list[torch.Tensor]] = []
        for sample_idx in range(N_train):
            if load_snps:
                per_sample = [
                    true_snps_blocks[blk_idx][sample_idx]
                    for blk_idx in range(N_blocks)
                ]
            else:
                per_sample = []
            self.true_snps.append(per_sample)

    def __len__(self):
        return self.block_embs.size(0)

    def __getitem__(self, idx):
        # return one sample: block_embs, spans
        emb, spans, true_snps = self.block_embs[idx], self.spans, self.true_snps[idx]
        return emb, spans, true_snps

def load_pca_blocks(embeddings_dir):
    """Load PCA blocks for SNP reconstruction with dimension metadata."""
    logger.info("Loading PCA blocks...")
    pca_blocks = []
    load_dir = os.path.join(embeddings_dir, "loadings")
    mean_dir = os.path.join(embeddings_dir, "means")
    metadata_dir = os.path.join(embeddings_dir, "metadata")
    
    # Get all PCA loading files and sort them numerically by block number
    load_files = glob.glob(os.path.join(load_dir, "*_pca_loadings.pt"))
    
    # Extract block numbers and sort numerically
    def extract_block_number(filepath):
        filename = os.path.basename(filepath)
        match = re.search(r'block(\d+)', filename)
        return int(match.group(1)) if match else 0
    
    load_files_sorted = sorted(load_files, key=extract_block_number)
    
    for load_file in load_files_sorted:
        block_base = os.path.basename(load_file).replace("_pca_loadings.pt", "")
        means_file = os.path.join(mean_dir, f"{block_base}_pca_means.pt")
        metadata_file = os.path.join(metadata_dir, f"{block_base}_pca_metadata.pt")
        
        loadings = torch.load(load_file, map_location="cuda", weights_only=False)    # (n_snps, k)
        means = torch.load(means_file, map_location="cuda", weights_only=False)      # (n_snps,)
        
        # Load metadata if available (backwards compatibility)
        if os.path.exists(metadata_file):
            metadata = torch.load(metadata_file, map_location="cpu", weights_only=False)
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

def create_embedding_masks(pca_blocks, max_dim, device="cuda"):
    """Create masks for valid (non-padded) dimensions in block embeddings."""
    masks = []
    for pca_block in pca_blocks:
        mask = torch.zeros(max_dim, device=device)
        mask[:pca_block.actual_k] = 1.0  # Mark valid dimensions as 1
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

def train(args):
    setup_logging()
    
    # Create two datasets: one without SNPs (faster), one with SNPs (for SNP loss phases)
    ds_no_snps = SNPBlocksDataset(args.spans_file, args.recoded_dir, load_snps=False)
    ds_with_snps = SNPBlocksDataset(args.spans_file, args.recoded_dir, load_snps=True)
    # For evaluation, always use dataset with SNPs
    ds_eval = ds_with_snps
    
    # Get the actual embedding dimension from the dataset
    actual_block_dim = ds_no_snps.block_embs.shape[-1]
    logger.info(f"Using actual block embedding dimension: {actual_block_dim}")
    
    # Override args.block_dim if it doesn't match the actual embeddings
    if args.block_dim != actual_block_dim:
        logger.warning(f"Config block_dim ({args.block_dim}) != actual embeddings ({actual_block_dim}). Using actual dimension.")
        args.block_dim = actual_block_dim
    
    # model
    model = SNPVAE(
        grid_size       = (args.grid_h, args.grid_w),
        block_emb_dim   = args.block_dim,
        pos_emb_dim     = args.pos_dim,
        latent_channels = args.latent_channels
    ).cuda()

    # Load PCA blocks for SNP reconstruction
    pca_blocks = load_pca_blocks(args.embeddings_dir)
    
    # Create embedding dimension masks for padded dimensions
    embedding_mask = create_embedding_masks(pca_blocks, args.block_dim, device="cuda")
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Load initial checkpoint if specified (optional - for resuming training)
    start_epoch = 1
    if (hasattr(args, 'checkpoint_path') and 
        args.checkpoint_path and 
        args.checkpoint_path.strip() and 
        os.path.exists(args.checkpoint_path)):
        logger.info(f"Loading initial checkpoint from {args.checkpoint_path}")
        try:
            checkpoint = torch.load(args.checkpoint_path, map_location='cuda')
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
    mse_loss  = nn.MSELoss(reduction='mean')
    
    # Training parameters
    kld_weight = args.kld_weight
    lambda_mse = args.decoded_mse_weight

    for epoch in range(start_epoch, args.epochs + 1):
        # Determine whether to use SNP loss for this epoch
        use_snp_loss = args.reconstruct_snps and (epoch > args.snp_start_epoch)
        
        # Create loader based on whether we need SNP data
        current_ds = ds_with_snps if use_snp_loss else ds_no_snps
        loader = DataLoader(
            current_ds, 
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
            if use_snp_loss:
                emb, spans, raw_snps = batch_data
            else:
                emb, spans, _ = batch_data  # raw_snps will be empty list
            
            # Move to GPU and ensure correct types
            emb = emb.float().cuda()
            spans = spans.float().cuda()
            
            # Forward pass
            recon_emb, dist = model(emb, spans)
            
            # Reconstruction loss (embedding space) with masking for padded dimensions
            recon_loss = masked_mse_loss(recon_emb, emb, embedding_mask)
            
            # KLD loss
            kld_loss = dist.kl().mean()
            
            # Start with basic VAE loss
            loss = recon_loss + kld_weight * kld_loss
            
            # Add SNP reconstruction loss if in SNP phase - EXACTLY like original train.py
            if use_snp_loss:
                # Follow original train.py EXACTLY:
                decoded_snps = [
                    pca_blocks[idx].decode(recon_emb[:, idx, :])
                    for idx in range(len(pca_blocks))
                ]
                
                decoded_mse = sum(
                    mse_loss(r, raw_snps[i].float().to(r.device))
                    for i, r in enumerate(decoded_snps)
                ) / len(decoded_snps)
                
                # Warm up SNP loss weight over first 50 epochs after snp_start_epoch
                if epoch <= args.snp_start_epoch + 50:
                    lambda_snp = lambda_mse * (epoch - args.snp_start_epoch) / 50
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
            if epoch <= args.snp_start_epoch + 50:
                lambda_snp = lambda_mse * (epoch - args.snp_start_epoch) / 50
            else:
                lambda_snp = lambda_mse
            logger.info(f"Epoch {epoch} ({phase}): Recon={avg_recon:.4f} | KLD={avg_kld:.4f} | SNP_MSE={avg_snp_mse:.4f} | λ_SNP={lambda_snp:.2e} | KLD_weight={kld_weight:.2e} | LR={args.lr:.2e}")
        else:
            logger.info(f"Epoch {epoch} ({phase}): Recon={avg_recon:.4f} | KLD={avg_kld:.4f} | KLD_weight={kld_weight:.2e} | LR={args.lr:.2e}")
            
        # Log transition to SNP phase
        if epoch == args.snp_start_epoch + 1:
            logger.info(f">>> TRANSITION: Starting SNP loss phase at epoch {epoch} <<<")
        
        # Comprehensive evaluation at specified frequency (or at end)
        eval_freq = getattr(args, 'eval_frequency', 50)  # Default to 10 if not specified
        if epoch % eval_freq == 0 or epoch == args.epochs:
            eval_metrics = evaluate_model(model, ds_eval, pca_blocks, args.batch_size)
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
    torch.save(model.state_dict(), args.model_save_path)
    logger.info(f"Model saved to {args.model_save_path}")

def evaluate_model(model, dataset, pca_blocks, batch_size):
    """Comprehensive evaluation of model on SNP reconstruction."""
    model.eval()
    total_mse = 0.0
    total_acc = 0.0
    total_samples = 0
    
    eval_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=1,
        pin_memory=True
    )
    
    with torch.no_grad():
        for emb, spans, raw_snps in tqdm(eval_loader, desc="Evaluating"):
            emb = emb.float().cuda()
            spans = spans.float().cuda()
            
            # Forward pass
            recon_emb, _ = model(emb, spans)
            
            # Evaluation following original script exactly
            decoded = [
                pca_blocks[idx].decode(recon_emb[:, idx, :])
                for idx in range(len(pca_blocks))
            ]
            # MSE
            batch_mse = sum(
                F.mse_loss(r, raw_snps[i].float().to(r.device))
                for i, r in enumerate(decoded)
            ) / len(decoded)
            # accuracy = fraction of exact matches
            batch_acc = sum(
                (r.round() == raw_snps[i].to(r.device)).float().mean()
                for i, r in enumerate(decoded)
            ) / len(decoded)
            
            bs = emb.size(0)
            total_mse += batch_mse.item() * bs
            total_acc += batch_acc.item() * bs
            total_samples += bs
    
    model.train()  # Return to training mode
    return {
        'mse': total_mse / total_samples,
        'accuracy': total_acc / total_samples
    }

def main():
    """Main function for VAE training."""
    parser = argparse.ArgumentParser(description="Train VAE on block embeddings")
    parser.add_argument("--spans-file", type=str, required=True)
    parser.add_argument("--recoded-dir", type=str, required=True)
    parser.add_argument("--embeddings-dir", type=str, required=True)
    parser.add_argument("--model-save-path", type=str, required=True)
    parser.add_argument("--checkpoint-path", type=str, default=None, help="Path to checkpoint for resuming training")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--grid-h", type=int, default=32)
    parser.add_argument("--grid-w", type=int, default=32)
    parser.add_argument("--block-dim", type=int, default=3)
    parser.add_argument("--pos-dim", type=int, default=16)
    parser.add_argument("--latent-channels", type=int, default=32)
    parser.add_argument("--kld-weight", type=float, default=1e-3)
    parser.add_argument("--reconstruct-snps", action="store_true")
    parser.add_argument("--snp-start-epoch", type=int, default=10)
    parser.add_argument("--decoded-mse-weight", type=float, default=1e-2)
    parser.add_argument("--eval-frequency", type=int, default=10, help="Evaluation and checkpoint frequency (epochs)")
    
    args = parser.parse_args()
    train(args)

if __name__ == "__main__":
    main()
