#!/usr/bin/env python

import argparse
import torch
import os

from ..utils import setup_logging, get_logger
from .vae import SNPVAE
from .train import SNPBlocksDataset, unscale_embeddings

logger = get_logger(__name__)

def inference(args):
    setup_logging()
    
    # Handle both old and new model save formats to determine if PC scaling was used
    checkpoint = torch.load(args.model_path, map_location='cpu')
    if isinstance(checkpoint, dict) and 'config' in checkpoint:
        scale_pc_embeddings = checkpoint['config'].get('scale_pc_embeddings', False)
    else:
        scale_pc_embeddings = False
    
    # Load dataset first to get actual embedding dimension
    dataset = SNPBlocksDataset(args.spans_file, args.recoded_dir, load_snps=False, scale_pc_embeddings=scale_pc_embeddings)
    
    # Get the actual embedding dimension from the dataset
    actual_block_dim = dataset.block_embs.shape[-1]
    logger.info(f"Using actual block embedding dimension: {actual_block_dim}")
    
    # Override args.block_dim if it doesn't match the actual embeddings
    if args.block_dim != actual_block_dim:
        logger.warning(f"Config block_dim ({args.block_dim}) != actual embeddings ({actual_block_dim}). Using actual dimension.")
        args.block_dim = actual_block_dim
    
    # Load model
    model = SNPVAE(
        grid_size=(args.grid_h, args.grid_w),
        block_emb_dim=args.block_dim,
        pos_emb_dim=args.pos_dim,
        latent_channels=args.latent_channels
    ).cuda()
    
    # Handle both old and new model save formats
    checkpoint = torch.load(args.model_path, map_location='cuda')
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # New format with metadata
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info("Loaded model from new format with metadata")
        
        # Load PC scaling factors if they exist
        pc_means = None
        pc_scales = None
        if 'pc_scaling' in checkpoint and scale_pc_embeddings:
            pc_means = checkpoint['pc_scaling']['pc_means'].cuda()
            pc_scales = checkpoint['pc_scaling']['pc_scales'].cuda()
            logger.info("Loaded PC scaling factors from saved model")
        elif scale_pc_embeddings:
            # Use scaling factors from dataset if not in model (for compatibility)
            pc_means = dataset.pc_means.cuda() if dataset.pc_means is not None else None
            pc_scales = dataset.pc_scales.cuda() if dataset.pc_scales is not None else None
            logger.info("Using PC scaling factors from dataset")
    else:
        # Old format - direct state dict
        model.load_state_dict(checkpoint)
        logger.info("Loaded model from old format")
        pc_means = None
        pc_scales = None
    model.eval()
    
    # Encode all data
    all_latents = []
    with torch.no_grad():
        for i in range(len(dataset)):
            emb, spans, _ = dataset[i]
            emb = emb.unsqueeze(0).cuda()  # Add batch dimension
            spans = spans.unsqueeze(0).cuda()
            
            z, _ = model.encode(emb, spans)
            all_latents.append(z.cpu())
    
    # Stack and save
    all_latents = torch.cat(all_latents, dim=0)
    logger.info(f"Generated latents shape: {all_latents.shape}")
    
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    torch.save(all_latents, args.output_path)
    logger.info(f"Latents saved to {args.output_path}")

def main():
    """Main function for VAE inference."""
    parser = argparse.ArgumentParser(description="VAE inference to generate latents")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--spans-file", type=str, required=True)
    parser.add_argument("--recoded-dir", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--grid-h", type=int, default=32)
    parser.add_argument("--grid-w", type=int, default=32)
    parser.add_argument("--block-dim", type=int, default=3)
    parser.add_argument("--pos-dim", type=int, default=16)
    parser.add_argument("--latent-channels", type=int, default=32)
    
    args = parser.parse_args()
    inference(args)

if __name__ == "__main__":
    main()
