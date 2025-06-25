#!/usr/bin/env python

import argparse
import torch
import os

from ..utils import setup_logging, get_logger
from .vae import SNPVAE
from .train import SNPBlocksDataset

logger = get_logger(__name__)

def inference(args):
    setup_logging()
    
    # Load dataset first to get actual embedding dimension
    dataset = SNPBlocksDataset(args.spans_file, args.recoded_dir, load_snps=False)
    
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
    
    model.load_state_dict(torch.load(args.model_path, map_location='cuda'))
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
