#!/usr/bin/env python

import argparse
import torch
import os
import pandas as pd
from tqdm import tqdm

from ..utils import setup_logging, get_logger
from .vae import SNPVAE
from .memory_efficient_dataset import BlockPCDataset
from .precompute_embeddings import precompute_embeddings

logger = get_logger(__name__)

def inference(args):
    setup_logging()
    
    # Handle both old and new model save formats to determine if PC scaling was used
    checkpoint = torch.load(args.model_path, map_location='cpu')
    if isinstance(checkpoint, dict) and 'config' in checkpoint:
        scale_pc_embeddings = checkpoint['config'].get('scale_pc_embeddings', False)
        # Get n_blocks from saved config if available
        saved_n_blocks = checkpoint['config'].get('n_blocks', None)
    else:
        scale_pc_embeddings = False
        saved_n_blocks = None
    
    # Generate HDF5 path automatically based on spans file (same as training)
    spans_basename = os.path.splitext(os.path.basename(args.spans_file))[0]
    h5_dir = os.path.join(os.path.dirname(args.spans_file), "h5_cache")
    os.makedirs(h5_dir, exist_ok=True)
    emb_h5_path = os.path.join(h5_dir, f"{spans_basename}_embeddings.h5")
    
    # Create HDF5 file if it doesn't exist
    if not os.path.exists(emb_h5_path):
        logger.info(f"Creating HDF5 embeddings file for inference: {emb_h5_path}")
        # Need to determine n_train from one of the embedding files
        df_temp = pd.read_csv(args.spans_file)
        first_emb = torch.load(df_temp.iloc[0].block_file, weights_only=False)
        if not isinstance(first_emb, torch.Tensor):
            first_emb = torch.tensor(first_emb, dtype=torch.float32)
        n_train = first_emb.shape[0]
        
        precompute_embeddings(args.spans_file, emb_h5_path, n_train, args.block_dim)
    
    # Load dataset using memory-efficient approach
    dataset = BlockPCDataset(
        emb_h5_path=emb_h5_path,
        spans_file=args.spans_file,
        recoded_dir=args.recoded_dir,
        embeddings_dir=args.embeddings_dir,
        scale_pc_embeddings=scale_pc_embeddings
    )
    
    # Get the actual embedding dimension from the dataset
    actual_block_dim = dataset.pc_dim
    logger.info(f"Using actual block embedding dimension: {actual_block_dim}")
    
    # Override args.block_dim if it doesn't match the actual embeddings
    if args.block_dim != actual_block_dim:
        logger.warning(f"Config block_dim ({args.block_dim}) != actual embeddings ({actual_block_dim}). Using actual dimension.")
        args.block_dim = actual_block_dim
    
    # Get number of blocks from spans file
    df_spans = pd.read_csv(args.spans_file)
    n_blocks = len(df_spans)
    logger.info(f"Number of blocks from spans file: {n_blocks}")
    
    # Validate against saved config if available
    if saved_n_blocks is not None and saved_n_blocks != n_blocks:
        logger.warning(f"Saved model n_blocks ({saved_n_blocks}) != current spans file n_blocks ({n_blocks}). Using current spans file.")
    elif saved_n_blocks is not None:
        logger.info(f"n_blocks matches saved model config: {n_blocks}")
    
    # Load model
    model = SNPVAE(
        n_blocks=n_blocks,
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
    
    # Encode all data (in batches if needed)
    batch_size = 50500
    batch_count = 0
    all_latents = []
    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc="Encoding data"):
            emb, spans, sample_idx = dataset[i]
            emb = emb.unsqueeze(0).cuda()  # Add batch dimension
            spans = spans.unsqueeze(0).cuda()
            
            z, _ = model.encode(emb, spans)
            all_latents.append(z.cpu())
            
            if len(all_latents) >= batch_size:
                # save and clear
                batch_tensor = torch.cat(all_latents, dim=0)
                batch_output = args.output_path.replace('.pt', f'_batch{batch_count}.pt')
                os.makedirs(os.path.dirname(batch_output), exist_ok=True)
                torch.save(batch_tensor, batch_output)
                logger.info(f"Saved batch {batch_count} with {batch_tensor.shape[0]} latents to {batch_output}")
                batch_count += 1
                all_latents = []
                torch.cuda.empty_cache()
    
    # Save any remaining latents
    if len(all_latents) > 0:
        batch_tensor = torch.cat(all_latents, dim=0)
        batch_output = args.output_path.replace('.pt', f'_batch{batch_count}.pt')
        os.makedirs(os.path.dirname(batch_output), exist_ok=True)
        torch.save(batch_tensor, batch_output)
        logger.info(f"Saved final batch {batch_count} with {batch_tensor.shape[0]} latents to {batch_output}")

def main():
    """Main function for VAE inference."""
    parser = argparse.ArgumentParser(description="VAE inference to generate latents using memory-efficient approach")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--spans-file", type=str, required=True)
    parser.add_argument("--recoded-dir", type=str, required=True)
    parser.add_argument("--embeddings-dir", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--grid-h", type=int, default=64)
    parser.add_argument("--grid-w", type=int, default=64)
    parser.add_argument("--block-dim", type=int, default=4)
    parser.add_argument("--pos-dim", type=int, default=16)
    parser.add_argument("--latent-channels", type=int, default=128)
    
    args = parser.parse_args()
    inference(args)

if __name__ == "__main__":
    main()

# wd=/n/home03/ahmadazim/WORKING/genGen/UKB/
# python -m DiffuGene.joint_embed.infer --model-path ${wd}models/joint_embed/VAE_unrelWhite_allchr_4PC_64z_checkpoint_400.pt --spans-file ${wd}genomic_data/work_encode/encoding_ukb_allchr_unrel_britishWhite_conditional_diffusion_train/ukb_allchr_unrel_britishWhite_conditional_diffusion_train_blocks_4PC_inference.csv --recoded-dir ${wd}genomic_data/work_encode/encoding_ukb_allchr_unrel_britishWhite_conditional_diffusion_train/recoded_blocks/ --embeddings-dir ${wd}genomic_data/work_encode/encoding_ukb_allchr_unrel_britishWhite_conditional_diffusion_train/embeddings/ --output-path ${wd}genomic_data/VAE_embeddings/ukb_allchr_unrel_britishWhite_conditional_diffusion_train_VAE_latents_4PC_64z_checkpoint_400.pt --grid-h 64 --grid-w 64 --block-dim 4 --pos-dim 16 --latent-channels 128