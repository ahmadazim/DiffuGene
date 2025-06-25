#!/usr/bin/env python

import argparse
import torch
import os
from diffusers import DDPMScheduler
from tqdm import tqdm

from ..utils import setup_logging, get_logger
from .unet import LatentUNET2D

logger = get_logger(__name__)

def generate(args):
    setup_logging()
    
    # Load model
    model = LatentUNET2D(input_channels=16, output_channels=16).cuda()
    checkpoint = torch.load(args.model_path, map_location='cuda')
    
    if 'ema' in checkpoint:
        logger.info("Loading EMA weights")
        try:
            from timm.utils import ModelEmaV3
            
            # Create temporary EMA wrapper to load the saved EMA state
            ema = ModelEmaV3(model, decay=0.9999)
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
    scheduler = DDPMScheduler(
        num_train_timesteps=args.num_time_steps,
        beta_schedule="linear",
        beta_start=1e-4,
        beta_end=0.02
    )
    scheduler.set_timesteps(args.num_inference_steps, device="cuda")
    
    # Load normalization stats
    model_dir = os.path.dirname(args.model_path)
    # Extract model name from model path (without extension)
    model_name = os.path.splitext(os.path.basename(args.model_path))[0]
    channel_means = torch.load(os.path.join(model_dir, f"train_{model_name}_channel_means.pt"), map_location='cuda')
    channel_stds = torch.load(os.path.join(model_dir, f"train_{model_name}_channel_stds.pt"), map_location='cuda')
    
    # Generate samples
    logger.info(f"Generating {args.num_samples} samples...")
    all_samples = []
    
    for i in tqdm(range(0, args.num_samples, args.batch_size)):
        batch_size = min(args.batch_size, args.num_samples - i)
        
        # Start from random noise
        latents = torch.randn(batch_size, 16, 16, 16, device="cuda")
        
        # Denoising loop
        with torch.no_grad():
            for t in tqdm(scheduler.timesteps, desc=f"Denoising batch {i//args.batch_size + 1}"):
                # Model prediction
                noise_pred = model(latents, t.expand(batch_size))
                
                # Scheduler step
                latents = scheduler.step(noise_pred, t, latents).prev_sample
        
        # Denormalize
        latents = latents * channel_stds[None, :, None, None] + channel_means[None, :, None, None]
        all_samples.append(latents.cpu())
    
    # Concatenate all samples
    all_samples = torch.cat(all_samples, dim=0)
    logger.info(f"Generated samples shape: {all_samples.shape}")
    
    # Save
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    torch.save(all_samples, args.output_path)
    logger.info(f"Samples saved to {args.output_path}")

def main():
    """Main function for diffusion sample generation."""
    parser = argparse.ArgumentParser(description="Generate samples using trained diffusion model")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-time-steps", type=int, default=1000)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    
    args = parser.parse_args()
    generate(args)

if __name__ == "__main__":
    main()
