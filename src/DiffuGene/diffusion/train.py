#!/usr/bin/env python

import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from timm.utils import ModelEmaV3
from tqdm import tqdm
from diffusers import DDPMScheduler

from ..utils import setup_logging, get_logger
from .unet import LatentUNET2D, set_seed, noise_pred_loss

logger = get_logger(__name__)

def read_prepare_data(path, output_folder, model_output_path):
    # """Load and normalize training data."""
    # data = torch.load(path, weights_only=False)
    # channel_means = data.mean(dim=(0, 2, 3))   # shape (16,)
    # channel_stds  = data.std (dim=(0, 2, 3))   # shape (16,)
    
    # # Extract model name from model_output_path (without extension)
    # model_name = os.path.splitext(os.path.basename(model_output_path))[0]
    
    # # store channel_means and channel_stds with model name
    # os.makedirs(output_folder, exist_ok=True)
    # torch.save(channel_means, os.path.join(output_folder, f"train_{model_name}_channel_means.pt"))
    # torch.save(channel_stds, os.path.join(output_folder, f"train_{model_name}_channel_stds.pt"))
    # data = (data - channel_means[None, :, None, None]) / channel_stds [None, :, None, None]
    # return data
    """Load raw latent training data (normalize later)"""
    return torch.load(path, weights_only=False)

def train(
    batch_size: int=64,
    num_time_steps: int=1000,
    num_epochs: int=15,
    seed: int=-1,
    ema_decay: float=0.9999,  
    lr=2e-5,
    checkpoint_path: str=None,
    model_output_path: str=None,
    train_embed_dataset_path: str=None
):
    setup_logging()
    set_seed(random.randint(0, 2**32-1)) if seed == -1 else set_seed(seed)

    output_folder = os.path.dirname(model_output_path)
    train_dataset = read_prepare_data(train_embed_dataset_path, output_folder, model_output_path)
    
    # estimate scaling from the first batch
    stats_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=1)
    first_batch = next(iter(stats_loader)).float().cuda()
    sigma_hat = first_batch.std(unbiased=False)
    model_name = os.path.splitext(os.path.basename(model_output_path))[0]
    torch.save(sigma_hat, os.path.join(output_folder, f"train_{model_name}_sigma.pt"))
    logger.info(f"Estimated global sigma_hat = {sigma_hat:.4f}")
    
    logger.info(f"Training data shape: {train_dataset.shape}")
    logger.info(f"Training data std per channel: {train_dataset.std(dim=(0, 2, 3))}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=1)

    # switch to HF's scheduler for noising/denoising
    scheduler = DDPMScheduler(
        num_train_timesteps=num_time_steps,
        beta_schedule="linear",
        beta_start=1e-4,
        beta_end=0.02, 
        clip_sample=False
    )
    scheduler.set_timesteps(num_time_steps, device="cuda")
    scheduler.alphas_cumprod = scheduler.alphas_cumprod.to("cuda")
    scheduler.betas         = scheduler.betas.to("cuda")

    model = LatentUNET2D(input_channels=16, output_channels=16).cuda()
    # Start LR at 1e-4 and schedule down to user-supplied LR
    start_lr = 1e-4
    optimizer = optim.Adam(model.parameters(), lr=start_lr)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr)
    ema = ModelEmaV3(model, decay=ema_decay)
    
    # Load initial checkpoint if specified (optional - for resuming training)
    if checkpoint_path is not None and checkpoint_path.strip() and os.path.exists(checkpoint_path):
        logger.info(f"Loading initial checkpoint from {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['weights'])
            ema.load_state_dict(checkpoint['ema'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            if 'lr_scheduler' in checkpoint:
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            logger.info("Checkpoint loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}. Starting from scratch.")
    else:
        if checkpoint_path is not None and checkpoint_path.strip():
            logger.warning(f"Checkpoint path specified but file not found: {checkpoint_path}")
        logger.info("Starting training from scratch")

    for i in range(num_epochs):
        total_loss = 0
        for bidx, x in enumerate(tqdm(train_loader, desc=f"Epoch {i+1}/{num_epochs}")):
            # x: (B,16,16,16)
            x = x.float().cuda()
            x = x / sigma_hat
            #  # --- DEBUG: verify input normalization ---
            # if bidx == 0 and i == 0:
            #     logger.info(f"[DEBUG] normalized input overall std: {x.std().item():.4f}")
            #     logger.info(f"[DEBUG] normalized input per-channel std: {x.std(dim=(0,2,3)).cpu().tolist()}")

            # sample random timesteps and noise, then add noise via HF scheduler
            t = torch.randint(0, num_time_steps, (batch_size,), device=x.device, dtype=torch.long)
            noise = torch.randn_like(x)
            x_clean = x
            x = scheduler.add_noise(x_clean, noise, t)
            
            # # --- DEBUG: log noising stats every 50 batches ---
            # if bidx % 50 == 0:
            #     logger.info(f"[DEBUG] batch={bidx} | mean t={t.float().mean():.2f} | "
            #           f"noise.std={noise.std().item():.4f} | x_noised.std={x.std().item():.4f}")
            #     for t_dbg in [0, num_time_steps//2, num_time_steps-1]:
            #         x_dbg = scheduler.add_noise(
            #             x_clean,
            #             torch.randn_like(x_clean),
            #             torch.tensor([t_dbg], device=x.device)
            #         )
            #         logger.info(f"[DEBUG] simulated noised t={t_dbg:4d} std={x_dbg.std().item():.4f}")

            output = model(x, t)
            optimizer.zero_grad()
            loss = noise_pred_loss(output, noise, t, scheduler, simplified_loss=True)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            ema.update(model)
            
            del x, x_clean, noise, output, loss
            torch.cuda.empty_cache()
            
        dataset_size = len(train_loader.dataset)
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f'Epoch {i+1} | Loss {total_loss / (dataset_size/batch_size):.5f} | LR {current_lr:.2e}')
        lr_scheduler.step()

        # save model every args.checkpoint_every epochs
        if (i+1) % 50 == 0:
            checkpoint = {
                'weights': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'ema': ema.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict()
            }
            os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
            torch.save(checkpoint, model_output_path)
            logger.info(f"Model saved to {model_output_path}")

    # Save checkpoint
    checkpoint = {
        'weights': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'ema': ema.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict()
    }
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    torch.save(checkpoint, model_output_path)
    logger.info(f"Model saved to {model_output_path}")


def main():
    """Main function for diffusion training."""
    parser = argparse.ArgumentParser(description="Train DDPM")
    parser.add_argument("--model-output-path", type=str,   required=True)
    parser.add_argument("--train-embed-dataset-path", type=str, required=True)
    parser.add_argument("--batch-size",        type=int,   default=64)
    parser.add_argument("--num-time-steps",    type=int,   default=1000)
    parser.add_argument("--num-epochs",        type=int,   default=15)
    parser.add_argument("--seed",              type=int,   default=-1)
    parser.add_argument("--ema-decay",         type=float, default=0.9999)
    parser.add_argument("--lr",                type=float, default=2e-5)
    parser.add_argument("--checkpoint-path",   type=str,   default=None)
    args = parser.parse_args()

    train(
        batch_size        = args.batch_size,
        num_time_steps    = args.num_time_steps,
        num_epochs        = args.num_epochs,
        seed              = args.seed,
        ema_decay         = args.ema_decay,
        lr                = args.lr,
        checkpoint_path   = args.checkpoint_path,
        model_output_path = args.model_output_path,
        train_embed_dataset_path = args.train_embed_dataset_path
    )

if __name__ == "__main__":
    main()
