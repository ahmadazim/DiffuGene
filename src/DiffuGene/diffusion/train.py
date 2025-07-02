#!/usr/bin/env python

import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from timm.utils import ModelEmaV3
from tqdm import tqdm
from diffusers import DDPMScheduler
from typing import List

from ..utils import setup_logging, get_logger, prepare_covariates_for_training, save_covariate_metadata
from .unet import LatentUNET2D as ConditionalUNET
from .unet_unconditional import LatentUNET2D as UnconditionalUNET
from .unet import set_seed, noise_pred_loss

logger = get_logger(__name__)

def read_prepare_data(path, output_folder, model_output_path):
    """Load raw latent training data (normalize later)"""
    return torch.load(path, weights_only=False)

def prepare_conditional_data(latent_data, covariate_path, fam_path, 
                           binary_cols=None, categorical_cols=None,
                           output_folder=None, model_name=None):
    """Prepare covariates and create conditional dataset.
    
    Args:
        latent_data: Tensor of latent embeddings
        covariate_path: Path to covariate CSV file
        fam_path: Path to training fam file
        binary_cols: List of binary variable column names
        categorical_cols: List of categorical variable column names
        output_folder: Folder to save covariate metadata
        model_name: Model name for metadata filename
    
    Returns:
        TensorDataset with latents and covariates, covariate dimension
    """
    logger.info("Preparing conditional training data...")
    
    # Prepare covariates
    covariate_tensor, covariate_names, norm_params = prepare_covariates_for_training(
        covariate_path=covariate_path,
        fam_path=fam_path,
        binary_cols=binary_cols,
        categorical_cols=categorical_cols
    )
    
    # Verify sample count matches
    if len(latent_data) != len(covariate_tensor):
        raise ValueError(f"Latent data has {len(latent_data)} samples but "
                        f"covariate data has {len(covariate_tensor)} samples")
    
    # Save covariate metadata for generation
    if output_folder and model_name:
        metadata_path = os.path.join(output_folder, f"{model_name}_covariate_metadata.json")
        save_covariate_metadata(
            output_path=metadata_path,
            covariate_names=covariate_names,
            normalization_params=norm_params,
            fam_path=fam_path
        )
        logger.info(f"Saved covariate metadata to: {metadata_path}")
    
    # Create conditional dataset
    dataset = TensorDataset(latent_data, covariate_tensor)
    cond_dim = covariate_tensor.shape[1]
    
    logger.info(f"Conditional dataset prepared: {len(dataset)} samples, {cond_dim} covariates")
    logger.info(f"Covariate features: {covariate_names}")
    
    return dataset, cond_dim

def train(
    batch_size: int = 64,
    num_time_steps: int = 1000,
    num_epochs: int = 15,
    seed: int = -1,
    ema_decay: float = 0.9999,  
    lr = 2e-5,
    checkpoint_path: str = None,
    model_output_path: str = None,
    train_embed_dataset_path: str = None,
    # Conditional generation parameters
    conditional: bool = False,
    covariate_file: str = None,
    fam_file: str = None,
    cond_dim: int = 10,
    binary_cols: List[str] = None,
    categorical_cols: List[str] = None
):
    setup_logging()
    set_seed(random.randint(0, 2**32-1)) if seed == -1 else set_seed(seed)

    output_folder = os.path.dirname(model_output_path)
    model_name = os.path.splitext(os.path.basename(model_output_path))[0]
    
    # Load latent training data
    train_dataset_raw = read_prepare_data(train_embed_dataset_path, output_folder, model_output_path)
    
    # Prepare conditional or unconditional dataset
    if conditional:
        logger.info("Training conditional diffusion model")
        if not covariate_file or not fam_file:
            raise ValueError("Conditional training requires covariate_file and fam_file")
        
        train_dataset, actual_cond_dim = prepare_conditional_data(
            latent_data=train_dataset_raw,
            covariate_path=covariate_file,
            fam_path=fam_file,
            binary_cols=binary_cols,
            categorical_cols=categorical_cols,
            output_folder=output_folder,
            model_name=model_name
        )
        cond_dim = actual_cond_dim  # Use actual dimension from data
        
    else:
        logger.info("Training unconditional diffusion model")
        train_dataset = train_dataset_raw
    
    # Estimate scaling from the first batch
    if conditional:
        # For conditional, extract latents only for scaling estimation
        stats_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=1)
        first_batch_latents, _ = next(iter(stats_loader))
        first_batch = first_batch_latents.float().cuda()
    else:
        stats_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=1)
        first_batch = next(iter(stats_loader)).float().cuda()
    
    sigma_hat = first_batch.std(unbiased=False)
    torch.save(sigma_hat, os.path.join(output_folder, f"train_{model_name}_sigma.pt"))
    logger.info(f"Estimated global sigma_hat = {sigma_hat:.4f}")
    
    logger.info(f"Training data shape: {train_dataset_raw.shape}")
    logger.info(f"Training data std per channel: {train_dataset_raw.std(dim=(0, 2, 3))}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=1)

    # Switch to HF's scheduler for noising/denoising
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

    # Create conditional or unconditional model
    if conditional:
        model = ConditionalUNET(
            input_channels=64, 
            output_channels=64, 
            cond_dim=cond_dim,
            time_steps=num_time_steps
        ).cuda()
        logger.info(f"Created conditional UNet with {cond_dim} covariate dimensions")
    else:
        model = UnconditionalUNET(
            input_channels=64, 
            output_channels=64,
            time_steps=num_time_steps
        ).cuda()
        logger.info("Created unconditional UNet")
    
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
        for bidx, batch_data in enumerate(tqdm(train_loader, desc=f"Epoch {i+1}/{num_epochs}")):
            
            # Handle conditional vs unconditional data
            if conditional:
                x, covariates = batch_data
                x = x.float().cuda()
                covariates = covariates.float().cuda()
            else:
                x = batch_data.float().cuda()
                covariates = None
            
            # Normalize latents by sigma
            x = x / sigma_hat

            # Sample random timesteps and noise, then add noise via HF scheduler
            t = torch.randint(0, num_time_steps, (batch_size,), device=x.device, dtype=torch.long)
            noise = torch.randn_like(x)
            x_clean = x
            x = scheduler.add_noise(x_clean, noise, t)
            
            # Forward pass through model
            if conditional:
                output = model(x, t, covariates)
            else:
                output = model(x, t)
            
            optimizer.zero_grad()
            loss = noise_pred_loss(output, noise, t, scheduler, simplified_loss=True)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            ema.update(model)
            
            del x, x_clean, noise, output, loss
            if conditional:
                del covariates
            torch.cuda.empty_cache()
            
        dataset_size = len(train_loader.dataset)
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f'Epoch {i+1} | Loss {total_loss / (dataset_size/batch_size):.5f} | LR {current_lr:.2e}')
        lr_scheduler.step()

        # Save model every 50 epochs
        if (i+1) % 50 == 0:
            checkpoint = {
                'weights': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'ema': ema.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'conditional': conditional,
                'cond_dim': cond_dim if conditional else None
            }
            os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
            torch.save(checkpoint, model_output_path)
            logger.info(f"Model saved to {model_output_path}")

    # Save final checkpoint
    checkpoint = {
        'weights': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'ema': ema.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'conditional': conditional,
        'cond_dim': cond_dim if conditional else None
    }
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    torch.save(checkpoint, model_output_path)
    logger.info(f"Model saved to {model_output_path}")


def main():
    """Main function for diffusion training."""
    parser = argparse.ArgumentParser(description="Train DDPM")
    parser.add_argument("--model-output-path", type=str, required=True)
    parser.add_argument("--train-embed-dataset-path", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-time-steps", type=int, default=1000)
    parser.add_argument("--num-epochs", type=int, default=15)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--ema-decay", type=float, default=0.9999)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--checkpoint-path", type=str, default=None)
    
    # Conditional generation arguments
    parser.add_argument("--conditional", action='store_true', help="Enable conditional generation")
    parser.add_argument("--covariate-file", type=str, help="Path to covariate CSV file")
    parser.add_argument("--fam-file", type=str, help="Path to training fam file")
    parser.add_argument("--cond-dim", type=int, default=10, help="Covariate dimension (auto-detected if conditional)")
    parser.add_argument("--binary-cols", type=str, nargs='+', help="List of binary variable column names")
    parser.add_argument("--categorical-cols", type=str, nargs='+', help="List of categorical variable column names")
    
    args = parser.parse_args()

    train(
        batch_size=args.batch_size,
        num_time_steps=args.num_time_steps,
        num_epochs=args.num_epochs,
        seed=args.seed,
        ema_decay=args.ema_decay,
        lr=args.lr,
        checkpoint_path=args.checkpoint_path,
        model_output_path=args.model_output_path,
        train_embed_dataset_path=args.train_embed_dataset_path,
        conditional=args.conditional,
        covariate_file=args.covariate_file,
        fam_file=args.fam_file,
        cond_dim=args.cond_dim,
        binary_cols=args.binary_cols,
        categorical_cols=args.categorical_cols
    )

if __name__ == "__main__":
    main()
