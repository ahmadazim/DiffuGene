#!/usr/bin/env python

import argparse
import os
import random
import gc
from tqdm import tqdm
from typing import List
import glob
import re
import numpy as np
from timm.utils import ModelEmaV3
from diffusers import DDPMScheduler

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.cuda.amp as amp
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from ..utils import setup_logging, get_logger, prepare_covariates_for_training, save_covariate_metadata
from .unet import LatentUNET2D as ConditionalUNET
from .unet_unconditional import LatentUNET2D as UnconditionalUNET
from .unet import set_seed, noise_pred_loss, v_pred_loss

logger = get_logger(__name__)

class MemmapDataset(Dataset):
    """Memory-mapped dataset for efficient loading of large latent arrays."""
    def __init__(self, memmap_path, shape=None):
        # Load the shape info file if shape not provided
        if shape is None:
            shape_file = memmap_path.replace('.npy', '_shape.txt')
            if os.path.exists(shape_file):
                with open(shape_file, 'r') as f:
                    shape = tuple(map(int, f.read().strip().split(',')))
            else:
                raise ValueError(f"Shape file not found: {shape_file}. Cannot determine memmap shape.")
        
        self.arr = np.memmap(memmap_path, dtype='float32', mode='r', shape=shape)
        logger.info(f"Loaded memmap dataset: {self.arr.shape} from {memmap_path}")
    
    def __len__(self):
        return len(self.arr)
    
    def __getitem__(self, i):
        x = self.arr[i]  # fast slice
        return torch.from_numpy(x)

class ConditionalMemmapDataset(Dataset):
    """Memory-mapped dataset for conditional training with latents and covariates."""
    def __init__(self, memmap_path, covariate_tensor, shape=None):
        # Load the shape info file if shape not provided
        if shape is None:
            shape_file = memmap_path.replace('.npy', '_shape.txt')
            if os.path.exists(shape_file):
                with open(shape_file, 'r') as f:
                    shape = tuple(map(int, f.read().strip().split(',')))
            else:
                raise ValueError(f"Shape file not found: {shape_file}. Cannot determine memmap shape.")
        
        self.arr = np.memmap(memmap_path, dtype='float32', mode='r', shape=shape)
        self.covariates = covariate_tensor
        
        if len(self.arr) != len(self.covariates):
            raise ValueError(f"Latent data has {len(self.arr)} samples but "
                           f"covariate data has {len(self.covariates)} samples")
        
        logger.info(f"Loaded conditional memmap dataset: {self.arr.shape} latents + {self.covariates.shape} covariates")
    
    def __len__(self):
        return len(self.arr)
    
    def __getitem__(self, i):
        x = self.arr[i]  # fast slice
        c = self.covariates[i]  # covariate for this sample
        return torch.from_numpy(x), c

def create_memmap_from_batches(batch_files, memmap_path):
    """Create a memory-mapped numpy array from batch files."""
    # Check if memmap file already exists
    shape_file = memmap_path.replace('.npy', '_shape.txt')
    if os.path.exists(memmap_path) and os.path.exists(shape_file):
        logger.info(f"Memmap file already exists: {memmap_path}")
        # Load existing shape and return sample count
        with open(shape_file, 'r') as f:
            shape = tuple(map(int, f.read().strip().split(',')))
        logger.info(f"Existing memmap shape: {shape}")
        return shape[0], shape  # Return (total_samples, full_shape)
    
    logger.info(f"Creating memmap file from {len(batch_files)} batch files...")
    
    # First pass: determine total size and shape
    total_samples = 0
    sample_shape = None
    
    for batch_file in batch_files:
        batch_data = torch.load(batch_file, weights_only=False)
        if not isinstance(batch_data, torch.Tensor):
            batch_data = torch.tensor(batch_data, dtype=torch.float32)
        
        if sample_shape is None:
            sample_shape = batch_data.shape[1:]  # all dims except batch
        
        total_samples += batch_data.shape[0]
        logger.info(f"Batch {batch_file}: {batch_data.shape}")
    
    # Create memmap array
    full_shape = (total_samples,) + sample_shape
    logger.info(f"Creating memmap array with shape: {full_shape}")
    
    memmap_array = np.memmap(memmap_path, dtype='float32', mode='w+', shape=full_shape)
    
    # Second pass: copy data to memmap
    offset = 0
    for batch_file in tqdm(batch_files, desc="Writing to memmap"):
        batch_data = torch.load(batch_file, weights_only=False)
        if not isinstance(batch_data, torch.Tensor):
            batch_data = torch.tensor(batch_data, dtype=torch.float32)
        
        batch_np = batch_data.numpy()
        batch_size = batch_np.shape[0]
        memmap_array[offset:offset + batch_size] = batch_np
        offset += batch_size
    
    # Flush to disk
    del memmap_array
    
    # Save shape information
    with open(shape_file, 'w') as f:
        f.write(','.join(map(str, full_shape)))
    
    logger.info(f"Memmap file created successfully: {memmap_path}")
    logger.info(f"Shape file created: {shape_file}")
    return total_samples, full_shape

def read_prepare_data(path, output_folder, model_output_path):
    # Case 1: direct memmap path
    if os.path.exists(path) and path.endswith('.npy'):
        shape_file = path.replace('.npy', '_shape.txt')
        if not os.path.exists(shape_file):
            raise ValueError(f"Shape file missing for memmap: {shape_file}")
        logger.info(f"Using existing memmap file: {path}")
        return MemmapDataset(path)

    # Case 2: single tensor file to be converted to memmap
    if os.path.exists(path):
        logger.info(f"Loading single latent file: {path}")
        data = torch.load(path, weights_only=False)
        memmap_path = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(model_output_path))[0]}_memmap.npy")
        shape_file = memmap_path.replace('.npy', '_shape.txt')
        if not os.path.exists(memmap_path) or not os.path.exists(shape_file):
            if not isinstance(data, torch.Tensor):
                data = torch.tensor(data, dtype=torch.float32)
            logger.info(f"Creating memmap from single file: {data.shape}")
            memmap_array = np.memmap(memmap_path, dtype='float32', mode='w+', shape=data.shape)
            memmap_array[:] = data.numpy()
            del memmap_array
            with open(shape_file, 'w') as f:
                f.write(','.join(map(str, data.shape)))
            logger.info(f"Memmap file created: {memmap_path}")
            logger.info(f"Shape file created: {shape_file}")
            return MemmapDataset(memmap_path, data.shape)
        else:
            logger.info(f"Using existing memmap file: {memmap_path}")
            return MemmapDataset(memmap_path)
    
    # Check for batch files
    base_dir = os.path.dirname(path)
    base_name = os.path.splitext(os.path.basename(path))[0]
    
    batch_pattern = os.path.join(base_dir, f"{base_name}_batch*.pt")
    batch_files = glob.glob(batch_pattern)
    
    if not batch_files:
        raise FileNotFoundError(f"No training data found at {path} or batch files matching {batch_pattern}")
    
    def extract_batch_num(path):
        match = re.search(r'_batch(\d+)\.pt$', path)
        return int(match.group(1)) if match else 0
    
    batch_files.sort(key=extract_batch_num)
    logger.info(f"Found {len(batch_files)} batch files to process")
    
    # Create memmap file path
    memmap_path = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(model_output_path))[0]}_memmap.npy")
    
    # Create memmap file if it doesn't exist
    shape_file = memmap_path.replace('.npy', '_shape.txt')
    if not os.path.exists(memmap_path) or not os.path.exists(shape_file):
        logger.info(f"Creating memmap file: {memmap_path}")
        total_samples, full_shape = create_memmap_from_batches(batch_files, memmap_path)
        logger.info(f"Memmap file created with {total_samples} total samples")
        return MemmapDataset(memmap_path, full_shape)
    else:
        logger.info(f"Using existing memmap file: {memmap_path}")
        return MemmapDataset(memmap_path)

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

def prepare_conditional_memmap_data(memmap_dataset, covariate_path, fam_path, 
                                  binary_cols=None, categorical_cols=None,
                                  output_folder=None, model_name=None):
    """Prepare covariates and create conditional memmap dataset.
    
    Args:
        memmap_dataset: MemmapDataset of latent embeddings
        covariate_path: Path to covariate CSV file
        fam_path: Path to training fam file
        binary_cols: List of binary variable column names
        categorical_cols: List of categorical variable column names
        output_folder: Folder to save covariate metadata
        model_name: Model name for metadata filename
    
    Returns:
        ConditionalMemmapDataset with latents and covariates, covariate dimension
    """
    logger.info("Preparing conditional training data for memmap dataset...")
    
    # Prepare covariates
    covariate_tensor, covariate_names, norm_params = prepare_covariates_for_training(
        covariate_path=covariate_path,
        fam_path=fam_path,
        binary_cols=binary_cols,
        categorical_cols=categorical_cols
    )
    
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
    
    # Create conditional memmap dataset (the dataset will handle size verification)
    # Get the memmap file path and shape from the existing memmap dataset
    memmap_path = memmap_dataset.arr.filename
    memmap_shape = memmap_dataset.arr.shape
    dataset = ConditionalMemmapDataset(memmap_path, covariate_tensor, memmap_shape)
    cond_dim = covariate_tensor.shape[1]
    
    logger.info(f"Conditional memmap dataset prepared: {len(dataset)} samples, {cond_dim} covariates")
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
    categorical_cols: List[str] = None, 
    cfg_drop_prob: float = 0.1, 
):
    setup_logging()
    set_seed(random.randint(0, 2**32-1)) if seed == -1 else set_seed(seed)

    output_folder = os.path.dirname(model_output_path)
    model_name = os.path.splitext(os.path.basename(model_output_path))[0]
    
    # Initialize distributed training if launched with torchrun/torch.distributed
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    distributed = world_size > 1
    if distributed:
        logger.info(f"Distributed training with {world_size} GPUs")
        logger.info(f"Using local rank: {local_rank}")
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank if distributed else 0)
    logger.info(f"Using device: {device}")

    # Memory/performance toggles: use channels_last, bf16, and enable Flash/SDP attention
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)

    # Load latent training data
    train_dataset_raw = read_prepare_data(train_embed_dataset_path, output_folder, model_output_path)
    
    # Always memory-mapped dataset path
    if conditional:
        logger.info("Training conditional diffusion model with memory-mapped data")
        if not covariate_file or not fam_file:
            raise ValueError("Conditional training requires covariate_file and fam_file")
        train_dataset, actual_cond_dim = prepare_conditional_memmap_data(
            memmap_dataset=train_dataset_raw,
            covariate_path=covariate_file,
            fam_path=fam_file,
            binary_cols=binary_cols,
            categorical_cols=categorical_cols,
            output_folder=output_folder,
            model_name=model_name
        )
        cond_dim = actual_cond_dim
    else:
        logger.info("Training unconditional diffusion model with memory-mapped data")
        train_dataset = train_dataset_raw
    
    # Estimate scaling from the first batch(es) using memmap loader
    stats_loader = DataLoader(train_dataset, 
                              batch_size=batch_size, 
                              shuffle=False, 
                              drop_last=True, 
                              num_workers=0, 
                              pin_memory=False)
    
    sigma_estimates = []
    for i, batch_data in enumerate(stats_loader):
        if conditional:
            batch_latents, _ = batch_data
            batch = batch_latents.to(dtype=torch.float32, device="cpu")
        else:
            batch = batch_data.to(dtype=torch.float32, device="cpu")
        sigma_estimates.append(batch.std(unbiased=False).item())
        if conditional:
            del batch_latents
        if i >= 4:
            break
    sigma_hat = torch.tensor(sum(sigma_estimates) / len(sigma_estimates), device=device)
    if distributed:
        dist.all_reduce(sigma_hat, op=dist.ReduceOp.SUM)
        sigma_hat /= world_size
    logger.info(f"Estimated sigma from {len(sigma_estimates)} small batches: {sigma_hat:.4f}")
    total_samples = len(train_dataset)
    if conditional:
        sample_latent_shape = train_dataset[0][0].shape
        logger.info(f"Training data shape: ({total_samples},) + {sample_latent_shape} latents + covariates")
    else:
        sample_shape = train_dataset[0].shape
        logger.info(f"Training data shape: ({total_samples},) + {sample_shape}")
    del stats_loader, batch, batch_data
    gc.collect()
    torch.cuda.empty_cache()

    torch.save(sigma_hat, os.path.join(output_folder, f"train_{model_name}_sigma.pt"))
    logger.info(f"Estimated global sigma_hat = {sigma_hat:.4f}")

    # Create data loader
    if distributed: 
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=dist.get_rank(), shuffle=True)
        loader_args = dict(
            batch_size=batch_size,
            sampler=train_sampler,
            drop_last=True,
            num_workers=0,
            pin_memory=False,
            # prefetch_factor=1,
            persistent_workers=False
        )
    else: 
        train_sampler = None
        loader_args = dict(
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=0,
            pin_memory=False,
            # prefetch_factor=1, 
            persistent_workers=False
        )
    train_loader = DataLoader(train_dataset, **loader_args)
    logger.info(f"Created memmap DataLoader with {0} workers")

    # Switch to HF's scheduler for noising/denoising
    scheduler = DDPMScheduler(
        num_train_timesteps=num_time_steps,
        beta_schedule="linear",
        beta_start=1e-4,
        beta_end=0.02, 
        clip_sample=False
    )
    scheduler.set_timesteps(num_time_steps, device=device)
    scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(device)
    scheduler.betas         = scheduler.betas.to(device)

    # Create conditional or unconditional model
    if conditional:
        model = ConditionalUNET(
            input_channels=4, 
            output_channels=4, 
            cond_dim=cond_dim,
            time_steps=num_time_steps
        )
        logger.info(f"Created conditional UNet with {cond_dim} covariate dimensions")
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
        logger.info(f"Model size: {sum(p.numel() for p in model.parameters()) * 4 / 1024**2:.2f} MB")
    else:
        model = UnconditionalUNET(
            input_channels=4, 
            output_channels=4,
            time_steps=num_time_steps
        )
        logger.info("Created unconditional UNet")
    
    # Enable gradient checkpointing
    model.unet.enable_gradient_checkpointing()
    logger.info("Enabled gradient checkpointing for UNet")

    # Convert to channels_last and bf16 for memory efficiency
    model.to(device, non_blocking=True)
    model = model.to(memory_format=torch.channels_last, dtype=torch.bfloat16)

    # Wrap in DDP 
    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False, find_unused_parameters=True)
        logger.info("Wrapped model in DDP")
        ddp_wrapped = True
    else:
        ddp_wrapped = False
    
    # Start LR at 1e-4 and schedule down to user-supplied LR
    start_lr = 1e-4
    optimizer = optim.Adam(model.parameters(), lr=start_lr)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr)
    
    # Create EMA model on CPU to save GPU memory
    ema_src = model.module if ddp_wrapped else model
    ema = ModelEmaV3(ema_src, decay=ema_decay, device='cpu')
    logger.info("Created EMA model on CPU to save GPU memory")
    
    # Create mixed precision scaler (disabled for bfloat16)
    scaler = amp.GradScaler(enabled=False)
    logger.info("Mixed precision enabled with bfloat16 autocast; GradScaler disabled for bf16")
    
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
            if 'scaler' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler'])
            logger.info("Checkpoint loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}. Starting from scratch.")
    else:
        if checkpoint_path is not None and checkpoint_path.strip():
            logger.warning(f"Checkpoint path specified but file not found: {checkpoint_path}")
        logger.info("Starting training from scratch")

    # classifier‐free guidance drop rate
    p_uncond = cfg_drop_prob if conditional else 0.0
    
    sigma_dq = 0.01  # small dequantization noise std (to stabilize training, particularly for spiky latent dimensions)
    
    for i in range(num_epochs):
        if distributed:
            train_loader.sampler.set_epoch(i)
        total_loss = 0
        for bidx, batch_data in enumerate(tqdm(train_loader, desc=f"Epoch {i+1}/{num_epochs}")):
            
            # Handle conditional vs unconditional data
            if conditional:
                x, covariates = batch_data
                x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last, dtype=torch.bfloat16)
                covariates = covariates.float().to(device, non_blocking=True).to(dtype=torch.bfloat16)
            else:
                x = batch_data.float().to(device, non_blocking=True).to(memory_format=torch.channels_last, dtype=torch.bfloat16)
                covariates = None
            
            # Normalize latents by sigma
            x = x / sigma_hat

            # Sample random timesteps and noise, then add noise via HF scheduler
            t = torch.randint(0, num_time_steps, (batch_size,), device=x.device, dtype=torch.long)
            noise = torch.randn_like(x)
            
            # x_clean = x
            x_clean = x + sigma_dq * torch.randn_like(x)
            
            x = scheduler.add_noise(x_clean, noise, t)
            
            optimizer.zero_grad()
            
            # Forward pass with mixed precision
            with amp.autocast(dtype=torch.bfloat16):
                # Access custom attributes whether wrapped by DDP or not
                net = model.module if isinstance(model, DDP) else model
                if conditional:
                    # CFG Training: drop + double-batch approach
                    B = x.size(0)
                    
                    # 1) Embed covariates
                    e_cond = net.cond_emb(covariates).unsqueeze(1)        # (B,1,128)
                    hidden_dim = e_cond.size(-1)
                    e_null = net.null_cond_emb.unsqueeze(0).expand(B,1,hidden_dim)  # (B,1,hidden_dim)
                    # Ensure dtype match for masked assignment under AMP
                    if e_null.dtype != e_cond.dtype:
                        e_null = e_null.to(dtype=e_cond.dtype)
                    
                    # 2) Randomly drop some examples
                    mask = (torch.rand(B, device=x.device) < p_uncond)   # (B,)
                    e_drop = e_cond.clone()
                    e_drop[mask] = e_null[mask]               # replace those with null
                    
                    # 3) Build double-batch
                    x_in = torch.cat([x, x], dim=0)
                    t_in = torch.cat([t, t], dim=0)
                    emb_in = torch.cat([e_drop, e_cond], dim=0)
                    
                    # 4) Forward through UNet directly
                    h = net.input_proj(x_in)  
                    out = net.unet(h, t_in, encoder_hidden_states=emb_in).sample
                    out = net.output_proj(out)              # (2B,4,512,512)
                    
                    # 5) Split & compute loss
                    out_uncond, out_cond = out.chunk(2, dim=0)
                    
                    # loss = noise_pred_loss(out_uncond, noise, t, scheduler, simplified_loss=True)
                    # loss += noise_pred_loss(out_cond, noise, t, scheduler, simplified_loss=True)
                    loss = v_pred_loss(out_uncond, x_clean, noise, t, scheduler, gamma=10.0)
                    loss += v_pred_loss(out_cond, x_clean, noise, t, scheduler, gamma=10.0)
                else:
                    output = model(x, t)
                    # loss = noise_pred_loss(output, noise, t, scheduler, simplified_loss=True)
                    loss = v_pred_loss(output, x_clean, noise, t, scheduler, gamma=10.0)
            
            total_loss += loss.item()
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            ema.update(ema_src)
            
            # Clean up
            if conditional:
                del x, x_clean, noise, out, loss, t, covariates
            else:
                del x, x_clean, noise, output, loss, t
            
        dataset_size = len(train_loader.dataset)
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f'Epoch {i+1} | Loss {total_loss / (dataset_size/batch_size):.5f} | LR {current_lr:.2e}')
        lr_scheduler.step()

        # Save model every epoch, only on the first rank
        if (not distributed) or (distributed and dist.get_rank() == 0):
            if (i+1) % 1 == 0:
                checkpoint = {
                    'weights': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'ema': ema.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'scaler': scaler.state_dict(),
                    'conditional': conditional,
                    'cond_dim': cond_dim if conditional else None
                }
                os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
                model_output_path_epoch = model_output_path.replace(".pth", f"_epoch{i+1}.pth")
                torch.save(checkpoint, model_output_path_epoch)
                logger.info(f"Model saved to {model_output_path_epoch}")

    # Save final checkpoint
    if (not distributed) or (distributed and dist.get_rank() == 0):
        checkpoint = {
            'weights': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'ema': ema.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'scaler': scaler.state_dict(),
            'conditional': conditional,
            'cond_dim': cond_dim if conditional else None
        }
        os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
        torch.save(checkpoint, model_output_path)
        logger.info(f"Model saved to {model_output_path}")

    if distributed:
        dist.destroy_process_group()


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
    parser.add_argument("--cfg-drop-prob", type=float, default=0.1, help="Drop probability for classifier-free guidance (unconditional branch)")

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
        categorical_cols=args.categorical_cols, 
        cfg_drop_prob=args.cfg_drop_prob
    )

if __name__ == "__main__":
    main()
