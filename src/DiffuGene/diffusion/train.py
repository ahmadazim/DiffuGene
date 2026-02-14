#!/usr/bin/env python

import argparse
import os
import random
import gc
from tqdm import tqdm
from typing import List, Tuple, Optional
import glob
import re
import numpy as np
from timm.utils import ModelEmaV3
from diffusers import DDPMScheduler
import math

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
from .unet import set_seed, v_pred_loss

logger = get_logger(__name__)

class MemmapDataset(Dataset):
    """Memory-mapped dataset for efficient loading of large latent arrays."""
    def __init__(self, memmap_path, shape=None):
        self.memmap_path = memmap_path
        # Load the shape info file if shape not provided
        if shape is None:
            shape_file = memmap_path.replace('.npy', '_shape.txt')
            if os.path.exists(shape_file):
                with open(shape_file, 'r') as f:
                    shape = tuple(map(int, f.read().strip().split(',')))
            else:
                raise ValueError(f"Shape file not found: {shape_file}. Cannot determine memmap shape.")
        self.shape = shape
        self.arr = np.memmap(memmap_path, dtype='float32', mode='r', shape=self.shape)
        logger.info(f"Loaded memmap dataset: {self.arr.shape} from {memmap_path}")
    
    def __len__(self):
        return len(self.arr)
    
    def __getitem__(self, i):
        x = self.arr[i]  # fast slice
        return torch.from_numpy(x)

class ConditionalMemmapDataset(Dataset):
    """Memory-mapped dataset for conditional training with latents and covariates."""
    def __init__(self, memmap_path, covariate_tensor, shape=None):
        self.memmap_path = memmap_path
        # Load the shape info file if shape not provided
        if shape is None:
            shape_file = memmap_path.replace('.npy', '_shape.txt')
            if os.path.exists(shape_file):
                with open(shape_file, 'r') as f:
                    shape = tuple(map(int, f.read().strip().split(',')))
            else:
                raise ValueError(f"Shape file not found: {shape_file}. Cannot determine memmap shape.")
        
        self.shape = shape
        self.arr = np.memmap(memmap_path, dtype='float32', mode='r', shape=self.shape)
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


def compute_channel_stats_from_memmap(
    memmap_array: np.memmap,
    chunk_size: int = 512,
    rank: int = 0,
    world_size: int = 1,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute per-channel mean/std over an entire (N,C,H,W) memmap without loading it fully into RAM.
    When running under DDP, each rank processes a disjoint slice of samples and the partial
    statistics are aggregated via all_reduce to avoid long idle waits on non-zero ranks.
    """
    if memmap_array.ndim != 4:
        raise ValueError(f"Expected a 4D memmap for channel stats, got shape {memmap_array.shape}")

    n_samples, n_channels, height, width = memmap_array.shape
    world_size = max(1, int(world_size))
    rank = max(0, int(rank))
    chunk = max(1, int(chunk_size))

    samples_per_rank = (n_samples + world_size - 1) // world_size
    start_idx = min(n_samples, samples_per_rank * rank)
    end_idx = min(n_samples, start_idx + samples_per_rank)

    sum_c = np.zeros((n_channels,), dtype=np.float64)
    sumsq_c = np.zeros((n_channels,), dtype=np.float64)
    local_pixels = float(max(0, end_idx - start_idx)) * float(height) * float(width)

    for start in range(start_idx, end_idx, chunk):
        end = min(end_idx, start + chunk)
        if start >= end:
            break
        block = np.asarray(memmap_array[start:end])  # (b,C,H,W)
        sum_c += block.sum(axis=(0, 2, 3))
        sumsq_c += np.square(block).sum(axis=(0, 2, 3))

    sum_t = torch.from_numpy(sum_c)
    sumsq_t = torch.from_numpy(sumsq_c)
    pixel_t = torch.tensor([local_pixels], dtype=torch.float64)

    if world_size > 1 and dist.is_available() and dist.is_initialized():
        if device is None:
            raise RuntimeError("Distributed channel-stat computation requires a CUDA device reference.")
        sum_t = sum_t.to(device)
        sumsq_t = sumsq_t.to(device)
        pixel_t = pixel_t.to(device)
        dist.all_reduce(sum_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(sumsq_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(pixel_t, op=dist.ReduceOp.SUM)
        sum_t = sum_t.cpu()
        sumsq_t = sumsq_t.cpu()
        total_pixels = float(pixel_t.cpu().item())
    else:
        total_pixels = local_pixels

    total_pixels = max(total_pixels, 1e-9)
    mean_t = sum_t / total_pixels
    var_t = torch.clamp(sumsq_t / total_pixels - mean_t * mean_t, min=1e-12)
    std_t = torch.sqrt(var_t)

    return mean_t.to(dtype=torch.float32), std_t.to(dtype=torch.float32)

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
        memmap_path = os.path.join(
            output_folder,
            f"{os.path.splitext(os.path.basename(model_output_path))[0]}_memmap.npy",
        )
        shape_file = memmap_path.replace('.npy', '_shape.txt')

        # Only rank 0 is allowed to create the memmap for the first time.
        is_dist = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if is_dist else 0

        if (not is_dist) or rank == 0:
            if not os.path.exists(memmap_path) or not os.path.exists(shape_file):
                if not isinstance(data, torch.Tensor):
                    data = torch.tensor(data, dtype=torch.float32)
                logger.info(f"Creating memmap from single file: {data.shape}")
                memmap_array = np.memmap(
                    memmap_path, dtype='float32', mode='w+', shape=data.shape
                )
                memmap_array[:] = data.numpy()
                del memmap_array
                with open(shape_file, 'w') as f:
                    f.write(','.join(map(str, data.shape)))
                logger.info(f"Memmap file created: {memmap_path}")
                logger.info(f"Shape file created: {shape_file}")
        if is_dist:
            dist.barrier()
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
    memmap_path = os.path.join(
        output_folder,
        f"{os.path.splitext(os.path.basename(model_output_path))[0]}_memmap.npy",
    )
    shape_file = memmap_path.replace('.npy', '_shape.txt')

    is_dist = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if is_dist else 0

    # Only rank 0 should construct the memmap from batch files if needed.
    if (not is_dist) or rank == 0:
        if not os.path.exists(memmap_path) or not os.path.exists(shape_file):
            logger.info(f"Creating memmap file: {memmap_path}")
            total_samples, full_shape = create_memmap_from_batches(
                batch_files, memmap_path
            )
            logger.info(f"Memmap file created with {total_samples} total samples")

    # Synchronize so other ranks only try to read once creation is complete.
    if is_dist:
        dist.barrier()

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
        rank = int(os.environ.get("RANK", 0))
        logger.info(f"Distributed training: rank {rank}/{world_size}, local_rank={local_rank}")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size,
        )
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
    
    # # Estimate scaling from the first batch(es) using memmap loader (disabled)
    # stats_loader = DataLoader(train_dataset, 
    #                           batch_size=batch_size, 
    #                           shuffle=False, 
    #                           drop_last=True, 
    #                           num_workers=0, 
    #                           pin_memory=False)
    # sigma_estimates = []
    # for i, batch_data in enumerate(stats_loader):
    #     if conditional:
    #         batch_latents, _ = batch_data
    #         batch = batch_latents.to(dtype=torch.float32, device=\"cpu\")
    #     else:
    #         batch = batch_data.to(dtype=torch.float32, device=\"cpu\")
    #     sigma_estimates.append(batch.std(unbiased=False).item())
    #     if conditional:
    #         del batch_latents
    #     if i >= 4:
    #         break
    # sigma_hat = torch.tensor(sum(sigma_estimates) / len(sigma_estimates), device=device)
    # if distributed:
    #     dist.all_reduce(sigma_hat, op=dist.ReduceOp.SUM)
    #     sigma_hat /= world_size
    # logger.info(f"Estimated sigma from {len(sigma_estimates)} small batches: {sigma_hat:.4f}")
    total_samples = len(train_dataset)
    if conditional:
        sample_latent_shape = train_dataset[0][0].shape
        logger.info(f"Training data shape: ({total_samples},) + {sample_latent_shape} latents + covariates")
        inferred_channels = int(sample_latent_shape[0])
    else:
        sample_shape = train_dataset[0].shape
        logger.info(f"Training data shape: ({total_samples},) + {sample_shape}")
        inferred_channels = int(sample_shape[0])
    # del stats_loader, batch, batch_data
    gc.collect()
    torch.cuda.empty_cache()

    # torch.save(sigma_hat, os.path.join(output_folder, f"train_{model_name}_sigma.pt"))
    # logger.info(f"Estimated global sigma_hat = {sigma_hat:.4f}")

    # ------------------------------------------------------------------
    # Channel-wise normalization stats over the full latent canvas
    # ------------------------------------------------------------------
    if not hasattr(train_dataset, "arr"):
        raise ValueError("Channel normalization requires a memmap-backed dataset with an 'arr' attribute.")
    stats_chunk = max(int(batch_size), 256)
    if distributed:
        if dist.get_rank() == 0:
            logger.info(f"[NormStats] Computing per-channel mean/std over memmap with chunk={stats_chunk} (distributed)")
        channel_mean_cpu, channel_std_cpu = compute_channel_stats_from_memmap(
            train_dataset.arr,
            chunk_size=stats_chunk,
            rank=dist.get_rank(),
            world_size=world_size,
            device=device,
        )
        if dist.get_rank() == 0:
            logger.info(
                "[NormStats] mean(|mean|)=%.4e | mean(std)=%.4e | min(std)=%.4e",
                channel_mean_cpu.abs().mean().item(),
                channel_std_cpu.mean().item(),
                channel_std_cpu.min().item(),
            )
    else:
        logger.info(f"[NormStats] Computing per-channel mean/std over memmap with chunk={stats_chunk}")
        channel_mean_cpu, channel_std_cpu = compute_channel_stats_from_memmap(
            train_dataset.arr,
            chunk_size=stats_chunk,
        )
        logger.info(
            "[NormStats] mean(|mean|)=%.4e | mean(std)=%.4e | min(std)=%.4e",
            channel_mean_cpu.abs().mean().item(),
            channel_std_cpu.mean().item(),
            channel_std_cpu.min().item(),
        )

    channel_std_cpu = torch.clamp(channel_std_cpu, min=1e-6)
    mu_dev = channel_mean_cpu.view(1, inferred_channels, 1, 1).to(device)
    sd_dev = channel_std_cpu.view(1, inferred_channels, 1, 1).to(device)
    mu_dev = mu_dev.contiguous().to(memory_format=torch.channels_last)
    sd_dev = sd_dev.contiguous().to(memory_format=torch.channels_last)

    # Create data loader
    if distributed: 
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=dist.get_rank(), shuffle=True, drop_last=True)
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
    logger.info(f"Created memmap DataLoader with {loader_args.get('num_workers', 0)} workers")

    # Switch to HF's scheduler for noising/denoising
    scheduler = DDPMScheduler(
        num_train_timesteps=num_time_steps,
        beta_schedule="squaredcos_cap_v2",
        beta_start=1e-4,
        beta_end=0.02,
        clip_sample=True,
        clip_sample_range=10.0,
    )
    scheduler.config.prediction_type = "v_prediction"
    scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(device)
    scheduler.betas         = scheduler.betas.to(device)

    # Create conditional or unconditional model
    if conditional:
        model = ConditionalUNET(
            input_channels=inferred_channels,
            output_channels=inferred_channels,
            cond_dim=cond_dim,
            time_steps=num_time_steps
        )
        logger.info(f"Created conditional UNet with {cond_dim} covariate dimensions")
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
        logger.info(f"Model size: {sum(p.numel() for p in model.parameters()) * 4 / 1024**2:.2f} MB")
    else:
        model = UnconditionalUNET(
            input_channels=inferred_channels,
            output_channels=inferred_channels,
            time_steps=num_time_steps
        )
        logger.info("Created unconditional UNet")
    
    # Enable gradient checkpointing
    model.unet.enable_gradient_checkpointing()
    logger.info("Enabled gradient checkpointing for UNet")

    model.to(device, non_blocking=True)
    # model = model.to(memory_format=torch.channels_last, dtype=torch.bfloat16)
    model = model.to(memory_format=torch.channels_last)

    # Wrap in DDP 
    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False, find_unused_parameters=False)
        logger.info("Wrapped model in DDP")
        ddp_wrapped = True
    else:
        ddp_wrapped = False
    
    # Start LR at 1e-4 and schedule down to user-supplied LR
    start_lr = 1e-4
    # optimizer = optim.Adam(model.parameters(), lr=start_lr)
    optimizer = optim.AdamW(model.parameters(), lr=start_lr, betas=(0.9, 0.99), weight_decay=1e-4)
    
    # lr_scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr)
    from torch.optim.lr_scheduler import LambdaLR
    warmup_steps = 2000
    total_steps  = num_epochs * (len(train_loader))
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        # cosine to eta_min thereafter
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(lr / start_lr, cosine)
    lr_scheduler = LambdaLR(optimizer, lr_lambda)
    
    # Create EMA model on CPU to save GPU memory
    ema_src = model.module if ddp_wrapped else model
    for p in ema_src.parameters():
        p.data = p.data.float()
    ema = ModelEmaV3(ema_src, decay=ema_decay, device='cpu')
    for p in ema.module.parameters():
        p.data = p.data.float()
    # ema = ModelEmaV3(ema_src, decay=ema_decay, device='cpu')
    logger.info("Created EMA model on CPU to save GPU memory")
    
    # Create mixed precision scaler (disabled for bfloat16)
    scaler = amp.GradScaler(enabled=False)
    logger.info("Mixed precision enabled with float32 autocast; GradScaler disabled for float32")
    
    # Load initial checkpoint if specified (optional - for resuming training)
    if checkpoint_path is not None and checkpoint_path.strip() and os.path.exists(checkpoint_path):
        logger.info(f"Loading initial checkpoint from {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            weight_state = checkpoint.get('weights', {})
            has_module_prefix = any(k.startswith("module.") for k in weight_state.keys())
            is_ddp_wrapped = isinstance(model, DDP)

            if is_ddp_wrapped and not has_module_prefix:
                logger.info("Loaded checkpoint without DDP prefixes; applying to wrapped module.")
                model.module.load_state_dict(weight_state)
            elif (not is_ddp_wrapped) and has_module_prefix:
                logger.info("Loaded checkpoint with DDP prefixes; stripping 'module.' before load.")
                stripped_weights = {
                    k.split("module.", 1)[1] if k.startswith("module.") else k: v
                    for k, v in weight_state.items()
                }
                model.load_state_dict(stripped_weights)
            else:
                model.load_state_dict(weight_state)

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
    
    # sigma_dq = 0.01  # small dequantization noise std (to stabilize training, particularly for spiky latent dimensions)
    
    main_process = (not distributed) or (rank == 0)

    for i in range(num_epochs):
        if distributed:
            train_loader.sampler.set_epoch(i)
        total_loss = 0.0
        total_steps = 0
        epoch_iterator = tqdm(
            train_loader,
            desc=f"Epoch {i+1}/{num_epochs}",
            disable=not main_process,
        )
        for bidx, batch_data in enumerate(epoch_iterator):
            
            # Handle conditional vs unconditional data
            if conditional:
                x, covariates = batch_data
                x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last).float()
                covariates = covariates.float().to(device, non_blocking=True)
            else:
                x = batch_data.float().to(device, non_blocking=True).to(memory_format=torch.channels_last)
                covariates = None
            
            clean_latents = (x - mu_dev) / sd_dev

            # Sample random timesteps and noise, then add noise via HF scheduler
            B = clean_latents.size(0)
            t = torch.randint(0, num_time_steps, (B,), device=x.device, dtype=torch.long)
            noise = torch.randn_like(clean_latents)
            noisy_latents = scheduler.add_noise(clean_latents, noise, t)
            
            optimizer.zero_grad()
            
            if conditional:
                # Classifier-free guidance training (uses DDP-wrapped model so gradients sync)
                out_uncond, out_cond = model(
                    noisy_latents,
                    t,
                    covariates,
                    cfg_drop_prob=p_uncond,
                    return_pair=True,
                )
                loss_un = v_pred_loss(out_uncond, clean_latents, noise, t, scheduler)
                loss_co = v_pred_loss(out_cond,   clean_latents, noise, t, scheduler)
                loss = 0.5 * (loss_un + loss_co)
            else:
                output = model(noisy_latents, t)
                loss = v_pred_loss(output, clean_latents, noise, t, scheduler)
            
            total_loss += loss.item()
            total_steps += 1
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            # scaler.step(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()
            
            ema.update(ema_src)
            
            # Clean up
            if conditional:
                del x, clean_latents, noise, loss, t, covariates, noisy_latents
            else:
                del x, clean_latents, noise, output, loss, t, noisy_latents
            
        loss_tensor = torch.tensor(
            [total_loss, float(total_steps)],
            device=device,
            dtype=torch.float64,
        )
        if distributed:
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        global_steps = max(1.0, loss_tensor[1].item())
        mean_epoch_loss = loss_tensor[0].item() / global_steps

        if main_process:
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(f'Epoch {i+1} | Loss {mean_epoch_loss:.5f} | LR {current_lr:.2e}')
        # lr_scheduler.step()

        # Save model every epoch, only on the first rank
        if distributed:
            dist.barrier()
        if main_process:
            if (i+1) % 1 == 0:
                checkpoint = {
                    'weights': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'ema': ema.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'scaler': scaler.state_dict(),
                    'conditional': conditional,
                    'cond_dim': cond_dim if conditional else None,
                    'channel_mean': channel_mean_cpu.clone(),
                    'channel_std': channel_std_cpu.clone(),
                }
                os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
                model_output_path_epoch = model_output_path.replace(".pth", f"_epoch{i+1}.pth")
                torch.save(checkpoint, model_output_path_epoch)
                logger.info(f"Model saved to {model_output_path_epoch}")
        if distributed:
            dist.barrier()

    # Save final checkpoint
    if distributed:
        dist.barrier()
    if main_process:
        checkpoint = {
            'weights': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'ema': ema.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'scaler': scaler.state_dict(),
            'conditional': conditional,
            'cond_dim': cond_dim if conditional else None,
            'channel_mean': channel_mean_cpu.clone(),
            'channel_std': channel_std_cpu.clone(),
        }
        os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
        torch.save(checkpoint, model_output_path)
        logger.info(f"Model saved to {model_output_path}")
    if distributed:
        dist.barrier()

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
