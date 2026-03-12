#!/usr/bin/env python

import argparse
import os
import random
import gc
import sys
from datetime import timedelta
from tqdm import tqdm
from typing import List, Tuple, Optional, Dict
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

try:
    from ..utils import setup_logging, get_logger, prepare_covariates_for_training, save_covariate_metadata
    from .unet import LatentUNET2D as ConditionalUNET
    from .unet_unconditional import LatentUNET2D as UnconditionalUNET
    from .unet import set_seed, v_pred_loss
    from .SiT import SiTFlowModel
except ImportError:
    # Support direct execution: python path/to/DiffuGene/diffusion/train.py
    this_dir = os.path.dirname(__file__)
    src_root = os.path.abspath(os.path.join(this_dir, "..", "..", ".."))
    if src_root not in sys.path:
        sys.path.insert(0, src_root)
    from DiffuGene.utils import setup_logging, get_logger, prepare_covariates_for_training, save_covariate_metadata
    from DiffuGene.diffusion.unet import LatentUNET2D as ConditionalUNET
    from DiffuGene.diffusion.unet_unconditional import LatentUNET2D as UnconditionalUNET
    from DiffuGene.diffusion.unet import set_seed, v_pred_loss
    from DiffuGene.diffusion.SiT import SiTFlowModel

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


class MultiChromosomeMemmapDataset(Dataset):
    """
    Memory-mapped dataset that concatenates per-chromosome token latents on-the-fly.
    Expected per-chromosome memmap shape: (N, T_chr, D)
    """

    def __init__(self, memmap_paths_by_chr: Dict[int, str]):
        if not memmap_paths_by_chr:
            raise ValueError("No chromosome memmap paths were provided.")
        self.chromosomes = sorted(int(c) for c in memmap_paths_by_chr.keys())
        self.memmap_paths_by_chr = {int(k): v for k, v in memmap_paths_by_chr.items()}
        self.arrays: Dict[int, np.memmap] = {}
        self.sample_count: Optional[int] = None
        self.latent_dim: Optional[int] = None
        self.tokens_per_chr: Dict[int, int] = {}

        for chrom in self.chromosomes:
            memmap_path = self.memmap_paths_by_chr[chrom]
            shape_file = memmap_path.replace('.npy', '_shape.txt')
            if not os.path.exists(shape_file):
                raise ValueError(f"Shape file not found for chromosome {chrom}: {shape_file}")
            with open(shape_file, 'r') as f:
                shape = tuple(map(int, f.read().strip().split(',')))
            if len(shape) != 3:
                raise ValueError(f"Chromosome {chrom} memmap must be 3D (N,T,D), got shape {shape}")
            n_samples, n_tokens, d_model = shape
            if self.sample_count is None:
                self.sample_count = int(n_samples)
            elif int(n_samples) != int(self.sample_count):
                raise ValueError(
                    f"Sample-count mismatch: chr{chrom} has {n_samples} but expected {self.sample_count}"
                )
            if self.latent_dim is None:
                self.latent_dim = int(d_model)
            elif int(d_model) != int(self.latent_dim):
                raise ValueError(
                    f"Latent-dim mismatch: chr{chrom} has {d_model} but expected {self.latent_dim}"
                )
            self.tokens_per_chr[chrom] = int(n_tokens)
            self.arrays[chrom] = np.memmap(memmap_path, dtype='float32', mode='r', shape=shape)

        self.sample_count = int(self.sample_count)
        self.latent_dim = int(self.latent_dim)
        self.total_tokens = int(sum(self.tokens_per_chr[c] for c in self.chromosomes))
        self.source_chromosomes = list(self.chromosomes)
        source_ids = []
        for source_idx, chrom in enumerate(self.chromosomes):
            source_ids.append(torch.full((self.tokens_per_chr[chrom],), source_idx, dtype=torch.long))
        self.source_token_ids = torch.cat(source_ids, dim=0)
        self.num_sources = len(self.chromosomes)
        logger.info(
            "Loaded multi-chromosome memmap dataset: N=%d total_tokens=%d D=%d sources=%d (%s)",
            self.sample_count,
            self.total_tokens,
            self.latent_dim,
            self.num_sources,
            ",".join(f"chr{c}:{self.tokens_per_chr[c]}" for c in self.chromosomes),
        )

    def __len__(self):
        return self.sample_count

    def __getitem__(self, i):
        parts = [torch.from_numpy(self.arrays[c][i]) for c in self.chromosomes]
        return torch.cat(parts, dim=0)

    def compute_per_source_feature_stats(
        self,
        chunk_size: int = 512,
        rank: int = 0,
        world_size: int = 1,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        means = []
        stds = []
        for chrom in self.chromosomes:
            arr = self.arrays[chrom]
            mean_c, std_c = compute_feature_stats_from_memmap(
                arr,
                chunk_size=chunk_size,
                rank=rank,
                world_size=world_size,
                device=device,
            )
            means.append(mean_c)
            stds.append(std_c)
        return torch.stack(means, dim=0), torch.stack(stds, dim=0)

    def compute_feature_stats(
        self,
        chunk_size: int = 512,
        rank: int = 0,
        world_size: int = 1,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        means, stds = self.compute_per_source_feature_stats(
            chunk_size=chunk_size,
            rank=rank,
            world_size=world_size,
            device=device,
        )
        weighted_mean = None
        weighted_second = None
        total_token_weight = 0.0
        for source_idx, chrom in enumerate(self.chromosomes):
            mean_c = means[source_idx]
            std_c = stds[source_idx]
            token_weight = float(self.tokens_per_chr[chrom])
            second_c = std_c * std_c + mean_c * mean_c
            if weighted_mean is None:
                weighted_mean = mean_c * token_weight
                weighted_second = second_c * token_weight
            else:
                weighted_mean = weighted_mean + mean_c * token_weight
                weighted_second = weighted_second + second_c * token_weight
            total_token_weight += token_weight
        if total_token_weight <= 0.0:
            raise ValueError("Invalid token weighting while computing multi-chromosome stats.")
        global_mean = weighted_mean / total_token_weight
        global_var = torch.clamp((weighted_second / total_token_weight) - (global_mean * global_mean), min=1e-12)
        global_std = torch.sqrt(global_var)
        return global_mean.to(dtype=torch.float32), global_std.to(dtype=torch.float32)


class ConditionalLatentDataset(Dataset):
    """Attach covariates to any latent dataset with aligned sample order."""

    def __init__(self, latent_dataset: Dataset, covariate_tensor: torch.Tensor):
        self.latent_dataset = latent_dataset
        self.covariates = covariate_tensor
        if len(self.latent_dataset) != len(self.covariates):
            raise ValueError(
                f"Latent data has {len(self.latent_dataset)} samples but "
                f"covariate data has {len(self.covariates)} samples"
            )

    def __len__(self):
        return len(self.latent_dataset)

    def __getitem__(self, i):
        x = self.latent_dataset[i]
        c = self.covariates[i]
        return x, c

    @property
    def source_token_ids(self):
        return getattr(self.latent_dataset, "source_token_ids", None)

    @property
    def num_sources(self):
        return getattr(self.latent_dataset, "num_sources", None)

    def compute_feature_stats(
        self,
        chunk_size: int = 512,
        rank: int = 0,
        world_size: int = 1,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if hasattr(self.latent_dataset, "compute_feature_stats"):
            return self.latent_dataset.compute_feature_stats(
                chunk_size=chunk_size,
                rank=rank,
                world_size=world_size,
                device=device,
            )
        if hasattr(self.latent_dataset, "arr"):
            return compute_feature_stats_from_memmap(
                self.latent_dataset.arr,
                chunk_size=chunk_size,
                rank=rank,
                world_size=world_size,
                device=device,
            )
        raise ValueError("Underlying latent dataset does not support feature-stat computation.")


def extract_batch_num(path: str) -> int:
    basename = os.path.basename(path)
    match = re.search(r'batch(\d+)', basename)
    if match:
        return int(match.group(1))
    nums = re.findall(r'(\d+)', basename)
    return int(nums[-1]) if nums else 0


def discover_chr_latent_batches(root_dir: str) -> Dict[int, List[str]]:
    chr_batches: Dict[int, List[str]] = {}
    if not os.path.isdir(root_dir):
        return chr_batches
    for entry in sorted(os.listdir(root_dir)):
        m = re.match(r'^chr(\d+)$', entry)
        if not m:
            continue
        chrom = int(m.group(1))
        chr_dir = os.path.join(root_dir, entry)
        files = sorted(glob.glob(os.path.join(chr_dir, "batch*_latents.pt")), key=extract_batch_num)
        if not files:
            files = sorted(glob.glob(os.path.join(chr_dir, "batch*.pt")), key=extract_batch_num)
        if files:
            chr_batches[chrom] = files
    return chr_batches


def load_or_compute_per_source_stats(
    stats_dataset: Dataset,
    stats_path: str,
    flow_norm_eps: float,
    chunk_size: int,
    distributed: bool,
    world_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return per-source (mean,std) with shape (S,D), cached on disk.
    """
    is_dist = distributed and dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if is_dist else 0
    source_ids = getattr(stats_dataset, "source_token_ids", None)
    num_sources = int(getattr(stats_dataset, "num_sources", 0) or 0)
    if source_ids is None or num_sources <= 0:
        raise ValueError("Per-source stats requested but dataset does not expose source metadata.")

    if os.path.exists(stats_path):
        if is_dist:
            dist.barrier()
        payload = torch.load(stats_path, map_location="cpu")
        mean_s = payload["per_source_mean"].float()
        std_s = payload["per_source_std"].float()
    else:
        if not hasattr(stats_dataset, "compute_per_source_feature_stats"):
            raise ValueError("Dataset does not implement compute_per_source_feature_stats().")
        # In DDP mode all ranks must participate because compute uses all_reduce.
        mean_s, std_s = stats_dataset.compute_per_source_feature_stats(
            chunk_size=chunk_size,
            rank=rank,
            world_size=world_size,
            device=device if is_dist else None,
        )
        std_s = torch.clamp(std_s, min=float(flow_norm_eps))
        if (not is_dist) or rank == 0:
            payload = {
                "per_source_mean": mean_s.cpu(),
                "per_source_std": std_s.cpu(),
                "source_token_ids": source_ids.cpu().long(),
                "num_sources": int(num_sources),
                "source_chromosomes": list(getattr(stats_dataset, "source_chromosomes", [])),
                "flow_norm_eps": float(flow_norm_eps),
            }
            torch.save(payload, stats_path)
        if is_dist:
            dist.barrier()
    if int(mean_s.shape[0]) != num_sources:
        raise ValueError(
            f"Per-source stats source count mismatch: {mean_s.shape[0]} vs dataset {num_sources}"
        )
    if is_dist and not os.path.exists(stats_path):
        # Should never happen after barrier.
        raise FileNotFoundError(f"Expected per-source stats file was not created: {stats_path}")
    if is_dist and os.path.exists(stats_path):
        dist.barrier()
        payload = torch.load(stats_path, map_location="cpu")
        mean_s = payload["per_source_mean"].float()
        std_s = payload["per_source_std"].float()
    return mean_s, torch.clamp(std_s, min=float(flow_norm_eps))


def compute_feature_stats_from_memmap(
    memmap_array: np.memmap,
    chunk_size: int = 512,
    rank: int = 0,
    world_size: int = 1,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute per-feature mean/std over an entire latent memmap without loading it fully into RAM.
    Supported shapes:
      - 4D: (N, C, H, W) -> stats over C
      - 3D: (N, L, D)    -> stats over D
    When running under DDP, each rank processes a disjoint slice of samples and the partial
    statistics are aggregated via all_reduce to avoid long idle waits on non-zero ranks.
    """
    if memmap_array.ndim not in (3, 4):
        raise ValueError(f"Expected a 3D or 4D memmap for latent stats, got shape {memmap_array.shape}")

    if memmap_array.ndim == 4:
        n_samples, n_features, d2, d3 = memmap_array.shape
        reducer_axes = (0, 2, 3)
        elements_per_sample = float(d2) * float(d3)
    else:
        n_samples, d1, n_features = memmap_array.shape
        reducer_axes = (0, 1)
        elements_per_sample = float(d1)
    world_size = max(1, int(world_size))
    rank = max(0, int(rank))
    chunk = max(1, int(chunk_size))

    samples_per_rank = (n_samples + world_size - 1) // world_size
    start_idx = min(n_samples, samples_per_rank * rank)
    end_idx = min(n_samples, start_idx + samples_per_rank)

    sum_c = np.zeros((n_features,), dtype=np.float64)
    sumsq_c = np.zeros((n_features,), dtype=np.float64)
    local_pixels = float(max(0, end_idx - start_idx)) * elements_per_sample

    for start in range(start_idx, end_idx, chunk):
        end = min(end_idx, start + chunk)
        if start >= end:
            break
        block = np.asarray(memmap_array[start:end])
        sum_c += block.sum(axis=reducer_axes)
        sumsq_c += np.square(block).sum(axis=reducer_axes)

    sum_t = torch.from_numpy(sum_c)
    sumsq_t = torch.from_numpy(sumsq_c)
    pixel_t = torch.tensor([local_pixels], dtype=torch.float64)

    if world_size > 1 and dist.is_available() and dist.is_initialized():
        if device is None:
            raise RuntimeError("Distributed stat computation requires a CUDA device reference.")
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
    # Case 0: multi-chromosome latent directory (e.g., AE_embeddings/chr*/batch*_latents.pt)
    if os.path.isdir(path):
        chr_batches = discover_chr_latent_batches(path)
        if chr_batches:
            logger.info(f"Detected per-chromosome latent directory with {len(chr_batches)} sources: {path}")
            model_stem = os.path.splitext(os.path.basename(model_output_path))[0]
            memmap_paths_by_chr: Dict[int, str] = {}
            is_dist = dist.is_available() and dist.is_initialized()
            rank = dist.get_rank() if is_dist else 0
            for chrom, batch_files in sorted(chr_batches.items()):
                memmap_path = os.path.join(output_folder, f"{model_stem}_chr{chrom}_memmap.npy")
                shape_file = memmap_path.replace('.npy', '_shape.txt')
                if (not is_dist) or rank == 0:
                    if not os.path.exists(memmap_path) or not os.path.exists(shape_file):
                        logger.info(
                            "Creating chromosome memmap for chr%s from %d files -> %s",
                            chrom,
                            len(batch_files),
                            memmap_path,
                        )
                        create_memmap_from_batches(batch_files, memmap_path)
                memmap_paths_by_chr[chrom] = memmap_path
            if is_dist:
                dist.barrier()
            return MultiChromosomeMemmapDataset(memmap_paths_by_chr)
        batch_files = sorted(glob.glob(os.path.join(path, "batch*.pt")), key=extract_batch_num)
        if batch_files:
            logger.info(f"Detected batch-latent directory with {len(batch_files)} files: {path}")
            memmap_path = os.path.join(
                output_folder,
                f"{os.path.splitext(os.path.basename(model_output_path))[0]}_memmap.npy",
            )
            shape_file = memmap_path.replace('.npy', '_shape.txt')
            is_dist = dist.is_available() and dist.is_initialized()
            rank = dist.get_rank() if is_dist else 0
            if (not is_dist) or rank == 0:
                if not os.path.exists(memmap_path) or not os.path.exists(shape_file):
                    create_memmap_from_batches(batch_files, memmap_path)
            if is_dist:
                dist.barrier()
            return MemmapDataset(memmap_path)
        raise FileNotFoundError(
            f"Directory {path} does not contain supported latent files "
            "(expected chr*/batch*_latents.pt or batch*.pt)."
        )

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
    """Prepare covariates and create conditional dataset over latent sources.
    
    Args:
        memmap_dataset: MemmapDataset of latent embeddings
        covariate_path: Path to covariate CSV file
        fam_path: Path to training fam file
        binary_cols: List of binary variable column names
        categorical_cols: List of categorical variable column names
        output_folder: Folder to save covariate metadata
        model_name: Model name for metadata filename
    
    Returns:
        ConditionalLatentDataset with latents and covariates, covariate dimension
    """
    logger.info("Preparing conditional training data for latent dataset...")
    
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
    
    # Create conditional dataset wrapper (works for both single and multi-source latents)
    dataset = ConditionalLatentDataset(memmap_dataset, covariate_tensor)
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
    cfg_drop_prob: float = 0.0, 
    model_type: str = "usit",
    sit_num_layers: int = 9,
    sit_num_heads: int = 8,
    sit_mlp_ratio: int = 4,
    sit_dropout: float = 0.0,
    sit_qkv_bias: bool = False,
    sit_use_udit: bool = False,
    sit_hidden_dim: Optional[int] = None,
    flow_use_latent_normalization: bool = False,
    flow_norm_eps: float = 1e-6,
    dist_timeout_minutes: int = 60,
):
    setup_logging()
    set_seed(random.randint(0, 2**32-1)) if seed == -1 else set_seed(seed)
    model_type = str(model_type).lower()
    if model_type not in {"unet", "sit", "usit"}:
        raise ValueError(f"Unsupported model_type='{model_type}'. Expected one of: unet, sit, usit")
    if model_type in {"sit", "usit"}:
        if conditional:
            logger.warning("Conditional inputs are disabled for SiT/USiT; proceeding unconditionally.")
            conditional = False
        if cfg_drop_prob != 0.0:
            logger.warning("CFG drop is disabled for SiT/USiT; forcing cfg_drop_prob=0.")
            cfg_drop_prob = 0.0
        if sit_dropout != 0.0:
            logger.warning("SiT/USiT dropout is forced to 0.0 for this training mode.")
            sit_dropout = 0.0
    if dist_timeout_minutes <= 0:
        raise ValueError(f"dist_timeout_minutes must be > 0, got {dist_timeout_minutes}")

    output_folder = os.path.dirname(model_output_path)
    model_name = os.path.splitext(os.path.basename(model_output_path))[0]
    
    # Initialize distributed training if launched with torchrun/torch.distributed
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    distributed = world_size > 1
    rank = 0
    if distributed:
        rank = int(os.environ.get("RANK", 0))
        logger.info(f"Distributed training: rank {rank}/{world_size}, local_rank={local_rank}")
        torch.cuda.set_device(local_rank)
        pg_timeout = timedelta(minutes=int(dist_timeout_minutes))
        logger.info("Initializing NCCL process group with timeout=%s", pg_timeout)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size,
            timeout=pg_timeout,
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
        sample_shape = train_dataset[0][0].shape
        logger.info(f"Training data shape: ({total_samples},) + {sample_shape} latents + covariates")
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
    # Latent normalization stats
    #   - UNet / single-source: global per-feature stats
    #   - SiT/USiT multi-source: per-source per-feature stats cached on disk
    # ------------------------------------------------------------------
    stats_dataset = train_dataset.latent_dataset if isinstance(train_dataset, ConditionalLatentDataset) else train_dataset
    if flow_norm_eps <= 0.0:
        raise ValueError(f"flow_norm_eps must be > 0. Got {flow_norm_eps}")
    if flow_use_latent_normalization and (not hasattr(stats_dataset, "arr")) and (not hasattr(stats_dataset, "compute_feature_stats")):
        raise ValueError("Latent normalization requires a dataset with memmap storage or compute_feature_stats().")

    if model_type == "unet" and len(sample_shape) != 4:
        raise ValueError(
            f"UNet training expects latent shape (C,H,W); got sample shape {sample_shape}. "
            "Use model_type=sit/usit for token latents."
        )

    use_per_source_norm = (
        model_type in {"sit", "usit"}
        and hasattr(stats_dataset, "source_token_ids")
        and (getattr(stats_dataset, "source_token_ids", None) is not None)
        and hasattr(stats_dataset, "compute_per_source_feature_stats")
    )
    source_token_ids = getattr(stats_dataset, "source_token_ids", None)
    num_sources = getattr(stats_dataset, "num_sources", None)

    if (not flow_use_latent_normalization) and model_type in {"sit", "usit"}:
        logger.info("[NormStats] Flow latent normalization disabled by config.")
        if len(sample_shape) == 3:  # (C,H,W)
            channel_mean_cpu = torch.zeros(sample_shape[0], dtype=torch.float32)
            channel_std_cpu = torch.ones(sample_shape[0], dtype=torch.float32)
            per_source_mean_cpu = None
            per_source_std_cpu = None
        elif len(sample_shape) == 2:  # (L,D)
            if use_per_source_norm:
                s = int(num_sources)
                d = int(sample_shape[-1])
                per_source_mean_cpu = torch.zeros((s, d), dtype=torch.float32)
                per_source_std_cpu = torch.ones((s, d), dtype=torch.float32)
                channel_mean_cpu = per_source_mean_cpu.mean(dim=0)
                channel_std_cpu = per_source_std_cpu.mean(dim=0)
            else:
                channel_mean_cpu = torch.zeros(sample_shape[-1], dtype=torch.float32)
                channel_std_cpu = torch.ones(sample_shape[-1], dtype=torch.float32)
                per_source_mean_cpu = None
                per_source_std_cpu = None
        else:
            raise ValueError(f"Unsupported latent shape {sample_shape}.")
    else:
        stats_chunk = max(int(batch_size), 256)
        per_source_mean_cpu = None
        per_source_std_cpu = None
        if use_per_source_norm:
            stats_path = os.path.join(output_folder, f"{model_name}_flow_source_norm_stats.pt")
            if (not distributed) or (dist.get_rank() == 0):
                logger.info(
                    "[NormStats] Loading/computing per-source stats for SiT/USiT with chunk=%d -> %s",
                    stats_chunk,
                    stats_path,
                )
            per_source_mean_cpu, per_source_std_cpu = load_or_compute_per_source_stats(
                stats_dataset=stats_dataset,
                stats_path=stats_path,
                flow_norm_eps=flow_norm_eps,
                chunk_size=stats_chunk,
                distributed=distributed,
                world_size=world_size,
                device=device,
            )
            channel_mean_cpu = per_source_mean_cpu.mean(dim=0)
            channel_std_cpu = per_source_std_cpu.mean(dim=0)
            if (not distributed) or (dist.get_rank() == 0):
                logger.info(
                    "[NormStats] per-source mean(|mean|)=%.4e | mean(std)=%.4e | min(std)=%.4e",
                    per_source_mean_cpu.abs().mean().item(),
                    per_source_std_cpu.mean().item(),
                    per_source_std_cpu.min().item(),
                )
        elif distributed:
            if dist.get_rank() == 0:
                logger.info(f"[NormStats] Computing per-feature mean/std over memmap with chunk={stats_chunk} (distributed)")
            if hasattr(stats_dataset, "compute_feature_stats"):
                channel_mean_cpu, channel_std_cpu = stats_dataset.compute_feature_stats(
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
            logger.info(f"[NormStats] Computing per-feature mean/std over memmap with chunk={stats_chunk}")
            if hasattr(stats_dataset, "compute_feature_stats"):
                channel_mean_cpu, channel_std_cpu = stats_dataset.compute_feature_stats(
                    chunk_size=stats_chunk,
                    rank=0,
                    world_size=1,
                    device=None,
                )
            else:
                channel_mean_cpu, channel_std_cpu = compute_feature_stats_from_memmap(
                    stats_dataset.arr,
                    chunk_size=stats_chunk,
                )
            logger.info(
                "[NormStats] mean(|mean|)=%.4e | mean(std)=%.4e | min(std)=%.4e",
                channel_mean_cpu.abs().mean().item(),
                channel_std_cpu.mean().item(),
                channel_std_cpu.min().item(),
            )

    channel_std_cpu = torch.clamp(channel_std_cpu, min=float(flow_norm_eps))
    if len(sample_shape) == 3:  # UNet sample (C,H,W)
        mu_dev = channel_mean_cpu.view(1, inferred_channels, 1, 1).to(device)
        sd_dev = channel_std_cpu.view(1, inferred_channels, 1, 1).to(device)
        if model_type == "unet":
            mu_dev = mu_dev.contiguous().to(memory_format=torch.channels_last)
            sd_dev = sd_dev.contiguous().to(memory_format=torch.channels_last)
    elif len(sample_shape) == 2:  # token sample (L,D)
        if use_per_source_norm:
            sid = source_token_ids.long()
            mu_tokens = per_source_mean_cpu[sid]
            sd_tokens = torch.clamp(per_source_std_cpu[sid], min=float(flow_norm_eps))
            mu_dev = mu_tokens.unsqueeze(0).to(device)
            sd_dev = sd_tokens.unsqueeze(0).to(device)
        else:
            mu_dev = channel_mean_cpu.view(1, 1, int(sample_shape[-1])).to(device)
            sd_dev = channel_std_cpu.view(1, 1, int(sample_shape[-1])).to(device)
    else:
        raise ValueError(f"Unsupported latent sample shape {sample_shape}; expected 3D or 4D.")

    # Create data loader
    if distributed: 
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=dist.get_rank(), shuffle=True, drop_last=True)
        loader_args = dict(
            batch_size=batch_size,
            sampler=train_sampler,
            drop_last=True,
            num_workers=4,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True,
        )
    else: 
        train_sampler = None
        loader_args = dict(
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=4,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True,
        )
    train_loader = DataLoader(train_dataset, **loader_args)
    logger.info(f"Created memmap DataLoader with {loader_args.get('num_workers', 0)} workers")

    scheduler: Optional[DDPMScheduler] = None
    if model_type == "unet":
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
        scheduler.betas = scheduler.betas.to(device)

        # Create conditional or unconditional UNet
        if conditional:
            model = ConditionalUNET(
                input_channels=inferred_channels,
                output_channels=inferred_channels,
                cond_dim=cond_dim,
                time_steps=num_time_steps
            )
            logger.info(f"Created conditional UNet with {cond_dim} covariate dimensions")
        else:
            model = UnconditionalUNET(
                input_channels=inferred_channels,
                output_channels=inferred_channels,
                time_steps=num_time_steps
            )
            logger.info("Created unconditional UNet")

        model.unet.enable_gradient_checkpointing()
        logger.info("Enabled gradient checkpointing for UNet")
    else:
        # SiT/USiT:
        #   token latents: (L,D) -> latent_length=L, token_dim=D
        #   image latents: (C,H,W) -> token_dim=C, latent_length=H*W
        if len(sample_shape) == 2:
            latent_length = int(sample_shape[0])
            token_dim = int(sample_shape[1])
        elif len(sample_shape) == 3:
            token_dim = int(sample_shape[0])
            latent_length = int(sample_shape[1] * sample_shape[2])
        else:
            raise ValueError(f"Unsupported latent shape for SiT/USiT: {sample_shape}")

        source_token_ids = getattr(stats_dataset, "source_token_ids", None)
        num_sources = getattr(stats_dataset, "num_sources", None)
        if source_token_ids is not None:
            logger.info(
                "Enabling source adapters for %d sources across %d tokens",
                int(num_sources) if num_sources is not None else int(source_token_ids.max().item() + 1),
                int(source_token_ids.numel()),
            )
        hidden_dim = int(sit_hidden_dim) if sit_hidden_dim is not None else None
        model = SiTFlowModel(
            token_dim=token_dim,
            latent_length=latent_length,
            hidden_dim=hidden_dim,
            cond_dim=None,
            num_layers=sit_num_layers,
            num_heads=sit_num_heads,
            mlp_ratio=sit_mlp_ratio,
            dropout=sit_dropout,
            qkv_bias=sit_qkv_bias,
            use_udit=(sit_use_udit or model_type == "usit"),
            source_token_ids=source_token_ids,
            num_sources=num_sources,
        )
        logger.info(
            "Created %s flow model (token_dim=%d latent_length=%d cond=%s hidden_dim=%d num_layers=%d num_heads=%d mlp_ratio=%d)",
            model_type.upper(),
            token_dim,
            latent_length,
            conditional,
            hidden_dim,
            sit_num_layers,
            sit_num_heads,
            sit_mlp_ratio,
        )

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    logger.info(f"Model size: {sum(p.numel() for p in model.parameters()) * 4 / 1024**2:.2f} MB")

    model.to(device, non_blocking=True)
    if model_type == "unet":
        model = model.to(memory_format=torch.channels_last)

    # Wrap in DDP 
    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False, find_unused_parameters=False)
        logger.info("Wrapped model in DDP")
        ddp_wrapped = True
    else:
        ddp_wrapped = False
    
    # Warm up to a peak LR, then cosine decay to a floor LR.
    # This behaves correctly whether user lr is below or above start_lr.
    start_lr = 1e-4
    peak_lr = max(float(start_lr), float(lr))
    min_lr = min(float(start_lr), float(lr))
    optimizer = optim.AdamW(model.parameters(), lr=peak_lr, betas=(0.9, 0.99), weight_decay=1e-4)
    
    # lr_scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr)
    from torch.optim.lr_scheduler import LambdaLR
    warmup_steps = 2000
    total_steps  = num_epochs * (len(train_loader))
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        # Cosine decay from peak_lr -> min_lr after warmup.
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        current_lr = min_lr + (peak_lr - min_lr) * cosine
        return current_lr / peak_lr
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
    # scaler = amp.GradScaler(enabled=False)
    # logger.info("Mixed precision enabled with float32 autocast; GradScaler disabled for float32")
    use_bf16 = True
    scaler = amp.GradScaler(enabled=(not use_bf16))
    logger.info(f"Autocast enabled: bf16={use_bf16}, GradScaler enabled={scaler.is_enabled()}")
    
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
    global_step = 0
    mem_log_interval = 1000

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
                x = x.to(device, non_blocking=True).float()
                if model_type == "unet":
                    x = x.to(memory_format=torch.channels_last)
                covariates = covariates.float().to(device, non_blocking=True)
            else:
                x = batch_data.float().to(device, non_blocking=True)
                if model_type == "unet":
                    x = x.to(memory_format=torch.channels_last)
                covariates = None
            
            clean_latents = (x - mu_dev) / sd_dev

            optimizer.zero_grad()

            with amp.autocast(dtype=torch.bfloat16, enabled=use_bf16):
                if model_type == "unet":
                    # Sample random timesteps and noise, then add noise via DDPM scheduler
                    B = clean_latents.size(0)
                    t = torch.randint(0, num_time_steps, (B,), device=x.device, dtype=torch.long)
                    noise = torch.randn_like(clean_latents)
                    noisy_latents = scheduler.add_noise(clean_latents, noise, t)

                    if conditional:
                        out_uncond, out_cond = model(
                            noisy_latents,
                            t,
                            covariates,
                            cfg_drop_prob=p_uncond,
                            return_pair=True,
                        )
                        loss_un = v_pred_loss(out_uncond, clean_latents, noise, t, scheduler)
                        loss_co = v_pred_loss(out_cond, clean_latents, noise, t, scheduler)
                        loss = 0.5 * (loss_un + loss_co)
                    else:
                        output = model(noisy_latents, t)
                        loss = v_pred_loss(output, clean_latents, noise, t, scheduler)
                else:
                    # Flow matching objective in latent space.
                    B = clean_latents.size(0)
                    t = torch.rand(B, device=x.device, dtype=clean_latents.dtype)
                    t_view = t.view(B, *([1] * (clean_latents.dim() - 1)))
                    base_latents = torch.randn_like(clean_latents)
                    z_t = (1.0 - t_view) * base_latents + t_view * clean_latents
                    v_target = clean_latents - base_latents

                    if conditional:
                        out_uncond, out_cond = model(
                            z_t,
                            t,
                            covariates,
                            cfg_drop_prob=p_uncond,
                            return_pair=True,
                        )
                        loss = 0.5 * (
                            F.mse_loss(out_uncond, v_target) +
                            F.mse_loss(out_cond, v_target)
                        )
                    else:
                        output = model(z_t, t)
                        loss = F.mse_loss(output, v_target)
            
            total_loss += loss.item()
            total_steps += 1
            
            # Backward pass with gradient scaling
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
            # scaler.step(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            lr_scheduler.step()
            ema.update(ema_src)
            
            should_log_mem = (
                global_step == 0
                or global_step == warmup_steps
                or (global_step % mem_log_interval == 0)
            )
            if should_log_mem:
                alloc_gb = torch.cuda.max_memory_allocated(device) / 1024**3
                reserv_gb = torch.cuda.max_memory_reserved(device) / 1024**3
                free_bytes, total_bytes = torch.cuda.mem_get_info(device)
                free_gb = free_bytes / 1024**3
                total_gb = total_bytes / 1024**3
                logger.info(
                    "[mem][rank=%d] step=%d | max_alloc=%.2f GB | max_resv=%.2f GB | "
                    "free=%.2f/%.2f GB",
                    rank, global_step, alloc_gb, reserv_gb, free_gb, total_gb,
                )
                torch.cuda.reset_peak_memory_stats(device)

            global_step += 1

            # Clean up
            if conditional:
                if model_type == "unet":
                    del x, clean_latents, noise, loss, t, covariates, noisy_latents
                else:
                    del x, clean_latents, loss, t, covariates, base_latents, z_t, v_target
            else:
                if model_type == "unet":
                    del x, clean_latents, noise, output, loss, t, noisy_latents
                else:
                    del x, clean_latents, output, loss, t, base_latents, z_t, v_target
            
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
                    'model_type': model_type,
                    'conditional': conditional,
                    'cond_dim': cond_dim if conditional else None,
                    'channel_mean': channel_mean_cpu.clone(),
                    'channel_std': channel_std_cpu.clone(),
                    'per_source_mean': per_source_mean_cpu.clone() if ('per_source_mean_cpu' in locals() and per_source_mean_cpu is not None) else None,
                    'per_source_std': per_source_std_cpu.clone() if ('per_source_std_cpu' in locals() and per_source_std_cpu is not None) else None,
                    'source_token_ids': source_token_ids.clone() if ('source_token_ids' in locals() and source_token_ids is not None) else None,
                    'latent_shape': tuple(sample_shape),
                    'flow_matching': (model_type in {"sit", "usit"}),
                    'sit_config': {
                        'num_layers': int(sit_num_layers),
                        'num_heads': int(sit_num_heads),
                        'mlp_ratio': int(sit_mlp_ratio),
                        'dropout': float(sit_dropout),
                        'qkv_bias': bool(sit_qkv_bias),
                        'use_udit': bool(sit_use_udit or model_type == "usit"),
                        'num_sources': int(num_sources) if ('num_sources' in locals() and num_sources is not None) else None,
                        'flow_use_latent_normalization': bool(flow_use_latent_normalization),
                        'flow_norm_eps': float(flow_norm_eps),
                    },
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
            'model_type': model_type,
            'conditional': conditional,
            'cond_dim': cond_dim if conditional else None,
            'channel_mean': channel_mean_cpu.clone(),
            'channel_std': channel_std_cpu.clone(),
            'per_source_mean': per_source_mean_cpu.clone() if ('per_source_mean_cpu' in locals() and per_source_mean_cpu is not None) else None,
            'per_source_std': per_source_std_cpu.clone() if ('per_source_std_cpu' in locals() and per_source_std_cpu is not None) else None,
            'source_token_ids': source_token_ids.clone() if ('source_token_ids' in locals() and source_token_ids is not None) else None,
            'latent_shape': tuple(sample_shape),
            'flow_matching': (model_type in {"sit", "usit"}),
            'sit_config': {
                'num_layers': int(sit_num_layers),
                'num_heads': int(sit_num_heads),
                'mlp_ratio': int(sit_mlp_ratio),
                'dropout': float(sit_dropout),
                'qkv_bias': bool(sit_qkv_bias),
                'use_udit': bool(sit_use_udit or model_type == "usit"),
                'num_sources': int(num_sources) if ('num_sources' in locals() and num_sources is not None) else None,
                'flow_use_latent_normalization': bool(flow_use_latent_normalization),
                'flow_norm_eps': float(flow_norm_eps),
            },
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
    parser.add_argument("--cfg-drop-prob", type=float, default=0.0, help="Drop probability for classifier-free guidance (UNet only)")
    parser.add_argument("--model-type", type=str, default="usit", choices=["unet", "sit", "usit"], help="Diffusion architecture to train")
    parser.add_argument("--sit-num-layers", type=int, default=9, help="SiT/USiT transformer layers")
    parser.add_argument("--sit-num-heads", type=int, default=8, help="SiT/USiT attention heads")
    parser.add_argument("--sit-mlp-ratio", type=int, default=4, help="SiT/USiT MLP ratio")
    parser.add_argument("--sit-hidden-dim", type=int, default=None, help="SiT/USiT transformer hidden width (UViT hidden_size). If unset, defaults to token_dim.")
    parser.add_argument("--sit-dropout", type=float, default=0.0, help="SiT/USiT dropout")
    parser.add_argument("--sit-qkv-bias", action='store_true', help="Enable QKV bias in SiT/USiT attention")
    parser.add_argument("--sit-use-udit", action='store_true', help="Use U-shaped SiT even if model-type=sit")
    parser.add_argument("--flow-disable-latent-normalization", action='store_true', help="Disable latent normalization in flow matching mode")
    parser.add_argument("--flow-norm-eps", type=float, default=1e-6, help="Epsilon for latent normalization std clamp")
    parser.add_argument(
        "--dist-timeout-minutes",
        type=int,
        default=200,
        help="Distributed process group timeout in minutes (increase for long rank-0 preprocessing).",
    )

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
        cfg_drop_prob=args.cfg_drop_prob,
        model_type=args.model_type,
        sit_num_layers=args.sit_num_layers,
        sit_num_heads=args.sit_num_heads,
        sit_mlp_ratio=args.sit_mlp_ratio,
        sit_hidden_dim=args.sit_hidden_dim,
        sit_dropout=args.sit_dropout,
        sit_qkv_bias=args.sit_qkv_bias,
        sit_use_udit=args.sit_use_udit,
        flow_use_latent_normalization=(not args.flow_disable_latent_normalization),
        flow_norm_eps=args.flow_norm_eps,
        dist_timeout_minutes=args.dist_timeout_minutes,
    )

if __name__ == "__main__":
    main()
