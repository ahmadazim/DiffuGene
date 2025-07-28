"""
DiffuGene: A comprehensive package for genetic diffusion modeling.

This package provides a complete pipeline for:
1. Block-wise PCA embedding of genetic data
2. Joint VAE embedding of block embeddings  
3. Diffusion modeling on latent representations
"""

__version__ = "0.1.0"

# Suppress common warnings before any imports
import os
import warnings
import sys

# Comprehensive warning suppression
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')  # Suppress all TF messages except errors
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')  # Disable oneDNN warnings
os.environ.setdefault('PYTHONWARNINGS', 'ignore')
os.environ.setdefault('CUDA_LAUNCH_BLOCKING', '1')  # Reduce CUDA warnings
os.environ.setdefault('CUDA_CACHE_DISABLE', '1')   # Disable CUDA cache warnings

# Additional XLA/CUDA suppression
os.environ.setdefault('TF_XLA_FLAGS', '--tf_xla_enable_xla_devices=false')
os.environ.setdefault('XLA_FLAGS', '--xla_gpu_cuda_data_dir=/dev/null')

# Suppress Python warnings
warnings.filterwarnings('ignore')

# Import key components for easy access
from . import utils
from . import block_embed
from . import joint_embed  
from . import diffusion

# Main classes for direct import
from .block_embed import PCA_Block, load_pca_blocks
from .joint_embed import SNPVAE, BlockPCDataset, SNPLoader
from .diffusion import LatentUNET2D

__all__ = [
    # Modules
    "utils", "block_embed", "joint_embed", "diffusion",
    # Key classes
    "PCA_Block", "load_pca_blocks", "SNPVAE", "BlockPCDataset", "SNPLoader", "LatentUNET2D"
]
