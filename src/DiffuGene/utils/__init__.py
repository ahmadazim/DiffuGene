"""Utility functions for DiffuGene package."""

from .data_prep import RecodedData, RecodedInfo, read_raw
from .file_utils import (
    Block, find_raw_files, find_unique_raw_file, extract_block_number,
    load_blocks_for_chr, create_snplist_files, ensure_dir_exists,
    get_sorted_files_by_block
)
from .logging import setup_logging, get_logger
from .covariate_utils import (
    load_covariate_file, load_fam_file, match_covariates_to_fam,
    normalize_covariates, prepare_covariates_for_training,
    save_covariate_metadata, load_covariate_metadata, sample_training_covariates
)

__all__ = [
    # Data preparation
    "RecodedData", "RecodedInfo", "read_raw",
    # File utilities
    "Block", "find_raw_files", "find_unique_raw_file", "extract_block_number",
    "load_blocks_for_chr", "create_snplist_files", "ensure_dir_exists",
    "get_sorted_files_by_block",
    # Logging
    "setup_logging", "get_logger",
    # Covariate utilities
    "load_covariate_file", "load_fam_file", "match_covariates_to_fam",
    "normalize_covariates", "prepare_covariates_for_training",
    "save_covariate_metadata", "load_covariate_metadata", "sample_training_covariates"
]
