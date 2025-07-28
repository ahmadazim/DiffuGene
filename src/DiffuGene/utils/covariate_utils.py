#!/usr/bin/env python
"""
Covariate data utilities for conditional generation.

This module provides functions to load and process covariate data for conditional
diffusion generation, including matching covariates to genetic samples based on
family ID (FID) and individual ID (IID).
"""

import os
import pandas as pd
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Union
from .logging import get_logger

logger = get_logger(__name__)

def load_covariate_file(covariate_path: str) -> pd.DataFrame:
    """Load covariate CSV file and validate format.
    
    Args:
        covariate_path: Path to covariate CSV file
        
    Returns:
        DataFrame with covariates
        
    Raises:
        FileNotFoundError: If covariate file doesn't exist
        ValueError: If required columns are missing
    """
    if not os.path.exists(covariate_path):
        raise FileNotFoundError(f"Covariate file not found: {covariate_path}")
    
    logger.info(f"Loading covariate file: {covariate_path}")
    # Specify common missing value representations
    na_values = ['NA', 'N/A', 'NULL', 'null', 'NaN', 'nan', '', ' ', '.', 'missing', 'Missing', 'MISSING']
    df = pd.read_csv(covariate_path, na_values=na_values, keep_default_na=True)
    
    # Validate required columns
    required_cols = ['fid', 'iid']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in covariate file: {missing_cols}")
    
    logger.info(f"Loaded covariates: {df.shape[0]} individuals, {df.shape[1]-2} covariates")
    
    # Log missing value statistics
    covariate_cols = [col for col in df.columns if col not in ['fid', 'iid']]
    missing_stats = df[covariate_cols].isnull().sum()
    n_cols_with_missing = (missing_stats > 0).sum()
    
    if n_cols_with_missing > 0:
        logger.info(f"Missing value statistics: {n_cols_with_missing}/{len(covariate_cols)} columns have missing values")
        # Log the top 10 columns with most missing values
        top_missing = missing_stats[missing_stats > 0].sort_values(ascending=False).head(10)
        for col, count in top_missing.items():
            pct = count / len(df) * 100
            logger.info(f"  {col}: {count} missing ({pct:.1f}%)")
    else:
        logger.info("No missing values detected in covariate file")
    
    return df

def load_fam_file(fam_path: str) -> pd.DataFrame:
    """Load PLINK fam file and extract FID/IID.
    
    Args:
        fam_path: Path to PLINK .fam file
        
    Returns:
        DataFrame with FID, IID columns
    """
    if not os.path.exists(fam_path):
        raise FileNotFoundError(f"Fam file not found: {fam_path}")
    
    # PLINK .fam format: FID IID PID MID SEX PHENO
    fam_df = pd.read_csv(fam_path, sep='\s+', header=None, 
                         names=['fid', 'iid', 'pid', 'mid', 'sex', 'pheno'])
    
    # Keep only FID and IID, ensure string type for matching
    fam_df = fam_df[['fid', 'iid']].copy()
    fam_df['fid'] = fam_df['fid'].astype(str)
    fam_df['iid'] = fam_df['iid'].astype(str)
    
    logger.info(f"Loaded fam file: {len(fam_df)} individuals")
    return fam_df

def match_covariates_to_fam(covariate_df: pd.DataFrame, 
                           fam_df: pd.DataFrame,
                           binary_cols: List[str] = None,
                           categorical_cols: List[str] = None) -> Tuple[np.ndarray, List[str]]:
    """Match covariate data to individuals in fam file.
    
    Args:
        covariate_df: DataFrame with covariates (must have fid, iid columns)
        fam_df: DataFrame with genetic sample IDs (fid, iid columns)  
        binary_cols: List of binary variable column names (filled with 0 if missing)
        categorical_cols: List of categorical variable column names (filled with mode if missing)
        
    Returns:
        Tuple of (covariate_matrix, covariate_names)
        - covariate_matrix: (n_samples, n_covariates) numpy array
        - covariate_names: List of covariate column names
        
    Raises:
        ValueError: If no matches found
    """
    logger.info("Matching covariates to genetic samples...")
    
    # Default to empty lists if not provided
    binary_cols = binary_cols or []
    categorical_cols = categorical_cols or []
    
    # Ensure string types for matching
    covariate_df = covariate_df.copy()
    covariate_df['fid'] = covariate_df['fid'].astype(str)
    covariate_df['iid'] = covariate_df['iid'].astype(str)
    
    # Get covariate columns (exclude fid, iid)
    covariate_cols = [col for col in covariate_df.columns if col not in ['fid', 'iid']]
    if not covariate_cols:
        raise ValueError("No covariate columns found (all columns are fid/iid)")
    
    # Classify variable types
    quantitative_cols = [col for col in covariate_cols if col not in binary_cols and col not in categorical_cols]
    
    logger.info(f"Variable type classification:")
    logger.info(f"  Binary variables ({len(binary_cols)}): {binary_cols}")
    logger.info(f"  Categorical variables ({len(categorical_cols)}): {categorical_cols}")
    logger.info(f"  Quantitative variables ({len(quantitative_cols)}): {quantitative_cols}")
    
    # Create merge key for matching
    covariate_df['merge_key'] = covariate_df['fid'] + '_' + covariate_df['iid']
    fam_df['merge_key'] = fam_df['fid'] + '_' + fam_df['iid']
    
    # Merge to match samples
    merged = fam_df.merge(covariate_df, on='merge_key', how='left', suffixes=('_fam', '_cov'))
    
    # Check matching statistics
    n_total = len(fam_df)
    n_matched = merged[covariate_cols[0]].notna().sum()
    n_missing = n_total - n_matched
    
    logger.info(f"Covariate matching results:")
    logger.info(f"  Total genetic samples: {n_total}")
    logger.info(f"  Matched with covariates: {n_matched} ({n_matched/n_total*100:.1f}%)")
    logger.info(f"  Missing covariates: {n_missing} ({n_missing/n_total*100:.1f}%)")
    
    if n_matched == 0:
        raise ValueError("No genetic samples matched with covariate data")
    
    # Extract covariate matrix
    covariate_matrix = merged[covariate_cols].values.astype(float)
    
    # Always check for and handle missing values in each column
    logger.info("Checking and filling missing values by variable type...")
    
    for i, col in enumerate(covariate_cols):
        col_data = covariate_matrix[:, i]
        missing_mask = np.isnan(col_data)
        n_missing_col = missing_mask.sum()
        
        if n_missing_col == 0:
            logger.info(f"  '{col}': no missing values")
            continue
        
        logger.info(f"  '{col}': found {n_missing_col} missing values ({n_missing_col/len(col_data)*100:.1f}%)")
            
        if col in binary_cols:
            # Binary variables: fill with 0
            covariate_matrix[missing_mask, i] = 0.0
            logger.info(f"    Binary '{col}': filled {n_missing_col} missing values with 0")
            
        elif col in categorical_cols:
            # Categorical variables: fill with mode
            if missing_mask.sum() < len(col_data):  # If not all missing
                mode_val = pd.Series(col_data[~missing_mask]).mode()
                if len(mode_val) > 0:
                    fill_val = mode_val.iloc[0]
                else:
                    fill_val = 0.0  # Fallback if mode calculation fails
            else:
                fill_val = 0.0  # If all missing, use 0
            covariate_matrix[missing_mask, i] = fill_val
            logger.info(f"    Categorical '{col}': filled {n_missing_col} missing values with mode {fill_val}")
            
        else:
            # Quantitative variables: fill with mean
            if missing_mask.sum() < len(col_data):  # If not all missing
                mean_val = np.nanmean(col_data)
                if np.isnan(mean_val):  # Double-check if mean calculation failed
                    mean_val = 0.0
                    logger.warning(f"    Mean calculation failed for '{col}', using 0.0")
            else:
                mean_val = 0.0  # If all missing, use 0
            covariate_matrix[missing_mask, i] = mean_val
            logger.info(f"    Quantitative '{col}': filled {n_missing_col} missing values with mean {mean_val:.3f}")
    
    # Convert to float32 for model compatibility
    covariate_matrix = covariate_matrix.astype(np.float32)
    
    # Verify no NaN values remain after imputation
    final_nan_count = np.isnan(covariate_matrix).sum()
    if final_nan_count > 0:
        logger.error(f"ERROR: {final_nan_count} NaN values remain after imputation!")
        # Log which columns still have NaN values
        for i, col in enumerate(covariate_cols):
            col_nans = np.isnan(covariate_matrix[:, i]).sum()
            if col_nans > 0:
                logger.error(f"  Column '{col}': {col_nans} NaN values remaining")
    else:
        logger.info("Imputation successful: no NaN values remain in covariate matrix")
    
    logger.info(f"Final covariate matrix shape: {covariate_matrix.shape}")
    logger.info(f"Covariate names: {covariate_cols}")
    
    return covariate_matrix, covariate_cols

def normalize_covariates(covariate_matrix: np.ndarray, 
                        covariate_names: List[str],
                        binary_cols: List[str] = None,
                        categorical_cols: List[str] = None) -> Tuple[np.ndarray, Dict]:
    """Mean-center quantitative covariates only.
    
    Args:
        covariate_matrix: (n_samples, n_covariates) array
        covariate_names: List of covariate column names
        binary_cols: List of binary variable column names (not normalized)
        categorical_cols: List of categorical variable column names (not normalized)
        
    Returns:
        Tuple of (normalized_matrix, normalization_params)
    """
    logger.info("Mean-centering quantitative covariates...")
    
    # Default to empty lists if not provided
    binary_cols = binary_cols or []
    categorical_cols = categorical_cols or []
    
    # Create copy for modification
    normalized_matrix = covariate_matrix.copy()
    
    # Track which columns were normalized
    means = np.zeros(len(covariate_names))
    normalized_flags = np.zeros(len(covariate_names), dtype=bool)
    
    # Classify and normalize quantitative variables only
    for i, col_name in enumerate(covariate_names):
        if col_name in binary_cols:
            logger.info(f"  Binary '{col_name}': no normalization")
        elif col_name in categorical_cols:
            logger.info(f"  Categorical '{col_name}': no normalization")
        else:
            # Quantitative variable: mean-center
            col_mean = np.mean(covariate_matrix[:, i])
            normalized_matrix[:, i] = covariate_matrix[:, i] - col_mean
            means[i] = col_mean
            normalized_flags[i] = True
            logger.info(f"  Quantitative '{col_name}': mean-centered (mean={col_mean:.3f})")
    
    # Create normalization parameters
    params = {
        'method': 'mean_center',
        'means': means,
        'normalized_flags': normalized_flags,
        'binary_cols': binary_cols,
        'categorical_cols': categorical_cols,
        'covariate_names': covariate_names
    }
    
    n_normalized = normalized_flags.sum()
    logger.info(f"Normalization complete: {n_normalized}/{len(covariate_names)} variables mean-centered")
    
    return normalized_matrix.astype(np.float32), params

def prepare_covariates_for_training(covariate_path: str,
                                  fam_path: str,
                                  binary_cols: List[str] = None,
                                  categorical_cols: List[str] = None) -> Tuple[torch.Tensor, List[str], Dict]:
    """Complete pipeline to prepare covariates for training.
    
    Args:
        covariate_path: Path to covariate CSV file
        fam_path: Path to PLINK .fam file for training data
        binary_cols: List of binary variable column names
        categorical_cols: List of categorical variable column names
        
    Returns:
        Tuple of (covariate_tensor, covariate_names, normalization_params)
    """
    # Load data
    covariate_df = load_covariate_file(covariate_path)
    fam_df = load_fam_file(fam_path)
    
    # Match and extract covariates
    covariate_matrix, covariate_names = match_covariates_to_fam(
        covariate_df, fam_df, binary_cols=binary_cols, categorical_cols=categorical_cols
    )
    
    # Normalize (mean-center quantitative variables only)
    normalized_matrix, norm_params = normalize_covariates(
        covariate_matrix, covariate_names, binary_cols=binary_cols, categorical_cols=categorical_cols
    )
    
    # Convert to tensor
    covariate_tensor = torch.from_numpy(normalized_matrix)
    
    logger.info(f"Covariates prepared: {covariate_tensor.shape} tensor with {len(covariate_names)} features")
    
    return covariate_tensor, covariate_names, norm_params

def save_covariate_metadata(output_path: str,
                           covariate_names: List[str],
                           normalization_params: Dict,
                           fam_path: str = None):
    """Save covariate metadata for later use during generation.
    
    Args:
        output_path: Path to save metadata
        covariate_names: List of covariate feature names
        normalization_params: Parameters used for normalization
        fam_path: Optional path to fam file used for training
    """
    import json
    
    metadata = {
        'covariate_names': covariate_names,
        'n_covariates': len(covariate_names),
        'normalization_params': normalization_params,
        'fam_path': fam_path
    }
    
    # Convert numpy arrays to lists for JSON serialization
    if 'means' in normalization_params:
        metadata['normalization_params']['means'] = normalization_params['means'].tolist()
    if 'normalized_flags' in normalization_params:
        metadata['normalization_params']['normalized_flags'] = normalization_params['normalized_flags'].tolist()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Covariate metadata saved to: {output_path}")

def load_covariate_metadata(metadata_path: str) -> Tuple[List[str], Dict]:
    """Load covariate metadata saved during training.
    
    Args:
        metadata_path: Path to metadata JSON file
        
    Returns:
        Tuple of (covariate_names, normalization_params)
    """
    import json
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    covariate_names = metadata['covariate_names']
    norm_params = metadata['normalization_params']
    
    # Convert lists back to numpy arrays
    if 'means' in norm_params:
        norm_params['means'] = np.array(norm_params['means'])
    if 'normalized_flags' in norm_params:
        norm_params['normalized_flags'] = np.array(norm_params['normalized_flags'], dtype=bool)
    
    logger.info(f"Loaded covariate metadata: {len(covariate_names)} features")
    return covariate_names, norm_params

def sample_training_covariates(covariate_path: str,
                             fam_path: str,
                             num_samples: int,
                             binary_cols: List[str] = None,
                             categorical_cols: List[str] = None,
                             random_seed: int = None) -> Tuple[torch.Tensor, np.ndarray]:
    """Sample covariate profiles from actual training data.
    
    Args:
        covariate_path: Path to covariate CSV file
        fam_path: Path to training fam file  
        num_samples: Number of samples to generate
        binary_cols: List of binary variable column names
        categorical_cols: List of categorical variable column names
        random_seed: Random seed for reproducible sampling
        
    Returns:
        Tuple of (sampled_covariates_tensor, original_covariate_matrix)
        - sampled_covariates_tensor: (num_samples, n_covariates) normalized for model
        - original_covariate_matrix: (num_samples, n_covariates) original values for saving
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    logger.info(f"Sampling {num_samples} covariate profiles from training data...")
    
    # Load and prepare training covariates
    covariate_tensor, covariate_names, norm_params = prepare_covariates_for_training(
        covariate_path=covariate_path,
        fam_path=fam_path,
        binary_cols=binary_cols,
        categorical_cols=categorical_cols
    )
    
    n_train_samples = covariate_tensor.shape[0]
    logger.info(f"Training data has {n_train_samples} covariate profiles available")
    
    # Sample indices with replacement
    sample_indices = np.random.choice(n_train_samples, size=num_samples, replace=True)
    
    # Get sampled normalized covariates for model
    sampled_normalized = covariate_tensor[sample_indices]
    
    # Get original (unnormalized) covariates for saving
    # We need to reconstruct the original values
    original_matrix = covariate_tensor.numpy().copy()
    
    # Add back means for quantitative variables
    means = norm_params['means']
    normalized_flags = norm_params['normalized_flags']
    
    for i in range(len(covariate_names)):
        if normalized_flags[i]:  # If this variable was normalized
            original_matrix[:, i] += means[i]  # Add back the mean
    
    sampled_original = original_matrix[sample_indices]
    
    logger.info(f"Sampled covariate profiles: {sampled_normalized.shape}")
    logger.info(f"Sample indices range: {sample_indices.min()} to {sample_indices.max()}")
    
    return sampled_normalized, sampled_original 