#!/usr/bin/env python

import os
import sys
import glob
import numpy as np
import torch

from ..utils import read_raw, get_logger
from .pca import PCA_Block

logger = get_logger(__name__)

def main(chrNo=None, blockNo=None, config_paths=None):
    # Handle command-line arguments if not provided directly
    if chrNo is None or blockNo is None:
        if len(sys.argv) != 3:
            print("Usage: python fit_pca.py <chrNo> <blockNo>")
            print("Note: This script requires config_paths to be provided when called programmatically")
            sys.exit(1)
        chrNo, blockNo = sys.argv[1], sys.argv[2]
    
    # Config paths are required
    if config_paths is None:
        raise ValueError("config_paths parameter is required. This script must be called with configuration parameters.")
    
    recoded_dir = config_paths['recoded_dir']
    output_dirs = config_paths['output_dirs']
    basename = config_paths['basename']
    k = config_paths.get('pca_k')

    # find raw file
    pattern = os.path.join(
        recoded_dir,
        f"{basename}_chr{chrNo}_block{blockNo}_*.raw"
    )
    matches = glob.glob(pattern)
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one .raw, found {len(matches)} for {pattern}")
    raw_path = matches[0]

    # load & impute
    rec = read_raw(raw_path)
    X = rec.impute().get_variants()  # (n_samples, n_snps)
    n_samples, n_snps = X.shape

    # PCA parameters
    pca = PCA_Block(k)

    # fit, encode, decode
    scores = pca.fit(X)                   # (n_samples, k) - padded if actual_k < k
    X_recon = pca.decode(scores)          # (n_samples, n_snps)

    # compute errors
    mse = np.mean((X.astype(float) - X_recon) ** 2)
    X_round = np.clip(np.rint(X_recon), 0, 2).astype(int)
    accuracy = np.mean(X_round == X)

    # prepare outputs
    for d in output_dirs.values():
        os.makedirs(d, exist_ok=True)
        
    # Create metadata directory if it doesn't exist
    metadata_dir = output_dirs.get("metadata", os.path.join(os.path.dirname(list(output_dirs.values())[0]), "metadata"))
    os.makedirs(metadata_dir, exist_ok=True)

    prefix = f"{basename}_chr{chrNo}_block{blockNo}"
    torch.save(scores,      os.path.join(output_dirs["embeddings"],       f"{prefix}_embeddings.pt"))
    torch.save(X_recon,     os.path.join(output_dirs["reconstructed"], f"{prefix}_pca_reconstructed.pt"))
    torch.save(pca.components_.T, os.path.join(output_dirs["loadings"],    f"{prefix}_pca_loadings.pt"))
    torch.save(pca.means,   os.path.join(output_dirs["means"],       f"{prefix}_pca_means.pt"))
    
    # Save PCA metadata including actual_k
    pca_metadata = {
        'k': pca.k,               # Target number of components  
        'actual_k': pca.actual_k, # Actual number of components used
        'n_snps': n_snps,         # Number of SNPs in this block
        'n_samples': n_samples    # Number of samples
    }
    torch.save(pca_metadata, os.path.join(metadata_dir, f"{prefix}_pca_metadata.pt"))

    return {
        'mse': mse,
        'accuracy': accuracy,
        'n_samples': n_samples,
        'n_snps': n_snps,
        'k': pca.k,
        'actual_k': pca.actual_k,
        'block_no': blockNo
    }

if __name__ == "__main__":
    print("Error: This script cannot be run standalone.")
    print("Use the DiffuGene pipeline instead:")
    print("  python -m DiffuGene.pipeline --steps block_embed")
    print("  or")
    print("  diffugene --steps block_embed")
    sys.exit(1)
