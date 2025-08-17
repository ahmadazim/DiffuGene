#!/usr/bin/env python
"""
Convert decoded VAE tensor to PLINK BED format.

Simple script that:
1. Loads decoded tensor from decode_vae_latents.py 
2. Concatenates blocks and converts continuous values to discrete genotypes (0, 1, 2)
3. Copies provided BIM and FAM files (since SNP/sample info doesn't change)
4. Writes PLINK BED file
"""

import argparse
import os
import sys
import subprocess
import numpy as np
import torch
import shutil
from tqdm import tqdm
from pyplink import PyPlink

# Add the src directory to the path to import DiffuGene modules
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

try:
    from DiffuGene.utils import setup_logging, get_logger, ensure_dir_exists
except ImportError:
    # Fallback logging setup
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    def get_logger(name):
        return logging.getLogger(name)
    
    def ensure_dir_exists(path):
        os.makedirs(path, exist_ok=True)
    
    def setup_logging():
        pass

logger = get_logger(__name__)

def load_decoded_data(decoded_file):
    """Load decoded SNP data from VAE output file."""
    logger.info(f"Loading decoded data from: {decoded_file}")
    
    data = torch.load(decoded_file, map_location='cpu', weights_only=False)
    
    if isinstance(data, dict):
        reconstructed_snps = data['reconstructed_snps']
        n_samples = data['n_samples']
        n_blocks = data['n_blocks']
        block_snp_counts = data.get('block_snp_counts', [snps.shape[1] for snps in reconstructed_snps])
    else:
        # Fallback for simpler format
        reconstructed_snps = data
        n_samples = reconstructed_snps[0].shape[0] if len(reconstructed_snps) > 0 else 0
        n_blocks = len(reconstructed_snps)
        block_snp_counts = [snps.shape[1] for snps in reconstructed_snps]
    
    logger.info(f"Loaded {n_blocks} blocks with {n_samples} samples")
    logger.info(f"SNPs per block: {block_snp_counts[:20]}")
    logger.info(f"Total SNPs: {sum(block_snp_counts)}")
    
    return reconstructed_snps, n_samples, n_blocks, block_snp_counts


# def write_bed_from_tensor(reconstructed_snps, bim_file, fam_file, output_prefix):
#     """
#     Simple function to write BED file from decoded tensor.
    
#     Args:
#         reconstructed_snps: List of tensors with reconstructed SNP values
#         bim_file: Existing BIM file to copy
#         fam_file: Existing FAM file to copy  
#         output_prefix: Output prefix for PLINK files
#     """
#     logger.info("Converting decoded tensor to PLINK BED format...")
    
#     # 1. Concatenate all blocks into one matrix
#     logger.info("Concatenating blocks...")
#     all_blocks = torch.cat(reconstructed_snps, dim=1)  # (n_samples, total_snps)
#     n_samples, n_snps = all_blocks.shape
    
#     logger.info(f"Processing: {n_samples} samples × {n_snps} SNPs")
    
#     # 2. Copy BIM and FAM files
#     shutil.copy2(bim_file, f"{output_prefix}.bim")
#     # shutil.copy2(fam_file, f"{output_prefix}.fam") # already exists
#     logger.info(f"Copied BIM and FAM files")
    
#     # 3. Write PED file (convert on-the-fly to save memory)
#     logger.info("Writing PED file with on-the-fly conversion...")
    
#     # Create temporary PED file
#     ped_file = f"{output_prefix}_temp.ped"
    
#     # Read sample IDs from FAM file
#     sample_ids = []
#     with open(fam_file, 'r') as f:
#         for line in f:
#             parts = line.strip().split()
#             sample_ids.append(parts[1])  # IID
    
#     # Convert to numpy once
#     all_blocks_np = all_blocks.cpu().numpy()
    
#     # Track genotype counts for summary
#     geno_counts = np.zeros(3, dtype=int)
    
#     # Write PED file with vectorized operations
#     buffer_size = 10000
#     buffer = []
    
#     # Pre-compute allele mappings for speed
#     allele_map = {0: "\tA\tA", 1: "\tA\tT", 2: "\tT\tT"}
    
#     with open(ped_file, 'w') as f:
#         for i, sample_id in enumerate(tqdm(sample_ids, desc="Writing PED")):
#             # Vectorized round/clip for entire sample
#             sample_genos = np.clip(np.round(all_blocks_np[i, :]), 0, 2).astype(int)
            
#             # Update counts
#             geno_counts += np.bincount(sample_genos, minlength=3)
            
#             # Build line with vectorized allele conversion
#             line_parts = [f"{sample_id}\t{sample_id}\t0\t0\t0\t-9"]
#             line_parts.extend([allele_map.get(g, "\t0\t0") for g in sample_genos])
            
#             buffer.append("".join(line_parts) + "\n")
            
#             # Write buffer when full
#             if len(buffer) >= buffer_size:
#                 f.writelines(buffer)
#                 buffer = []
        
#         # Write any remaining buffer
#         if buffer:
#             f.writelines(buffer)
    
#     # 4. Convert PED to BED using PLINK
#     logger.info("Converting PED to BED...")
    
#     cmd = [
#         "plink",
#         "--ped", ped_file,
#         "--map", f"{output_prefix}.bim",  # Use BIM as MAP (same format for our purposes)
#         "--make-bed",
#         "--out", output_prefix
#     ]
    
#     result = subprocess.run(cmd, capture_output=True, text=True)
    
#     if result.returncode == 0:
#         logger.info("Successfully created PLINK BED file")
#         os.remove(ped_file)  # Clean up
#     else:
#         logger.error(f"PLINK conversion failed: {result.stderr}")
    
#     # 5. Log summary
#     total_genos = geno_counts.sum()
#     logger.info(f"=== SUMMARY ===")
#     logger.info(f"Samples: {n_samples}, SNPs: {n_snps}")
#     logger.info(f"Genotype distribution: 0={geno_counts[0]/total_genos:.1%}, 1={geno_counts[1]/total_genos:.1%}, 2={geno_counts[2]/total_genos:.1%}")
    
#     return output_prefix

def main():
    """Main entry point for the aggregation script."""
    parser = argparse.ArgumentParser(
        description="Convert decoded VAE tensor to PLINK BED format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python aggregate_decoded_blocks.py \\
    --decoded-file /path/to/decoded_snps.pt \\
    --bim-file /path/to/original.bim \\
    --fam-file /path/to/original.fam \\
    --output-prefix /path/to/output/reconstructed_data
        """
    )
    
    # Required arguments
    parser.add_argument("--decoded-file", type=str, required=True,
                       help="Path to decoded SNPs file (.pt) from decode_vae_latents.py")
    parser.add_argument("--bim-file", type=str, required=True,
                       help="Original BIM file to copy (SNP info doesn't change)")
    parser.add_argument("--fam-file", type=str, required=True,
                       help="Original FAM file to copy (sample info doesn't change)")
    parser.add_argument("--output-prefix", type=str, required=True,
                       help="Output prefix for PLINK files (will create .bed/.bim/.fam)")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    logger.info("Converting decoded tensor to PLINK format")
    
    # Validate input files
    if not os.path.exists(args.decoded_file):
        parser.error(f"Decoded file not found: {args.decoded_file}")
    if not os.path.exists(args.bim_file):
        parser.error(f"BIM file not found: {args.bim_file}")
    if not os.path.exists(args.fam_file):
        parser.error(f"FAM file not found: {args.fam_file}")
    
    # Create output directory
    output_dir = os.path.dirname(args.output_prefix)
    if output_dir:
        ensure_dir_exists(output_dir)
    
    # Load decoded data
    reconstructed_snps, n_samples, n_blocks, block_snp_counts = load_decoded_data(args.decoded_file)
    all_blocks = torch.cat(reconstructed_snps, dim=1)
    del reconstructed_snps 
    genotypes = np.clip(np.round(all_blocks), 0, 2).astype(int)
    
    # # Convert to BED format
    # output_prefix = write_bed_from_tensor(
    #     reconstructed_snps=reconstructed_snps,
    #     bim_file=args.bim_file,
    #     fam_file=args.fam_file,
    #     output_prefix=args.output_prefix
    # )

    with PyPlink(args.output_prefix, mode="w", bed_format="SNP-major") as pl:
        pl.write_genotypes(genotypes)
    
    logger.info(f"Conversion completed successfully!")
    logger.info(f"Output files: {args.output_prefix}.bed/.bim/.fam")

if __name__ == "__main__":
    main()



from pandas_plink import write_plink1_bin
import numpy as np
import xarray as xr
import torch 
import pandas as pd
from tqdm import tqdm


DECOODED = "/n/home03/ahmadazim/WORKING/genGen/DiffuGene/../UKB/synthData/VAE_embeddings/UKB_allchr_unrel_britishWhite_genome_encoder_train_IDPsubset_all_chr_VAE_decoded_4PCscale_batch1.pt"
bim_path = "/n/home03/ahmadazim/WORKING/genGen/DiffuGene/../UKB/genomic_data/geneticBinary/ukb_allchr_unrel_britishWhite.bim"
fam_path = "/n/home03/ahmadazim/WORKING/genGen/UKB/synthData//UKB_allchr_unrel_britishWhite_genome_encoder_train_IDPsubset_batch1.fam"
OUTPUT_PREFIX = "/n/home03/ahmadazim/WORKING/genGen/DiffuGene/../UKB/synthData/UKB_allchr_unrel_britishWhite_genome_encoder_train_IDPsubset_PCVAEdecoded_batch1"

# load output_data dict
output = torch.load(DECOODED, map_location="cpu")
blocks = output['reconstructed_snps']
n_samples = output['n_samples']

# Create a memmap for full genotype matrix
geno = np.concatenate([b.numpy() for b in blocks], axis=1) 
geno = np.rint(geno)
geno = np.clip(geno, 0, 2).astype(np.float32)

# Read .fam and .bim to get sample and variant coords
fam = pd.read_csv(
    fam_path, delim_whitespace=True, header=None,
    names=['chrom_fam','fid','iid','pid','mid','sex','phen']  # fam has 7 cols
)
bim = pd.read_csv(
    bim_path, delim_whitespace=True, header=None,
    names=['chrom','snp','cm','pos','a1','a2']
)
# subset bim to only include chr 1-22
bim = bim[bim['chrom'].isin(range(1,23))]

# Wrap in xarray
G = xr.DataArray(
    geno,
    dims=("sample","variant"),
    coords={
        "fid":   ("sample", fam['fid'].values),
        "iid":   ("sample", fam['iid'].values),
        "snp":   ("variant", bim['snp'].values),
        "chrom": ("variant", bim['chrom'].values),
        "pos":   ("variant", bim['pos'].values),
        "a1":    ("variant", bim['a1'].values),
        "a2":    ("variant", bim['a2'].values),
    }
)

# Write out PLINK1 binary trio
write_plink1_bin(
    G,
    OUTPUT_PREFIX+".bed",
    bim=bim_path,
    fam=fam_path,
    verbose=True
)