#!/usr/bin/env python

import argparse
import os
import glob
import pandas as pd
from tqdm import tqdm
import re
import subprocess

from ..utils import (
    setup_logging, get_logger, load_blocks_for_chr, 
    create_snplist_files, ensure_dir_exists
)

def calculate_allele_frequencies(args, logger):
    """Calculate allele frequencies if they don't exist."""
    global_bfile = getattr(args, 'global_bfile', f"{args.genetic_binary_folder}/{args.basename}")
    freq_file = f"{global_bfile}.frq"
    
    if os.path.exists(freq_file):
        logger.info(f"Frequency file already exists: {freq_file}")
        return freq_file
    
    logger.info(f"Calculating allele frequencies for {global_bfile}...")
    
    cmd = [
        "plink",
        "--bfile", global_bfile,
        "--freq",
        "--out", global_bfile
    ]
    
    # Add --keep if using a subset fam file
    if hasattr(args, 'keep_fam_file') and args.keep_fam_file:
        cmd.extend(["--keep", args.keep_fam_file])
        logger.info(f"Calculating frequencies using subset: {args.keep_fam_file}")
    
    logger.info(f"Running frequency calculation: {' '.join(cmd)}")
    
    # Just run the command - simple and direct
    return_code = os.system(' '.join(cmd))
    
    if return_code != 0:
        logger.error(f"PLINK frequency calculation failed with return code {return_code}")
        raise RuntimeError("PLINK frequency calculation failed")
    
    if not os.path.exists(freq_file):
        logger.error(f"PLINK completed but did not create frequency file: {freq_file}")
        raise RuntimeError("PLINK did not create frequency file")
    
    logger.info(f"Frequency calculation completed: {freq_file}")
    return freq_file

def run_plink_ld_blocks(args, logger):
    """Run PLINK to infer LD blocks if needed."""
    block_file = f"{args.block_folder}/{args.basename}_chr{args.chrNo}_blocks.blocks.det"
    
    if not os.path.exists(block_file):
        ensure_dir_exists(args.block_folder)
        logger.info(f"Block definition file not found; running PLINK to infer LD blocks for chromosome {args.chrNo}")
        
        # Base PLINK command using the original global bfile
        global_bfile = getattr(args, 'global_bfile', f"{args.genetic_binary_folder}/{args.basename}")
        
        cmd = [
            "plink",
            "--bfile", global_bfile,
            "--blocks", "no-pheno-req", "no-small-max-span",
            "--blocks-max-kb", str(args.plink_max_kb),
            "--blocks-min-maf", str(args.plink_min_maf),
            "--blocks-strong-lowci", str(args.plink_strong_lowci),
            "--blocks-strong-highci", str(args.plink_strong_highci),
            "--blocks-recomb-highci", str(args.plink_recomb_highci),
            "--blocks-inform-frac", str(args.plink_inform_frac),
            "--chr", str(args.chrNo),
            "--out", f"{args.block_folder}/{args.basename}_chr{args.chrNo}_blocks"
        ]
        
        # Add --keep if using a subset fam file (multi-dataset mode)
        if hasattr(args, 'keep_fam_file') and args.keep_fam_file:
            cmd.extend(["--keep", args.keep_fam_file])
            logger.info(f"Using --keep {args.keep_fam_file} to subset individuals")
        
        # Add --read-freq if frequency file exists
        freq_file = getattr(args, 'freq_file', f"{global_bfile}.frq")
        if os.path.exists(freq_file):
            cmd.extend(["--read-freq", freq_file])
            logger.info(f"Using pre-calculated frequencies from {freq_file}")
        else:
            logger.info(f"Frequency file {freq_file} not found, frequencies will be calculated")
        
        logger.info(f"Running PLINK command: {' '.join(cmd)}")
        
        # Just run the command - simple and direct
        return_code = os.system(' '.join(cmd))
        
        if return_code != 0:
            logger.error(f"PLINK failed with return code {return_code}")
            raise RuntimeError("PLINK LD block inference failed")
        
        # Basic file validation
        if not os.path.exists(block_file):
            logger.error(f"PLINK completed but did not create block definition file: {block_file}")
            raise RuntimeError(f"PLINK did not create block definition file for chromosome {args.chrNo}")
        
        if os.path.getsize(block_file) == 0:
            logger.error(f"PLINK created empty block definition file for chromosome {args.chrNo}")
            raise RuntimeError(f"No LD blocks found for chromosome {args.chrNo} - file is empty")
        
        logger.info(f"PLINK LD block inference completed for chromosome {args.chrNo}")
    else:
        logger.info(f"Block definition file found for chromosome {args.chrNo}.")
        
        # Also validate existing files
        if os.path.getsize(block_file) == 0:
            logger.error(f"Existing block definition file is empty for chromosome {args.chrNo}")
            raise RuntimeError(f"Block definition file exists but is empty for chromosome {args.chrNo}")
    
    return block_file

def recode_blocks(args, snpfiles, logger):
    """Recode individual blocks using PLINK."""
    ensure_dir_exists(args.recoded_block_folder)
    
    # Use the same global bfile approach as in LD block inference
    global_bfile = getattr(args, 'global_bfile', f"{args.genetic_binary_folder}/{args.basename}")
    freq_file = getattr(args, 'freq_file', f"{global_bfile}.frq")
    
    for snpfile in tqdm(snpfiles, desc="Recoding blocks"):
        blockNo = re.search(r'block(\d+)', snpfile).group(1)
        
        # Base PLINK command
        cmd = [
            "plink",
            "--bfile", global_bfile,
            "--chr", str(args.chrNo),
            "--extract", snpfile,
            "--recodeA",
            "--out", f"{args.recoded_block_folder}/{args.basename}_chr{args.chrNo}_block{blockNo}_recodeA"
        ]
        
        # Add --keep if using a subset fam file (multi-dataset mode)
        if hasattr(args, 'keep_fam_file') and args.keep_fam_file:
            cmd.extend(["--keep", args.keep_fam_file])
        
        # Add --read-freq if frequency file exists
        if os.path.exists(freq_file):
            cmd.extend(["--read-freq", freq_file])
        
        # Just run the command - simple and direct
        logger.debug(f"Recoding block {blockNo}: {' '.join(cmd)}")
        return_code = os.system(' '.join(cmd))
        
        if return_code != 0:
            logger.warning(f"PLINK recoding failed for block {blockNo} with return code {return_code}")



def main(args):
    logger = setup_logging()
    logger.info(f"Starting block embedding prep: basename={args.basename}, chrNo={args.chrNo}")

    # Step 0: Calculate allele frequencies if needed
    freq_file = calculate_allele_frequencies(args, logger)
    args.freq_file = freq_file  # Store for use in other functions

    # Step 1: Infer LD blocks with PLINK
    block_file = run_plink_ld_blocks(args, logger)
    
    # Step 2: Parse blocks and create SNP list files
    logger.info(f"Parsing block definition file '{block_file}'")
    LD_blocks = load_blocks_for_chr(block_file, args.chrNo)
    logger.info(f"{len(LD_blocks)} LD blocks parsed")
    
    ensure_dir_exists(args.snplist_folder)
    create_snplist_files(LD_blocks, args.snplist_folder, args.basename, args.chrNo)
    
    # Step 3: Recode individual blocks
    pattern = f"{args.snplist_folder}/{args.basename}_chr{args.chrNo}_block*.snplist"
    snpfiles = glob.glob(pattern)
    logger.info(f"Found {len(snpfiles)} SNP list files.")
    
    recode_blocks(args, snpfiles, logger)
    
    logger.info("Block preparation completed successfully")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prep data for block embedding.")
    parser.add_argument("--basename", type=str, required=True)
    parser.add_argument("--chrNo", type=int, required=True)
    parser.add_argument("--genetic-binary-folder", type=str, required=True)
    parser.add_argument("--block-folder", type=str, required=True)
    parser.add_argument("--recoded-block-folder", type=str, required=True)
    parser.add_argument("--embedding-folder", type=str, required=True)
    parser.add_argument("--snplist-folder", type=str, required=True)
    
    # PLINK LD block parameters
    parser.add_argument("--plink-max-kb", type=int, default=100000, help="PLINK --blocks-max-kb parameter")
    parser.add_argument("--plink-min-maf", type=float, default=0.01, help="PLINK --blocks-min-maf parameter")
    parser.add_argument("--plink-strong-lowci", type=float, default=0.5001, help="PLINK --blocks-strong-lowci parameter")
    parser.add_argument("--plink-strong-highci", type=float, default=0.8301, help="PLINK --blocks-strong-highci parameter")
    parser.add_argument("--plink-recomb-highci", type=float, default=0.60, help="PLINK --blocks-recomb-highci parameter")
    parser.add_argument("--plink-inform-frac", type=float, default=0.90, help="PLINK --blocks-inform-frac parameter")
    
    # Multi-dataset support
    parser.add_argument("--global-bfile", type=str, help="Global PLINK bfile to use (overrides default)")
    parser.add_argument("--keep-fam-file", type=str, help="Fam file for --keep to subset individuals")
    parser.add_argument("--freq-file", type=str, help="Pre-calculated frequency file to use with --read-freq")
    
    args = parser.parse_args()
    main(args)
