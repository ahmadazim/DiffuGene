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

def run_plink_ld_blocks(args, logger):
    """Run PLINK to infer LD blocks if needed."""
    block_file = f"{args.block_folder}/{args.basename}_chr{args.chrNo}_blocks.blocks.det"
    
    if not os.path.exists(block_file):
        ensure_dir_exists(args.block_folder)
        logger.info(f"Block definition file not found; running PLINK to infer LD blocks for chromosome {args.chrNo}")
        
        cmd = [
            "plink",
            "--bfile", f"{args.genetic_binary_folder}/{args.basename}",
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
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"PLINK failed: {result.stderr}")
            raise RuntimeError("PLINK LD block inference failed")
        
        logger.info(f"PLINK LD block inference completed for chromosome {args.chrNo}")
    else:
        logger.info(f"Block definition file found for chromosome {args.chrNo}.")
    
    return block_file

def recode_blocks(args, snpfiles, logger):
    """Recode individual blocks using PLINK."""
    ensure_dir_exists(args.recoded_block_folder)
    
    for snpfile in tqdm(snpfiles, desc="Recoding blocks"):
        blockNo = re.search(r'block(\d+)', snpfile).group(1)
        
        cmd = [
            "plink",
            "--bfile", f"{args.genetic_binary_folder}/{args.basename}",
            "--chr", str(args.chrNo),
            "--extract", snpfile,
            "--recodeA",
            "--out", f"{args.recoded_block_folder}/{args.basename}_chr{args.chrNo}_block{blockNo}_recodeA"
        ]
        
        with open(os.devnull, 'w') as devnull:
            subprocess.run(cmd, stdout=devnull, stderr=devnull)



def main(args):
    logger = setup_logging()
    logger.info(f"Starting block embedding prep: basename={args.basename}, chrNo={args.chrNo}")

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
    
    args = parser.parse_args()
    main(args)
