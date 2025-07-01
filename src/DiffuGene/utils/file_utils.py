import os
import glob
import re
from typing import List, Optional, Tuple
import pandas as pd
from collections import namedtuple

# Define the Block namedtuple used in data preparation
Block = namedtuple("Block", ["chr", "bp1", "bp2", "snps"])


def find_raw_files(pattern: str) -> List[str]:
    """Find .raw files matching a pattern.
    
    Args:
        pattern: Glob pattern to match files
        
    Returns:
        List of file paths
        
    Raises:
        FileNotFoundError: If no files or multiple files found when expecting one
    """
    matches = glob.glob(pattern)
    return matches


def find_unique_raw_file(pattern: str) -> str:
    """Find exactly one .raw file matching a pattern.
    
    Args:
        pattern: Glob pattern to match files
        
    Returns:
        Single file path
        
    Raises:
        FileNotFoundError: If no files or multiple files found
    """
    matches = find_raw_files(pattern)
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one .raw file, found {len(matches)} for pattern: {pattern}")
    return matches[0]


def extract_block_number(filepath: str) -> int:
    """Extract block number from filepath.
    
    Args:
        filepath: Path containing block number pattern
        
    Returns:
        Block number as integer
    """
    filename = os.path.basename(filepath)
    match = re.search(r'block(\d+)', filename)
    if not match:
        raise ValueError(f"Could not extract block number from: {filepath}")
    return int(match.group(1))


def load_blocks_for_chr(fn: str, chr_no: int) -> List[Block]:
    """Parse PLINK .blocks.det for one chromosome into a list of Block objects.
    
    Args:
        fn: Path to .blocks.det file
        chr_no: Chromosome number
        
    Returns:
        List of Block namedtuples
    """
    if not os.path.exists(fn):
        raise FileNotFoundError(f"Block definition file not found: {fn}")
    
    blocks = []
    with open(fn) as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split()
            bp1 = int(parts[1])
            bp2 = int(parts[2])
            snplist = parts[5].split("|")
            blocks.append(Block(chr_no, bp1, bp2, snplist))
    return blocks


def create_snplist_files(blocks: List[Block], snp_dir: str, basename: str, chr_no: int) -> None:
    """Write one .snplist file per Block.
    
    Args:
        blocks: List of Block objects
        snp_dir: Output directory for SNP list files
        basename: Base name for output files
        chr_no: Chromosome number
    """
    os.makedirs(snp_dir, exist_ok=True)
    for idx, blk in enumerate(blocks, start=1):
        fname = os.path.join(
            snp_dir,
            f"{basename}_chr{chr_no}_block{idx}.snplist"
        )
        with open(fname, "w") as out:
            for snp in blk.snps:
                out.write(snp + "\n")


def ensure_dir_exists(path: str) -> None:
    """Create directory if it doesn't exist.
    
    Args:
        path: Directory path to create
    """
    os.makedirs(path, exist_ok=True)


def get_sorted_files_by_block(pattern: str) -> List[str]:
    """Get files matching pattern, sorted by block number.
    
    Args:
        pattern: Glob pattern to match files
        
    Returns:
        List of file paths sorted by block number
    """
    files = glob.glob(pattern)
    return sorted(files, key=extract_block_number)
