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


def read_bim_file(bim_file: str, chr_no: int) -> pd.DataFrame:
    """Read PLINK BIM file and filter for specific chromosome.
    
    Args:
        bim_file: Path to .bim file
        chr_no: Chromosome number to filter
        
    Returns:
        DataFrame with columns: CHR, SNP, CM, BP, A1, A2
    """
    if not os.path.exists(bim_file):
        raise FileNotFoundError(f"BIM file not found: {bim_file}")
    
    # PLINK BIM format: CHR, SNP, CM, BP, A1, A2
    bim_df = pd.read_csv(bim_file, sep='\t', header=None, 
                        names=['CHR', 'SNP', 'CM', 'BP', 'A1', 'A2'])
    
    # Filter for the specific chromosome
    bim_chr = bim_df[bim_df['CHR'] == chr_no].copy()
    
    return bim_chr


def collect_snps_from_snplists(snplist_dir: str, basename: str, chr_no: int) -> set:
    """Collect all SNPs from existing snplist files for a chromosome.
    
    Args:
        snplist_dir: Directory containing SNP list files
        basename: Base name for SNP list files
        chr_no: Chromosome number
        
    Returns:
        Set of SNP IDs found in all snplist files
    """
    snps_in_blocks = set()
    
    # Find all snplist files for this chromosome
    pattern = os.path.join(snplist_dir, f"{basename}_chr{chr_no}_block*.snplist")
    snplist_files = glob.glob(pattern)
    
    for snplist_file in snplist_files:
        with open(snplist_file, 'r') as f:
            for line in f:
                snp = line.strip()
                if snp:  # Skip empty lines
                    snps_in_blocks.add(snp)
    
    return snps_in_blocks


def find_nearest_block(snp_bp: int, blocks: List[Block]) -> int:
    """Find the nearest LD block for a given SNP position.
    
    Args:
        snp_bp: Base pair position of the SNP
        blocks: List of Block objects
        
    Returns:
        Index of the nearest block (1-based)
    """
    if not blocks:
        raise ValueError("No blocks provided")
    
    min_distance = float('inf')
    nearest_block_idx = 1
    
    for idx, block in enumerate(blocks, start=1):
        # Calculate distance to block
        if snp_bp < block.bp1:
            # SNP is before block start
            distance = block.bp1 - snp_bp
        elif snp_bp > block.bp2:
            # SNP is after block end
            distance = snp_bp - block.bp2
        else:
            # SNP is within block boundaries (should not happen for missing SNPs)
            distance = 0
        
        if distance < min_distance:
            min_distance = distance
            nearest_block_idx = idx
    
    return nearest_block_idx


def assign_missing_snps_to_blocks(bim_file: str, snplist_dir: str, basename: str, 
                                chr_no: int, blocks: List[Block]) -> dict:
    """Assign SNPs missing from LD blocks to the nearest block.
    
    Args:
        bim_file: Path to PLINK BIM file
        snplist_dir: Directory containing SNP list files
        basename: Base name for files
        chr_no: Chromosome number
        blocks: List of Block objects
        
    Returns:
        Dictionary with assignment statistics
    """
    # Read BIM file for this chromosome
    bim_df = read_bim_file(bim_file, chr_no)
    all_snps_bim = set(bim_df['SNP'].tolist())
    
    # Collect SNPs already in LD blocks
    snps_in_blocks = collect_snps_from_snplists(snplist_dir, basename, chr_no)
    
    # Find missing SNPs
    missing_snps = all_snps_bim - snps_in_blocks
    
    if not missing_snps:
        return {
            'total_snps_bim': len(all_snps_bim),
            'snps_in_blocks': len(snps_in_blocks),
            'missing_snps': 0,
            'assigned_snps': 0,
            'assignments': {}
        }
    
    # Create a mapping from SNP to position for missing SNPs
    missing_snp_positions = {}
    for _, row in bim_df.iterrows():
        if row['SNP'] in missing_snps:
            missing_snp_positions[row['SNP']] = row['BP']
    
    # Assign each missing SNP to nearest block
    assignments = {}
    for snp, bp in missing_snp_positions.items():
        nearest_block_idx = find_nearest_block(bp, blocks)
        if nearest_block_idx not in assignments:
            assignments[nearest_block_idx] = []
        assignments[nearest_block_idx].append(snp)
    
    # Update the snplist files
    for block_idx, assigned_snps in assignments.items():
        snplist_file = os.path.join(snplist_dir, f"{basename}_chr{chr_no}_block{block_idx}.snplist")
        
        # Append the assigned SNPs to the existing snplist file
        with open(snplist_file, 'a') as f:
            for snp in assigned_snps:
                f.write(f"{snp}\n")
    
    return {
        'total_snps_bim': len(all_snps_bim),
        'snps_in_blocks': len(snps_in_blocks),
        'missing_snps': len(missing_snps),
        'assigned_snps': sum(len(snps) for snps in assignments.values()),
        'assignments': assignments
    }
