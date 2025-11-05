#!/usr/bin/env python
"""Create per-chromosome H5 caches for homogenized AE training/encoding."""

import argparse
import os
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create batched H5 caches for a single chromosome",
    )
    parser.add_argument("--bfile", required=True, help="PLINK bfile prefix or .bed path")
    parser.add_argument("--bim", required=True, help="Path to BIM file (used for SNP info)")
    parser.add_argument("--fam", required=True, help="Path to FAM file (controls IID ordering)")
    parser.add_argument("--out-dir", required=True, help="Output directory for batched H5 caches")
    parser.add_argument("--chromosome", required=True, type=int, help="Chromosome to process (1-22)")
    parser.add_argument("--batch-size", type=int, default=12000, help="Individuals per batch when writing H5 caches")
    return parser.parse_args()


def main() -> int:
    # Allow invoking from repository root without installing as a package
    this_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(this_dir, ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from DiffuGene.VAEembed.encode_batched_AE import create_batched_h5

    args = parse_args()
    if args.chromosome < 1 or args.chromosome > 22:
        raise ValueError("--chromosome must be in [1, 22]")

    per_chr = create_batched_h5(
        bfile=args.bfile,
        fam=args.fam,
        bim=args.bim,
        out_dir=args.out_dir,
        chromosomes=[int(args.chromosome)],
        batch_size=int(args.batch_size),
    )

    num_batches = per_chr.get(int(args.chromosome), 0)
    print(
        f"[create_h5_for_chr] Chromosome {args.chromosome} completed | batches={num_batches} | "
        f"out_dir={os.path.join(args.out_dir, f'chr{int(args.chromosome)}')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

