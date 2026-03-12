#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from typing import Dict, List, Tuple

try:
    from DiffuGene.utils import setup_logging, get_logger, ensure_dir_exists
    from DiffuGene.utils.file_utils import read_bim_file
    from DiffuGene.VAEembed.latentAllocTokens import solve_token_allocation_milp, organize_token_solution
except Exception:
    this_dir = os.path.dirname(__file__)
    src_root = os.path.abspath(os.path.join(this_dir, "..", ".."))
    if src_root not in sys.path:
        sys.path.insert(0, src_root)
    from DiffuGene.utils import setup_logging, get_logger, ensure_dir_exists
    from DiffuGene.utils.file_utils import read_bim_file
    from DiffuGene.VAEembed.latentAllocTokens import solve_token_allocation_milp, organize_token_solution


logger = get_logger(__name__)


def get_chromosome_list(chrom_spec: List[str]) -> List[int]:
    if len(chrom_spec) == 1 and str(chrom_spec[0]).lower() == "all":
        return list(range(1, 23))
    return [int(x) for x in chrom_spec]


def compute_variant_counts_per_chromosome(bim_file: str, chromosomes: List[int]) -> Dict[int, int]:
    out: Dict[int, int] = {}
    for c in chromosomes:
        df = read_bim_file(bim_file, c)
        out[c] = int(df.shape[0])
    return out


def build_training_command(
    h5_dir: str,
    val_h5_dir: str,
    chromosome: int,
    latent_length: int,
    latent_dim: int,
    model_out_path: str,
    embed_dim: int,
    max_c: int,
    dropout: float,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device: str,
    grad_clip: float,
) -> List[str]:
    return [
        sys.executable, "-m", "DiffuGene.VAEembed.train_tok",
        "--h5-dir", h5_dir,
        "--val-h5-dir", val_h5_dir,
        "--chromosome", str(int(chromosome)),
        "--latent-length", str(int(latent_length)),
        "--latent-dim", str(int(latent_dim)),
        "--embed-dim", str(int(embed_dim)),
        "--max-c", str(int(max_c)),
        "--dropout", str(float(dropout)),
        "--epochs", str(int(epochs)),
        "--batch-size", str(int(batch_size)),
        "--lr", str(float(lr)),
        "--weight-decay", str(float(weight_decay)),
        "--device", device,
        "--grad-clip", str(float(grad_clip)),
        "--save-path", model_out_path,
    ]


def main() -> None:
    p = argparse.ArgumentParser(description="Orchestrate per-chromosome token-AE training with token allocation.")
    p.add_argument("--h5-dir", required=True, help="H5 cache directory root with chr*/batch*.h5")
    p.add_argument("--val-h5-dir", required=True, help="Validation H5 root with chr*/batch*.h5")
    p.add_argument("--bim", required=True, help="BIM file used to infer per-chromosome variant counts")
    p.add_argument("--output-dir", required=True, help="Directory for token-AE checkpoints and allocation files")
    p.add_argument("--chromosomes", nargs="+", type=str, default=["all"])

    p.add_argument("--total-tokens", type=int, default=4096, help="Total unified token budget")
    p.add_argument("--latent-dim", type=int, default=256, help="Token channel dimension D")
    p.add_argument("--min-tokens", type=int, default=1, help="Minimum power-of-two tokens per chromosome")
    p.add_argument("--max-tokens", type=int, default=None, help="Maximum power-of-two tokens per chromosome")

    p.add_argument("--embed-dim", type=int, default=8)
    p.add_argument("--max-c", type=int, default=5)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--grad-clip", type=float, default=1.0)

    p.add_argument("--use-slurm", action="store_true", help="Submit sbatch jobs instead of local execution")
    p.add_argument("--slurm-script", type=str, default=None, help="Path to SLURM wrapper script")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    setup_logging()
    ensure_dir_exists(args.output_dir)

    chromosomes = get_chromosome_list(args.chromosomes)
    counts = compute_variant_counts_per_chromosome(args.bim, chromosomes)
    logger.info(f"Variant counts per chromosome: {counts}")

    layout_json = os.path.join(args.output_dir, "tok_milp_layout.json")
    layout_csv = os.path.join(args.output_dir, "tok_milp_layout.csv")

    weights = [counts[c] for c in chromosomes]
    result = solve_token_allocation_milp(
        w=weights,
        total_tokens=int(args.total_tokens),
        min_tokens=int(args.min_tokens),
        max_tokens=(None if args.max_tokens is None else int(args.max_tokens)),
    )
    alloc_map = organize_token_solution(result)

    layout_records = []
    for idx, chrom in enumerate(chromosomes):
        rec = alloc_map[idx]
        model_path = os.path.join(args.output_dir, f"ae_tok_chr{chrom}.pt")
        layout_records.append(
            {
                "chromosome": int(chrom),
                "variant_count": int(counts[chrom]),
                "latent_length": int(rec["tokens"]),
                "token_start": int(rec["token_start"]),
                "token_end": int(rec["token_end"]),
                "latent_dim": int(args.latent_dim),
                "model_file": model_path,
            }
        )

    layout_payload = {
        "status": result.get("status"),
        "objective": float(result.get("objective", 0.0)),
        "total_tokens": int(args.total_tokens),
        "latent_dim": int(args.latent_dim),
        "chromosomes": [int(c) for c in chromosomes],
        "layout": layout_records,
    }
    with open(layout_json, "w") as f:
        json.dump(layout_payload, f, indent=2)
    with open(layout_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "chromosome", "variant_count", "latent_length",
                "token_start", "token_end", "latent_dim", "model_file",
            ],
        )
        writer.writeheader()
        for rec in layout_records:
            writer.writerow(rec)
    logger.info(f"Saved token allocation JSON: {layout_json}")
    logger.info(f"Saved token allocation CSV: {layout_csv}")

    jobs: List[Tuple[int, List[str], str]] = []
    for rec in layout_records:
        chrom = int(rec["chromosome"])
        model_out = str(rec["model_file"])
        if os.path.exists(model_out):
            logger.info(f"[SKIP] chr{chrom}: checkpoint exists at {model_out}")
            continue
        cmd = build_training_command(
            h5_dir=args.h5_dir,
            val_h5_dir=args.val_h5_dir,
            chromosome=chrom,
            latent_length=int(rec["latent_length"]),
            latent_dim=int(args.latent_dim),
            model_out_path=model_out,
            embed_dim=int(args.embed_dim),
            max_c=int(args.max_c),
            dropout=float(args.dropout),
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
            device=args.device,
            grad_clip=float(args.grad_clip),
        )
        jobs.append((chrom, cmd, model_out))

    logger.info(f"Planned {len(jobs)} token-AE training jobs.")
    for chrom, cmd, _ in jobs:
        if args.use_slurm:
            slurm_script = args.slurm_script
            if slurm_script is None:
                candidate = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                    "scripts", "slurm_train_ae_single.sh"
                )
                slurm_script = candidate if os.path.exists(candidate) else None
            if slurm_script is None:
                logger.warning("No SLURM wrapper found; falling back to local execution.")
            else:
                sbatch_cmd = ["sbatch", slurm_script] + cmd
                logger.info(f"[SLURM] chr{chrom}: {' '.join(sbatch_cmd)}")
                if not args.dry_run:
                    subprocess.run(sbatch_cmd, check=True)
                continue
        logger.info(f"[LOCAL] chr{chrom}: {' '.join(cmd)}")
        if not args.dry_run:
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

