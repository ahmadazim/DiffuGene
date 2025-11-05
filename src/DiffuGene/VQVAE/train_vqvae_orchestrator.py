#!/usr/bin/env python
import os
import sys
import argparse
import subprocess
from typing import List, Dict, Tuple

import numpy as np
import json
import csv

# Support both package and script execution
try:
    from DiffuGene.utils import setup_logging, get_logger, ensure_dir_exists
    from DiffuGene.utils.file_utils import read_bim_file
    from DiffuGene.VAEembed.latentAlloc_MILP import solve_quadtree_milp, organize_quadtree_solution
except Exception:
    this_dir = os.path.dirname(__file__)
    src_root = os.path.abspath(os.path.join(this_dir, "..", "..", ".."))
    if src_root not in sys.path:
        sys.path.insert(0, src_root)
    from DiffuGene.utils import setup_logging, get_logger, ensure_dir_exists
    from DiffuGene.utils.file_utils import read_bim_file
    from DiffuGene.VAEembed.latentAlloc_MILP import solve_quadtree_milp, organize_quadtree_solution


logger = get_logger(__name__)


def get_chromosome_list(chromosome_spec) -> List[int]:
    if str(chromosome_spec).lower() == "all":
        return list(range(1, 23))
    return [int(chromosome_spec)]


def compute_variant_counts_per_chromosome(bim_file: str, chromosomes: List[int]) -> Dict[int, int]:
    counts: Dict[int, int] = {}
    for c in chromosomes:
        df = read_bim_file(bim_file, c)
        counts[c] = int(df.shape[0])
    return counts


def build_training_command(
    h5_dir: str,
    bim_file: str,
    val_h5_dir: str,
    chromosome: int,
    latent_grid_dim: int,
    model_out_path: str,
    latent_dim: int,
    codebook_size: int,
    num_quantizers: int,
    beta_commit: float,
    hidden_1d_channels: int,
    hidden_2d_channels: int,
    layers_at_final: int,
    ema_decay: float,
    ld_lambda: float,
    maf_lambda: float,
    ld_window: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    grad_clip: float,
    beta_warmup_steps: int,
    # Early stop & MMD knobs forwarded
    plateau_min_rel_improve: float,
    plateau_patience: int,
    plateau_mse_threshold: float,
    mmd_start_epoch: int,
    mmd_lambda: float,
    mmd_warmup_epochs: int,
    mmd_max_samples: int,
) -> List[str]:
    # Use module invocation so PYTHONPATH resolves correctly
    cmd = [
        sys.executable, "-m", "DiffuGene.VAEembed.train_vqvae",
        "--h5-dir", h5_dir,
        "--bim", bim_file,
        "--val-h5-dir", val_h5_dir,
        "--chromosome", str(int(chromosome)),
        "--save-path", model_out_path,
        "--latent-dim", str(int(latent_dim)),
        "--codebook-size", str(int(codebook_size)),
        "--num-quantizers", str(int(num_quantizers)),
        "--beta-commit", str(float(beta_commit)),
        "--latent-grid-dim", str(int(latent_grid_dim)),
        "--hidden-1d-channels", str(int(hidden_1d_channels)),
        "--hidden-2d-channels", str(int(hidden_2d_channels)),
        "--layers-at-final", str(int(layers_at_final)),
        "--ema-decay", str(float(ema_decay)),
        "--ld-lambda", str(float(ld_lambda)),
        "--maf-lambda", str(float(maf_lambda)),
        "--ld-window", str(int(ld_window)),
        "--epochs", str(int(epochs)),
        "--batch-size", str(int(batch_size)),
        "--lr", str(float(lr)),
        "--device", device,
        "--grad-clip", str(float(grad_clip)),
        "--beta-warmup-steps", str(int(beta_warmup_steps)),
        "--plateau-min-rel-improve", str(float(plateau_min_rel_improve)),
        "--plateau-patience", str(int(plateau_patience)),
        "--plateau-mse-threshold", str(float(plateau_mse_threshold)),
        "--mmd-start-epoch", str(int(mmd_start_epoch)),
        "--mmd-lambda", str(float(mmd_lambda)),
        "--mmd-warmup-epochs", str(int(mmd_warmup_epochs)),
        "--mmd-max-samples", str(int(mmd_max_samples)),
    ]
    return cmd


def main():
    p = argparse.ArgumentParser(description="Orchestrate per-chromosome VQ-VAE training with MILP-based tile allocation")
    p.add_argument("--h5-dir", required=True, help="H5 cache directory root (with chr*/batch*.h5)")
    p.add_argument("--bim", required=True, help="BIM file for counting variants per chromosome")
    p.add_argument("--output-dir", required=True, help="Directory to save per-chromosome models")
    p.add_argument("--val-h5-dir", required=True, help="Validation H5 cache root (chr*/batch*.h5)")
    p.add_argument("--chromosomes", nargs='+', type=str, default=["all"], help="Chromosomes to process or 'all'")
    p.add_argument("--total-grid-size", type=int, default=64, help="Total latent grid side (M) for MILP allocation")
    # Model/training hyperparameters
    p.add_argument("--latent-dim", type=int, default=64)
    p.add_argument("--codebook-size", type=int, default=1024)
    p.add_argument("--num-quantizers", type=int, default=2)
    p.add_argument("--beta-commit", type=float, default=0.25)
    p.add_argument("--hidden-1d-channels", type=int, default=8)
    p.add_argument("--hidden-2d-channels", type=int, default=64)
    p.add_argument("--layers-at-final", type=int, default=0)
    p.add_argument("--ema-decay", type=float, default=0.99)
    p.add_argument("--ld-lambda", type=float, default=1e-3)
    p.add_argument("--maf-lambda", type=float, default=0.0)
    p.add_argument("--ld-window", type=int, default=128)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--beta-warmup-steps", type=int, default=100)
    # Mirror early stop & MMD defaults
    p.add_argument("--plateau-min-rel-improve", type=float, default=0.005)
    p.add_argument("--plateau-patience", type=int, default=3)
    p.add_argument("--plateau-mse-threshold", type=float, default=0.01)
    p.add_argument("--mmd-start-epoch", type=int, default=6)
    p.add_argument("--mmd-lambda", type=float, default=1e-4)
    p.add_argument("--mmd-warmup-epochs", type=int, default=3)
    p.add_argument("--mmd-max-samples", type=int, default=8192)
    # Execution
    p.add_argument("--use-slurm", action='store_true', help="Submit SLURM jobs instead of running locally")
    p.add_argument("--slurm-script", type=str, default=None, help="Path to SLURM wrapper script; defaults to bundled script")
    p.add_argument("--dry-run", action='store_true', help="Print planned jobs without executing")

    args = p.parse_args()

    setup_logging()

    # Resolve chromosome list
    chrom_spec = args.chromosomes
    if len(chrom_spec) == 1 and str(chrom_spec[0]).lower() == "all":
        chromosomes = list(range(1, 23))
    else:
        chromosomes = [int(x) for x in chrom_spec]

    ensure_dir_exists(args.output_dir)

    # 1) Compute weights from BIM (#variants per chromosome)
    counts = compute_variant_counts_per_chromosome(args.bim, chromosomes)
    weights = [counts[c] for c in chromosomes]
    logger.info(f"Variant counts per chromosome: {counts}")

    # 2) Load existing MILP layout if present; otherwise solve and persist
    json_path = os.path.join(args.output_dir, "vqvae_milp_layout.json")
    placements = None
    used_existing = False
    if os.path.exists(json_path):
        try:
            with open(json_path, "r") as jf:
                layout_summary = json.load(jf)
            layout_list = layout_summary.get("layout", [])
            # Map chromosome -> (side, x0, y0, M_eff)
            chr_to_place = {int(rec["chromosome"]): (int(rec.get("tile_side", rec.get("latent_grid_dim", 0))),
                                                      int(rec.get("x0", 0)),
                                                      int(rec.get("y0", 0)),
                                                      int(rec.get("grid_M", layout_summary.get("grid_M", args.total_grid_size))))
                             for rec in layout_list}
            # Build placements aligned to requested chromosomes
            missing = [c for c in chromosomes if c not in chr_to_place]
            if not missing:
                placements = [chr_to_place[c] for c in chromosomes]
                used_existing = True
                logger.info(f"Loaded existing MILP layout from {json_path}")
            else:
                logger.info(f"Existing MILP layout missing chromosomes {missing}; will solve anew.")
        except Exception as e:
            logger.warning(f"Failed to load existing MILP layout at {json_path}: {e}; will solve anew.")

    if not used_existing:
        logger.info(f"Solving MILP for M={args.total_grid_size} with N={len(chromosomes)}")
        result = solve_quadtree_milp(weights, M=int(args.total_grid_size))
        placements = organize_quadtree_solution(result)  # index -> (side, x0, y0, M_eff)
        logger.info(f"MILP solution: {placements}")

        # Persist MILP layout so downstream encode/decode can reuse without re-solving
        try:
            layout_records = []
            for idx, c in enumerate(chromosomes):
                side, x0, y0, M_eff = placements[idx]
                model_out = os.path.join(args.output_dir, f"vqvae_chr{c}.pt")
                layout_records.append({
                    "chromosome": int(c),
                    "latent_grid_dim": int(side),
                    "tile_side": int(side),
                    "x0": int(x0),
                    "y0": int(y0),
                    "grid_M": int(M_eff),
                    "model_file": model_out,
                })
            layout_summary = {
                "solver_status": result.get("status"),
                "objective": float(result.get("objective", 0.0)),
                "grid_M": int(result.get("M_eff", args.total_grid_size)),
                "chromosomes": [int(c) for c in chromosomes],
                "layout": layout_records,
            }
            with open(json_path, "w") as jf:
                json.dump(layout_summary, jf, indent=2)
            logger.info(f"Saved MILP layout JSON: {json_path}")
            csv_path = os.path.join(args.output_dir, "vqvae_milp_layout.csv")
            with open(csv_path, "w", newline="") as cf:
                writer = csv.DictWriter(cf, fieldnames=[
                    "chromosome", "latent_grid_dim", "tile_side", "x0", "y0", "grid_M", "model_file"
                ])
                writer.writeheader()
                for rec in layout_records:
                    writer.writerow(rec)
            logger.info(f"Saved MILP layout CSV: {csv_path}")
        except Exception as e:
            logger.warning(f"Failed to persist MILP layout: {e}")

    # 3) Build training plans
    job_specs: List[Tuple[int, int, str, List[str]]] = []
    for idx, c in enumerate(chromosomes):
        side, _x0, _y0, _M = placements[idx]
        model_out = os.path.join(args.output_dir, f"vqvae_chr{c}.pt")
        # Skip if model checkpoint already exists
        if os.path.exists(model_out):
            logger.info(f"[SKIP] chr{c} tile={side}: model exists at {model_out}")
            continue
        cmd = build_training_command(
            h5_dir=args.h5_dir,
            bim_file=args.bim,
            val_h5_dir=args.val_h5_dir,
            chromosome=c,
            latent_grid_dim=int(side),
            model_out_path=model_out,
            latent_dim=args.latent_dim,
            codebook_size=args.codebook_size,
            num_quantizers=args.num_quantizers,
            beta_commit=args.beta_commit,
            hidden_1d_channels=args.hidden_1d_channels,
            hidden_2d_channels=args.hidden_2d_channels,
            layers_at_final=args.layers_at_final,
            ema_decay=args.ema_decay,
            ld_lambda=args.ld_lambda,
            maf_lambda=args.maf_lambda,
            ld_window=args.ld_window,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
            grad_clip=args.grad_clip,
            beta_warmup_steps=args.beta_warmup_steps,
            plateau_min_rel_improve=args.plateau_min_rel_improve,
            plateau_patience=args.plateau_patience,
            plateau_mse_threshold=args.plateau_mse_threshold,
            mmd_start_epoch=args.mmd_start_epoch,
            mmd_lambda=args.mmd_lambda,
            mmd_warmup_epochs=args.mmd_warmup_epochs,
            mmd_max_samples=args.mmd_max_samples,
        )
        job_specs.append((c, int(side), model_out, cmd))

    # 4) Execute jobs
    logger.info(f"Planned {len(job_specs)} chromosome trainings")
    for c, side, out_path, cmd in job_specs:
        if args.use_slurm:
            slurm_script = args.slurm_script
            if slurm_script is None:
                # default bundled script path
                slurm_script = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                    "scripts", "slurm_train_vqvae_single.sh"
                )
            # Pass only the training flags to the wrapper (exclude module token)
            flags_only = cmd[3:]  # drop: [python, -m, module]
            sbatch_cmd = ["sbatch", slurm_script] + flags_only
            logger.info(f"[SLURM] chr{c} tile={side}: {' '.join(sbatch_cmd)}")
            if not args.dry_run:
                subprocess.run(sbatch_cmd, check=True)
        else:
            logger.info(f"[LOCAL] chr{c} tile={side}: {' '.join(cmd)}")
            if not args.dry_run:
                subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()


# python train_vqvae_orchestrator.py \
#   --h5-dir /n/home03/ahmadazim/WORKING/genGen/UKBVQVAE/genomic_data/vqvae_h5_cache \
#   --bim /n/home03/ahmadazim/WORKING/genGen/UKBVQVAE/genomic_data/geneticBinary/ukb_allchr_unrel_britishWhite.bim \
#   --output-dir /n/home03/ahmadazim/WORKING/genGen/UKBVQVAE/models/vqvae \
#   --chromosomes all \
#   --total-grid-size 256 \
#   --latent-dim 8 \
#   --codebook-size 128 \
#   --num-quantizers 2 \
#   --beta-commit 0.5 \
#   --hidden-1d-channels 8 \
#   --hidden-2d-channels 32 \
#   --layers-at-final 2 \
#   --ema-decay 0.9 \
#   --ld-lambda 1e-3 \
#   --maf-lambda 1e-3 \
#   --ld-window 128 \
#   --epochs 50 \
#   --batch-size 256 \
#   --lr 5e-3 \
#   --device cuda \
#   --use-slurm \
#   --slurm-script /n/home03/ahmadazim/WORKING/genGen/DiffuGene/scripts/slurm_train_vqvae_single.sh

# python train_vqvae_orchestrator.py \
#   --h5-dir /n/home03/ahmadazim/WORKING/genGen/UKBVQVAE/genomic_data/vqvae_h5_cache \
#   --bim /n/home03/ahmadazim/WORKING/genGen/UKBVQVAE/genomic_data/geneticBinary/ukb_allchr_unrel_britishWhite.bim \
#   --output-dir /n/home03/ahmadazim/WORKING/genGen/UKBVQVAE/models/vqvae512 \
#   --chromosomes all \
#   --total-grid-size 512 \
#   --latent-dim 4 \
#   --codebook-size 128 \
#   --num-quantizers 2 \
#   --beta-commit 0.5 \
#   --hidden-1d-channels 8 \
#   --hidden-2d-channels 16 \
#   --layers-at-final 2 \
#   --ema-decay 0.9 \
#   --ld-lambda 1e-3 \
#   --maf-lambda 1e-3 \
#   --ld-window 128 \
#   --epochs 50 \
#   --batch-size 256 \
#   --lr 5e-3 \
#   --device cuda \
#   --use-slurm \
#   --slurm-script /n/home03/ahmadazim/WORKING/genGen/DiffuGene/scripts/slurm_train_vqvae_single.sh