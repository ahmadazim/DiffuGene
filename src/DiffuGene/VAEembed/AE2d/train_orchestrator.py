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
    val_h5_dir: str,
    chromosome: int,
    spatial1d: int,
    spatial2d: int,
    model_out_path: str,
    latent_channels: int,
    embed_dim: int,
    ld_lambda: float,
    maf_lambda: float,
    ld_window: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device: str,
    grad_clip: float,
    # Early stop forwarded
    plateau_min_rel_improve: float,
    plateau_patience: int,
    plateau_mse_threshold: float,
) -> List[str]:
    # Use module invocation so PYTHONPATH resolves correctly
    cmd = [
        sys.executable, "-m", "DiffuGene.VAEembed.train",
        "--h5-dir", h5_dir,
        "--val-h5-dir", val_h5_dir,
        "--chromosome", str(int(chromosome)),
        "--save-path", model_out_path,
        "--spatial1d", str(int(spatial1d)),
        "--spatial2d", str(int(spatial2d)),
        "--latent-channels", str(int(latent_channels)),
        "--embed-dim", str(int(embed_dim)),
        "--ld-lambda", str(float(ld_lambda)),
        "--maf-lambda", str(float(maf_lambda)),
        "--ld-window", str(int(ld_window)),
        "--epochs", str(int(epochs)),
        "--batch-size", str(int(batch_size)),
        "--lr", str(float(lr)),
        "--weight-decay", str(float(weight_decay)),
        "--device", device,
        "--grad-clip", str(float(grad_clip)),
        "--plateau-min-rel-improve", str(float(plateau_min_rel_improve)),
        "--plateau-patience", str(int(plateau_patience)),
        "--plateau-mse-threshold", str(float(plateau_mse_threshold)),
    ]
    return cmd


def main():
    p = argparse.ArgumentParser(description="Orchestrate per-chromosome AE training with MILP-based tile allocation")
    p.add_argument("--h5-dir", required=True, help="H5 cache directory root (with chr*/batch*.h5)")
    p.add_argument("--bim", required=True, help="BIM file for counting variants per chromosome")
    p.add_argument("--output-dir", required=True, help="Directory to save per-chromosome models")
    p.add_argument("--val-h5-dir", required=True, help="Validation H5 cache root (chr*/batch*.h5)")
    p.add_argument("--chromosomes", nargs='+', type=str, default=["all"], help="Chromosomes to process or 'all'")
    p.add_argument("--total-grid-size", type=int, default=64, help="Total latent grid side (M) for MILP allocation")
    # ---- AE model/training hyperparameters to forward to trainer ----
    p.add_argument("--spatial1d-options", nargs=2, type=int, default=[1024, 512], help="Two options for 1D spatial size K1: [large, small], e.g., 1024 512")
    p.add_argument("--latent-channels", type=int, default=64, help="Final 2D latent channels (e.g., 64, 32, 16)")
    p.add_argument("--embed-dim", type=int, default=8, help="1D embedding channels")
    p.add_argument("--ld-lambda", type=float, default=0.0)
    p.add_argument("--maf-lambda", type=float, default=0.0)
    p.add_argument("--ld-window", type=int, default=128)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--grad-clip", type=float, default=1.0)
    # Early stopping (trainer handles these)
    p.add_argument("--plateau-min-rel-improve", type=float, default=0.005)
    p.add_argument("--plateau-patience", type=int, default=3)
    p.add_argument("--plateau-mse-threshold", type=float, default=0.01)
    # Execution
    p.add_argument("--use-slurm", action='store_true', help="Submit SLURM jobs instead of running locally")
    p.add_argument("--slurm-script", type=str, default=None, help="Path to SLURM wrapper script for AE; if missing, fall back to local")
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
    json_path = os.path.join(args.output_dir, "ae_milp_layout.json")
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
                model_out = os.path.join(args.output_dir, f"ae_chr{c}.pt")
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
            csv_path = os.path.join(args.output_dir, "ae_milp_layout.csv")
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
        side, _x0, _y0, grid_M = placements[idx]
        model_out = os.path.join(args.output_dir, f"ae_chr{c}.pt")
        if os.path.exists(model_out):
            logger.info(f"[SKIP] chr{c} tile={side}: model exists at {model_out}")
            continue
        # Infer spatial1d from tile size relative to grid_M (prefer exact M/4 vs M/8)
        try:
            grid_M = int(grid_M)
        except Exception:
            grid_M = int(args.total_grid_size)
        large_threshold = max(1, grid_M // 4)
        small_threshold = max(1, grid_M // 8)
        large_k1, small_k1 = int(args.spatial1d_options[0]), int(args.spatial1d_options[1])
        if int(side) == large_threshold:
            spatial1d_val = large_k1
        elif int(side) == small_threshold:
            spatial1d_val = small_k1
        else:
            logger.error(
                f"chr{c} tile_side={side} not equal to M/4({large_threshold}) or M/8({small_threshold}); exiting"
            )
            sys.exit(1)
        cmd = build_training_command(
            h5_dir=args.h5_dir,
            val_h5_dir=args.val_h5_dir,
            chromosome=c,
            spatial1d=int(spatial1d_val),
            spatial2d=int(side),
            model_out_path=model_out,
            latent_channels=int(args.latent_channels),
            embed_dim=int(args.embed_dim),
            ld_lambda=float(args.ld_lambda),
            maf_lambda=float(args.maf_lambda),
            ld_window=int(args.ld_window),
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
            device=args.device,
            grad_clip=float(args.grad_clip),
            plateau_min_rel_improve=float(args.plateau_min_rel_improve),
            plateau_patience=int(args.plateau_patience),
            plateau_mse_threshold=float(args.plateau_mse_threshold),
        )
        job_specs.append((c, int(side), model_out, cmd))

    # 4) Execute jobs
    logger.info(f"Planned {len(job_specs)} chromosome trainings")
    for c, side, out_path, cmd in job_specs:
        if args.use_slurm:
            slurm_script = args.slurm_script
            if slurm_script is None:
                # Prefer AE-specific wrapper if present
                candidate = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                    "scripts", "slurm_train_ae_single.sh"
                )
                slurm_script = candidate if os.path.exists(candidate) else None
            if slurm_script is None:
                logger.warning("SLURM wrapper for AE not found; falling back to local execution for this job")
            else:
                # Pass only the training flags to the wrapper (exclude module token)
                flags_only = cmd[3:]  # drop: [python, -m, module]
                sbatch_cmd = ["sbatch", slurm_script] + flags_only
                logger.info(f"[SLURM] chr{c} tile={side}: {' '.join(sbatch_cmd)}")
                if not args.dry_run:
                    subprocess.run(sbatch_cmd, check=True)
                continue
        # Fallback/local execution
        logger.info(f"[LOCAL] chr{c} tile={side}: {' '.join(cmd)}")
        if not args.dry_run:
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
