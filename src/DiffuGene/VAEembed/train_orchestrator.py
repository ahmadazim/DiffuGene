#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from typing import Dict, List, Tuple
import numpy as np

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


def _pow2_options(min_tokens: int, max_tokens: int) -> List[int]:
    out: List[int] = []
    k = 0
    while (1 << k) <= int(max_tokens):
        v = 1 << k
        if v >= int(min_tokens):
            out.append(int(v))
        k += 1
    if len(out) == 0:
        raise ValueError("No power-of-two options in requested [min_tokens, max_tokens] range.")
    return out


def solve_token_allocation_dp(
    w: List[float],
    total_tokens: int,
    min_tokens: int,
    max_tokens: int | None,
) -> Dict:
    """
    Exact dynamic-programming allocation without external MILP solvers.
    Minimizes sum_i |alloc_i - target_i| with power-of-two alloc_i and exact token budget.
    """
    w_arr = np.asarray(w, dtype=float)
    if w_arr.ndim != 1 or len(w_arr) == 0:
        raise ValueError("w must be a non-empty 1D list.")
    if np.any(w_arr < 0) or float(w_arr.sum()) <= 0.0:
        raise ValueError("weights must be nonnegative and sum to > 0.")

    n = int(len(w_arr))
    T = int(total_tokens)
    if T <= 0:
        raise ValueError("total_tokens must be > 0.")
    lo = max(1, int(min_tokens))
    hi = int(max_tokens) if max_tokens is not None else int(total_tokens)
    options = _pow2_options(lo, hi)
    if n * min(options) > T or n * max(options) < T:
        raise ValueError("No feasible allocation satisfies bounds and exact token budget.")

    targets = (w_arr / float(w_arr.sum())) * float(T)
    inf = float("inf")
    dp = np.full((n + 1, T + 1), inf, dtype=np.float64)
    prev_s = np.full((n + 1, T + 1), -1, dtype=np.int32)
    prev_o = np.full((n + 1, T + 1), -1, dtype=np.int32)
    dp[0, 0] = 0.0

    for i in range(1, n + 1):
        tgt = float(targets[i - 1])
        for s in range(0, T + 1):
            best = inf
            best_prev = -1
            best_opt = -1
            for opt in options:
                p = s - int(opt)
                if p < 0:
                    continue
                base = float(dp[i - 1, p])
                if not np.isfinite(base):
                    continue
                cand = base + abs(float(opt) - tgt)
                if cand < best:
                    best = cand
                    best_prev = p
                    best_opt = int(opt)
            dp[i, s] = best
            prev_s[i, s] = best_prev
            prev_o[i, s] = best_opt

    if not np.isfinite(dp[n, T]):
        raise RuntimeError("DP failed to find a feasible exact allocation.")

    allocations = np.zeros((n,), dtype=int)
    s = T
    for i in range(n, 0, -1):
        opt = int(prev_o[i, s])
        if opt <= 0:
            raise RuntimeError("DP backtracking failed.")
        allocations[i - 1] = opt
        s = int(prev_s[i, s])
    if s != 0:
        raise RuntimeError("DP backtracking ended with non-zero remainder.")

    assignments = []
    offsets = []
    cursor = 0
    for i in range(n):
        tok = int(allocations[i])
        assignments.append(
            {
                "i": int(i),
                "tokens": tok,
                "target": float(targets[i]),
                "abs_error": float(abs(float(tok) - float(targets[i]))),
            }
        )
        offsets.append({"i": int(i), "token_start": int(cursor), "token_end": int(cursor + tok)})
        cursor += tok

    return {
        "status": "Optimal",
        "objective": float(dp[n, T]),
        "total_tokens": int(T),
        "allocations": allocations,
        "targets": targets,
        "abs_errors": np.abs(allocations.astype(float) - targets),
        "assignments": assignments,
        "offsets": offsets,
        "options": options,
        "sum_allocations": int(allocations.sum()),
    }


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
    p = argparse.ArgumentParser(description="Orchestrate per-chromosome token-AE training with MILP token allocation.")
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
    p.add_argument(
        "--token-solver",
        type=str,
        default="AUTO",
        choices=["AUTO", "DP", "CBC", "GUROBI", "GLPK", "SCIP"],
        help="Solver for token allocation MILP; AUTO falls back to pure-Python DP if CBC is unavailable.",
    )
    args = p.parse_args()

    setup_logging()
    ensure_dir_exists(args.output_dir)

    chromosomes = get_chromosome_list(args.chromosomes)
    counts = compute_variant_counts_per_chromosome(args.bim, chromosomes)
    logger.info(f"Variant counts per chromosome: {counts}")

    layout_json = os.path.join(args.output_dir, "tok_milp_layout.json")
    layout_csv = os.path.join(args.output_dir, "tok_milp_layout.csv")

    weights = [counts[c] for c in chromosomes]
    token_solver = str(args.token_solver).upper()
    if token_solver == "DP":
        result = solve_token_allocation_dp(
            w=weights,
            total_tokens=int(args.total_tokens),
            min_tokens=int(args.min_tokens),
            max_tokens=(None if args.max_tokens is None else int(args.max_tokens)),
        )
    elif token_solver == "AUTO":
        try:
            result = solve_token_allocation_milp(
                w=weights,
                total_tokens=int(args.total_tokens),
                min_tokens=int(args.min_tokens),
                max_tokens=(None if args.max_tokens is None else int(args.max_tokens)),
                solver_name="CBC",
                verbose=False,
            )
        except Exception as ex:
            logger.warning(f"CBC unavailable ({ex}); falling back to pure-Python DP allocation.")
            result = solve_token_allocation_dp(
                w=weights,
                total_tokens=int(args.total_tokens),
                min_tokens=int(args.min_tokens),
                max_tokens=(None if args.max_tokens is None else int(args.max_tokens)),
            )
    else:
        result = solve_token_allocation_milp(
            w=weights,
            total_tokens=int(args.total_tokens),
            min_tokens=int(args.min_tokens),
            max_tokens=(None if args.max_tokens is None else int(args.max_tokens)),
            solver_name=token_solver,
            verbose=False,
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

