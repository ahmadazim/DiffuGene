#!/usr/bin/env python
"""
Decode unified latent canvases into genotype calls using the per-chromosome
autoencoders trained in stage-2 (HomogenizedAE).

This wraps the helpers inside `decode_utils.py` so that the notebook procedure
documented in `notebooks/newAE.ipynb` (cells 1-35) can be run from the CLI.

Example:

python -m DiffuGene.generate.decode \
    --latents-path /path/to/generated_samples.pt \
    --layout-json /path/to/ae_milp_layout.json \
    --tile-stats /path/to/tile_norm_stats.pt \
    --models-dir /path/to/models/AE \
    --model-pattern "ae_chr{chr}_homog.pt" \
    --return-logits

The script will save a `{latents_path}_decoded.pt` file containing the hard
genotype calls (and optionally logits) for every chromosome represented in the
layout JSON.
"""

import argparse
import logging
import os
from typing import Dict, Iterable, List, Optional

import torch

from .decode_utils import (
    decode_unified_latents,
    load_homogenized_ae_models,
    load_layout_json,
    load_tile_stats,
)


logger = logging.getLogger(__name__)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decode unified latent canvases into genotype calls.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--latents-path",
        type=str,
        required=True,
        help="Path to a .pt file containing the generated latent canvases.",
    )
    parser.add_argument(
        "--latents-key",
        type=str,
        default="latents",
        help="Key within the latents .pt payload. Set to '' to auto-detect.",
    )
    parser.add_argument(
        "--layout-json",
        type=str,
        required=True,
        help="Path to the MILP layout JSON describing the latent tile placement.",
    )
    parser.add_argument(
        "--tile-stats",
        type=str,
        required=True,
        help="Path to the saved `tile_norm_stats.pt` used for per-chr destandardization.",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        required=True,
        help="Directory containing the per-chromosome HomogenizedAE checkpoints.",
    )
    parser.add_argument(
        "--model-pattern",
        type=str,
        default="ae_chr{chr}_homog.pt",
        help="Filename pattern for per-chromosome checkpoints. Must include '{chr}'.",
    )
    parser.add_argument(
        "--batch-decode",
        type=int,
        default=128,
        help="Mini-batch size used when iterating over the latent batch dimension.",
    )
    parser.add_argument(
        "--chromosomes",
        type=str,
        nargs="*",
        default=None,
        help="Subset of chromosomes to decode (space or comma separated). Defaults to all 22 autosomes.",
    )
    parser.add_argument(
        "--return-logits",
        action="store_true",
        help="If set, also save the per-allele logits in the output file.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run decoding on: 'auto', 'cpu', 'cuda', or e.g. 'cuda:1'.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Optional explicit output path. Defaults to `<latents_path>_decoded.pt`.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing decoded output file.",
    )
    return parser.parse_args()


def resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device_str)


def parse_chromosomes(raw: Optional[Iterable[str]]) -> Optional[List[int]]:
    if not raw:
        return None
    values: List[int] = []
    for token in raw:
        pieces = token.split(",")
        for piece in pieces:
            piece = piece.strip()
            if not piece:
                continue
            try:
                chrom = int(piece)
            except ValueError as exc:
                raise ValueError(f"Invalid chromosome '{piece}'") from exc
            if chrom < 1:
                raise ValueError(f"Chromosome must be >=1, got {chrom}")
            if chrom not in values:
                values.append(chrom)
    if not values:
        return None
    return values


def load_latents_tensor(path: str, key: Optional[str]) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu")
    if torch.is_tensor(payload):
        latents = payload
    elif isinstance(payload, dict):
        candidates: List[str] = []
        if key:
            candidates.append(key)
        for default_key in ("latents", "samples", "generated", "x_final", "x"):
            if default_key not in candidates:
                candidates.append(default_key)
        latents = None
        for cand in candidates:
            if cand and cand in payload and torch.is_tensor(payload[cand]):
                latents = payload[cand]
                break
        if latents is None:
            for value in payload.values():
                if torch.is_tensor(value) and value.ndim == 4:
                    latents = value
                    break
        if latents is None:
            raise KeyError(
                f"Could not find a tensor latents entry inside {path}. "
                f"Tried keys: {candidates}"
            )
    else:
        raise TypeError(
            f"Unsupported latents payload type: {type(payload)}. "
            "Expecting a tensor or a dict containing a tensor."
        )

    if latents.ndim != 4:
        raise ValueError(f"Latents tensor must be 4D (B,C,H,W); got {tuple(latents.shape)}")

    latents = latents.to(dtype=torch.float32)
    return latents.contiguous()


def derive_output_path(latents_path: str, explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    base, ext = os.path.splitext(latents_path)
    ext = ext or ".pt"
    return f"{base}_decoded{ext}"


def main() -> None:
    configure_logging()
    args = parse_args()
    device = resolve_device(args.device)

    if args.model_pattern and "{chr}" not in args.model_pattern:
        raise ValueError("--model-pattern must contain '{chr}' placeholder.")

    output_path = derive_output_path(args.latents_path, args.output_path)
    if os.path.exists(output_path) and not args.overwrite:
        raise FileExistsError(
            f"Output file {output_path} already exists. Use --overwrite to replace it."
        )

    logger.info(f"[SETUP] Loading latents from {args.latents_path}")
    latents = load_latents_tensor(args.latents_path, args.latents_key or None)
    logger.info(f"[SETUP] Latents tensor shape: {tuple(latents.shape)}")

    logger.info(f"[SETUP] Loading layout JSON from {args.layout_json}")
    tiles_by_chr, chr_canvas_geom = load_layout_json(args.layout_json)

    logger.info(f"[SETUP] Loading tile stats from {args.tile_stats}")
    stats_by_chr = load_tile_stats(args.tile_stats)

    logger.info(f"[SETUP] Loading HomogenizedAE checkpoints from {args.models_dir}")
    models_by_chr = load_homogenized_ae_models(
        models_by_chr_dir=args.models_dir,
        filename_pattern=args.model_pattern,
        device=device,
    )

    chromosomes = parse_chromosomes(args.chromosomes)
    if chromosomes is not None:
        logger.info(f"[DECODE] Restricting to chromosomes: {chromosomes}")

    logger.info("[DECODE] Starting decode_unified_latents()")
    hard_calls_by_chr, logits_by_chr = decode_unified_latents(
        latents_unified=latents,
        models_by_chr=models_by_chr,
        tiles_by_chr=tiles_by_chr,
        chr_canvas_geom=chr_canvas_geom,
        stats_by_chr=stats_by_chr,
        device=device,
        batch_decode=args.batch_decode,
        return_logits=args.return_logits,
        chromosomes=chromosomes,
    )

    payload: Dict[str, object] = {
        "hard_calls_by_chr": hard_calls_by_chr,
        "meta": {
            "latents_path": args.latents_path,
            "layout_json": args.layout_json,
            "tile_stats": args.tile_stats,
            "models_dir": args.models_dir,
            "model_pattern": args.model_pattern,
            "batch_decode": args.batch_decode,
            "device": str(device),
            "chromosomes": chromosomes,
        },
    }
    if args.return_logits and logits_by_chr is not None:
        payload["logits_by_chr"] = logits_by_chr

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.save(payload, output_path)
    logger.info(f"[DONE] Saved decoded genotypes to {output_path}")


if __name__ == "__main__":
    main()

