#!/usr/bin/env python
"""
Streamed decoder for unified diffusion latents.

Given a 4D latent canvas (B,64,128,128), a MILP layout JSON, and per-chromosome
homogenized AE checkpoints, this script:
  1) slices each chromosome canvas from the unified latent according to layout,
  2) destandardizes using saved tile stats,
  3) applies the chromosome-specific decode head + AE decoder,
  4) accumulates across GPU decode batches and writes exactly two files per chromosome:
     - chr{c}_calls.pt  : LongTensor (B, L_chr)
     - chr{c}_logits.pt : FloatTensor (B, 3, L_chr) if --return-logits
"""

import argparse
import logging
import os
from typing import Dict, Iterable, List, Optional, Tuple

import torch

import sys
this_dir = os.path.dirname(__file__)
src_root = os.path.abspath(os.path.join(this_dir, "..", "..", ".."))
if src_root not in sys.path:
    sys.path.insert(0, src_root)

from DiffuGene.generate.decode_utils import (
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

def resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device_str)

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


def parse_chromosomes(raw: Optional[Iterable[str]]) -> Optional[List[int]]:
    if not raw:
        return None
    out: List[int] = []
    for token in raw:
        for piece in token.split(","):
            piece = piece.strip()
            if not piece:
                continue
            chrom = int(piece)
            if chrom < 1:
                raise ValueError(f"Chromosome must be >=1, got {chrom}")
            if chrom not in out:
                out.append(chrom)
    return out or None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Decode unified diffusion latents and write one pair of files per chromosome.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--latents-path", required=True, help="Path to .pt file from diffusion generation.")
    p.add_argument("--latents-key", default="latents", help="Key inside latents payload ('' to auto-detect).")
    p.add_argument("--layout-json", required=True, help="MILP layout JSON describing tile placement.")
    p.add_argument("--tile-stats", required=True, help="tile_norm_stats.pt used for destandardization.")
    p.add_argument("--models-dir", required=True, help="Directory with per-chrom homogenized AE checkpoints.")
    p.add_argument("--model-pattern", default="ae_chr{chr}_homog.pt", help="Filename pattern with '{chr}' placeholder.")
    p.add_argument("--batch-decode", type=int, default=64, help="Decode batch size along latent batch dimension.")
    p.add_argument("--chromosomes", nargs="*", default=None, help="Optional subset of chromosomes to decode.")
    p.add_argument("--device", default="auto", help="Device string: auto|cpu|cuda|cuda:1 ...")
    p.add_argument("--output-dir", default=None, help="Directory for final per-chromosome outputs.")
    p.add_argument("--return-logits", action="store_true", help="Also persist per-allele logits.")
    p.add_argument("--overwrite", action="store_true", help="Allow writing into a non-empty output dir.")
    return p.parse_args()


@torch.no_grad()
def decode_and_write(
    latents: torch.Tensor,
    tiles_by_chr: Dict[int, List[dict]],
    chr_canvas_geom: Dict[int, Tuple[int, int, int, int, int, int]],
    stats_by_chr: Dict[int, Dict[str, torch.Tensor]],
    models_by_chr: Dict[int, torch.nn.Module],
    output_dir: str,
    chrom_list: List[int],
    batch_decode: int,
    device: torch.device,
    return_logits: bool,
) -> None:
    B, C, _, _ = latents.shape
    device_is_cuda = device.type == "cuda"

    # Prepare stats (CPU) once
    stats_prepared = {
        c: (
            rec["mean"].to(dtype=torch.float32).view(1, -1, 1, 1),
            rec["std"].to(dtype=torch.float32).clamp_min(1e-6).view(1, -1, 1, 1),
        )
        for c, rec in stats_by_chr.items()
    }

    # Precompute per-chromosome sequence lengths and preallocate holders on CPU
    calls_holders: Dict[int, torch.Tensor] = {}
    logits_holders: Dict[int, torch.Tensor] = {} if return_logits else {}
    with torch.no_grad():
        for c in chrom_list:
            if c not in tiles_by_chr or c not in chr_canvas_geom:
                continue
            y_min, _, x_min, _, Hc, Wc = chr_canvas_geom[c]
            # Tiny dummy to determine L_chr
            dummy = torch.zeros(1, C, Hc, Wc, device=device)
            z_dec = models_by_chr[c].decode_head(dummy, c - 1)
            L_chr = int(models_by_chr[c].aes[0].decode(z_dec).size(-1))
            calls_holders[c] = torch.empty((B, L_chr), dtype=torch.long)
            if return_logits:
                logits_holders[c] = torch.empty((B, 3, L_chr), dtype=torch.float32)
            del dummy, z_dec
        if device_is_cuda:
            torch.cuda.empty_cache()

    for b0 in range(0, B, batch_decode):
        b1 = min(B, b0 + batch_decode)
        Bb = b1 - b0
        logger.info(f"[BATCH] decoding samples {b0}:{b1}")

        for c in chrom_list:
            if c not in tiles_by_chr or c not in chr_canvas_geom:
                continue
            y_min, _, x_min, _, Hc, Wc = chr_canvas_geom[c]

            # assemble per-chromosome canvas
            z_chr_cpu = torch.empty((Bb, C, Hc, Wc), dtype=latents.dtype)
            for t in tiles_by_chr[c]:
                y0g, y1g, x0g, x1g = t["y0"], t["y1"], t["x0"], t["x1"]
                y0l, y1l = y0g - y_min, y1g - y_min
                x0l, x1l = x0g - x_min, x1g - x_min
                z_chr_cpu[:, :, y0l:y1l, x0l:x1l] = latents[b0:b1, :, y0g:y1g, x0g:x1g]

            mu, sd = stats_prepared[c]
            z_chr_cpu = z_chr_cpu.to(dtype=torch.float32)
            z_chr_cpu.mul_(sd).add_(mu)

            z_chr = z_chr_cpu.to(device, non_blocking=device_is_cuda)
            model_c = models_by_chr[c]
            chrom_idx = c - 1
            z_dec = model_c.decode_head(z_chr, chrom_idx)
            logits = model_c.aes[0].decode(z_dec)  # [Bb, 3, L]

            calls_cpu = logits.argmax(dim=1).to(torch.int64).cpu()
            calls_holders[c][b0:b1, :] = calls_cpu
            if return_logits:
                logits_holders[c][b0:b1, :] = logits.cpu()

            del z_chr_cpu, z_chr, z_dec, logits, calls_cpu
            if device_is_cuda:
                torch.cuda.empty_cache()

    # write final per-chromosome files
    for c, calls_tensor in calls_holders.items():
        calls_path = os.path.join(output_dir, f"chr{c}_calls.pt")
        torch.save(calls_tensor, calls_path)
    if return_logits:
        for c, logits_tensor in logits_holders.items():
            logits_path = os.path.join(output_dir, f"chr{c}_logits.pt")
            torch.save(logits_tensor, logits_path)


def main() -> None:
    configure_logging()
    args = parse_args()
    device = resolve_device(args.device)

    output_dir = args.output_dir or f"{os.path.splitext(args.latents_path)[0]}_decoded"
    if os.path.exists(output_dir) and os.listdir(output_dir) and not args.overwrite:
        raise FileExistsError(f"Output dir {output_dir} is not empty; use --overwrite to proceed.")
    os.makedirs(output_dir, exist_ok=True)

    latents = load_latents_tensor(args.latents_path, args.latents_key or None)
    tiles_by_chr, chr_canvas_geom = load_layout_json(args.layout_json)
    stats_by_chr = load_tile_stats(args.tile_stats)
    models_by_chr = load_homogenized_ae_models(
        models_by_chr_dir=args.models_dir,
        filename_pattern=args.model_pattern,
        device=device,
    )

    chrom_list = parse_chromosomes(args.chromosomes)
    if chrom_list is None:
        chrom_list = [c for c in range(1, 23) if c in models_by_chr and c in tiles_by_chr]
    logger.info(f"[SETUP] decoding chromosomes: {chrom_list}")

    decode_and_write(
        latents=latents,
        tiles_by_chr=tiles_by_chr,
        chr_canvas_geom=chr_canvas_geom,
        stats_by_chr=stats_by_chr,
        models_by_chr=models_by_chr,
        output_dir=output_dir,
        chrom_list=chrom_list,
        batch_decode=int(args.batch_decode),
        device=device,
        return_logits=bool(args.return_logits),
    )
    logger.info(f"[DONE] wrote per-chromosome outputs to {output_dir}")


if __name__ == "__main__":
    main()