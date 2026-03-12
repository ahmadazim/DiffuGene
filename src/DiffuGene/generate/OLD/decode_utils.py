#!/usr/bin/env python
import os
import json
from typing import Dict, List, Tuple, Optional

import torch
import pandas as pd

import sys
this_dir = os.path.dirname(__file__)
src_root = os.path.abspath(os.path.join(this_dir, "..", "..", ".."))
if src_root not in sys.path:
    sys.path.insert(0, src_root)

from DiffuGene.VAEembed.ae import GenotypeAutoencoder, VAEConfig
from DiffuGene.VAEembed.train import H5ChromosomeDataset
from DiffuGene.VAEembed.sharedEmbed import HomogenizedAE, FiLM2D


# ---------------------------
# Layout helpers (JSON format)
# ---------------------------
def load_layout_json(layout_json_path: str) -> Tuple[Dict[int, List[dict]], Dict[int, Tuple[int,int,int,int,int,int]]]:
    """
    Read the MILP layout JSON and return:
      tiles_by_chr: chr -> list of tiles {chromosome, x0,y0,tile_side}
      chr_canvas_geom: chr -> (y_min, y_max, x_min, x_max, Hc, Wc)
    """
    with open(layout_json_path, "r") as f:
        layout = json.load(f)

    recs = layout["layout"]
    tiles_by_chr: Dict[int, List[dict]] = {}
    for r in recs:
        c = int(r["chromosome"])
        tiles_by_chr.setdefault(c, []).append(
            {
                "x0": int(r["x0"]),
                "y0": int(r["y0"]),
                "x1": int(r["x0"]) + int(r["tile_side"]),
                "y1": int(r["y0"]) + int(r["tile_side"]),
                "tile_side": int(r["tile_side"]),
            }
        )

    chr_canvas_geom: Dict[int, Tuple[int,int,int,int,int,int]] = {}
    for c, tiles in tiles_by_chr.items():
        y_min = min(t["y0"] for t in tiles)
        y_max = max(t["y1"] for t in tiles)
        x_min = min(t["x0"] for t in tiles)
        x_max = max(t["x1"] for t in tiles)
        Hc = y_max - y_min
        Wc = x_max - x_min
        chr_canvas_geom[c] = (y_min, y_max, x_min, x_max, Hc, Wc)

    return tiles_by_chr, chr_canvas_geom


# --------------------------------------------
# Per-chromosome normalization stats (saved)
# --------------------------------------------
def load_tile_stats(stats_pt_path: str) -> Dict[int, Dict[str, torch.Tensor]]:
    """
    Load per-chromosome stats saved by unify (tile_norm_stats.pt).
    Returns: {chr: {"mean": Tensor[C], "std": Tensor[C]}}
    """
    stats = torch.load(stats_pt_path, map_location="cpu")
    # sanity
    for c, rec in stats.items():
        if "mean" not in rec or "std" not in rec:
            raise ValueError(f"Stats for chr {c} missing 'mean'/'std'")
    return {int(k): {"mean": v["mean"], "std": v["std"]} for k, v in stats.items()}


# ---------------------------------------
# Load per-chromosome VQ-VAE decoders
# ---------------------------------------
def load_homogenized_ae_models(models_by_chr_dir: str,
                               filename_pattern: str,
                               device: torch.device) -> Dict[int, HomogenizedAE]:
    """
    Load per-chromosome HomogenizedAE checkpoints saved by stage-2 training.
    filename_pattern example: 'ae_chr{chr}_homog.pt'
    Returns: {chr: HomogenizedAE} with frozen params, on requested device.
    """
    models: Dict[int, HomogenizedAE] = {}
    for c in range(1, 23):
        ckpt_path = os.path.join(models_by_chr_dir, filename_pattern.format(chr=c))
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Missing homogenized AE ckpt for chr{c}: {ckpt_path}")
        payload = torch.load(ckpt_path, map_location="cpu")
        meta = payload.get("meta", {})
        cfg_dict = meta.get("config")
        if cfg_dict is None:
            raise KeyError(f"Checkpoint {ckpt_path} missing meta['config'] for AE reconstruction")
        cfg = VAEConfig(**cfg_dict)
        # Rebuild AE backbone
        ae = GenotypeAutoencoder(
            input_length=cfg.input_length,
            K1=cfg.K1,
            K2=cfg.K2,
            C=cfg.C,
            embed_dim=cfg.embed_dim,
        )
        # Build HomogenizedAE wrapper (single AE)
        homog = HomogenizedAE([ae])
        state = payload.get("model_state") if isinstance(payload, dict) else None
        if state is None:
            state = payload if isinstance(payload, dict) else {}
        missing = homog.load_state_dict(state, strict=True)
        # Freeze parameters and move to device
        for p in homog.parameters():
            p.requires_grad = False
        homog.to(device).eval()
        models[c] = homog
    return models


# -------------------------------------------------------
# Core: decode unified latents → per-chromosome genotypes
# -------------------------------------------------------
@torch.no_grad()
def decode_unified_latents(
    latents_unified: torch.Tensor,           # (B, C, H=128, W=128)
    models_by_chr: Dict[int, HomogenizedAE],
    tiles_by_chr: Dict[int, List[dict]],
    chr_canvas_geom: Dict[int, Tuple[int,int,int,int,int,int]],
    stats_by_chr: Dict[int, Dict[str, torch.Tensor]],
    device: torch.device,
    batch_decode: int = 128,
    return_logits: bool = False,
    chromosomes : Optional[List[int]] = None,
) -> Tuple[Dict[int, torch.Tensor], Optional[Dict[int, torch.Tensor]]]:
    """
    Returns:
      hard_calls_by_chr: {chr: LongTensor [B, L_chr]}
      logits_by_chr (optional): {chr: FloatTensor [B, 3, L_chr]}
    """

    if latents_unified.ndim != 4:
        raise ValueError(f"Expected 4D tensor (B,C,H,W); got {latents_unified.shape}")
    B, C, H, W = latents_unified.shape

    # Sanity: ensure AE latent_channels = C
    for c, m in models_by_chr.items():
        ae_c = getattr(m, "aes")[0]
        if int(ae_c.latent_channels) != C:
            raise RuntimeError(f"chr{c}: AE latent_channels={ae_c.latent_channels} != channels C={C}")

    device_is_cuda = (device.type == "cuda")
    latents_unified = latents_unified.contiguous()  # slicing stability

    # outputs
    hard_calls_by_chr: Dict[int, torch.Tensor] = {}
    logits_by_chr: Optional[Dict[int, torch.Tensor]] = {} if return_logits else None

    # Preallocate output buffers once we know sizes
    prealloc = False
    holders_hard: Dict[int, torch.Tensor] = {}
    holders_logits: Dict[int, torch.Tensor] = {}

    # Iterate batch-wise to keep mem reasonable
    for b0 in range(0, B, batch_decode):
        b1 = min(B, b0 + batch_decode)
        Bb = b1 - b0

        for c in chromosomes if chromosomes is not None else range(1, 23):
            if c not in tiles_by_chr or c not in chr_canvas_geom:
                # if a chromosome had no tiles in layout, skip
                continue
            model_c = models_by_chr[c]
            y_min, y_max, x_min, x_max, Hc, Wc = chr_canvas_geom[c]

            # Assemble per-chr canvas from tiles
            z_chr_cpu = torch.empty((Bb, C, Hc, Wc), dtype=latents_unified.dtype)
            wrote = torch.zeros((Bb, Hc, Wc), dtype=torch.bool)
            for t in tiles_by_chr[c]:
                y0g, y1g, x0g, x1g = t["y0"], t["y1"], t["x0"], t["x1"]
                y0l, y1l = y0g - y_min, y1g - y_min
                x0l, x1l = x0g - x_min, x1g - x_min
                z_chr_cpu[:, :, y0l:y1l, x0l:x1l] = latents_unified[b0:b1, :, y0g:y1g, x0g:x1g]
                wrote[:, y0l:y1l, x0l:x1l] = True
            if not wrote.all():
                miss = (~wrote).sum().item()
                raise RuntimeError(f"chr{c}: canvas not fully covered; missing {miss} px")

            # De-standardize per-chromosome, per-channel
            st = stats_by_chr.get(c, None)
            if st is None:
                raise KeyError(f"No norm stats for chr{c} in provided stats dict.")
            mu = st["mean"].to(dtype=torch.float32).view(1, -1, 1, 1)
            sd = st["std"].to(dtype=torch.float32).clamp_min(1e-6).view(1, -1, 1, 1)
            z_chr_cpu = z_chr_cpu.to(dtype=torch.float32)
            z_chr_cpu = z_chr_cpu * sd + mu

            # Decode using homogenized AE: apply decode_head FiLM then AE.decode
            model_c = models_by_chr[c]
            chrom_idx = c - 1
            z_chr = z_chr_cpu.to(device, non_blocking=device_is_cuda)
            z_dec = model_c.decode_head(z_chr, chrom_idx)
            ae_c = model_c.aes[0]
            logits_b = ae_c.decode(z_dec)   # [Bb, 3, L_chr]
            if logits_b.dim() != 3 or logits_b.size(1) != 3:
                raise RuntimeError(f"chr{c}: unexpected logits shape {tuple(logits_b.shape)}")

            hard_b = logits_b.argmax(dim=1).to(torch.int64).to("cpu")  # [Bb, L_chr]

            # Preallocate once
            if not prealloc:
                L_chr = int(logits_b.size(-1))
                for cc in range(1, 23):
                    if cc in models_by_chr and cc in tiles_by_chr and cc in chr_canvas_geom:
                        # Decode a tiny dummy to get per-chr sequence length
                        Hcc = chr_canvas_geom[cc][4]
                        Wcc = chr_canvas_geom[cc][5]
                        dummy = torch.zeros(1, C, Hcc, Wcc, device=device, dtype=z_dec.dtype)
                        z_dec_cc = models_by_chr[cc].decode_head(dummy, cc-1)
                        ae_cc = models_by_chr[cc].aes[0]
                        L_cc = int(ae_cc.decode(z_dec_cc).size(-1))
                        holders_hard[cc] = torch.empty((B, L_cc), dtype=torch.long)
                        if return_logits:
                            holders_logits[cc] = torch.empty((B, 3, L_cc), dtype=logits_b.dtype)
                prealloc = True

            holders_hard[c][b0:b1, :] = hard_b
            if return_logits:
                holders_logits[c][b0:b1, :] = logits_b.to("cpu")

            # cleanup
            del z_chr_cpu, z_chr, z_dec, logits_b, hard_b, wrote
            if device_is_cuda:
                torch.cuda.empty_cache()

    # Move holders into final dicts
    for c, arr in holders_hard.items():
        hard_calls_by_chr[c] = arr
    if return_logits:
        out_logits: Dict[int, torch.Tensor] = {}
        for c, arr in holders_logits.items():
            out_logits[c] = arr
        return hard_calls_by_chr, out_logits
    else:
        return hard_calls_by_chr, None