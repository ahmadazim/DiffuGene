#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
import os
import random
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from DiffuGene.VAEembed.aeTok import TokenAutoencoder1D, TokenAEConfig
    from DiffuGene.VAEembed.train_tok import H5ChromosomeDataset
    from DiffuGene.VAEembed.sharedEmbed_tok import (
        FiLM1D,
        HomogenizedTokenAE,
        Stage2PenaltyConfigTok,
        compute_shared_head_penalties_tok,
    )
except Exception:
    import sys
    this_dir = os.path.dirname(__file__)
    src_root = os.path.abspath(os.path.join(this_dir, "..", "..", ".."))
    if src_root not in sys.path:
        sys.path.insert(0, src_root)
    from DiffuGene.VAEembed.aeTok import TokenAutoencoder1D, TokenAEConfig
    from DiffuGene.VAEembed.train_tok import H5ChromosomeDataset
    from DiffuGene.VAEembed.sharedEmbed_tok import (
        FiLM1D,
        HomogenizedTokenAE,
        Stage2PenaltyConfigTok,
        compute_shared_head_penalties_tok,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage2 shared-head homogenization for token AEs.")
    p.add_argument("--ae-checkpoints", required=True, help="Directory containing ae_tok_chr{chrom}.pt")
    p.add_argument("--h5-dir", required=True)
    p.add_argument("--val-h5-dir", required=True)
    p.add_argument("--chromosomes", nargs="+", type=str, default=["all"])
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--val-batch-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--latent-loss-weight", type=float, default=0.05)
    p.add_argument("--tv-lambda", type=float, default=2e-2)
    p.add_argument("--robust-lambda", type=float, default=1.0)
    p.add_argument("--stable-lambda", type=float, default=3e-1)
    p.add_argument("--latent-noise-std", type=float, default=0.05)
    p.add_argument("--embed-noise-std", type=float, default=0.03)
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def resolve_chromosomes(spec: List[str]) -> List[int]:
    if len(spec) == 1 and spec[0].lower() == "all":
        return list(range(1, 23))
    return [int(x) for x in spec]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_ae_model(path: str, device: torch.device) -> Tuple[TokenAutoencoder1D, Dict]:
    ckpt = torch.load(path, map_location="cpu")
    cfg_dict = ckpt["config"]
    cfg = TokenAEConfig(**cfg_dict)
    model = TokenAutoencoder1D(
        input_length=cfg.input_length,
        latent_length=cfg.latent_length,
        latent_dim=cfg.latent_dim,
        embed_dim=cfg.embed_dim,
        max_c=cfg.max_c,
        dropout=cfg.dropout,
    )
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, cfg_dict


def build_loaders(h5_root: str, chromosomes: List[int], batch_size: int, num_workers: int, shuffle: bool) -> List[DataLoader]:
    loaders = []
    for c in chromosomes:
        ds = H5ChromosomeDataset(h5_root, c)
        loaders.append(
            DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=(num_workers > 0),
            )
        )
    return loaders


def save_homog_per_chr(
    encode_state: Dict[str, torch.Tensor],
    decode_state: Dict[str, torch.Tensor],
    chromosomes: List[int],
    ckpt_dir: str,
    epoch: int,
    mean_val_ce: float,
    mean_val_acc: float,
    best_epoch: int,
    best_val_ce: float,
    best_val_acc: float,
) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for c in chromosomes:
        ckpt_path = os.path.join(ckpt_dir, f"ae_tok_chr{c}.pt")
        ae, cfg_dict = load_ae_model(ckpt_path, torch.device("cpu"))
        single = HomogenizedTokenAE([ae]).to("cpu")
        single.encode_head.load_state_dict(encode_state, strict=True)
        single.decode_head.load_state_dict(decode_state, strict=True)
        for p in single.parameters():
            p.requires_grad = False
        out = os.path.join(ckpt_dir, f"ae_tok_chr{c}_homog.pt")
        payload = {
            "model_state": single.state_dict(),
            "meta": {
                "chromosome": int(c),
                "config": cfg_dict,
                "stage2": {
                    "current_epoch": int(epoch),
                    "current_mean_val_ce": float(mean_val_ce),
                    "current_mean_val_acc": float(mean_val_acc),
                    "best_epoch": int(best_epoch),
                    "best_mean_val_ce": float(best_val_ce),
                    "best_mean_val_acc": float(best_val_acc),
                    "timestamp": ts,
                },
            },
        }
        torch.save(payload, out)
        print(f"[TOK-STAGE2] Saved {out}")


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    set_seed(args.seed)
    chromosomes = resolve_chromosomes(args.chromosomes)

    train_loaders = build_loaders(args.h5_dir, chromosomes, args.batch_size, args.num_workers, shuffle=True)
    val_loaders = build_loaders(args.val_h5_dir, chromosomes, max(args.val_batch_size, args.batch_size), max(1, args.num_workers // 2), shuffle=False)

    ae_map: Dict[int, TokenAutoencoder1D] = {}
    for c in chromosomes:
        p = os.path.join(args.ae_checkpoints, f"ae_tok_chr{c}.pt")
        ae, _ = load_ae_model(p, device)
        ae_map[c] = ae

    latent_dim = next(iter(ae_map.values())).latent_dim
    encode_head = FiLM1D(latent_dim).to(device)
    decode_head = FiLM1D(latent_dim).to(device)
    optimizer = torch.optim.AdamW(
        list(encode_head.parameters()) + list(decode_head.parameters()),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    penalty_cfg = Stage2PenaltyConfigTok(
        tv_lambda=float(args.tv_lambda),
        robust_lambda=float(args.robust_lambda),
        stable_lambda=float(args.stable_lambda),
        latent_noise_std=float(args.latent_noise_std),
        embed_noise_std=float(args.embed_noise_std),
    )

    best_epoch = 0
    best_val_ce = math.inf
    best_val_acc = 0.0
    best_encode_state = {k: v.detach().cpu().clone() for k, v in encode_head.state_dict().items()}
    best_decode_state = {k: v.detach().cpu().clone() for k, v in decode_head.state_dict().items()}

    for epoch in range(1, int(args.epochs) + 1):
        encode_head.train()
        decode_head.train()
        train_loss = 0.0
        train_steps = 0
        for local_idx, c in enumerate(chromosomes):
            ae = ae_map[c]
            for xb in train_loaders[local_idx]:
                xb = xb.to(device, non_blocking=True)
                with torch.no_grad():
                    _, z_orig = ae(xb)

                chrom_embed_idx = c - 1
                chrom_vec = torch.full((z_orig.size(0),), int(chrom_embed_idx), dtype=torch.long, device=device)

                z_hom = encode_head(z_orig, chrom_vec)
                z_dec = decode_head(z_hom, chrom_vec)
                logits = ae.decode(z_dec)

                ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), xb.reshape(-1).long())
                lat = F.mse_loss(z_hom, z_orig)
                pen, _ = compute_shared_head_penalties_tok(
                    ae=ae,
                    shared_forward=lambda latent: (
                        encode_head(latent, chrom_vec),
                        decode_head(encode_head(latent, chrom_vec), chrom_vec),
                    ),
                    z_input=z_orig,
                    z_dec=z_dec,
                    x_batch=xb,
                    penalty_cfg=penalty_cfg,
                )
                loss = ce + float(args.latent_loss_weight) * lat + pen
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                train_loss += float(loss.item())
                train_steps += 1

        encode_head.eval()
        decode_head.eval()
        val_ce_sum = 0.0
        val_corr = 0.0
        val_sites = 0.0
        with torch.no_grad():
            for local_idx, c in enumerate(chromosomes):
                ae = ae_map[c]
                chrom_embed_idx = c - 1
                for xb in val_loaders[local_idx]:
                    xb = xb.to(device, non_blocking=True)
                    _, z_orig = ae(xb)
                    chrom_vec = torch.full((z_orig.size(0),), int(chrom_embed_idx), dtype=torch.long, device=device)
                    z_hom = encode_head(z_orig, chrom_vec)
                    z_dec = decode_head(z_hom, chrom_vec)
                    logits = ae.decode(z_dec)
                    ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), xb.reshape(-1).long(), reduction="sum")
                    pred = logits.argmax(dim=-1)
                    val_ce_sum += float(ce.item())
                    val_corr += float((pred == xb).sum().item())
                    val_sites += float(xb.numel())

        mean_val_ce = val_ce_sum / max(1.0, val_sites)
        mean_val_acc = val_corr / max(1.0, val_sites)
        print(
            f"[TOK-STAGE2] epoch={epoch}/{args.epochs} train_loss={train_loss/max(1,train_steps):.6f} "
            f"val_ce={mean_val_ce:.6f} val_acc={mean_val_acc:.6f}"
        )

        if mean_val_ce < best_val_ce:
            best_val_ce = mean_val_ce
            best_val_acc = mean_val_acc
            best_epoch = epoch
            best_encode_state = {k: v.detach().cpu().clone() for k, v in encode_head.state_dict().items()}
            best_decode_state = {k: v.detach().cpu().clone() for k, v in decode_head.state_dict().items()}

        save_homog_per_chr(
            best_encode_state,
            best_decode_state,
            chromosomes,
            args.ae_checkpoints,
            epoch=epoch,
            mean_val_ce=mean_val_ce,
            mean_val_acc=mean_val_acc,
            best_epoch=best_epoch,
            best_val_ce=best_val_ce,
            best_val_acc=best_val_acc,
        )

    print(f"[TOK-STAGE2] Done. best_epoch={best_epoch} best_val_ce={best_val_ce:.6f}")


if __name__ == "__main__":
    main()
