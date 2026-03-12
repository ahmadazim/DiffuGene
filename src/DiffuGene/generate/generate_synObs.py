#!/usr/bin/env python
"""
Synthetic-observed generation for chained SiT/USiT flow models.

This script starts from observed training latents instead of fresh Gaussian noise:
1. Load observed training latents.
2. Normalize into the main-model training space.
3. Flow backward through the main model.
4. Convert into the inverse-model training space.
5. Flow backward through the inverse model to obtain inverse-inverse-noise.
6. Flow forward through the inverse model and then the main model.
7. Save generated latents and optionally decode them per chromosome.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

try:
    from ..utils import setup_logging, get_logger
    from ..diffusion.SiT import SiTFlowModel
    from ..diffusion.train import MultiChromosomeMemmapDataset, read_prepare_data
    from .generate import (
        SOLVER_MAP,
        autocast_ctx,
        decode_per_chromosome,
        get_norm_tensors,
        is_identity_norm,
        load_ae_for_chr,
        load_flow_model,
        make_token_offsets,
        resolve_amp_dtype,
        save_outputs,
        validate_compatible_checkpoints,
    )
except ImportError:
    this_dir = os.path.dirname(__file__)
    src_root = os.path.abspath(os.path.join(this_dir, "..", "..", ".."))
    if src_root not in sys.path:
        sys.path.insert(0, src_root)
    from DiffuGene.utils import setup_logging, get_logger
    from DiffuGene.diffusion.SiT import SiTFlowModel
    from DiffuGene.diffusion.train import MultiChromosomeMemmapDataset, read_prepare_data
    from DiffuGene.generate.generate import (
        SOLVER_MAP,
        autocast_ctx,
        decode_per_chromosome,
        get_norm_tensors,
        is_identity_norm,
        load_ae_for_chr,
        load_flow_model,
        make_token_offsets,
        resolve_amp_dtype,
        save_outputs,
        validate_compatible_checkpoints,
    )


logger = get_logger(__name__)

torch.backends.cuda.matmul.allow_tf32 = True
if torch.cuda.is_available():
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)


def reverse_euler(model: SiTFlowModel, z: torch.Tensor, steps: int) -> torch.Tensor:
    if int(steps) < 1:
        return z
    dt = 1.0 / float(steps)
    for k in range(int(steps)):
        t_cur = 1.0 - float(k) * dt
        t_batch = torch.full((z.shape[0],), t_cur, device=z.device, dtype=z.dtype)
        z = z - dt * model(z, t_batch)
    return z


def reverse_rk2(model: SiTFlowModel, z: torch.Tensor, steps: int) -> torch.Tensor:
    if int(steps) < 1:
        return z
    batch = z.shape[0]
    dt = 1.0 / float(steps)
    for k in range(int(steps)):
        t1 = 1.0 - dt * k
        t0 = 1.0 - dt * (k + 1)
        t1_batch = torch.full((batch,), float(t1), device=z.device, dtype=z.dtype)
        t0_batch = torch.full((batch,), float(t0), device=z.device, dtype=z.dtype)
        v1 = model(z, t1_batch)
        z_euler = z - dt * v1
        v0 = model(z_euler, t0_batch)
        z = z - 0.5 * dt * (v1 + v0)
    return z


def reverse_dpm(model: SiTFlowModel, z: torch.Tensor, steps: int) -> torch.Tensor:
    if int(steps) < 1:
        return z
    dt = 1.0 / float(steps)
    t_steps = torch.linspace(0.0, 1.0, int(steps) + 1, device=z.device, dtype=z.dtype)
    v_prev = None
    for i in range(int(steps)):
        t = t_steps[i]
        t_batch = torch.full((z.shape[0],), t, device=z.device, dtype=z.dtype)
        v = model(z, 1.0 - t_batch)
        if i == 0:
            z = z - dt * v
        else:
            z = z - dt * (1.5 * v - 0.5 * v_prev)
        v_prev = v
    return z


REVERSE_SOLVER_MAP = {
    "euler": reverse_euler,
    "rk2": reverse_rk2,
    "dpm": reverse_dpm,
}


def get_layout_from_dataset(dataset: MultiChromosomeMemmapDataset) -> Tuple[List[int], Dict[int, int], int, int]:
    chroms = list(dataset.chromosomes)
    chr_tokens = {int(chrom): int(dataset.tokens_per_chr[chrom]) for chrom in chroms}
    total_tokens = int(dataset.total_tokens)
    token_dim = int(dataset.latent_dim)
    return chroms, chr_tokens, total_tokens, token_dim


def load_observed_slice(dataset: MultiChromosomeMemmapDataset, start: int, end: int) -> torch.Tensor:
    parts = [torch.from_numpy(np.asarray(dataset.arrays[chrom][start:end])) for chrom in dataset.chromosomes]
    return torch.cat(parts, dim=1).contiguous().float()


def prepare_observed_dataset(args: argparse.Namespace) -> MultiChromosomeMemmapDataset:
    cache_dir = args.memmap_cache_dir or args.output_dir
    os.makedirs(cache_dir, exist_ok=True)
    train_latent_path = os.path.abspath(os.path.expanduser(str(args.train_latent_path)))
    if not os.path.exists(train_latent_path):
        raise FileNotFoundError(
            f"Observed training latent path does not exist: {train_latent_path}. "
            "Expected a directory containing chr*/batch*_latents.pt, an existing memmap, "
            "or another path format accepted by diffusion.train.read_prepare_data()."
        )
    dataset = read_prepare_data(
        path=train_latent_path,
        output_folder=cache_dir,
        model_output_path=args.main_checkpoint_path,
    )
    if not isinstance(dataset, MultiChromosomeMemmapDataset):
        raise ValueError(
            "Observed synthetic generation expects multi-chromosome token latents. "
            f"Got dataset type: {type(dataset).__name__}"
        )
    return dataset


def generate(args: argparse.Namespace) -> None:
    setup_logging()

    if args.seed is not None:
        random.seed(int(args.seed))
        np.random.seed(int(args.seed))
        torch.manual_seed(int(args.seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(args.seed))

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    amp_dtype = resolve_amp_dtype(args.amp_dtype)
    use_amp = (not args.disable_amp) and device.type == "cuda"

    inverse_forward_solver = SOLVER_MAP[args.inverse_solver]
    main_forward_solver = SOLVER_MAP[args.main_solver]
    inverse_backward_solver = REVERSE_SOLVER_MAP[args.inverse_backward_solver or args.inverse_solver]
    main_backward_solver = REVERSE_SOLVER_MAP[args.main_backward_solver or args.main_solver]

    logger.info("Using device=%s | amp=%s dtype=%s", device, use_amp, amp_dtype)

    observed_dataset = prepare_observed_dataset(args)
    chroms, chr_tokens, total_tokens, token_dim = get_layout_from_dataset(observed_dataset)
    offsets = make_token_offsets(chr_tokens)
    logger.info(
        "Observed training layout: %s",
        ", ".join(f"chr{chrom}:{chr_tokens[chrom]}" for chrom in chroms),
    )

    total_available = len(observed_dataset)
    if args.data_start_index < 1:
        raise ValueError("--data-start-index must be >= 1 (1-indexed).")
    start_idx0 = int(args.data_start_index) - 1
    end_idx = start_idx0 + int(args.num_samples)
    if end_idx > total_available:
        raise ValueError(
            f"Requested {args.num_samples} samples starting at row {args.data_start_index} "
            f"requires rows up to {end_idx}, but only {total_available} rows are available."
        )

    main_model, main_payload = load_flow_model(
        checkpoint_path=args.main_checkpoint_path,
        device=device,
        use_ema=(not args.use_main_model_weights),
        hidden_dim_override=args.main_hidden_dim,
    )
    inverse_model, inverse_payload = load_flow_model(
        checkpoint_path=args.inverse_checkpoint_path,
        device=device,
        use_ema=(not args.use_inverse_model_weights),
        hidden_dim_override=args.inverse_hidden_dim,
    )
    validate_compatible_checkpoints(main_payload, inverse_payload)

    latent_shape = tuple(main_payload.get("latent_shape", ()))
    if latent_shape != (total_tokens, token_dim):
        raise ValueError(
            f"Observed dataset token layout {(total_tokens, token_dim)} does not match checkpoint latent_shape {latent_shape}"
        )

    mu_inverse, sd_inverse = get_norm_tensors(inverse_payload, device=device)
    mu_main, sd_main = get_norm_tensors(main_payload, device=device)
    inverse_has_norm = not is_identity_norm(mu_inverse, sd_inverse)
    main_has_norm = not is_identity_norm(mu_main, sd_main)

    if inverse_has_norm:
        logger.info("Inverse checkpoint normalization detected.")
    else:
        logger.info("No norm detected in inverse checkpoint; skipping inverse normalization transforms.")

    if main_has_norm:
        logger.info("Main checkpoint normalization detected.")
    else:
        logger.info("No norm detected in main checkpoint; skipping main normalization transforms.")

    ae_models: Optional[Dict[int, torch.nn.Module]] = None
    if args.ae_models_dir:
        ae_models = {chrom: load_ae_for_chr(args.ae_models_dir, chrom, device) for chrom in chroms}
        logger.info("Loaded AE models for chromosomes: %s", chroms)

    latents_by_chr: Dict[int, List[torch.Tensor]] = {chrom: [] for chrom in chroms}
    calls_by_chr: Optional[Dict[int, List[np.ndarray]]] = (
        {chrom: [] for chrom in chroms} if ae_models is not None else None
    )

    for start in tqdm(range(start_idx0, end_idx, int(args.batch_size)), desc="Generating synthetic observed"):
        stop = min(end_idx, start + int(args.batch_size))
        x_obs_cpu = load_observed_slice(observed_dataset, start, stop)
        z = x_obs_cpu.to(device=device, dtype=torch.float32, non_blocking=(device.type == "cuda"))

        # Raw observed latents -> main normalized training space.
        if main_has_norm:
            z = (z - mu_main) / sd_main

        # Backward through main model: observed data -> inverse-noise shared space.
        with autocast_ctx(device=device, enabled=use_amp, dtype=amp_dtype):
            z = main_backward_solver(main_model, z, int(args.main_steps))

        z = z.float()

        # Shared space -> inverse-model normalized training space.
        if inverse_has_norm:
            z = (z - mu_inverse) / sd_inverse

        # Backward through inverse model: inverse noise -> inverse-inverse noise (~Gaussian).
        with autocast_ctx(device=device, enabled=use_amp, dtype=amp_dtype):
            z = inverse_backward_solver(inverse_model, z, int(args.inverse_steps))

        z = z.float()

        # Forward again as in generate.py.
        with autocast_ctx(device=device, enabled=use_amp, dtype=amp_dtype):
            z = inverse_forward_solver(inverse_model, z, int(args.inverse_steps))

        z = z.float()
        if inverse_has_norm:
            z = z * sd_inverse + mu_inverse
        if main_has_norm:
            z = (z - mu_main) / sd_main

        with autocast_ctx(device=device, enabled=use_amp, dtype=amp_dtype):
            z = main_forward_solver(main_model, z, int(args.main_steps))

        z = z.float()
        if main_has_norm:
            z = z * sd_main + mu_main
        z_cpu = z.detach().cpu()

        offset = 0
        for chrom in chroms:
            next_offset = offset + chr_tokens[chrom]
            latents_by_chr[chrom].append(z_cpu[:, offset:next_offset, :].contiguous())
            offset = next_offset

        if ae_models is not None and calls_by_chr is not None:
            decoded_calls = decode_per_chromosome(
                latents_cpu=z_cpu,
                chroms=chroms,
                offsets=offsets,
                ae_models=ae_models,
                device=device,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
            )
            for chrom in chroms:
                calls_by_chr[chrom].append(decoded_calls[chrom])

        del x_obs_cpu, z, z_cpu
        if device.type == "cuda":
            torch.cuda.empty_cache()

    metadata = {
        "mode": "synthetic_observed",
        "main_checkpoint_path": args.main_checkpoint_path,
        "inverse_checkpoint_path": args.inverse_checkpoint_path,
        "train_latent_path": args.train_latent_path,
        "memmap_cache_dir": args.memmap_cache_dir,
        "data_start_index": int(args.data_start_index),
        "num_samples": int(args.num_samples),
        "batch_size": int(args.batch_size),
        "inverse_steps": int(args.inverse_steps),
        "main_steps": int(args.main_steps),
        "inverse_solver": args.inverse_solver,
        "main_solver": args.main_solver,
        "inverse_backward_solver": args.inverse_backward_solver or args.inverse_solver,
        "main_backward_solver": args.main_backward_solver or args.main_solver,
        "chromosomes": chroms,
        "tokens_per_chr": chr_tokens,
        "token_dim": token_dim,
        "device": str(device),
        "amp_enabled": bool(use_amp),
        "amp_dtype": args.amp_dtype,
        "seed": args.seed,
    }

    save_outputs(
        output_dir=args.output_dir,
        output_prefix=args.output_prefix,
        latents_by_chr=latents_by_chr,
        calls_by_chr=calls_by_chr,
        metadata=metadata,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate synthetic observed samples by cycling training latents backward and forward through chained flows."
    )
    parser.add_argument("--main-checkpoint-path", type=str, required=True, help="Main flow checkpoint")
    parser.add_argument("--inverse-checkpoint-path", type=str, required=True, help="Noise-correction flow checkpoint")
    parser.add_argument(
        "--train-latent-path",
        type=str,
        required=True,
        help="Observed training latents: per-chromosome batch directory or existing memmap-backed source supported by diffusion.train.read_prepare_data().",
    )
    parser.add_argument(
        "--memmap-cache-dir",
        type=str,
        default=None,
        help="Optional directory for memmaps created from batch latents. Defaults to output-dir.",
    )
    parser.add_argument("--output-dir", type=str, required=True, help="Directory for generated outputs")
    parser.add_argument("--output-prefix", type=str, default="synthetic_observed", help="Prefix for saved files")
    parser.add_argument("--num-samples", type=int, default=1000, help="Number of observed rows to process")
    parser.add_argument(
        "--data-start-index",
        type=int,
        default=1,
        help="1-indexed start row within the observed training latents.",
    )
    parser.add_argument("--batch-size", type=int, default=128, help="Number of observed samples processed at once")
    parser.add_argument("--inverse-steps", type=int, default=50, help="ODE steps for the inverse model")
    parser.add_argument("--main-steps", type=int, default=50, help="ODE steps for the main model")
    parser.add_argument(
        "--inverse-solver",
        type=str,
        default="dpm",
        choices=sorted(SOLVER_MAP.keys()),
        help="Forward solver for the inverse model.",
    )
    parser.add_argument(
        "--main-solver",
        type=str,
        default="dpm",
        choices=sorted(SOLVER_MAP.keys()),
        help="Forward solver for the main model.",
    )
    parser.add_argument(
        "--inverse-backward-solver",
        type=str,
        default=None,
        choices=sorted(REVERSE_SOLVER_MAP.keys()),
        help="Optional backward solver for the inverse model. Defaults to --inverse-solver.",
    )
    parser.add_argument(
        "--main-backward-solver",
        type=str,
        default=None,
        choices=sorted(REVERSE_SOLVER_MAP.keys()),
        help="Optional backward solver for the main model. Defaults to --main-solver.",
    )
    parser.add_argument(
        "--ae-models-dir",
        type=str,
        default=None,
        help="Optional AE directory. If provided, decoded genotype calls are also saved per chromosome.",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device, e.g. cuda or cpu")
    parser.add_argument("--amp-dtype", type=str, default="bf16", help="Autocast dtype: bf16, fp16, fp32")
    parser.add_argument("--disable-amp", action="store_true", help="Disable autocast during flow/decoding")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--use-main-model-weights",
        action="store_true",
        help="Use raw main model weights instead of EMA weights.",
    )
    parser.add_argument(
        "--use-inverse-model-weights",
        action="store_true",
        help="Use raw inverse model weights instead of EMA weights.",
    )
    parser.add_argument(
        "--main-hidden-dim",
        type=int,
        default=None,
        help="Optional hidden_dim override for older main checkpoints that did not save it.",
    )
    parser.add_argument(
        "--inverse-hidden-dim",
        type=int,
        default=None,
        help="Optional hidden_dim override for older inverse checkpoints that did not save it.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    generate(args)


if __name__ == "__main__":
    main()
