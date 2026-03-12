#!/usr/bin/env python
"""
Two-stage latent generation for SiT/USiT flow models.

Workflow:
1. Sample Gaussian noise.
2. Transport it with an inverse-noise flow model.
3. Convert that output into the main model's normalized latent space.
4. Transport with the main flow model into final latent samples.
5. Optionally decode per chromosome with trained AEs.

This script is written to mirror the normalization and architecture choices used
in `DiffuGene.diffusion.train`, rather than assuming notebook-only settings.
"""

from __future__ import annotations

import argparse
import glob
import os
import random
import re
import sys
from contextlib import nullcontext
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

try:
    from ..utils import setup_logging, get_logger
    from ..diffusion.SiT import SiTFlowModel
    from ..VAEembed.ae import TokenAutoencoder1D
    from ..VAEembed.AE2d.ae import GenotypeAutoencoder
except ImportError:
    this_dir = os.path.dirname(__file__)
    src_root = os.path.abspath(os.path.join(this_dir, "..", "..", ".."))
    if src_root not in sys.path:
        sys.path.insert(0, src_root)
    from DiffuGene.utils import setup_logging, get_logger
    from DiffuGene.diffusion.SiT import SiTFlowModel
    from DiffuGene.VAEembed.ae import TokenAutoencoder1D
    from DiffuGene.VAEembed.AE2d.ae import GenotypeAutoencoder


logger = get_logger(__name__)

torch.backends.cuda.matmul.allow_tf32 = True
if torch.cuda.is_available():
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)


def resolve_amp_dtype(name: str) -> torch.dtype:
    name = str(name).lower().strip()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported amp dtype '{name}'. Use one of: bf16, fp16, fp32.")


def autocast_ctx(device: torch.device, enabled: bool, dtype: torch.dtype):
    if device.type != "cuda" or (not enabled) or dtype == torch.float32:
        return nullcontext()
    return torch.autocast(device_type="cuda", enabled=True, dtype=dtype)


def checkpoint_base_stem(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    return re.sub(r"_epoch\d+$", "", stem)


def extract_chr(path: str) -> int:
    name = os.path.basename(path)
    match = re.search(r"_chr(\d+)_memmap\.npy$", name)
    if match is None:
        raise ValueError(f"Could not parse chromosome from memmap path: {path}")
    return int(match.group(1))


def load_memmap(path: str) -> np.memmap:
    shape_file = path.replace(".npy", "_shape.txt")
    if not os.path.exists(shape_file):
        raise FileNotFoundError(f"Missing shape file for memmap: {shape_file}")
    with open(shape_file, "r", encoding="utf-8") as handle:
        shape = tuple(int(x) for x in handle.read().strip().split(","))
    if len(shape) != 3:
        raise ValueError(f"Expected memmap shape (N,T,D). Got {shape} for {path}")
    return np.memmap(path, dtype="float32", mode="r", shape=shape)


def make_token_offsets(chr_to_tokens: Dict[int, int]) -> Dict[int, Tuple[int, int]]:
    offsets: Dict[int, Tuple[int, int]] = {}
    start = 0
    for chrom in sorted(chr_to_tokens.keys()):
        n_tokens = int(chr_to_tokens[chrom])
        offsets[chrom] = (start, start + n_tokens)
        start += n_tokens
    return offsets


def genotype_calls_from_logits(logits: torch.Tensor) -> torch.Tensor:
    if logits.dim() != 3:
        raise ValueError(f"logits must be 3D. Got shape {tuple(logits.shape)}")
    if logits.shape[-1] == 3:
        return logits.argmax(dim=-1)
    if logits.shape[1] == 3:
        return logits.argmax(dim=1)
    raise ValueError(f"Could not infer class dimension for logits shape {tuple(logits.shape)}")


def discover_memmaps(memmap_dir: str, main_checkpoint_path: str, pattern: Optional[str] = None) -> List[str]:
    if pattern:
        memmap_paths = sorted(glob.glob(pattern), key=extract_chr)
        if memmap_paths:
            return memmap_paths
        raise FileNotFoundError(f"No memmaps matched --memmap-pattern: {pattern}")

    run_stem = checkpoint_base_stem(main_checkpoint_path)
    default_pattern = os.path.join(memmap_dir, f"{run_stem}_chr*_memmap.npy")
    memmap_paths = sorted(glob.glob(default_pattern), key=extract_chr)
    if memmap_paths:
        return memmap_paths

    fallback_pattern = os.path.join(memmap_dir, "*_chr*_memmap.npy")
    memmap_paths = sorted(glob.glob(fallback_pattern), key=extract_chr)
    if memmap_paths:
        logger.warning(
            "No memmaps matched checkpoint stem '%s'. Falling back to all chromosome memmaps in %s.",
            run_stem,
            memmap_dir,
        )
        return memmap_paths

    raise FileNotFoundError(
        f"No chromosome memmaps found in {memmap_dir}. Tried {default_pattern} and {fallback_pattern}."
    )


def load_ae_for_chr(ae_models_dir: str, chrom: int, device: torch.device) -> torch.nn.Module:
    ckpt = os.path.join(ae_models_dir, f"ae_chr{chrom}.pt")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"AE checkpoint not found: {ckpt}")

    payload = torch.load(ckpt, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid AE checkpoint payload: {ckpt}")

    cfg = payload.get("config", {})
    state = payload.get("model_state", payload.get("weights"))
    if state is None:
        raise KeyError(f"AE checkpoint missing 'model_state'/'weights': {ckpt}")

    if isinstance(cfg, dict) and ("latent_length" in cfg and "latent_dim" in cfg):
        ae = TokenAutoencoder1D(
            input_length=int(cfg["input_length"]),
            latent_length=int(cfg["latent_length"]),
            latent_dim=int(cfg["latent_dim"]),
            embed_dim=int(cfg.get("embed_dim", 8)),
            max_c=int(cfg.get("max_c", 5)),
            dropout=float(cfg.get("dropout", 0.0)),
        )
    elif isinstance(cfg, dict) and all(k in cfg for k in ("input_length", "K1", "K2", "C")):
        ae = GenotypeAutoencoder(
            input_length=int(cfg["input_length"]),
            K1=int(cfg["K1"]),
            K2=int(cfg["K2"]),
            C=int(cfg["C"]),
            embed_dim=int(cfg.get("embed_dim", 8)),
        )
    else:
        raise ValueError(f"Unsupported AE config schema in {ckpt}. config keys={list(cfg.keys())}")

    ae.load_state_dict(state, strict=True)
    ae = ae.to(device)
    ae.eval()
    for param in ae.parameters():
        param.requires_grad_(False)
    return ae


def load_flow_model(
    checkpoint_path: str,
    device: torch.device,
    use_ema: bool = True,
    hidden_dim_override: Optional[int] = None,
) -> Tuple[SiTFlowModel, Dict]:
    payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid checkpoint payload at {checkpoint_path}")

    model_type = str(payload.get("model_type", "")).lower()
    if model_type not in {"sit", "usit"}:
        raise ValueError(f"Checkpoint model_type must be sit/usit, got '{model_type}'")

    latent_shape = payload.get("latent_shape")
    if latent_shape is None or len(latent_shape) != 2:
        raise ValueError(f"Expected token latent_shape=(L,D) in checkpoint. Got {latent_shape}")
    latent_length = int(latent_shape[0])
    token_dim = int(latent_shape[1])

    sit_cfg = payload.get("sit_config", {}) or {}
    source_token_ids = payload.get("source_token_ids", None)
    num_sources = sit_cfg.get("num_sources", None)
    if source_token_ids is not None:
        source_token_ids = source_token_ids.long()
    if num_sources is not None:
        num_sources = int(num_sources)

    hidden_dim = (
        int(hidden_dim_override)
        if hidden_dim_override is not None
        else int(sit_cfg.get("hidden_dim", token_dim))
    )

    model = SiTFlowModel(
        token_dim=token_dim,
        latent_length=latent_length,
        hidden_dim=hidden_dim,
        cond_dim=None,
        num_layers=int(sit_cfg.get("num_layers", 9)),
        num_heads=int(sit_cfg.get("num_heads", 8)),
        mlp_ratio=int(sit_cfg.get("mlp_ratio", 4)),
        dropout=float(sit_cfg.get("dropout", 0.0)),
        qkv_bias=bool(sit_cfg.get("qkv_bias", False)),
        use_udit=bool(sit_cfg.get("use_udit", model_type == "usit")),
        source_token_ids=source_token_ids,
        num_sources=num_sources,
    ).to(device)

    weights = payload.get("ema") if use_ema else payload.get("weights")
    if weights is None:
        raise KeyError("Checkpoint missing requested weights ('ema' or 'weights').")
    if any(key.startswith("module.") for key in weights.keys()):
        weights = {key.split("module.", 1)[1]: value for key, value in weights.items()}

    model.load_state_dict(weights, strict=True)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model, payload


def get_norm_tensors(payload: Dict, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    latent_shape = payload.get("latent_shape")
    if latent_shape is None or len(latent_shape) != 2:
        raise ValueError(f"Checkpoint missing valid latent_shape. Got {latent_shape}")
    latent_length = int(latent_shape[0])
    token_dim = int(latent_shape[1])

    source_token_ids = payload.get("source_token_ids")
    per_source_mean = payload.get("per_source_mean")
    per_source_std = payload.get("per_source_std")
    channel_mean = payload.get("channel_mean")
    channel_std = payload.get("channel_std")

    if source_token_ids is not None and per_source_mean is not None and per_source_std is not None:
        sid = source_token_ids.long().view(-1)
        if sid.numel() != latent_length:
            raise ValueError(f"source_token_ids length {sid.numel()} != latent_length {latent_length}")
        mu = per_source_mean.float()[sid].unsqueeze(0)
        sd = per_source_std.float()[sid].unsqueeze(0)
    elif channel_mean is not None and channel_std is not None:
        mu = channel_mean.float().view(1, 1, token_dim).expand(1, latent_length, token_dim).contiguous()
        sd = channel_std.float().view(1, 1, token_dim).expand(1, latent_length, token_dim).contiguous()
    else:
        mu = torch.zeros(1, latent_length, token_dim, dtype=torch.float32)
        sd = torch.ones(1, latent_length, token_dim, dtype=torch.float32)

    sd = torch.clamp(sd, min=1e-6)
    return mu.to(device=device, dtype=torch.float32), sd.to(device=device, dtype=torch.float32)


def is_identity_norm(mu: torch.Tensor, sd: torch.Tensor, atol: float = 1e-6) -> bool:
    mu_cpu = mu.detach().float().cpu()
    sd_cpu = sd.detach().float().cpu()
    return bool(
        torch.allclose(mu_cpu, torch.zeros_like(mu_cpu), atol=atol, rtol=0.0)
        and torch.allclose(sd_cpu, torch.ones_like(sd_cpu), atol=atol, rtol=0.0)
    )


def validate_compatible_checkpoints(main_payload: Dict, inverse_payload: Dict) -> None:
    main_shape = tuple(main_payload.get("latent_shape", ()))
    inverse_shape = tuple(inverse_payload.get("latent_shape", ()))
    if main_shape != inverse_shape:
        raise ValueError(
            f"Main and inverse checkpoints must share latent_shape. Got main={main_shape}, inverse={inverse_shape}"
        )

    main_sid = main_payload.get("source_token_ids")
    inverse_sid = inverse_payload.get("source_token_ids")
    if (main_sid is None) != (inverse_sid is None):
        logger.warning(
            "Only one checkpoint carries source_token_ids. Sampling will still proceed using per-checkpoint norms."
        )
        return
    if main_sid is not None and not torch.equal(main_sid.long().view(-1), inverse_sid.long().view(-1)):
        logger.warning(
            "source_token_ids differ between checkpoints. Proceeding, but token ordering must still correspond exactly."
        )


@torch.no_grad()
def solve_euler(model: SiTFlowModel, z: torch.Tensor, steps: int) -> torch.Tensor:
    if int(steps) < 1:
        return z
    batch = z.shape[0]
    dt = 1.0 / float(steps)
    for k in range(int(steps)):
        t_cur = float(k) * dt
        t_batch = torch.full((batch,), t_cur, device=z.device, dtype=z.dtype)
        z = z + dt * model(z, t_batch)
    return z


@torch.no_grad()
def solve_rk2(model: SiTFlowModel, z: torch.Tensor, steps: int) -> torch.Tensor:
    if int(steps) < 1:
        return z
    batch = z.shape[0]
    dt = 1.0 / float(steps)
    for k in range(int(steps)):
        t0 = float(k) * dt
        t1 = float(k + 1) * dt
        t0_batch = torch.full((batch,), t0, device=z.device, dtype=z.dtype)
        t1_batch = torch.full((batch,), t1, device=z.device, dtype=z.dtype)
        v0 = model(z, t0_batch)
        z_euler = z + dt * v0
        v1 = model(z_euler, t1_batch)
        z = z + 0.5 * dt * (v0 + v1)
    return z


@torch.no_grad()
def solve_dpm(model: SiTFlowModel, z: torch.Tensor, steps: int) -> torch.Tensor:
    if int(steps) < 1:
        return z
    t_steps = torch.linspace(0.0, 1.0, int(steps) + 1, device=z.device, dtype=z.dtype)
    dt = 1.0 / float(steps)
    v_prev = None
    for i in range(int(steps)):
        t_batch = torch.full((z.shape[0],), t_steps[i], device=z.device, dtype=z.dtype)
        v_pred = model(z, t_batch)
        if i == 0:
            z = z + dt * v_pred
        else:
            z = z + dt * (1.5 * v_pred - 0.5 * v_prev)
        v_prev = v_pred
    return z


SOLVER_MAP = {
    "euler": solve_euler,
    "rk2": solve_rk2,
    "dpm": solve_dpm,
}


def decode_per_chromosome(
    latents_cpu: torch.Tensor,
    chroms: List[int],
    offsets: Dict[int, Tuple[int, int]],
    ae_models: Dict[int, torch.nn.Module],
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> Dict[int, np.ndarray]:
    calls_by_chr: Dict[int, np.ndarray] = {}
    for chrom in chroms:
        start, end = offsets[chrom]
        zc = latents_cpu[:, start:end, :].to(device=device, dtype=torch.float32, non_blocking=(device.type == "cuda"))
        with autocast_ctx(device=device, enabled=use_amp, dtype=amp_dtype):
            logits = ae_models[chrom].decode(zc)
        calls = genotype_calls_from_logits(logits).detach().cpu().numpy().astype(np.int64)
        calls_by_chr[chrom] = calls
        del zc, logits
    return calls_by_chr


def save_outputs(
    output_dir: str,
    output_prefix: str,
    latents_by_chr: Dict[int, List[torch.Tensor]],
    calls_by_chr: Optional[Dict[int, List[np.ndarray]]],
    metadata: Dict,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    final_latents_by_chr: Dict[int, torch.Tensor] = {
        chrom: torch.cat(chunks, dim=0) for chrom, chunks in latents_by_chr.items()
    }
    merged_latents = torch.cat([final_latents_by_chr[chrom] for chrom in sorted(final_latents_by_chr.keys())], dim=1)

    latent_payload = {
        "latents": merged_latents,
        "latents_by_chr": final_latents_by_chr,
        "metadata": metadata,
    }
    latent_path = os.path.join(output_dir, f"{output_prefix}_latents.pt")
    torch.save(latent_payload, latent_path)
    logger.info("Saved merged latent payload to %s", latent_path)

    for chrom, tensor in final_latents_by_chr.items():
        chr_path = os.path.join(output_dir, f"{output_prefix}_chr{chrom}_latents.pt")
        torch.save(tensor, chr_path)
        logger.info("Saved chr%s latents to %s", chrom, chr_path)

    if calls_by_chr is not None:
        for chrom, chunk_list in calls_by_chr.items():
            calls = np.concatenate(chunk_list, axis=0)
            calls_path = os.path.join(output_dir, f"{output_prefix}_chr{chrom}_calls.npy")
            np.save(calls_path, calls)
            logger.info("Saved chr%s decoded calls to %s", chrom, calls_path)


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
    solver_inverse = SOLVER_MAP[args.inverse_solver]
    solver_main = SOLVER_MAP[args.main_solver]

    logger.info("Using device=%s | amp=%s dtype=%s", device, use_amp, amp_dtype)

    memmap_paths = discover_memmaps(
        memmap_dir=args.memmap_dir,
        main_checkpoint_path=args.main_checkpoint_path,
        pattern=args.memmap_pattern,
    )
    chroms = [extract_chr(path) for path in memmap_paths]
    memmaps = {extract_chr(path): load_memmap(path) for path in memmap_paths}
    chr_tokens = {chrom: int(memmaps[chrom].shape[1]) for chrom in chroms}
    token_dims = {int(memmaps[chrom].shape[2]) for chrom in chroms}
    if len(token_dims) != 1:
        raise ValueError(f"Expected a single token_dim across chromosomes, got {sorted(token_dims)}")
    token_dim = int(next(iter(token_dims)))
    total_tokens = int(sum(chr_tokens.values()))
    offsets = make_token_offsets(chr_tokens)

    logger.info(
        "Chromosome layout: %s",
        ", ".join(f"chr{chrom}:{chr_tokens[chrom]}" for chrom in chroms),
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
            f"Memmap-derived token layout {(total_tokens, token_dim)} does not match checkpoint latent_shape {latent_shape}"
        )

    mu_inverse, sd_inverse = get_norm_tensors(inverse_payload, device=device)
    mu_main, sd_main = get_norm_tensors(main_payload, device=device)
    inverse_has_norm = not is_identity_norm(mu_inverse, sd_inverse)
    main_has_norm = not is_identity_norm(mu_main, sd_main)

    if inverse_has_norm:
        logger.info("Inverse checkpoint normalization detected.")
    else:
        logger.info("No norm detected in inverse checkpoint; skipping inverse de-normalization.")

    if main_has_norm:
        logger.info("Main checkpoint normalization detected.")
    else:
        logger.info("No norm detected in main checkpoint; skipping main normalization/de-normalization.")

    ae_models: Optional[Dict[int, torch.nn.Module]] = None
    if args.ae_models_dir:
        ae_models = {chrom: load_ae_for_chr(args.ae_models_dir, chrom, device) for chrom in chroms}
        logger.info("Loaded AE models for chromosomes: %s", chroms)

    latents_by_chr: Dict[int, List[torch.Tensor]] = {chrom: [] for chrom in chroms}
    calls_by_chr: Optional[Dict[int, List[np.ndarray]]] = (
        {chrom: [] for chrom in chroms} if ae_models is not None else None
    )

    for start in tqdm(range(0, int(args.num_samples), int(args.batch_size)), desc="Generating"):
        batch_n = min(int(args.batch_size), int(args.num_samples) - start)
        z = torch.randn((batch_n, total_tokens, token_dim), device=device, dtype=torch.float32)

        with autocast_ctx(device=device, enabled=use_amp, dtype=amp_dtype):
            z = solver_inverse(inverse_model, z, int(args.inverse_steps))

        # Convert inverse-model output back to the shared latent coordinate system,
        # then into the normalization expected by the main model only when needed.
        z = z.float()
        if inverse_has_norm:
            z = z * sd_inverse + mu_inverse
        if main_has_norm:
            z = (z - mu_main) / sd_main

        with autocast_ctx(device=device, enabled=use_amp, dtype=amp_dtype):
            z = solver_main(main_model, z, int(args.main_steps))

        z = z.float()
        if main_has_norm:
            z = z * sd_main + mu_main
        z_cpu = z.detach().cpu()

        for chrom in chroms:
            s_idx, e_idx = offsets[chrom]
            latents_by_chr[chrom].append(z_cpu[:, s_idx:e_idx, :].contiguous())

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

        del z, z_cpu
        if device.type == "cuda":
            torch.cuda.empty_cache()

    metadata = {
        "main_checkpoint_path": args.main_checkpoint_path,
        "inverse_checkpoint_path": args.inverse_checkpoint_path,
        "memmap_dir": args.memmap_dir,
        "memmap_pattern": args.memmap_pattern,
        "num_samples": int(args.num_samples),
        "batch_size": int(args.batch_size),
        "inverse_steps": int(args.inverse_steps),
        "main_steps": int(args.main_steps),
        "inverse_solver": args.inverse_solver,
        "main_solver": args.main_solver,
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
        description="Generate latents by chaining inverse-noise and main SiT/USiT flow models."
    )
    parser.add_argument("--main-checkpoint-path", type=str, required=True, help="Main flow checkpoint")
    parser.add_argument("--inverse-checkpoint-path", type=str, required=True, help="Noise->inverse-noise checkpoint")
    parser.add_argument("--memmap-dir", type=str, required=True, help="Directory containing chromosome memmaps")
    parser.add_argument(
        "--memmap-pattern",
        type=str,
        default=None,
        help="Optional explicit glob for chromosome memmaps. Overrides checkpoint-stem discovery.",
    )
    parser.add_argument("--output-dir", type=str, required=True, help="Directory for generated outputs")
    parser.add_argument("--output-prefix", type=str, default="generated", help="Prefix for saved files")
    parser.add_argument("--num-samples", type=int, default=1000, help="Number of samples to generate")
    parser.add_argument("--batch-size", type=int, default=128, help="Sampling batch size")
    parser.add_argument("--inverse-steps", type=int, default=50, help="ODE steps for inverse-noise model")
    parser.add_argument("--main-steps", type=int, default=50, help="ODE steps for main model")
    parser.add_argument(
        "--inverse-solver",
        type=str,
        default="dpm",
        choices=sorted(SOLVER_MAP.keys()),
        help="Solver for the inverse-noise flow",
    )
    parser.add_argument(
        "--main-solver",
        type=str,
        default="dpm",
        choices=sorted(SOLVER_MAP.keys()),
        help="Solver for the main flow",
    )
    parser.add_argument(
        "--ae-models-dir",
        type=str,
        default=None,
        help="Optional AE directory. If provided, decoded genotype calls are also saved per chromosome.",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device, e.g. cuda or cpu")
    parser.add_argument("--amp-dtype", type=str, default="bf16", help="Autocast dtype: bf16, fp16, fp32")
    parser.add_argument("--disable-amp", action="store_true", help="Disable autocast during sampling/decoding")
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
