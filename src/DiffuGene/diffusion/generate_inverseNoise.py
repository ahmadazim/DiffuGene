#!/usr/bin/env python
import argparse
import glob
import os
import re
from typing import Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

try:
    from .SiT import SiTFlowModel
except ImportError:
    this_dir = os.path.dirname(__file__)
    src_root = os.path.abspath(os.path.join(this_dir, "..", "..", ".."))
    if src_root not in os.sys.path:
        os.sys.path.insert(0, src_root)
    from DiffuGene.diffusion.SiT import SiTFlowModel


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)


def extract_batch_num(path: str) -> int:
    name = os.path.basename(path)
    m = re.search(r"batch(\d+)", name)
    if m:
        return int(m.group(1))
    nums = re.findall(r"(\d+)", name)
    return int(nums[-1]) if nums else -1


def discover_chr_latent_batches(root_dir: str) -> Dict[int, List[str]]:
    chr_batches: Dict[int, List[str]] = {}
    if not os.path.isdir(root_dir):
        return chr_batches
    for entry in sorted(os.listdir(root_dir)):
        m = re.match(r"^chr(\d+)$", entry)
        if m is None:
            continue
        chrom = int(m.group(1))
        chr_dir = os.path.join(root_dir, entry)
        files = sorted(glob.glob(os.path.join(chr_dir, "batch*_latents.pt")), key=extract_batch_num)
        if not files:
            files = sorted(glob.glob(os.path.join(chr_dir, "batch*.pt")), key=extract_batch_num)
        if files:
            chr_batches[chrom] = files
    return chr_batches


def map_batches_by_index(chr_batches: Dict[int, List[str]]) -> Tuple[List[int], Dict[int, Dict[int, str]]]:
    if not chr_batches:
        raise ValueError("No chromosome batches found.")
    chroms = sorted(chr_batches.keys())
    batch_maps: Dict[int, Dict[int, str]] = {}
    batch_sets = []
    for c in chroms:
        m: Dict[int, str] = {}
        for p in chr_batches[c]:
            idx = extract_batch_num(p)
            if idx < 0:
                raise ValueError(f"Could not parse batch index from: {p}")
            m[idx] = p
        if not m:
            raise ValueError(f"No batch files found for chr{c}")
        batch_maps[c] = m
        batch_sets.append(set(m.keys()))
    common = sorted(set.intersection(*batch_sets))
    if not common:
        raise ValueError("No common batch indices across chromosomes.")
    missing_report = []
    for c in chroms:
        missing = sorted(set(common) - set(batch_maps[c].keys()))
        if missing:
            missing_report.append(f"chr{c}: missing {missing[:5]}")
    if missing_report:
        raise ValueError("Inconsistent chromosome batches. " + " | ".join(missing_report))
    by_batch: Dict[int, Dict[int, str]] = {}
    for idx in common:
        by_batch[idx] = {c: batch_maps[c][idx] for c in chroms}
    return chroms, by_batch


def load_latent_batch(path: str) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, torch.Tensor):
        x = obj
    elif isinstance(obj, dict) and "latents" in obj:
        x = obj["latents"]
    elif isinstance(obj, dict) and "data" in obj:
        x = obj["data"]
    else:
        x = torch.as_tensor(obj)
    if x.dim() != 3:
        raise ValueError(f"Expected latent batch tensor shape (N,T,D), got {tuple(x.shape)} from {path}")
    return x.float().contiguous()


def load_flow_model(checkpoint_path: str, device: torch.device, use_ema: bool = True) -> Tuple[SiTFlowModel, Dict]:
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

    model = SiTFlowModel(
        token_dim=token_dim,
        latent_length=latent_length,
        hidden_dim=int(sit_cfg.get("hidden_dim", 1152)),
        cond_dim=None,
        num_layers=int(sit_cfg.get("num_layers", 9)),
        num_heads=int(sit_cfg.get("num_heads", 8)),
        mlp_ratio=int(sit_cfg.get("mlp_ratio", 4)),
        dropout=float(sit_cfg.get("dropout", 0.0)),
        use_udit=bool(sit_cfg.get("use_udit", model_type == "usit")),
        source_token_ids=source_token_ids,
        num_sources=num_sources,
        qkv_bias=bool(sit_cfg.get("qkv_bias", True)),
    ).to(device)

    weights = payload.get("ema") if use_ema else payload.get("weights")
    if weights is None:
        raise KeyError("Checkpoint missing requested weights ('ema' or 'weights').")
    if any(k.startswith("module.") for k in weights.keys()):
        weights = {k.split("module.", 1)[1]: v for k, v in weights.items()}
    model.load_state_dict(weights, strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, payload


@torch.no_grad()
def solve_dpm_backward(model: SiTFlowModel, x: torch.Tensor, n_steps: int) -> torch.Tensor:
    if int(n_steps) < 1:
        return x
    dt = 1.0 / float(n_steps)
    t_steps = torch.linspace(0.0, 1.0, int(n_steps) + 1, device=x.device, dtype=x.dtype)
    v_prev = None
    for i in range(int(n_steps)):
        t = t_steps[i]
        t_batch = torch.full((x.shape[0],), t, device=x.device, dtype=x.dtype)
        v = model(x, (1.0 - t_batch))
        if i == 0:
            x = x - dt * v
        else:
            x = x - dt * (1.5 * v - 0.5 * v_prev)
        v_prev = v
    return x


def subset_batches(all_ids: List[int], chosen: Optional[List[int]]) -> List[int]:
    if not chosen:
        return all_ids
    chosen_set = set(int(x) for x in chosen)
    selected = [b for b in all_ids if b in chosen_set]
    missing = sorted(chosen_set - set(selected))
    if missing:
        raise ValueError(f"Requested batch indices not found in all chromosomes: {missing}")
    return selected


@torch.no_grad()
def run_batch(
    model: SiTFlowModel,
    files_by_chr: Dict[int, str],
    chroms: List[int],
    device: torch.device,
    model_batch_size: int,
    n_steps: int,
) -> Tuple[torch.Tensor, Dict[int, int]]:
    tensors_by_chr: Dict[int, torch.Tensor] = {}
    n_samples = None
    token_dim = None
    tokens_per_chr: Dict[int, int] = {}

    for c in chroms:
        x = load_latent_batch(files_by_chr[c])
        if n_samples is None:
            n_samples = int(x.shape[0])
        elif int(x.shape[0]) != int(n_samples):
            raise ValueError(f"Sample mismatch for chr{c}: {x.shape[0]} vs {n_samples}")
        if token_dim is None:
            token_dim = int(x.shape[2])
        elif int(x.shape[2]) != int(token_dim):
            raise ValueError(f"Token-dim mismatch for chr{c}: {x.shape[2]} vs {token_dim}")
        tensors_by_chr[c] = x
        tokens_per_chr[c] = int(x.shape[1])

    chunks: List[torch.Tensor] = []
    for start in tqdm(range(0, int(n_samples), int(model_batch_size)), desc="Reverse batches", leave=False):
        end = min(int(n_samples), start + int(model_batch_size))
        x_cpu = torch.cat([tensors_by_chr[c][start:end] for c in chroms], dim=1).contiguous()
        x = x_cpu.to(device=device, dtype=torch.float32, non_blocking=(device.type == "cuda"))
        z0 = solve_dpm_backward(model, x, n_steps=n_steps)
        chunks.append(z0.detach().cpu())
        del x_cpu, x, z0
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return torch.cat(chunks, dim=0), tokens_per_chr


def main():
    parser = argparse.ArgumentParser(description="Generate inverse noise batches from chromosome latent batches.")
    parser.add_argument("--checkpoint-path", type=str, required=True, help="SiT/USiT training checkpoint path")
    parser.add_argument("--input-root", type=str, required=True, help="Root directory containing chr*/batch*.pt")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for inverse-noise batches")
    parser.add_argument("--steps", type=int, default=25, help="Reverse DPM steps")
    parser.add_argument("--model-batch-size", type=int, default=128, help="Mini-batch size for model stepping")
    parser.add_argument("--samples-per-save", type=int, default=12000, help="Expected samples per source batch")
    parser.add_argument("--batch-indices", type=int, nargs="+", default=None, help="Optional subset of batch indices to process")
    parser.add_argument("--device", type=str, default="cuda", help="Device, e.g. cuda or cpu")
    parser.add_argument("--use-model-weights", action="store_true", help="Use raw model weights instead of EMA")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    args = parser.parse_args()

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    chr_batches = discover_chr_latent_batches(args.input_root)
    if not chr_batches:
        raise FileNotFoundError(
            f"No chromosome batch files found under {args.input_root}. Expected chr*/batch*_latents.pt or chr*/batch*.pt"
        )
    chroms, by_batch = map_batches_by_index(chr_batches)
    all_batch_ids = sorted(by_batch.keys())
    selected_batch_ids = subset_batches(all_batch_ids, args.batch_indices)

    print(f"[INFO] device={device}")
    print(f"[INFO] chromosomes={chroms}")
    print(f"[INFO] discovered_batches={len(all_batch_ids)} selected_batches={len(selected_batch_ids)}")
    print(f"[INFO] reverse_solver=DPMSolver steps={args.steps} model_batch_size={args.model_batch_size}")

    model, payload = load_flow_model(
        checkpoint_path=args.checkpoint_path,
        device=device,
        use_ema=(not args.use_model_weights),
    )
    ckpt_latent_shape = payload.get("latent_shape", None)
    if ckpt_latent_shape is None or len(ckpt_latent_shape) != 2:
        raise ValueError(f"Checkpoint missing valid latent_shape. Got {ckpt_latent_shape}")
    ckpt_tokens = int(ckpt_latent_shape[0])
    ckpt_dim = int(ckpt_latent_shape[1])

    for batch_idx in selected_batch_ids:
        out_file = os.path.join(args.output_dir, f"batch{batch_idx:04d}_inverseNoise.pt")
        if os.path.exists(out_file) and (not args.overwrite):
            print(f"[INFO] skipping existing: {out_file}")
            continue

        print(f"[INFO] processing batch index {batch_idx}")
        inv_noise, tokens_per_chr = run_batch(
            model=model,
            files_by_chr=by_batch[batch_idx],
            chroms=chroms,
            device=device,
            model_batch_size=int(args.model_batch_size),
            n_steps=int(args.steps),
        )

        total_tokens = int(sum(tokens_per_chr[c] for c in chroms))
        if int(inv_noise.shape[1]) != ckpt_tokens or int(inv_noise.shape[2]) != ckpt_dim:
            raise ValueError(
                f"Batch {batch_idx}: merged shape {tuple(inv_noise.shape)} incompatible with checkpoint latent_shape "
                f"({ckpt_tokens}, {ckpt_dim}); merged_total_tokens={total_tokens}"
            )
        if int(args.samples_per_save) > 0 and int(inv_noise.shape[0]) != int(args.samples_per_save):
            print(
                f"[WARN] batch {batch_idx} has {inv_noise.shape[0]} samples (expected {args.samples_per_save}). "
                "Saving anyway."
            )

        torch.save(inv_noise, out_file)
        print(f"[INFO] saved {tuple(inv_noise.shape)} to {out_file}")

    print("[INFO] done")


if __name__ == "__main__":
    main()