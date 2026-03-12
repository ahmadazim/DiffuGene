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
if torch.cuda.is_available():
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)


def extract_batch_num(path: str) -> int:
    name = os.path.basename(path)
    match = re.search(r"batch(\d+)", name)
    if match:
        return int(match.group(1))
    nums = re.findall(r"(\d+)", name)
    return int(nums[-1]) if nums else -1


def discover_inverse_noise_batches(root_dir: str) -> Dict[int, str]:
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"Input directory not found: {root_dir}")

    patterns = [
        os.path.join(root_dir, "batch*_inverseNoise.pt"),
        os.path.join(root_dir, "batch*.pt"),
    ]
    files: List[str] = []
    for pattern in patterns:
        files = sorted(glob.glob(pattern), key=extract_batch_num)
        if files:
            break

    if not files:
        raise FileNotFoundError(
            f"No inverse-noise batches found under {root_dir}. "
            "Expected files like batch0001_inverseNoise.pt or batch0001.pt"
        )

    by_batch: Dict[int, str] = {}
    for path in files:
        batch_idx = extract_batch_num(path)
        if batch_idx < 0:
            raise ValueError(f"Could not parse batch index from {path}")
        by_batch[batch_idx] = path
    return by_batch


def subset_batches(all_ids: List[int], chosen: Optional[List[int]]) -> List[int]:
    if not chosen:
        return all_ids
    chosen_set = set(int(x) for x in chosen)
    selected = [b for b in all_ids if b in chosen_set]
    missing = sorted(chosen_set - set(selected))
    if missing:
        raise ValueError(f"Requested batch indices not found: {missing}")
    return selected


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
        else int(sit_cfg.get("hidden_dim", 1152))
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
        use_udit=bool(sit_cfg.get("use_udit", model_type == "usit")),
        source_token_ids=source_token_ids,
        num_sources=num_sources,
        qkv_bias=bool(sit_cfg.get("qkv_bias", True)),
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


@torch.no_grad()
def run_batch(
    model: SiTFlowModel,
    batch_file: str,
    device: torch.device,
    model_batch_size: int,
    n_steps: int,
) -> torch.Tensor:
    x_all = load_latent_batch(batch_file)
    chunks: List[torch.Tensor] = []
    for start in tqdm(range(0, int(x_all.shape[0]), int(model_batch_size)), desc="Reverse batches", leave=False):
        end = min(int(x_all.shape[0]), start + int(model_batch_size))
        x = x_all[start:end].to(device=device, dtype=torch.float32, non_blocking=(device.type == "cuda"))
        z0 = solve_dpm_backward(model, x, n_steps=n_steps)
        chunks.append(z0.detach().cpu())
        del x, z0
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return torch.cat(chunks, dim=0)


def main():
    parser = argparse.ArgumentParser(
        description="Generate inverse-inverse-noise batches from existing inverse-noise batches."
    )
    parser.add_argument("--checkpoint-path", type=str, required=True, help="SiT/USiT inverse-noise model checkpoint")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing batch*_inverseNoise.pt")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for inverse-inverse-noise")
    parser.add_argument("--steps", type=int, default=25, help="Reverse DPM steps")
    parser.add_argument("--model-batch-size", type=int, default=128, help="Mini-batch size for model stepping")
    parser.add_argument("--samples-per-save", type=int, default=12000, help="Expected samples per saved batch")
    parser.add_argument("--batch-indices", type=int, nargs="+", default=None, help="Optional subset of batch indices to process")
    parser.add_argument("--device", type=str, default="cuda", help="Device, e.g. cuda or cpu")
    parser.add_argument("--use-model-weights", action="store_true", help="Use raw model weights instead of EMA")
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=None,
        help="Optional hidden_dim override for older checkpoints that did not save it.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    args = parser.parse_args()

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    batches_by_index = discover_inverse_noise_batches(args.input_dir)
    all_batch_ids = sorted(batches_by_index.keys())
    selected_batch_ids = subset_batches(all_batch_ids, args.batch_indices)

    print(f"[INFO] device={device}")
    print(f"[INFO] discovered_batches={len(all_batch_ids)} selected_batches={len(selected_batch_ids)}")
    print(f"[INFO] reverse_solver=DPMSolver steps={args.steps} model_batch_size={args.model_batch_size}")

    model, payload = load_flow_model(
        checkpoint_path=args.checkpoint_path,
        device=device,
        use_ema=(not args.use_model_weights),
        hidden_dim_override=args.hidden_dim,
    )
    ckpt_latent_shape = payload.get("latent_shape", None)
    if ckpt_latent_shape is None or len(ckpt_latent_shape) != 2:
        raise ValueError(f"Checkpoint missing valid latent_shape. Got {ckpt_latent_shape}")
    ckpt_tokens = int(ckpt_latent_shape[0])
    ckpt_dim = int(ckpt_latent_shape[1])

    for batch_idx in selected_batch_ids:
        in_file = batches_by_index[batch_idx]
        out_file = os.path.join(args.output_dir, f"batch{batch_idx:04d}_invInverseNoise.pt")
        if os.path.exists(out_file) and (not args.overwrite):
            print(f"[INFO] skipping existing: {out_file}")
            continue

        print(f"[INFO] processing batch index {batch_idx} from {in_file}")
        inv_inv_noise = run_batch(
            model=model,
            batch_file=in_file,
            device=device,
            model_batch_size=int(args.model_batch_size),
            n_steps=int(args.steps),
        )

        if int(inv_inv_noise.shape[1]) != ckpt_tokens or int(inv_inv_noise.shape[2]) != ckpt_dim:
            raise ValueError(
                f"Batch {batch_idx}: shape {tuple(inv_inv_noise.shape)} incompatible with checkpoint latent_shape "
                f"({ckpt_tokens}, {ckpt_dim})"
            )
        if int(args.samples_per_save) > 0 and int(inv_inv_noise.shape[0]) != int(args.samples_per_save):
            print(
                f"[WARN] batch {batch_idx} has {inv_inv_noise.shape[0]} samples (expected {args.samples_per_save}). "
                "Saving anyway."
            )

        torch.save(inv_inv_noise, out_file)
        print(f"[INFO] saved {tuple(inv_inv_noise.shape)} to {out_file}")

    print("[INFO] done")


if __name__ == "__main__":
    main()
