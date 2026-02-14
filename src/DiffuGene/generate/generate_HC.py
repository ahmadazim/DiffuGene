#!/usr/bin/env python
"""
Generate samples from a trained DDPM UNet using DDIM sampling (from pure Gaussian noise).

- Loads EMA weights from your training checkpoint.
- Supports conditional generation with optional classifier-free guidance (CFG).
- Works with epsilon- or v_prediction training.
- Starts from x_T ~ N(0, I) and runs DDIM for num_steps down to t=0.
- Saves a single .pt file with the generated latents and (optionally) the covariates used.

Usage example:

python /n/home03/ahmadazim/WORKING/genGen/DiffuGene/scripts/generate_HC.py \
  --model-path /n/home03/ahmadazim/WORKING/genGen/UKBVQVAE/models/diffusion/DDPM_unrelWhite_allchr_128z_epoch19.pth \
  --prediction-type epsilon \
  --num-steps 50 \
  --num-samples 256 \
  --batch-size 32 \
  --eta 0.0 \
  --guidance-scale 7.0 \
  --save-path /n/home03/ahmadazim/WORKING/genGen/UKBVQVAE/genomic_data/generated_samples/test_256.pt \
  --cond \
  --covariate-file /n/home03/ahmadazim/WORKING/genGen/UKBVQVAE/covariates/all_covariates.csv \
  --fam-file /n/home03/ahmadazim/WORKING/genGen/UKBVQVAE/genomic_data/geneticBinary/ukb_allchr_unrel_britishWhite_conditional_diffusion_train.fam \
  --binary-cols SEX_MALE CAD HYPERTENSION T2D T1D STROKE CKD HYPERLIPIDEMIA LUNG_CANCER BREAST_CANCER PROSTATE_CANCER COLORECTAL_CANCER PANCREATIC_CANCER BIPOLAR MAJOR_DEPRESSION RA IBD AD_DIMENTIA PARKINSONS ATRIAL_FIBRILLATION CHOL_LOW_MEDS ASSESSMENT_CENTER_10003 ASSESSMENT_CENTER_11001 ASSESSMENT_CENTER_11002 ASSESSMENT_CENTER_11003 ASSESSMENT_CENTER_11004 ASSESSMENT_CENTER_11005 ASSESSMENT_CENTER_11006 ASSESSMENT_CENTER_11007 ASSESSMENT_CENTER_11008 ASSESSMENT_CENTER_11009 ASSESSMENT_CENTER_11010 ASSESSMENT_CENTER_11011 ASSESSMENT_CENTER_11012 ASSESSMENT_CENTER_11013 ASSESSMENT_CENTER_11014 ASSESSMENT_CENTER_11016 ASSESSMENT_CENTER_11017 ASSESSMENT_CENTER_11018 ASSESSMENT_CENTER_11020 ASSESSMENT_CENTER_11021 ASSESSMENT_CENTER_11022 ASSESSMENT_CENTER_11023 \
  --categorical-cols SMOKING_STATUS ALCOHOL_INTAKE \
  --cov-start-index 1
"""

import os
import json
import argparse
import hashlib
from typing import Optional, List, Tuple

import sys
import torch
from torch.utils.data import DataLoader, TensorDataset
from timm.utils import ModelEmaV3
from diffusers import DDIMScheduler

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from DiffuGene.diffusion.unet import LatentUNET2D as ConditionalUNET
from DiffuGene.utils import (
    setup_logging, get_logger,
    prepare_covariates_for_training,  # builds normalized covariate tensor aligned to fam
    sample_training_covariates,       # optional helper if you have metadata file
)

logger = get_logger(__name__)


def ckpt_signature(ckpt) -> Tuple[str, list]:
    try:
        keys = sorted(list(ckpt.keys()))
        sig = hashlib.md5(json.dumps(keys).encode()).hexdigest()
        return sig, keys[:12]
    except Exception:
        return "NA", []


def load_model_from_ckpt(model_path: str, device: torch.device) -> Tuple[torch.nn.Module, bool, Optional[int]]:
    """
    Load ConditionalUNET and EMA weights from checkpoint.
    Returns (model, is_conditional, cond_dim)
    """
    checkpoint = torch.load(model_path, map_location="cpu")
    sig, head = ckpt_signature(checkpoint)
    logger.info(f"[CKPT] path={model_path} keyset_md5={sig} sample_keys={head}")

    is_conditional = bool(checkpoint.get("conditional", False))
    cond_dim = checkpoint.get("cond_dim", None)
    logger.info(f"[CKPT] conditional={is_conditional} cond_dim_in_ckpt={cond_dim}")

    if "ema" not in checkpoint:
        raise KeyError("EMA weights not found in checkpoint. Training saves EMA; generation expects it.")

    # Build model
    model = ConditionalUNET(input_channels=64, output_channels=64, cond_dim=(cond_dim or 0))
    ema = ModelEmaV3(model, decay=0.9)
    ema.load_state_dict(checkpoint["ema"])
    model.load_state_dict(ema.module.state_dict())

    # Move to device
    model = model.to(device)
    model.eval()

    nparams = sum(p.numel() for p in model.parameters())
    logger.info(f"[MODEL] {model.__class__.__name__} params={nparams:,}")
    logger.info(f"[MODEL] has cond_emb={hasattr(model,'cond_emb')} null_cond_emb={hasattr(model,'null_cond_emb')}")

    # cleanup
    del checkpoint, ema
    return model, is_conditional, cond_dim


def build_covariates_if_needed(
    cond: bool,
    covariate_file: Optional[str],
    fam_file: Optional[str],
    binary_cols: Optional[List[str]],
    categorical_cols: Optional[List[str]],
    device: torch.device,
) -> Optional[torch.Tensor]:
    """
    Prepares a normalized covariate matrix aligned to fam.
    Returns a CPU tensor (we'll slice per batch and send to device).
    """
    if not cond:
        return None

    if covariate_file and fam_file:
        cov_t, cov_names, norm_params = prepare_covariates_for_training(
            covariate_path=covariate_file,
            fam_path=fam_file,
            binary_cols=binary_cols or [],
            categorical_cols=categorical_cols or [],
        )
        logger.info(f"[COV] Prepared covariates: shape={tuple(cov_t.shape)} features={len(cov_names)}")
        return cov_t  # CPU tensor
    else:
        raise ValueError("--cond was set but no --covariate-file/--fam-file provided.")


@torch.no_grad()
def ddim_generate(
    model: torch.nn.Module,
    num_samples: int,
    batch_size: int,
    num_steps: int,
    prediction_type: str,
    eta: float,
    guidance_scale: float,
    covariates_ordered: Optional[torch.Tensor],
    device: torch.device,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Generate latents from pure Gaussian noise, using DDIM.

    Returns:
      latents_all: (N, 64, 128, 128)
      cov_used:    (N, D) or None
    """
    assert prediction_type in ("epsilon", "v_prediction"), "prediction_type must be epsilon or v_prediction"

    # Setup DDIM scheduler consistent with training (betas schedule must match!)
    # You trained with linear betas in your training code.
    ddim = DDIMScheduler(
        num_train_timesteps=1000,
        beta_schedule="linear",
        beta_start=1e-4,
        beta_end=0.02,
        clip_sample=False,
        prediction_type=prediction_type,
    )
    ddim.set_timesteps(num_inference_steps=num_steps, device=device)  # descending

    C, H, W = 64, 128, 128
    out = []
    cov_out = []

    # helper to run unet once
    def unet_once(x_t: torch.Tensor, t: torch.Tensor, cond_vec: Optional[torch.Tensor]):
        h = model.input_proj(x_t)
        if cond_vec is not None and hasattr(model, "cond_emb"):
            e = model.cond_emb(cond_vec).unsqueeze(1)  # (B,1,hidden)
            u = model.unet(h, t, encoder_hidden_states=e).sample
        else:
            u = model.unet(h, t).sample
        return model.output_proj(u)

    # optional CFG at inference (only if model is conditional AND user requests >1.0 scale)
    use_cfg = (guidance_scale is not None and guidance_scale > 1.0 and hasattr(model, "cond_emb") and hasattr(model, "null_cond_emb"))

    remaining = int(num_samples)
    cursor = 0  # position within covariates_ordered
    while remaining > 0:
        bs = min(batch_size, remaining)
        remaining -= bs

        # pick cond rows sequentially to preserve order (no replacement)
        if covariates_ordered is not None:
            cov_batch_cpu = covariates_ordered[cursor:cursor + bs]
            if cov_batch_cpu.shape[0] != bs:
                raise ValueError(f"Insufficient covariate rows for batch size {bs} at cursor {cursor}.")
            cov_batch = cov_batch_cpu.to(device=device, dtype=next(model.parameters()).dtype)
            cursor += bs
        else:
            cov_batch = None

        # x_T ~ N(0,I)
        x = torch.randn(bs, C, H, W, device=device, dtype=next(model.parameters()).dtype).to(memory_format=torch.channels_last)

        # run DDIM
        for t in ddim.timesteps:
            t_vec = t.expand(bs).to(device)
            if use_cfg:
                # compute shared input projection once
                h = model.input_proj(x)

                # Unconditional branch uses learned null embedding already in cross-attn space
                null_emb = getattr(model, "null_cond_emb").expand(bs, -1).unsqueeze(1)  # (B,1,hidden)
                u_uncond = model.unet(h, t_vec, encoder_hidden_states=null_emb).sample
                y_uncond = model.output_proj(u_uncond)

                # Conditional branch uses embedded covariates
                if cov_batch is None:
                    raise ValueError("CFG requested but covariates are missing.")
                e = model.cond_emb(cov_batch).unsqueeze(1)  # (B,1,hidden)
                u_cond = model.unet(h, t_vec, encoder_hidden_states=e).sample
                y_cond = model.output_proj(u_cond)

                # Guidance: y = u + s (c - u)
                model_out = y_uncond + guidance_scale * (y_cond - y_uncond)
            else:
                model_out = unet_once(x, t_vec, cov_batch)

            step = ddim.step(model_output=model_out, timestep=t, sample=x, eta=eta)
            x = step.prev_sample

        out.append(x.to("cpu"))
        if cov_batch is not None:
            cov_out.append(cov_batch.to("cpu"))

        # free
        del x
        torch.cuda.empty_cache()
        
        print(f"Generated {cursor} samples")

    latents_all = torch.cat(out, dim=0)
    cov_used = torch.cat(cov_out, dim=0) if cov_out else None
    return latents_all, cov_used


def main():
    setup_logging()
    p = argparse.ArgumentParser(description="Generate samples from a trained DDPM UNet using DDIM (from Gaussian noise).")

    # Model / checkpoint
    p.add_argument("--model-path", type=str, required=True)

    # Sampling
    p.add_argument("--prediction-type", type=str, default="epsilon", choices=["epsilon", "v_prediction"])
    p.add_argument("--num-steps", type=int, default=50, help="DDIM steps")
    p.add_argument("--eta", type=float, default=0.0, help="DDIM stochasticity (0.0 = deterministic)")
    p.add_argument("--num-samples", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=32)

    # Conditional settings
    p.add_argument("--cond", action="store_true", help="Enable conditional generation")
    p.add_argument("--guidance-scale", type=float, default=1.0, help="CFG scale; >1 enables CFG at inference")
    p.add_argument("--covariate-file", type=str, default=None)
    p.add_argument("--fam-file", type=str, default=None)
    p.add_argument("--binary-cols", nargs="*", default=None)
    p.add_argument("--categorical-cols", nargs="*", default=None)
    p.add_argument(
        "--cov-start-index",
        type=int,
        default=1,
        help="1-indexed start row in the covariate file to begin sampling from (inclusive).",
    )

    # Output
    p.add_argument("--save-path", type=str, required=True)

    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model (+ EMA)
    model, is_conditional, cond_dim = load_model_from_ckpt(args.model_path, device=device)

    if args.cond and not is_conditional:
        raise ValueError("You requested --cond but the loaded checkpoint is UNCONDITIONAL.")
    if (not args.cond) and is_conditional:
        logger.warning("Checkpoint is conditional but --cond not set; generation will ignore covariates.")

    # Prepare covariates (CPU tensor), if needed
    covariates_full = build_covariates_if_needed(
        cond=args.cond,
        covariate_file=args.covariate_file,
        fam_file=args.fam_file,
        binary_cols=args.binary_cols,
        categorical_cols=args.categorical_cols,
        device=device,
    )

    # Slice covariates to preserve file order from a 1-indexed start row
    if args.cond:
        if args.cov_start_index < 1:
            raise ValueError("--cov-start-index must be >= 1 (1-indexed).")
        start_idx_0 = args.cov_start_index - 1
        total_rows = int(covariates_full.shape[0])
        end_idx = start_idx_0 + int(args.num_samples)
        if end_idx > total_rows:
            raise ValueError(
                f"Requested {args.num_samples} samples starting at row {args.cov_start_index} "
                f"requires rows up to {end_idx} but only {total_rows} covariate rows are available."
            )
        covariates_ordered = covariates_full[start_idx_0:end_idx]
        logger.info(f"[COV] Using covariate rows [{args.cov_start_index}..{end_idx}] (1-indexed start, exclusive end).")
    else:
        covariates_ordered = None

    # Generate
    latents, cov_used = ddim_generate(
        model=model,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        num_steps=args.num_steps,
        prediction_type=args.prediction_type,
        eta=args.eta,
        guidance_scale=args.guidance_scale,
        covariates_ordered=covariates_ordered,
        device=device,
    )

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    payload = {
        "latents": latents,          # (N,64,128,128), CPU
        "cond": bool(args.cond),
        "guidance_scale": float(args.guidance_scale),
        "prediction_type": args.prediction_type,
        "num_steps": int(args.num_steps),
        "eta": float(args.eta),
    }
    if cov_used is not None:
        payload["covariates"] = cov_used  # (N, D)

    torch.save(payload, args.save_path)
    logger.info(f"[DONE] Saved samples to {args.save_path}  |  latents={tuple(latents.shape)}")


if __name__ == "__main__":
    main()