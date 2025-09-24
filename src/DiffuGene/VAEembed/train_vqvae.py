#!/usr/bin/env python

import os
import argparse
import glob
from typing import List
import bisect

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import h5py

from .vae import SNPVQVAE, build_vqvae
from ..utils.file_utils import read_bim_file
from ..utils import setup_logging, get_logger


logger = get_logger(__name__)


def log_model_summary(model: SNPVQVAE, x_sample: torch.Tensor, device: torch.device) -> None:
    """Run a dry forward through encoder/decoder and log activation shapes layer-by-layer.
    Uses module hooks on encoder/decoder blocks and explicit logs around quantizer.
    """
    model.eval()
    x_sample = x_sample.to(device)

    # Collect shapes
    enc_logs = []
    dec_logs = []

    def make_hook(store_list, name):
        def _hook(_mod, _inp, out):
            try:
                if isinstance(out, (tuple, list)):
                    out = out[0]
                if torch.is_tensor(out):
                    store_list.append((name, tuple(out.shape)))
            except Exception:
                pass
        return _hook

    # Register hooks on encoder and decoder_up
    enc_handles = []
    for name, mod in model.encoder.named_modules():
        if name == "":
            continue
        enc_handles.append(mod.register_forward_hook(make_hook(enc_logs, f"encoder.{name}")))

    dec_handles = []
    for name, mod in model.decoder_up.named_modules():
        if name == "":
            continue
        dec_handles.append(mod.register_forward_hook(make_hook(dec_logs, f"decoder_up.{name}")))

    try:
        with torch.no_grad():
            # Pad/reorder and embed path
            x_grid, _ = model.pad_and_reorder(x_sample)
            # Encoder
            h_enc = model.encoder(x_grid)
            B, C, H, W = h_enc.shape
            logger.info(f"[ModelSummary] encoder output: (B,C,H,W)=({B},{C},{H},{W})")
            for n, s in enc_logs:
                logger.info(f"[ModelSummary] {n}: {s}")
            # Tokens
            z_e_seq = h_enc.view(B, C, H * W)
            logger.info(f"[ModelSummary] z_e_seq: {tuple(z_e_seq.shape)}")
            # Quantizer
            z_q_seq, commit_loss, _, _ = model.quantizer(z_e_seq)
            logger.info(f"[ModelSummary] z_q_seq: {tuple(z_q_seq.shape)} | commit={commit_loss.item():.6f}")
            # Decoder path (mirror of decode_logits without reorder)
            T = z_q_seq.size(2)
            grid_dim = int(T ** 0.5)
            h = z_q_seq.view(B, z_q_seq.size(1), grid_dim, grid_dim)
            h_dec = model.decoder_up(h)
            logger.info(f"[ModelSummary] decoder_up output: {tuple(h_dec.shape)}")
            for n, s in dec_logs:
                logger.info(f"[ModelSummary] {n}: {s}")
            # Final head
            y = model.out_head(h_dec)
            logger.info(f"[ModelSummary] out_head output: {tuple(y.shape)}")
    except Exception as e:
        logger.warning(f"[ModelSummary] Failed to log shapes due to: {e}")
    finally:
        for h in enc_handles + dec_handles:
            try:
                h.remove()
            except Exception:
                pass
    model.train()


class H5BatchConcatDataset(Dataset):
    """Dataset that reads per-chromosome per-batch H5 caches and concatenates across chromosomes per sample.
    Directory structure: <h5_dir>/chr<no>/batchXXXXX.h5 with datasets X (B, L_chr), iid (B), bp (L_chr).
    """
    def __init__(self, h5_dir: str, chromosomes: List[int]):
        self.h5_dir = h5_dir
        self.chromosomes = chromosomes
        # Discover batch ids from one chromosome directory
        first_chr_dir = os.path.join(h5_dir, f"chr{chromosomes[0]}")
        batch_files = sorted(glob.glob(os.path.join(first_chr_dir, "batch*.h5")))
        if not batch_files:
            raise FileNotFoundError(f"No H5 caches found in {first_chr_dir}")
        self.batches = [os.path.splitext(os.path.basename(p))[0] for p in batch_files]  # batchXXXXX
        # Load per-chromosome lengths
        self.chr_lengths = {}
        for c in chromosomes:
            f0 = os.path.join(h5_dir, f"chr{c}", f"{self.batches[0]}.h5")
            if not os.path.exists(f0):
                raise FileNotFoundError(f"Missing H5 for chr{c}: {f0}")
            with h5py.File(f0, 'r') as f:
                self.chr_lengths[c] = f['X'].shape[1]
        self.total_len = sum(self.chr_lengths[c] for c in chromosomes)
        # Determine per-batch sample sizes from the first chromosome directory.
        # Assumption (enforced below): per-chromosome row counts match within a batch.
        self.batch_sizes = []
        for bname in self.batches:
            fpath_first = os.path.join(first_chr_dir, f"{bname}.h5")
            with h5py.File(fpath_first, 'r') as f:
                rows = int(f['X'].shape[0])
            if rows <= 0:
                raise ValueError(f"Empty batch detected for {bname}")
            # Sanity check across all selected chromosomes
            for c in chromosomes:
                fpath = os.path.join(h5_dir, f"chr{c}", f"{bname}.h5")
                with h5py.File(fpath, 'r') as f:
                    r_c = int(f['X'].shape[0])
                if r_c != rows:
                    raise ValueError(
                        f"Row count mismatch within batch {bname}: chr{chromosomes[0]}={rows} vs chr{c}={r_c}. "
                        f"Ensure caches are complete and consistent."
                    )
            self.batch_sizes.append(rows)
        # Build cumulative sizes for fast index mapping (variable batch sizes)
        self.cum_sizes = []
        total = 0
        for sz in self.batch_sizes:
            total += sz
            self.cum_sizes.append(total)

    def __len__(self):
        return self.cum_sizes[-1]

    def _load_sample(self, batch_idx: int, sample_idx_in_batch: int) -> torch.Tensor:
        arrays = []
        for c in self.chromosomes:
            p = os.path.join(self.h5_dir, f"chr{c}", f"{self.batches[batch_idx]}.h5")
            with h5py.File(p, 'r') as f:
                x = f['X'][sample_idx_in_batch]  # (L_chr,)
                arrays.append(torch.from_numpy(x.astype('int64')))
        return torch.cat(arrays, dim=0)

    def __getitem__(self, idx: int) -> torch.Tensor:
        # Map global index to (batch_idx, sample_idx) with variable batch sizes
        if idx < 0 or idx >= self.cum_sizes[-1]:
            raise IndexError(idx)
        batch_idx = bisect.bisect_right(self.cum_sizes, idx)
        prev_cum = 0 if batch_idx == 0 else self.cum_sizes[batch_idx - 1]
        sample_idx = idx - prev_cum
        x = self._load_sample(batch_idx, sample_idx)
        return x


def main():
    p = argparse.ArgumentParser(description="Train VQ-VAE on H5 caches")
    p.add_argument("--h5-dir", required=True)
    p.add_argument("--chromosomes", nargs='+', type=int, default=list(range(1,23)))
    p.add_argument("--bim", type=str, default=None, help="Optional BIM file to infer input length")
    p.add_argument("--latent-dim", type=int, default=64)
    p.add_argument("--codebook-size", type=int, default=1024)
    p.add_argument("--num-quantizers", type=int, default=2)
    p.add_argument("--beta-commit", type=float, default=0.25)
    p.add_argument("--latent-grid-dim", type=int, default=16)
    p.add_argument("--hidden-channels", type=int, default=64)
    p.add_argument("--width-mult-per-stage", type=float, default=1.0)
    p.add_argument("--ema-decay", type=float, default=0.99)
    p.add_argument("--ema-eps", type=float, default=1e-5)
    p.add_argument("--ld-lambda", type=float, default=1e-3)
    p.add_argument("--maf-lambda", type=float, default=0.0)
    p.add_argument("--ld-window", type=int, default=128)
    # Optional direct S->T mapping parameters
    p.add_argument("--init-down-kernel", type=int, default=0)
    p.add_argument("--init-down-stride", type=int, default=0)
    p.add_argument("--init-down-padding", type=int, default=0)
    p.add_argument("--init-down-out-channels", type=int, default=0)
    p.add_argument("--keep-layers-at-T", type=int, default=0)
    p.add_argument("--dec-up-kernel", type=int, default=0)
    p.add_argument("--dec-up-stride", type=int, default=0)
    p.add_argument("--dec-up-padding", type=int, default=0)
    p.add_argument("--dec-up-output-padding", type=int, default=0)
    p.add_argument("--dec-up-out-channels", type=int, default=0)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--save-path", required=True)
    args = p.parse_args()

    setup_logging()

    dataset = H5BatchConcatDataset(args.h5_dir, args.chromosomes)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)

    # Infer input length (L) either from BIM across selected chromosomes or from H5 shapes
    if args.bim:
        L = 0
        for c in args.chromosomes:
            bim_chr = read_bim_file(args.bim, c)
            L += int(bim_chr.shape[0])
    else:
        # Sum lengths from first batch H5 files
        L = dataset.total_len

    # Report dataset stats
    total_records = len(dataset)
    logger.info(
        f"[VQ-VAE Train] samples={total_records} | features(L)={L} | batch_size={args.batch_size} | "
        f"num_h5_batches={len(dataset.batches)} | chromosomes={args.chromosomes}"
    )

    model, optim = build_vqvae(
        input_length=L,
        latent_dim=args.latent_dim,
        codebook_size=args.codebook_size,
        num_quantizers=args.num_quantizers,
        beta_commit=args.beta_commit,
        lr=args.lr,
        latent_grid_dim=args.latent_grid_dim,
        ld_lambda=args.ld_lambda,
        maf_lambda=args.maf_lambda,
        ld_window=args.ld_window,
        ema_decay=args.ema_decay,
        ema_eps=args.ema_eps,
        hidden_channels=args.hidden_channels,
        width_mult_per_stage=args.width_mult_per_stage,
        init_down_kernel=int(args.init_down_kernel),
        init_down_stride=int(args.init_down_stride),
        init_down_padding=int(args.init_down_padding),
        init_down_out_channels=int(args.init_down_out_channels),
        keep_layers_at_T=int(args.keep_layers_at_T),
        dec_up_kernel=int(args.dec_up_kernel),
        dec_up_stride=int(args.dec_up_stride),
        dec_up_padding=int(args.dec_up_padding),
        dec_up_output_padding=int(args.dec_up_output_padding),
        dec_up_out_channels=int(args.dec_up_out_channels)
    )

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith('cuda') else 'cpu')
    model.to(device)
    model.train()

    # Log a one-batch model summary with user-provided params
    try:
        # Prepare a small sample batch from dataset
        first_batch = next(iter(DataLoader(dataset, batch_size=min(4, max(1, args.batch_size)), shuffle=False)))
        log_model_summary(model, first_batch, device)
        # Also log config parameters that control S->T/T->~S mapping
        logger.info(
            f"[ModelParams] init_down: k={model.cfg.init_down_kernel}, s={model.cfg.init_down_stride}, "
            f"p={model.cfg.init_down_padding}, out_ch={model.cfg.init_down_out_channels}, keep_at_T={model.cfg.keep_layers_at_T}; "
            f"dec_up: k={model.cfg.dec_up_kernel}, s={model.cfg.dec_up_stride}, p={model.cfg.dec_up_padding}, out_ch={model.cfg.dec_up_out_channels}, op={model.cfg.dec_up_output_padding}"
        )
    except Exception as e:
        logger.warning(f"Skipped detailed model summary due to: {e}")

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

    for epoch in range(args.epochs):
        total = 0.0
        tot_recon = 0.0
        tot_commit = 0.0
        tot_maf = 0.0
        tot_ld = 0.0
        tot_n = 0
        for x in tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            x = x.to(device)
            optim.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=(device.type=='cuda')):
                logits3, z_q_seq, commit_loss, _, _, mask_tokens = model(x)
                loss, metrics = model.loss_function(logits3, x, commit_loss, mask_tokens)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optim)
            scaler.update()
            B = x.size(0)
            tot_n += B
            total += loss.item() * B
            tot_recon += float(metrics["recon"]) * B
            tot_commit += float(metrics["commit"]) * B
            tot_maf += float(metrics["aux_maf"]) * B
            tot_ld += float(metrics["aux_ld"]) * B
        logger.info(
            f"Epoch {epoch+1}: loss={total/max(1,tot_n):.4f}, "
            f"recon={tot_recon/max(1,tot_n):.4f}, commit={tot_commit/max(1,tot_n):.4f}, "
            f"aux_maf={tot_maf/max(1,tot_n):.6f}, aux_ld={tot_ld/max(1,tot_n):.6f}"
        )
        if device.type == 'cuda':
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, 'ipc_collect'):
                torch.cuda.ipc_collect()

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save({
        'model_state': model.state_dict(),
        'config': model.cfg.__dict__,
    }, args.save_path)
    logger.info(f"Saved VQ-VAE to {args.save_path}")


if __name__ == "__main__":
    main()


