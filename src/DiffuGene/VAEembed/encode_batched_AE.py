#!/usr/bin/env python
import os
import sys
import argparse
import glob
import json
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import h5py
from contextlib import nullcontext

this_dir = os.path.dirname(__file__)
src_root = os.path.abspath(os.path.join(this_dir, "..", "..", ".."))
if src_root not in sys.path:
    sys.path.insert(0, src_root)

from DiffuGene.VAEembed.vae import GenotypeAutoencoder, VAEConfig
from DiffuGene.VAEembed.sharedEmbed import FiLM2D
from DiffuGene.utils import ensure_dir_exists
from DiffuGene.utils.file_utils import read_bim_file
try:
    from torch.cuda.amp import autocast as cuda_autocast
except Exception:  # CPU-only environments
    cuda_autocast = None


def get_chromosomes(spec: List[str]) -> List[int]:
    if len(spec) == 1 and str(spec[0]).lower() == 'all':
        return list(range(1, 23))
    return [int(x) for x in spec]


def _read_raw_header(raw_path: str) -> List[str]:
    with open(raw_path, 'r') as f:
        header = f.readline().strip().split()
    return header


def write_raw_to_h5_fast(raw_path: str,
                         h5_path: str,
                         expected_rows: int,
                         bp: np.ndarray,
                         snp_ids: List[str],
                         chunk_rows: int = 10000) -> None:
    """Stream PLINK .raw file chunks into an H5 dataset, replacing NA with 0."""
    header = _read_raw_header(raw_path)
    L_raw = len(header) - 6
    if L_raw <= 0:
        raise ValueError(f"Unexpected .raw format: no SNP columns in {raw_path}")
    if bp.size == 0:
        raise ValueError("BP array is empty; cannot determine SNP count")

    L = min(L_raw, int(bp.shape[0]))
    if L_raw != bp.shape[0]:
        print(f"[ENC-AE] SNP count mismatch .raw={L_raw} vs BIM {bp.shape[0]}; using min={L}")

    ensure_dir_exists(os.path.dirname(h5_path))
    with h5py.File(h5_path, "w") as f:
        dset_X = f.create_dataset("X", shape=(expected_rows, L), dtype='i1', compression="gzip", compression_opts=4)
        dset_iid = f.create_dataset("iid", shape=(expected_rows,), dtype=h5py.string_dtype(encoding='utf-8'))
        f.create_dataset("bp", data=bp[:L])
        try:
            f.create_dataset(
                "snp_ids",
                data=np.array(snp_ids[:L], dtype=object),
                dtype=h5py.string_dtype(encoding='utf-8'),
            )
        except Exception:
            pass

        offset = 0
        usecols = list(range(0, 2)) + list(range(6, 6 + L))
        for df in pd.read_csv(
            raw_path,
            sep=r"\s+",
            header=0,
            usecols=usecols,
            chunksize=chunk_rows,
            na_values=['NA'],
        ):
            iids = df.iloc[:, 1].astype(str).to_numpy()
            snp_df = df.iloc[:, 2:]
            X_chunk = snp_df.fillna(0).to_numpy(dtype=np.int16, copy=False)
            X_chunk = np.asarray(X_chunk, dtype=np.int8)
            n = X_chunk.shape[0]
            dset_X[offset:offset + n, :L] = X_chunk
            dset_iid[offset:offset + n] = iids
            offset += n

        if offset != expected_rows:
            print(f"[ENC-AE] Row count mismatch for {raw_path}: expected {expected_rows}, wrote {offset}")


def build_snp_info(bim_file: str, chromosomes: List[int]) -> Tuple[List[str], np.ndarray, Dict[int, Tuple[int, int]]]:
    """Return SNP ids, base-pair positions, and column offsets per chromosome."""
    all_ids: List[str] = []
    all_bp: List[np.ndarray] = []
    chr_offsets: Dict[int, Tuple[int, int]] = {}
    col_start = 0
    for chr_no in chromosomes:
        bim_chr = read_bim_file(bim_file, chr_no)
        snp_ids_chr = bim_chr["SNP"].astype(str).tolist()
        bp_chr = bim_chr["BP"].astype(np.int64).values
        all_ids.extend(snp_ids_chr)
        all_bp.append(bp_chr)
        chr_len = len(snp_ids_chr)
        chr_offsets[chr_no] = (col_start, col_start + chr_len)
        col_start += chr_len

    bp = np.concatenate(all_bp, axis=0) if all_bp else np.array([], dtype=np.int64)
    return all_ids, bp, chr_offsets


def run_plink_recode_batch(bfile_prefix: str, chromosomes: List[int], keep_tsv: str, out_prefix: str) -> str:
    import subprocess
    cmd = ["plink", "--bfile", bfile_prefix]
    if chromosomes:
        cmd += ["--chr"] + [str(c) for c in chromosomes]
    cmd += ["--keep", keep_tsv, "--recode", "A", "--out", out_prefix]
    print(f"[ENC] Running: {' '.join(cmd)}")
    env = os.environ.copy(); env.setdefault("LC_ALL", "C")
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, env=env)
    raw_path = f"{out_prefix}.raw"
    if not os.path.exists(raw_path):
        raise FileNotFoundError(raw_path)
    return raw_path


def create_batched_h5(bfile: str, fam: str, bim: str, out_dir: str, chromosomes: List[int], batch_size: int) -> Dict[int, int]:
    ensure_dir_exists(out_dir)
    # Load all IIDs
    fam_cols = ["FID", "IID", "PAT", "MAT", "SEX", "PHENOTYPE"]
    fam_df = pd.read_csv(fam, sep=r"\s+", header=None, names=fam_cols)
    iids = fam_df.iloc[:, :2].copy()

    # For each chromosome, create batched H5 files under out_dir/chr{c}
    per_chr_batches: Dict[int, int] = {}
    bfile_prefix = bfile[:-4] if bfile.endswith('.bed') else bfile
    for chr_no in chromosomes:
        chr_dir = os.path.join(out_dir, f"chr{chr_no}")
        ensure_dir_exists(chr_dir)
        # Determine SNP ids and bp for this chromosome
        ids_chr, bp_chr, _ = build_snp_info(bim, [chr_no])
        # Batch individuals by batch_size
        num_batches = int(np.ceil(len(iids) / float(batch_size)))
        for bi in range(1, num_batches + 1):
            s = (bi - 1) * batch_size
            e = min(len(iids), bi * batch_size)
            keep_tsv = os.path.join(chr_dir, f"keep_batch{bi:05d}.tsv")
            iids.iloc[s:e, :].to_csv(keep_tsv, sep='\t', header=False, index=False)
            out_prefix = os.path.join(chr_dir, f"tmp_batch{bi:05d}")
            h5_path = os.path.join(chr_dir, f"batch{bi:05d}.h5")
            if os.path.exists(h5_path):
                print(f"[ENC-AE] Found existing cache, skipping: {h5_path}")
                try:
                    os.remove(keep_tsv)
                except Exception:
                    pass
                continue
            # Recode and write H5
            raw_path = run_plink_recode_batch(bfile_prefix, [chr_no], keep_tsv, out_prefix)
            try:
                write_raw_to_h5_fast(
                    raw_path=raw_path,
                    h5_path=h5_path,
                    expected_rows=(e - s),
                    bp=np.array(bp_chr, dtype=np.int64),
                    snp_ids=list(ids_chr),
                    chunk_rows=10000,
                )
            finally:
                for ext in [".raw", ".log", ".nosex"]:
                    pth = out_prefix + ext
                    if os.path.exists(pth):
                        try:
                            os.remove(pth)
                        except Exception:
                            pass
                try:
                    os.remove(keep_tsv)
                except Exception:
                    pass
        per_chr_batches[chr_no] = num_batches
    return per_chr_batches


def _extract_prefixed_state(state_dict: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    """Return a sub-state-dict stripping the provided prefix."""
    plen = len(prefix)
    return {k[plen:]: v for k, v in state_dict.items() if k.startswith(prefix)}


class HomogenizedChromosomeEncoder:
    """Utility to encode batches with a homogenized AE checkpoint."""

    def __init__(self, chrom_no: int, model_path: str, device: torch.device, *, use_amp: bool = True) -> None:
        self.chrom_no = int(chrom_no)
        self.chrom_embed_idx = max(0, self.chrom_no - 1)
        self.device = device

        payload = torch.load(model_path, map_location='cpu')
        if not isinstance(payload, dict):
            raise ValueError(f"Checkpoint {model_path} is not a dict payload; got {type(payload)}")

        state_dict = payload.get('model_state')
        if state_dict is None:
            raise KeyError(f"Checkpoint {model_path} missing 'model_state'")

        meta = payload.get('meta', {})
        cfg_dict = meta.get('config') if isinstance(meta, dict) else None
        if cfg_dict is None:
            raise KeyError(f"Checkpoint {model_path} missing meta['config'] for AE reconstruction")

        cfg = VAEConfig(**cfg_dict)
        self.cfg = cfg

        self.ae = GenotypeAutoencoder(
            input_length=cfg.input_length,
            K1=cfg.K1,
            K2=cfg.K2,
            C=cfg.C,
            embed_dim=cfg.embed_dim,
        ).to(device)
        ae_state = _extract_prefixed_state(state_dict, 'aes.0.')
        missing = self.ae.load_state_dict(ae_state, strict=True)
        if getattr(missing, 'missing_keys', None) or getattr(missing, 'unexpected_keys', None):
            raise RuntimeError(
                f"Failed to load AE weights for chr{self.chrom_no} from {model_path}: {missing}"
            )
        self.ae.eval()
        for p in self.ae.parameters():
            p.requires_grad = False

        self.encode_head = FiLM2D(self.ae.latent_channels).to(device)
        encode_state = _extract_prefixed_state(state_dict, 'encode_head.')
        enc_missing = self.encode_head.load_state_dict(encode_state, strict=True)
        if getattr(enc_missing, 'missing_keys', None) or getattr(enc_missing, 'unexpected_keys', None):
            raise RuntimeError(
                f"Failed to load encode head for chr{self.chrom_no} from {model_path}: {enc_missing}"
            )
        self.encode_head.eval()
        for p in self.encode_head.parameters():
            p.requires_grad = False

        amp_on_cuda = bool(device.type == 'cuda' and cuda_autocast is not None)
        self.amp_enabled = bool(use_amp and amp_on_cuda)

    @property
    def latent_shape(self) -> Tuple[int, int, int]:
        return (self.ae.latent_channels, self.ae.M2D, self.ae.M2D)

    def encode_batch(self, batch_cpu: torch.Tensor) -> torch.Tensor:
        """Encode a CPU batch (B, L) into homogenized latents on CPU."""
        x_dev = batch_cpu.to(self.device, non_blocking=(self.device.type == 'cuda'))
        # amp_ctx = cuda_autocast(device_type='cuda') if self.amp_enabled else nullcontext()
        amp_ctx = nullcontext()
        with torch.no_grad():
            with amp_ctx:
                _, z_orig = self.ae(x_dev)
                z_hom = self.encode_head(z_orig, self.chrom_embed_idx)
        latents_cpu = z_hom.detach().to('cpu').float()
        del x_dev, z_orig, z_hom
        return latents_cpu


def encode_per_chr_batches(models_dir: str,
                           model_pattern: str,
                           h5_root: str,
                           out_root: str,
                           chromosomes: List[int],
                           device: torch.device,
                           encode_batch_size: int = 128,
                           amp: bool = True) -> Dict[int, int]:
    ensure_dir_exists(out_root)
    per_chr_batches: Dict[int, int] = {}
    for chr_no in chromosomes:
        model_path = os.path.join(models_dir, model_pattern.format(chr=chr_no))
        if not os.path.exists(model_path):
            print(f"[ENC-AE] Missing homogenized model for chr{chr_no}: {model_path}; skipping")
            continue

        try:
            encoder = HomogenizedChromosomeEncoder(chr_no, model_path, device, use_amp=amp)
        except Exception as exc:
            print(f"[ENC-AE] Failed to initialize encoder for chr{chr_no}: {exc}")
            continue

        chr_h5_dir = os.path.join(h5_root, f"chr{chr_no}")
        if not os.path.isdir(chr_h5_dir):
            print(f"[ENC-AE] No H5 directory for chr{chr_no}: {chr_h5_dir}; skipping")
            del encoder
            continue

        chr_out_dir = os.path.join(out_root, f"chr{chr_no}")
        ensure_dir_exists(chr_out_dir)

        h5_files = sorted(glob.glob(os.path.join(chr_h5_dir, 'batch*.h5')))
        if not h5_files:
            print(f"[ENC-AE] No H5 batches found for chr{chr_no} in {chr_h5_dir}; skipping")
            del encoder
            continue

        per_chr_batches[chr_no] = len(h5_files)
        print(f"[ENC-AE] Encoding chr{chr_no} ({len(h5_files)} batches) from {model_path}")

        with torch.no_grad():
            for h5p in h5_files:
                bn = os.path.splitext(os.path.basename(h5p))[0]
                out_pt = os.path.join(chr_out_dir, f"{bn}_latents.pt")
                if os.path.exists(out_pt):
                    print(f"[ENC-AE] Found existing latents, skipping: {out_pt}")
                    continue

                with h5py.File(h5p, 'r') as f:
                    X = f['X'][:].astype('int64')

                batch_cpu = torch.from_numpy(X)
                if device.type == 'cuda':
                    batch_cpu = batch_cpu.pin_memory()

                n_samples = batch_cpu.size(0)
                if n_samples == 0:
                    print(f"[ENC-AE] Empty batch detected in {h5p}; skipping")
                    continue

                latents_chunks: List[torch.Tensor] = []
                bs = int(max(1, encode_batch_size))
                for s in range(0, n_samples, bs):
                    e = min(n_samples, s + bs)
                    lat_cpu = encoder.encode_batch(batch_cpu[s:e])
                    latents_chunks.append(lat_cpu)
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()

                latents = torch.cat(latents_chunks, dim=0)
                torch.save(latents, out_pt)

                del latents, latents_chunks, batch_cpu, X
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

        del encoder
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    return per_chr_batches


def unify_batches(layout_json: str,
                  latents_root: str,
                  out_unified_root: str,
                  chromosomes: List[int],
                  embed_dtype: str = 'float32') -> None:
    ensure_dir_exists(out_unified_root)
    with open(layout_json, 'r') as jf:
        layout = json.load(jf)
    grid_M = int(layout.get('grid_M', 0))
    if grid_M == 0:
        raise ValueError(f"Grid size 0 used, check {layout_json}")
    layout_map = {int(rec['chromosome']): rec for rec in layout['layout']}
    available_chr = [chr_no for chr_no in chromosomes if os.path.isdir(os.path.join(latents_root, f"chr{chr_no}"))]
    if not available_chr:
        raise RuntimeError(f"No latent directories found under {latents_root}")

    reference_chr: Optional[int] = None
    num_batches: Optional[int] = None
    for chr_no in available_chr:
        lat_dir = os.path.join(latents_root, f"chr{chr_no}")
        batch_pts = sorted(glob.glob(os.path.join(lat_dir, 'batch*_latents.pt')))
        if batch_pts:
            reference_chr = chr_no
            num_batches = len(batch_pts)
            break

    if reference_chr is None or num_batches is None:
        raise RuntimeError(f"No latent batches (*.pt) found in {latents_root} for chromosomes {available_chr}")

    print(f"[ENC-AE] Unifying {num_batches} batches into {grid_M}x{grid_M} grid using chr{reference_chr} as reference")
    for bi in range(1, num_batches + 1):
        # Prepare unified tensor shape: (N, C, M, M)
        # Load N and channel from any chr batch
        first_pt = os.path.join(latents_root, f"chr{reference_chr}", f"batch{bi:05d}_latents.pt")
        if not os.path.exists(first_pt):
            raise FileNotFoundError(f"Reference latent batch missing: {first_pt}")
        L0 = torch.load(first_pt, map_location='cpu')  # (N, C, s, s)
        N, C, _, _ = L0.shape
        unified = torch.zeros((N, C, grid_M, grid_M), dtype=getattr(torch, embed_dtype))
        # Place each chromosome tile
        for chr_no in chromosomes:
            rec = layout_map.get(int(chr_no))
            if rec is None:
                continue
            tile_path = os.path.join(latents_root, f"chr{chr_no}", f"batch{bi:05d}_latents.pt")
            if not os.path.exists(tile_path):
                continue
            tile = torch.load(tile_path, map_location='cpu')
            s = int(rec['tile_side'])
            x0 = int(rec['x0']); y0 = int(rec['y0'])
            unified[:, :, y0:y0+s, x0:x0+s] = tile.to(unified.dtype)
        out_pt = os.path.join(out_unified_root, f"batch{bi:05d}_unified.pt")
        torch.save(unified, out_pt)
        # free per-batch tensors
        del unified, L0
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def cleanup_temp(paths: List[str]) -> None:
    import shutil
    for p in paths:
        try:
            if os.path.isdir(p):
                shutil.rmtree(p)
            elif os.path.exists(p):
                os.remove(p)
        except Exception:
            pass


def main():
    p = argparse.ArgumentParser(description="Encode batched homogenized AE latents per chromosome and unify via MILP layout")
    p.add_argument('--bfile', required=True)
    p.add_argument('--bim', required=True)
    p.add_argument('--fam', required=True)
    p.add_argument('--chromosomes', nargs='+', default=['all'])
    p.add_argument('--batch-size', type=int, default=12000, help='Individuals per batch when creating H5 caches')
    p.add_argument('--h5-out-root', required=True, help='Root dir for temporary batched H5 caches (genomic_data/unet_prep)')
    p.add_argument('--models-dir', required=True, help='Directory with per-chromosome homogenized AE checkpoints (ae_chr{chr}_homog.pt)')
    p.add_argument('--model-pattern', default='ae_chr{chr}_homog.pt', help='Filename pattern for homogenized AE checkpoints')
    p.add_argument('--latents-out-root', required=True, help='Root dir for temporary per-chromosome latents')
    p.add_argument('--layout-json', required=True, help='Path to MILP layout JSON produced by latentAlloc_MILP.py')
    p.add_argument('--unified-out-root', required=True, help='Root dir for unified batched embeddings')
    p.add_argument('--device', default='cuda')
    p.add_argument('--encode-batch-size', type=int, default=128, help='Micro-batch size for GPU forward pass during encoding')
    p.add_argument('--cleanup', action='store_true', help='Delete temporary H5 caches and per-chr latents after unifying')
    p.add_argument('--no-amp', dest='amp', action='store_false', help='Disable CUDA autocast during encoding (enabled by default)')
    p.set_defaults(amp=True)
    args = p.parse_args()

    chromosomes = get_chromosomes(args.chromosomes)
    ensure_dir_exists(args.h5_out_root)
    ensure_dir_exists(args.latents_out_root)
    ensure_dir_exists(args.unified_out_root)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith('cuda') else 'cpu')

    # 1) Create batched H5 caches per chromosome
    per_chr_batches = create_batched_h5(
        bfile=args.bfile,
        fam=args.fam,
        bim=args.bim,
        out_dir=args.h5_out_root,
        chromosomes=chromosomes,
        batch_size=int(args.batch_size)
    )
    print(f"[ENC-AE] H5 caches created per chromosome: {per_chr_batches}")

    # 2) Encode per-chromosome batches into latents (.pt)
    per_chr_batches = encode_per_chr_batches(
        models_dir=args.models_dir,
        model_pattern=args.model_pattern,
        h5_root=args.h5_out_root,
        out_root=args.latents_out_root,
        chromosomes=chromosomes,
        device=device,
        encode_batch_size=int(args.encode_batch_size),
        amp=args.amp,
    )
    print(f"[ENC-AE] Encoded latents per chromosome: {per_chr_batches}")

    # 3) Unify batched embeddings using saved MILP layout
    unify_batches(
        layout_json=args.layout_json,
        latents_root=args.latents_out_root,
        out_unified_root=args.unified_out_root,
        chromosomes=chromosomes,
    )
    print("[ENC-AE] Unification complete")

    # 4) Cleanup temporary files
    # if args.cleanup:
    #     print("[ENC-AE] Cleaning up temporary caches and per-chromosome latents")
    #     cleanup_temp([args.h5_out_root, args.latents_out_root])


if __name__ == '__main__':
    main()


# Example:
# python encode_batched_AE.py \
#   --bfile ${DATA_ROOT}/geneticBinary/ukb_allchr_unrel_britishWhite \
#   --bim ${DATA_ROOT}/geneticBinary/ukb_allchr_unrel_britishWhite.bim \
#   --fam ${DATA_ROOT}/geneticBinary/ukb_allchr_unrel_britishWhite.fam \
#   --chromosomes all \
#   --batch-size 12000 \
#   --h5-out-root /n/home03/ahmadazim/WORKING/genGen/UKBAE/genomic_data/ae_h5 \
#   --models-dir /n/home03/ahmadazim/WORKING/genGen/UKBVQVAE/models/AE \
#   --model-pattern ae_chr{chr}_homog.pt \
#   --latents-out-root /n/home03/ahmadazim/WORKING/genGen/UKBAE/genomic_data/AE_embeddings \
#   --layout-json /n/home03/ahmadazim/WORKING/genGen/UKBAE/models/ae_milp_layout.json \
#   --unified-out-root /n/home03/ahmadazim/WORKING/genGen/UKBAE/AE_embeddings/unified \
#   --device cuda 