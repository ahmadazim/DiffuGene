#!/usr/bin/env python
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
from contextlib import nullcontext
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
import torch

this_dir = os.path.dirname(__file__)
src_root = os.path.abspath(os.path.join(this_dir, "..", "..", ".."))
if src_root not in sys.path:
    sys.path.insert(0, src_root)

from DiffuGene.VAEembed.ae import TokenAutoencoder1D, TokenAEConfig
from DiffuGene.VAEembed.sharedEmbed import FiLM1D
from DiffuGene.utils import ensure_dir_exists
from DiffuGene.utils.file_utils import read_bim_file

try:
    from torch.cuda.amp import autocast as cuda_autocast
except Exception:
    cuda_autocast = None


def get_chromosomes(spec: List[str]) -> List[int]:
    if len(spec) == 1 and str(spec[0]).lower() == "all":
        return list(range(1, 23))
    return [int(x) for x in spec]


def _read_raw_header(raw_path: str) -> List[str]:
    with open(raw_path, "r") as f:
        return f.readline().strip().split()


def write_raw_to_h5_fast(
    raw_path: str,
    h5_path: str,
    expected_rows: int,
    bp: np.ndarray,
    snp_ids: List[str],
    chunk_rows: int = 10000,
) -> None:
    header = _read_raw_header(raw_path)
    l_raw = len(header) - 6
    if l_raw <= 0:
        raise ValueError(f"No SNP columns in {raw_path}")
    if bp.size == 0:
        raise ValueError("Empty bp array")
    l_use = min(l_raw, int(bp.shape[0]))

    ensure_dir_exists(os.path.dirname(h5_path))
    with h5py.File(h5_path, "w") as f:
        dset_x = f.create_dataset("X", shape=(expected_rows, l_use), dtype="i1", compression="gzip", compression_opts=4)
        dset_iid = f.create_dataset("iid", shape=(expected_rows,), dtype=h5py.string_dtype(encoding="utf-8"))
        f.create_dataset("bp", data=bp[:l_use])
        try:
            f.create_dataset("snp_ids", data=np.array(snp_ids[:l_use], dtype=object), dtype=h5py.string_dtype(encoding="utf-8"))
        except Exception:
            pass
        offset = 0
        usecols = list(range(0, 2)) + list(range(6, 6 + l_use))
        for df in pd.read_csv(
            raw_path,
            sep=r"\s+",
            header=0,
            usecols=usecols,
            chunksize=chunk_rows,
            na_values=["NA"],
        ):
            iids = df.iloc[:, 1].astype(str).to_numpy()
            x_chunk = df.iloc[:, 2:].fillna(0).to_numpy(dtype=np.int16, copy=False).astype(np.int8, copy=False)
            n = x_chunk.shape[0]
            dset_x[offset:offset + n, :] = x_chunk
            dset_iid[offset:offset + n] = iids
            offset += n


def build_snp_info(bim_file: str, chromosomes: List[int]) -> Tuple[List[str], np.ndarray]:
    all_ids: List[str] = []
    all_bp: List[np.ndarray] = []
    for c in chromosomes:
        bim = read_bim_file(bim_file, c)
        all_ids.extend(bim["SNP"].astype(str).tolist())
        all_bp.append(bim["BP"].astype(np.int64).values)
    bp = np.concatenate(all_bp, axis=0) if all_bp else np.array([], dtype=np.int64)
    return all_ids, bp


def run_plink_recode_batch(bfile_prefix: str, chromosomes: List[int], keep_tsv: str, out_prefix: str) -> str:
    cmd = ["plink", "--bfile", bfile_prefix]
    if chromosomes:
        cmd += ["--chr"] + [str(c) for c in chromosomes]
    cmd += ["--keep", keep_tsv, "--recode", "A", "--out", out_prefix]
    print(f"[ENC-TOK] Running: {' '.join(cmd)}")
    env = os.environ.copy()
    env.setdefault("LC_ALL", "C")
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, env=env)
    raw_path = f"{out_prefix}.raw"
    if not os.path.exists(raw_path):
        raise FileNotFoundError(raw_path)
    return raw_path


def create_batched_h5(
    bfile: str,
    fam: str,
    bim: str,
    out_dir: str,
    chromosomes: List[int],
    batch_size: int,
) -> Dict[int, int]:
    ensure_dir_exists(out_dir)
    fam_cols = ["FID", "IID", "PAT", "MAT", "SEX", "PHENOTYPE"]
    fam_df = pd.read_csv(fam, sep=r"\s+", header=None, names=fam_cols)
    iids = fam_df.iloc[:, :2].copy()
    bfile_prefix = bfile[:-4] if bfile.endswith(".bed") else bfile

    per_chr: Dict[int, int] = {}
    for c in chromosomes:
        chr_dir = os.path.join(out_dir, f"chr{c}")
        ensure_dir_exists(chr_dir)
        ids_chr, bp_chr = build_snp_info(bim, [c])
        n_batches = int(np.ceil(len(iids) / float(batch_size)))
        for bi in range(1, n_batches + 1):
            s = (bi - 1) * batch_size
            e = min(len(iids), bi * batch_size)
            keep_tsv = os.path.join(chr_dir, f"keep_batch{bi:05d}.tsv")
            iids.iloc[s:e, :].to_csv(keep_tsv, sep="\t", header=False, index=False)
            out_prefix = os.path.join(chr_dir, f"tmp_batch{bi:05d}")
            h5_path = os.path.join(chr_dir, f"batch{bi:05d}.h5")
            if os.path.exists(h5_path):
                try:
                    os.remove(keep_tsv)
                except Exception:
                    pass
                continue
            raw_path = run_plink_recode_batch(bfile_prefix, [c], keep_tsv, out_prefix)
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
        per_chr[c] = n_batches
    return per_chr


def _extract_prefixed_state(state_dict: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    plen = len(prefix)
    return {k[plen:]: v for k, v in state_dict.items() if k.startswith(prefix)}


class HomogenizedChromosomeEncoderTok:
    def __init__(self, chrom_no: int, model_path: str, device: torch.device, use_amp: bool = True) -> None:
        self.chrom_no = int(chrom_no)
        self.chrom_embed_idx = max(0, self.chrom_no - 1)
        self.device = device
        payload = torch.load(model_path, map_location="cpu")
        state = payload.get("model_state")
        if state is None:
            raise KeyError(f"{model_path} missing model_state")
        meta = payload.get("meta", {})
        cfg_dict = meta.get("config", payload.get("config"))
        if cfg_dict is None:
            raise KeyError(f"{model_path} missing config metadata")
        cfg = TokenAEConfig(**cfg_dict)

        self.ae = TokenAutoencoder1D(
            input_length=cfg.input_length,
            latent_length=cfg.latent_length,
            latent_dim=cfg.latent_dim,
            embed_dim=cfg.embed_dim,
            max_c=cfg.max_c,
            dropout=cfg.dropout,
        ).to(device)
        ae_state = _extract_prefixed_state(state, "aes.0.")
        self.ae.load_state_dict(ae_state, strict=True)
        self.ae.eval()
        for p in self.ae.parameters():
            p.requires_grad = False

        self.encode_head = FiLM1D(self.ae.latent_dim).to(device)
        eh_state = _extract_prefixed_state(state, "encode_head.")
        self.encode_head.load_state_dict(eh_state, strict=True)
        self.encode_head.eval()
        for p in self.encode_head.parameters():
            p.requires_grad = False

        self.amp_enabled = bool(use_amp and device.type == "cuda" and cuda_autocast is not None)

    @property
    def latent_shape(self) -> Tuple[int, int]:
        return (int(self.ae.latent_length), int(self.ae.latent_dim))

    def encode_batch(self, batch_cpu: torch.Tensor) -> torch.Tensor:
        device_is_cuda = self.device.type == "cuda"
        x_dev = batch_cpu.to(self.device, non_blocking=device_is_cuda).long()
        amp_ctx = cuda_autocast() if (self.amp_enabled and device_is_cuda) else nullcontext()
        with torch.no_grad():
            with amp_ctx:
                z = self.ae.encode(x_dev)
                chrom_vec = torch.full((z.size(0),), int(self.chrom_embed_idx), dtype=torch.long, device=z.device)
                z_hom = self.encode_head(z, chrom_vec)
        out = z_hom.detach().to("cpu").float()
        del x_dev, z, z_hom
        return out


def encode_per_chr_batches(
    models_dir: str,
    model_pattern: str,
    h5_root: str,
    out_root: str,
    chromosomes: List[int],
    device: torch.device,
    encode_batch_size: int = 128,
    amp: bool = True,
) -> Dict[int, int]:
    ensure_dir_exists(out_root)
    per_chr_batches: Dict[int, int] = {}
    for c in chromosomes:
        model_path = os.path.join(models_dir, model_pattern.format(chr=c))
        if not os.path.exists(model_path):
            print(f"[ENC-TOK] Missing homogenized model for chr{c}: {model_path}; skipping")
            continue
        try:
            enc = HomogenizedChromosomeEncoderTok(c, model_path, device, use_amp=amp)
        except Exception as exc:
            print(f"[ENC-TOK] Failed to init encoder for chr{c}: {exc}")
            continue
        chr_h5_dir = os.path.join(h5_root, f"chr{c}")
        if not os.path.isdir(chr_h5_dir):
            continue
        chr_out_dir = os.path.join(out_root, f"chr{c}")
        ensure_dir_exists(chr_out_dir)
        h5_files = sorted(glob.glob(os.path.join(chr_h5_dir, "batch*.h5")))
        if not h5_files:
            continue
        per_chr_batches[c] = len(h5_files)
        for h5p in h5_files:
            bn = os.path.splitext(os.path.basename(h5p))[0]
            out_pt = os.path.join(chr_out_dir, f"{bn}_latents.pt")
            if os.path.exists(out_pt):
                continue
            with h5py.File(h5p, "r") as f:
                x = f["X"][:].astype("int64")
            batch_cpu = torch.from_numpy(x)
            if device.type == "cuda":
                batch_cpu = batch_cpu.pin_memory()
            n = batch_cpu.size(0)
            parts: List[torch.Tensor] = []
            bs = int(max(1, encode_batch_size))
            for s in range(0, n, bs):
                e = min(n, s + bs)
                parts.append(enc.encode_batch(batch_cpu[s:e]))
            z_all = torch.cat(parts, dim=0)  # (N, T_chr, D)
            torch.save(z_all, out_pt)
            del z_all, parts, batch_cpu, x
            if device.type == "cuda":
                torch.cuda.empty_cache()
        del enc
    return per_chr_batches


def unify_batches(
    layout_json: str,
    latents_root: str,
    out_unified_root: str,
    chromosomes: List[int],
    embed_dtype: str = "float32",
) -> None:
    ensure_dir_exists(out_unified_root)
    with open(layout_json, "r") as f:
        layout = json.load(f)
    total_tokens = int(layout["total_tokens"])
    latent_dim = int(layout["latent_dim"])
    map_by_chr = {int(r["chromosome"]): r for r in layout["layout"]}

    available_chr = [c for c in chromosomes if os.path.isdir(os.path.join(latents_root, f"chr{c}"))]
    if not available_chr:
        raise RuntimeError(f"No per-chr latent dirs in {latents_root}")
    ref = None
    num_batches = None
    for c in available_chr:
        pts = sorted(glob.glob(os.path.join(latents_root, f"chr{c}", "batch*_latents.pt")))
        if pts:
            ref = c
            num_batches = len(pts)
            break
    if ref is None or num_batches is None:
        raise RuntimeError("No encoded batch files found.")

    for bi in range(1, num_batches + 1):
        ref_pt = os.path.join(latents_root, f"chr{ref}", f"batch{bi:05d}_latents.pt")
        if not os.path.exists(ref_pt):
            raise FileNotFoundError(ref_pt)
        ref_tensor = torch.load(ref_pt, map_location="cpu")
        n = int(ref_tensor.shape[0])
        unified = torch.zeros((n, total_tokens, latent_dim), dtype=getattr(torch, embed_dtype))
        for c in chromosomes:
            rec = map_by_chr.get(int(c))
            if rec is None:
                continue
            pt = os.path.join(latents_root, f"chr{c}", f"batch{bi:05d}_latents.pt")
            if not os.path.exists(pt):
                continue
            z = torch.load(pt, map_location="cpu")  # (N, T_chr, D)
            t_start = int(rec["token_start"])
            t_end = int(rec["token_end"])
            if z.shape[1] != (t_end - t_start):
                raise ValueError(
                    f"chr{c} token mismatch: encoded {z.shape[1]} vs layout span {t_end - t_start}"
                )
            if z.shape[2] != latent_dim:
                raise ValueError(f"chr{c} latent_dim mismatch: encoded {z.shape[2]} vs {latent_dim}")
            unified[:, t_start:t_end, :] = z.to(unified.dtype)
        out_pt = os.path.join(out_unified_root, f"batch{bi:05d}_unified.pt")
        torch.save(unified, out_pt)
        del unified, ref_tensor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    map_out = os.path.join(out_unified_root, "token_ranges.json")
    with open(map_out, "w") as f:
        json.dump(
            {
                "total_tokens": total_tokens,
                "latent_dim": latent_dim,
                "layout": layout["layout"],
            },
            f,
            indent=2,
        )
    print(f"[ENC-TOK] Saved token-range metadata: {map_out}")


def main() -> None:
    p = argparse.ArgumentParser(description="Encode batched homogenized token-AE latents and unify to (B,T,D).")
    p.add_argument("--bfile", required=True)
    p.add_argument("--bim", required=True)
    p.add_argument("--fam", required=True)
    p.add_argument("--chromosomes", nargs="+", default=["all"])
    p.add_argument("--batch-size", type=int, default=12000)
    p.add_argument("--h5-out-root", required=True)
    p.add_argument("--models-dir", required=True)
    p.add_argument("--model-pattern", default="ae_tok_chr{chr}_homog.pt")
    p.add_argument("--latents-out-root", required=True)
    p.add_argument("--layout-json", required=True)
    p.add_argument("--unified-out-root", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--encode-batch-size", type=int, default=128)
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.set_defaults(amp=True)
    args = p.parse_args()

    chromosomes = get_chromosomes(args.chromosomes)
    ensure_dir_exists(args.h5_out_root)
    ensure_dir_exists(args.latents_out_root)
    ensure_dir_exists(args.unified_out_root)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")

    per_chr_batches = create_batched_h5(
        bfile=args.bfile,
        fam=args.fam,
        bim=args.bim,
        out_dir=args.h5_out_root,
        chromosomes=chromosomes,
        batch_size=int(args.batch_size),
    )
    print(f"[ENC-TOK] H5 caches per chromosome: {per_chr_batches}")

    per_chr_batches = encode_per_chr_batches(
        models_dir=args.models_dir,
        model_pattern=args.model_pattern,
        h5_root=args.h5_out_root,
        out_root=args.latents_out_root,
        chromosomes=chromosomes,
        device=device,
        encode_batch_size=int(args.encode_batch_size),
        amp=bool(args.amp),
    )
    print(f"[ENC-TOK] Encoded batches per chromosome: {per_chr_batches}")

    unify_batches(
        layout_json=args.layout_json,
        latents_root=args.latents_out_root,
        out_unified_root=args.unified_out_root,
        chromosomes=chromosomes,
    )
    print("[ENC-TOK] Unification complete.")


if __name__ == "__main__":
    main()
