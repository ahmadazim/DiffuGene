import argparse, os, sys, time, gc, itertools, json, traceback
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.append("/n/home03/ahmadazim/WORKING/genGen/DiffuGene/")
from src.DiffuGene.utils import read_raw
from src.DiffuGene.VAEembed.vae import build_vqvae, train_vqvae, eval_batch_categorical

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", type=str, required=True, help="Path to .raw file (read_raw compatible).")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--limit_N", type=int, default=None, help="Optionally limit #individuals (rows).")
    ap.add_argument("--results_csv", type=str, default="results_vqvae_grid.csv")
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--beta_commit", type=float, default=0.25)
    ap.add_argument("--ema_decay", type=float, default=0.9)
    ap.add_argument("--ema_eps", type=float, default=1e-5)
    ap.add_argument("--ld_lambda", type=float, default=1e-3)
    ap.add_argument("--maf_lambda", type=float, default=1e-3)
    ap.add_argument("--ld_window", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--pin_memory", action="store_true")
    ap.add_argument("--eval_batch", type=int, default=512, help="Eval on this many samples from train.")
    return ap.parse_args()

def ensure_header(path, header_cols):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        with open(path, "w") as f:
            f.write(",".join(header_cols) + "\n")

def append_row(path, row_vals):
    with open(path, "a") as f:
        f.write(",".join(map(str, row_vals)) + "\n")

def dump_temp_pickle(path, records):
    try:
        with open(path, "wb") as f:
            pickle.dump(records, f)
    except Exception:
        pass

def main():
    args = parse_args()
    print(f"Using device: {args.device}")

    # --- Load data
    print("Loading data...")
    t0 = time.time()
    ds = read_raw(args.data_path).impute().get_variants()  # numpy (N x L) expected
    if args.limit_N is not None:
        ds = ds[:args.limit_N]
    N, L = ds.shape
    print(f"Data shape: {ds.shape}")

    # Torch tensor on CPU; train loop moves to device
    x_tensor = torch.tensor(ds, dtype=torch.float32)
    dataset = TensorDataset(x_tensor)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=False,
    )
    del ds
    gc.collect()

    # --- Grid
    latent_grid_dim_list   = [8, 16, 32, 64, 128]
    latent_dim_list        = [4, 8, 16, 32, 64]
    codebook_size_list     = [128, 256, 512, 1024]
    num_quantizers_list    = [1, 2]
    hidden_channels_list   = [32, 64]
    width_mult_list        = [1.0, 2.0]

    combos = list(itertools.product(
        latent_grid_dim_list,
        latent_dim_list,
        codebook_size_list,
        num_quantizers_list,
        hidden_channels_list,
        width_mult_list
    ))
    print(f"Total combinations: {len(combos)}")

    header = [
        "status", "elapsed_sec", "cuda_peak_mem_MB",
        "latent_grid_dim", "latent_dim", "codebook_size",
        "num_quantizers", "hidden_channels", "width_mult_per_stage",
        "lr", "beta_commit", "ema_decay", "ema_eps",
        "ld_lambda", "maf_lambda", "ld_window",
        "acc", "mse", "commit", "perplexities", "notes"
    ]
    ensure_header(args.results_csv, header)
    temp_pickle_path = args.results_csv + ".tmp.pkl"
    temp_records = []

    for i, (gdim, d, K, nq, hc, wmult) in enumerate(combos, 1):
        print(f"\n[{i}/{len(combos)}] gdim={gdim} d={d} K={K} nq={nq} hc={hc} wmult={wmult}")
        t_start = time.time()
        cuda_peak_mb = 0
        status = "ok"
        acc = mse = commit = None
        perpl = None
        notes = ""

        try:
            if torch.cuda.is_available() and "cuda" in args.device:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

            # Build model + optimizer
            vqvae, optimizer = build_vqvae(
                input_length=L,
                latent_dim=d,
                codebook_size=K,
                num_quantizers=nq,
                beta_commit=args.beta_commit,
                lr=args.lr,
                latent_grid_dim=gdim,
                ld_lambda=args.ld_lambda,
                maf_lambda=args.maf_lambda,
                ld_window=args.ld_window,
                ema_decay=args.ema_decay,
                ema_eps=args.ema_eps,
                hidden_channels=hc,
                width_mult_per_stage=wmult,
            )

            # Overwrite config options that build_vqvae sets by default
            vqvae.cfg.hidden_channels = hc
            vqvae.cfg.width_mult_per_stage = wmult
            vqvae.cfg.ema_decay = args.ema_decay
            vqvae.cfg.ema_eps = args.ema_eps
            vqvae.cfg.ld_lambda = args.ld_lambda
            vqvae.cfg.maf_lambda = args.maf_lambda
            vqvae.cfg.ld_window = args.ld_window

            # Train
            train_vqvae(
                vqvae, dataloader, optimizer,
                device=torch.device(args.device),
                num_epochs=args.epochs,
                grad_clip=1.0
            )

            # Evaluate on a small slice
            with torch.no_grad():
                eval_B = min(args.eval_batch, len(x_tensor))
                x_eval = x_tensor[:eval_B]
                metrics = eval_batch_categorical(vqvae, x_eval, device=torch.device(args.device), verbose=False)
                acc = metrics["acc"]
                mse = metrics["mse"]
                commit = metrics["commit"]
                perpl = metrics["perplexities"]  # list or None

            if torch.cuda.is_available() and "cuda" in args.device:
                cuda_peak_mb = torch.cuda.max_memory_allocated() / (1024**2)

        except RuntimeError as e:
            emsg = str(e)
            tb = traceback.format_exc(limit=1)
            if "CUDA out of memory" in emsg:
                status = "oom"
                notes = f"oom; {tb.strip()}"
            else:
                status = "runtime_error"
                notes = f"{emsg.splitlines()[-1]}"
            # Always try to clean up and move on
            try:
                del vqvae, optimizer
            except Exception:
                pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        except Exception as e:
            status = "error"
            notes = f"{repr(e)}"
            try:
                del vqvae, optimizer
            except Exception:
                pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        elapsed = time.time() - t_start
        row = [
            status,
            f"{elapsed:.2f}",
            f"{cuda_peak_mb:.1f}",
            gdim, d, K, nq, hc, wmult,
            args.lr, args.beta_commit, args.ema_decay, args.ema_eps,
            args.ld_lambda, args.maf_lambda, args.ld_window,
            "" if acc is None else f"{acc:.6f}",
            "" if mse is None else f"{mse:.6f}",
            "" if commit is None else f"{commit:.6f}",
            "" if perpl is None else json.dumps([float(x) for x in perpl]),
            notes.replace(",", ";")
        ]
        append_row(args.results_csv, row)

        # Accumulate and dump a temporary pickle snapshot after each iteration
        temp_records.append({
            "status": status,
            "elapsed_sec": elapsed,
            "cuda_peak_mem_MB": cuda_peak_mb,
            "latent_grid_dim": gdim,
            "latent_dim": d,
            "codebook_size": K,
            "num_quantizers": nq,
            "hidden_channels": hc,
            "width_mult_per_stage": wmult,
            "lr": args.lr,
            "beta_commit": args.beta_commit,
            "ema_decay": args.ema_decay,
            "ema_eps": args.ema_eps,
            "ld_lambda": args.ld_lambda,
            "maf_lambda": args.maf_lambda,
            "ld_window": args.ld_window,
            "acc": None if acc is None else float(acc),
            "mse": None if mse is None else float(mse),
            "commit": None if commit is None else float(commit),
            "perplexities": None if perpl is None else [float(x) for x in perpl],
            "notes": notes,
            "iteration_index": i,
            "total_combinations": len(combos),
        })
        dump_temp_pickle(temp_pickle_path, temp_records)

        # Extra GC between runs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    print("\nDone. Results written to:", args.results_csv)

if __name__ == "__main__":
    main()