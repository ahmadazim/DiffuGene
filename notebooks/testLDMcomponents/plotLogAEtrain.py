import re
import math
from typing import Dict, Any, Optional, List, Tuple
import matplotlib.pyplot as plt
import argparse
import glob
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-path", type=str, required=False, help="Optional: explicit log file to parse (single plot).")
    parser.add_argument("--out-path", type=str, required=True)
    parser.add_argument("--chromosomes", type=str, default="1-22", help="Chromosome list, e.g. '1-22' or '1,2,5,22'.")
    parser.add_argument("--model-name", type=str, default="veryhighbetaKL", help="Model name suffix, e.g. 'veryhighbetaKL'.")
    parser.add_argument("--logs-dir", type=str, default="/n/home03/ahmadazim/WORKING/log_err", help="Directory containing training log files.")
    return parser.parse_args()


def parse_ae_training_log(log_path: str) -> Dict[str, Any]:
    """
    Parse the AE training log and collect:
      - train loss terms per epoch: loss, recon, maf, ld, kl, lr
      - validation MSE per epoch: val_mse
      - latent penalties (train + val): tv, robust, stable, stable_norm
      - stage-2 start / end epochs

    Returns a dict:
      {
        "epochs": [...],
        "train": {
            "loss": [...],
            "recon": [...],
            "maf": [...],
            "ld": [...],
            "kl": [...],
            "lr": [...],
        },
        "val": {
            "val_mse": [...],
        },
        "latent": {
            "train": {
                "tv": [...],
                "robust": [...],
                "stable": [...],
                "stable_norm": [...],
            },
            "val": {
                "tv": [...],
                "robust": [...],
                "stable": [...],
                "stable_norm": [...],
            },
        },
        "stage2_start_epoch": int or None,
        "stage2_end_epoch": int or None,
      }
    """
    # Stage-2 schedule line
    stage2_re = re.compile(
        r"Stage-2 latent regularization .* from epoch (\d+) to (\d+)"
    )

    # Core training loss line
    train_re = re.compile(
        r"Epoch\s+(\d+)/(\d+):\s+"
        r"loss=([0-9eE\.+-]+)\s+\|\s+"
        r"recon=([0-9eE\.+-]+)\s+\|\s+"
        r"maf=([0-9eE\.+-]+)\s+\|\s+"
        r"ld=([0-9eE\.+-]+)\s+\|\s+"
        r"kl=([0-9eE\.+-]+)\s+\|\s+"
        r"lr=([0-9eE\.+-]+)"
    )

    # Validation MSE line
    val_re = re.compile(
        r"Epoch\s+(\d+)/(\d+):\s+val_mse=([0-9eE\.+-]+)"
    )

    # Latent penalties line
    latent_re = re.compile(
        r"Epoch\s+(\d+)/(\d+)\s+latent penalties:\s+"
        r"train_tv=([0-9eE\.+-]+)\s+\|\s+"
        r"train_robust=([0-9eE\.+-]+)\s+\|\s+"
        r"train_stable=([0-9eE\.+-]+)\s+\|\s+"
        r"train_stable_norm=([0-9eE\.+-]+)\s+\|\s+"
        r"val_tv=([0-9eE\.+-]+)\s+\|\s+"
        r"val_robust=([0-9eE\.+-]+)\s+\|\s+"
        r"val_stable=([0-9eE\.+-]+)\s+\|\s+"
        r"val_stable_norm=([0-9eE\.+-]+)"
    )

    stage2_start: Optional[int] = None
    stage2_end: Optional[int] = None

    # Temporary storage keyed by epoch
    train_by_epoch: Dict[int, Dict[str, float]] = {}
    val_by_epoch: Dict[int, Dict[str, float]] = {}
    latent_by_epoch: Dict[int, Dict[str, float]] = {}

    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()

            # Stage-2 schedule
            m_stage2 = stage2_re.search(line)
            if m_stage2:
                stage2_start = int(m_stage2.group(1))
                stage2_end = int(m_stage2.group(2))
                continue

            # Train line
            m_train = train_re.search(line)
            if m_train:
                epoch = int(m_train.group(1))
                loss = float(m_train.group(3))
                recon = float(m_train.group(4))
                maf = float(m_train.group(5))
                ld = float(m_train.group(6))
                kl = float(m_train.group(7))
                lr = float(m_train.group(8))
                train_by_epoch[epoch] = {
                    "loss": loss,
                    "recon": recon,
                    "maf": maf,
                    "ld": ld,
                    "kl": kl,
                    "lr": lr,
                }
                continue

            # Validation line
            m_val = val_re.search(line)
            if m_val:
                epoch = int(m_val.group(1))
                val_mse = float(m_val.group(3))
                val_by_epoch[epoch] = {"val_mse": val_mse}
                continue

            # Latent penalties line
            m_latent = latent_re.search(line)
            if m_latent:
                epoch = int(m_latent.group(1))
                train_tv = float(m_latent.group(3))
                train_robust = float(m_latent.group(4))
                train_stable = float(m_latent.group(5))
                train_stable_norm = float(m_latent.group(6))
                val_tv = float(m_latent.group(7))
                val_robust = float(m_latent.group(8))
                val_stable = float(m_latent.group(9))
                val_stable_norm = float(m_latent.group(10))
                latent_by_epoch[epoch] = {
                    "train_tv": train_tv,
                    "train_robust": train_robust,
                    "train_stable": train_stable,
                    "train_stable_norm": train_stable_norm,
                    "val_tv": val_tv,
                    "val_robust": val_robust,
                    "val_stable": val_stable,
                    "val_stable_norm": val_stable_norm,
                }
                continue

    # Build sorted epoch list
    epochs = sorted(train_by_epoch.keys())

    # Initialize structures
    train_dict: Dict[str, list] = {
        "loss": [],
        "recon": [],
        "maf": [],
        "ld": [],
        "kl": [],
        "lr": [],
    }
    val_dict: Dict[str, list] = {
        "val_mse": [],
    }
    latent_train_dict: Dict[str, list] = {
        "tv": [],
        "robust": [],
        "stable": [],
        "stable_norm": [],
    }
    latent_val_dict: Dict[str, list] = {
        "tv": [],
        "robust": [],
        "stable": [],
        "stable_norm": [],
    }

    for e in epochs:
        # Core losses
        t = train_by_epoch.get(e)
        if t is None:
            # Shouldn't happen if logging is consistent
            continue
        for k in train_dict.keys():
            train_dict[k].append(t[k])

        # Validation MSE (NaN if missing)
        v = val_by_epoch.get(e)
        if v is not None:
            val_dict["val_mse"].append(v["val_mse"])
        else:
            val_dict["val_mse"].append(float("nan"))

        # Latent penalties (NaN if missing)
        lat = latent_by_epoch.get(e)
        if lat is not None:
            latent_train_dict["tv"].append(lat["train_tv"])
            latent_train_dict["robust"].append(lat["train_robust"])
            latent_train_dict["stable"].append(lat["train_stable"])
            latent_train_dict["stable_norm"].append(lat["train_stable_norm"])
            latent_val_dict["tv"].append(lat["val_tv"])
            latent_val_dict["robust"].append(lat["val_robust"])
            latent_val_dict["stable"].append(lat["val_stable"])
            latent_val_dict["stable_norm"].append(lat["val_stable_norm"])
        else:
            for k in latent_train_dict.keys():
                latent_train_dict[k].append(float("nan"))
                latent_val_dict[k].append(float("nan"))

    return {
        "epochs": epochs,
        "train": train_dict,
        "val": val_dict,
        "latent": {
            "train": latent_train_dict,
            "val": latent_val_dict,
        },
        "stage2_start_epoch": stage2_start,
        "stage2_end_epoch": stage2_end,
    }


def _parse_chromosome_list(spec: str) -> List[int]:
    """
    Parse chromosome spec string like '1-22' or '1,2,5,22' into a sorted list of ints.
    """
    spec = spec.strip()
    if "-" in spec:
        a, b = spec.split("-", 1)
        a = int(a.strip())
        b = int(b.strip())
        if a <= b:
            return list(range(a, b + 1))
        else:
            return list(range(b, a + 1))
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    vals: List[int] = []
    for p in parts:
        try:
            vals.append(int(p))
        except ValueError:
            continue
    vals = sorted(set(vals))
    return vals


def find_log_for_chromosome(
    logs_dir: str,
    chromosome: int,
    model_name: str,
) -> Optional[str]:
    """
    Search through all matching log files under logs_dir for a line containing:
    'Saved last model to /n/home03/ahmadazim/WORKING/genGen/UKBVQVAE/models/VAEmasked_regLat/vae_chr{chromosome}_{model_name}.pt'
    Return the first log file path that contains this marker.
    """
    target_path = f"/n/home03/ahmadazim/WORKING/genGen/UKBVQVAE/models/VAEmasked_regLat/vae_chr{chromosome}_{model_name}.pt"
    pattern = os.path.join(logs_dir, "train_VAEmasked_regLat*")
    for log_path in glob.glob(pattern):
        try:
            with open(log_path, "r") as fh:
                for line in fh:
                    if "Saved last model to" in line and target_path in line:
                        return log_path
        except Exception:
            continue
    return None


def plot_ae_losses(log_stats: Dict[str, Any], out_path: str) -> None:
    """
    Given the dictionary returned by `parse_ae_training_log`,
    plot train/validation losses and latent penalties over epochs.

    - One panel (subplot) per quantity:
        core:   loss, recon (+ val_mse), maf, ld, kl, val_mse
        latent: tv, robust, stable, stable_norm (each with train+val)
    - Each subplot has epochs on x-axis.
    - Add a vertical dashed line at the stage-2 start epoch (if available).
    """
    epochs = log_stats["epochs"]
    train = log_stats["train"]
    val = log_stats["val"]
    latent_train = log_stats["latent"]["train"]
    latent_val = log_stats["latent"]["val"]
    stage2_start = log_stats.get("stage2_start_epoch", None)

    # Build a list of panel specs for metrics, plus one extra slot for legend
    metric_panels = []

    # Core loss terms (no total loss, recon and val_mse separate)
    metric_panels.append(("recon", lambda ax: ax.plot(epochs, train["recon"], label="train_recon")))
    metric_panels.append(("maf", lambda ax: ax.plot(epochs, train["maf"], label="train_maf")))
    metric_panels.append(("ld", lambda ax: ax.plot(epochs, train["ld"], label="train_ld")))
    metric_panels.append(("kl", lambda ax: ax.plot(epochs, train["kl"], label="train_kl")))
    metric_panels.append(("val_mse", lambda ax: ax.plot(epochs, val["val_mse"], label="val_mse")))

    # Latent penalties: only validation values (exclude normalized stability)
    for metric in ["tv", "robust", "stable"]:
        def make_plot_fn(m=metric):
            def _plot(ax):
                ax.plot(epochs, latent_val[m], label=f"val_{m}")
            return _plot
        metric_panels.append((f"latent_{metric}", make_plot_fn(metric)))

    n_metrics = len(metric_panels)
    ncols = 3
    nslots = n_metrics + 1  # extra slot for legend
    nrows = math.ceil(nslots / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 3.5 * nrows), squeeze=False)
    axes_flat = axes.reshape(-1)

    # Title / y-label mapping
    title_map = {
        "recon": "Reconstruction Loss",
        "maf": "MAF Penalty",
        "ld": "LD Penalty",
        "kl": "KL Divergence",
        "val_mse": "Validation MSE",
        "latent_tv": "Latent Spatial Smoothness",
        "latent_robust": "Latent Robustness",
        "latent_stable": "Latent Stability",
        "latent_stable_norm": "Latent Stability (Normalized)",
    }
    ylabel_map = {
        "recon": "Recon loss",
        "maf": "MAF loss",
        "ld": "LD loss",
        "kl": "KL loss",
        "val_mse": "Val MSE",
        "latent_tv": "TV loss",
        "latent_robust": "Robust loss",
        "latent_stable": "Stable loss",
        "latent_stable_norm": "Stable (normed) loss",
    }

    all_handles = []
    all_labels = []

    for idx, (panel_key, plot_fn) in enumerate(metric_panels):
        ax = axes_flat[idx]

        plot_fn(ax)

        # Stage-2 start marker
        if stage2_start is not None:
            ax.axvline(stage2_start, linestyle="--", linewidth=1.0)

        title = title_map.get(panel_key, panel_key.replace("_", " ").title())
        ylabel = ylabel_map.get(panel_key, title)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        h, l = ax.get_legend_handles_labels()
        all_handles.extend(h)
        all_labels.extend(l)
        ax.grid(True, alpha=0.3)

    # Legend-only subplot lives in the extra slot
    legend_idx = n_metrics
    if legend_idx < len(axes_flat):
        legend_ax = axes_flat[legend_idx]
        legend_ax.axis("off")
        uniq = {}
        for h, lab in zip(all_handles, all_labels):
            if lab not in uniq:
                uniq[lab] = h
        if uniq:
            legend_ax.legend(
                uniq.values(),
                uniq.keys(),
                loc="center",
                fontsize=8,
                ncol=min(len(uniq), 3),
            )

    # Hide any remaining unused axes beyond legend slot
    for idx in range(n_metrics + 1, nrows * ncols):
        axes_flat[idx].axis("off")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved plot to {out_path}")


def plot_ae_losses_multi(logs_by_chr: Dict[int, Dict[str, Any]], out_path: str) -> None:
    """
    Overlay AE training curves for multiple chromosomes on the same set of panels.
    Each chromosome is plotted with a different color/label.
    """
    if not logs_by_chr:
        print("No logs found to plot.")
        return

    # Build union of epochs across chromosomes; we will plot as-is per chromosome
    # and rely on each chromosome's own epoch indexing.
    # Panel specs similar to single-plot version (no total loss, recon and val_mse separate)
    metric_panels: List[Tuple[str, Optional[str], Optional[str]]] = []
    metric_panels.append(("recon", "train.recon", None))
    metric_panels.append(("maf", "train.maf", None))
    metric_panels.append(("ld", "train.ld", None))
    metric_panels.append(("kl", "train.kl", None))
    metric_panels.append(("val_mse", None, "val.val_mse"))
    # latent metrics (validation only, exclude normalized stability)
    latent_metrics = ["tv", "robust", "stable"]
    for m in latent_metrics:
        metric_panels.append((f"latent_{m}", None, f"latent.val.{m}"))

    n_metrics = len(metric_panels)
    ncols = 3
    nslots = n_metrics + 1  # extra slot for legend
    nrows = math.ceil(nslots / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 3.5 * nrows), squeeze=False)
    axes_flat = axes.reshape(-1)

    title_map = {
        "recon": "Reconstruction Loss",
        "maf": "MAF Penalty",
        "ld": "LD Penalty",
        "kl": "KL Divergence",
        "val_mse": "Validation MSE",
        "latent_tv": "Latent Spatial Smoothness",
        "latent_robust": "Latent Robustness",
        "latent_stable": "Latent Stability",
        "latent_stable_norm": "Latent Stability (Normalized)",
    }
    ylabel_map = {
        "recon": "Recon loss",
        "maf": "MAF loss",
        "ld": "LD loss",
        "kl": "KL loss",
        "val_mse": "Val MSE",
        "latent_tv": "TV loss",
        "latent_robust": "Robust loss",
        "latent_stable": "Stable loss",
        "latent_stable_norm": "Stable (normed) loss",
    }

    all_handles = []
    all_labels = []

    def _get(series_dict: Dict[str, Any], keypath: str) -> Optional[List[float]]:
        # keypath like "train.loss" or "latent.train.tv"
        parts = keypath.split(".")
        cur: Any = series_dict
        for p in parts:
            if p not in cur:
                return None
            cur = cur[p]
        return cur  # should be list

    for idx, (panel_key, train_key, val_key) in enumerate(metric_panels):
        ax = axes_flat[idx]
        for chrom, stats in sorted(logs_by_chr.items(), key=lambda x: x[0]):
            epochs = stats["epochs"]
            label = f"Chromosome {chrom}"
            if train_key is not None:
                train_vals = _get(stats, train_key)
                if train_vals is not None:
                    ax.plot(epochs, train_vals, label=label)
            if val_key is not None:
                val_vals = _get(stats, val_key)
                if val_vals is not None:
                    ax.plot(epochs, val_vals, label=label)
        title = title_map.get(panel_key, panel_key.replace("_", " ").title())
        ylabel = ylabel_map.get(panel_key, title)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        h, l = ax.get_legend_handles_labels()
        all_handles.extend(h)
        all_labels.extend(l)
        ax.grid(True, alpha=0.3)

    # Legend-only subplot
    legend_idx = n_metrics
    if legend_idx < len(axes_flat):
        legend_ax = axes_flat[legend_idx]
        legend_ax.axis("off")
        uniq = {}
        for h, lab in zip(all_handles, all_labels):
            if lab not in uniq:
                uniq[lab] = h
        if uniq:
            legend_ax.legend(
                uniq.values(),
                uniq.keys(),
                loc="center",
                fontsize=8,
                ncol=min(len(uniq), 3),
            )

    # Hide any remaining unused axes beyond legend slot
    for idx in range(n_metrics + 1, nrows * ncols):
        axes_flat[idx].axis("off")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved multi-chromosome plot to {out_path}")


def main():
    args = parse_args()
    # If explicit log-path provided, plot single
    if args.log_path:
        print(f"Parsing log file: {args.log_path}")
        log_stats = parse_ae_training_log(args.log_path)
        print(f"Plotting losses to: {args.out_path}")
        plot_ae_losses(log_stats, args.out_path)
        return

    # Else, discover logs per chromosome based on model name and saved-last marker
    chroms = _parse_chromosome_list(args.chromosomes)
    logs_by_chr: Dict[int, Dict[str, Any]] = {}
    for chrom in chroms:
        log_path = find_log_for_chromosome(args.logs_dir, chrom, args.model_name)
        if log_path is None:
            print(f"[WARN] No log found for chr{chrom} (model {args.model_name}) under {args.logs_dir}")
            continue
        try:
            stats = parse_ae_training_log(log_path)
            logs_by_chr[chrom] = stats
            print(f"[OK] Using log for chr{chrom}: {log_path}")
        except Exception as ex:
            print(f"[WARN] Failed to parse log for chr{chrom} at {log_path}: {ex}")
            continue

    print(f"Plotting multi-chromosome losses to: {args.out_path}")
    plot_ae_losses_multi(logs_by_chr, args.out_path)

if __name__ == "__main__":
    main()