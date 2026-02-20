import argparse, os, json, ast
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--csv", type=str, required=True)
parser.add_argument("--outdir", type=str, default="viz_vqvae_grid")
args = parser.parse_args()
os.makedirs(args.outdir, exist_ok=True)

def _split_csv_line_bracket_aware(line: str):
    line = line.rstrip("\n")
    parts, buf, depth = [], [], 0
    for ch in line:
        if ch == '[':
            depth += 1
            buf.append(ch)
        elif ch == ']':
            depth = max(0, depth - 1)
            buf.append(ch)
        elif ch == ',' and depth == 0:
            parts.append(''.join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    parts.append(''.join(buf).strip())
    return parts

def robust_read_results_csv(path: str) -> pd.DataFrame:
    with open(path, 'r') as f:
        lines = f.readlines()
    header = _split_csv_line_bracket_aware(lines[0])
    rows = []
    for ln in lines[1:]:
        if not ln.strip():
            continue
        fields = _split_csv_line_bracket_aware(ln)
        # Normalize field count
        if len(fields) < len(header):
            fields += [""] * (len(header) - len(fields))
        if len(fields) > len(header):
            # If there's an extra due to trailing comma, drop empties at end
            while len(fields) > len(header) and fields[-1] == "":
                fields.pop()
            # If still too many, merge extras into the last column (notes)
            if len(fields) > len(header):
                fields = fields[:len(header)-1] + [",".join(fields[len(header)-1:])]
        rows.append(fields)
    df = pd.DataFrame(rows, columns=header)

    # Coerce numerics
    num_cols = [
        "elapsed_sec","cuda_peak_mem_MB",
        "latent_grid_dim","latent_dim","codebook_size",
        "num_quantizers","hidden_channels","width_mult_per_stage",
        "lr","beta_commit","ema_decay","ema_eps",
        "ld_lambda","maf_lambda","ld_window",
        "acc","mse","commit"
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Parse perplexities
    def parse_perpl(x):
        if pd.isna(x) or x == "":
            return np.nan
        s = str(x).strip()
        try:
            if not (s.startswith('[') and s.endswith(']')):
                return np.nan
            try:
                vals = json.loads(s)
            except json.JSONDecodeError:
                vals = ast.literal_eval(s)
            vals = np.array(vals, dtype=float)
            if vals.size == 0:
                return np.nan
            return pd.Series({
                "perpl_mean": np.mean(vals),
                "perpl_median": np.median(vals),
                "perpl_min": np.min(vals),
                "perpl_max": np.max(vals),
                "perpl_len": len(vals),
            })
        except Exception:
            return pd.Series({"perpl_mean": np.nan, "perpl_median": np.nan, "perpl_min": np.nan, "perpl_max": np.nan, "perpl_len": np.nan})
    perpl_df = df["perplexities"].apply(parse_perpl)
    if isinstance(perpl_df, pd.Series):
        # all NaN
        df["perpl_mean"] = np.nan
        df["perpl_median"] = np.nan
        df["perpl_min"] = np.nan
        df["perpl_max"] = np.nan
        df["perpl_len"] = np.nan
    else:
        df = pd.concat([df, perpl_df], axis=1)

    df["okish"] = df["status"].isin(["ok"])
    for c in ["latent_grid_dim","latent_dim","codebook_size","num_quantizers","hidden_channels","ld_window"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    return df

df = robust_read_results_csv(args.csv)
df_ok = df[df["okish"] & df["acc"].notna()].copy()

sns.set(context="talk", style="whitegrid")

# Leaderboard
topN = 30
leader = df_ok.sort_values("acc", ascending=False).head(topN)
leader.to_csv(os.path.join(args.outdir, "leaderboard_top.csv"), index=False)
plt.figure(figsize=(12, 8))
ax = sns.barplot(data=leader, y="acc", x="latent_grid_dim", hue="latent_dim", dodge=True)
ax.set_title("Top models by accuracy (acc)")
ax.set_xlabel("latent_grid_dim"); ax.set_ylabel("Accuracy")
plt.legend(title="latent_dim", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout(); plt.savefig(os.path.join(args.outdir, "01_top_by_acc_bar.png"), dpi=150); plt.close()

# Heatmap
if not df_ok.empty:
    pv = df_ok.pivot_table(index="latent_grid_dim", columns="latent_dim", values="acc", aggfunc="max")
    plt.figure(figsize=(10, 7))
    sns.heatmap(pv, annot=True, fmt=".3f", cbar_kws={"label": "acc"})
    plt.title("Best acc by (latent_grid_dim × latent_dim)")
    plt.ylabel("latent_grid_dim"); plt.xlabel("latent_dim")
    plt.tight_layout(); plt.savefig(os.path.join(args.outdir, "02_heatmap_gdim_x_ldim_acc.png"), dpi=150); plt.close()

# Relplot
if {"codebook_size","num_quantizers"}.issubset(df_ok.columns):
    g = sns.relplot(data=df_ok, kind="line", x="latent_grid_dim", y="acc",
                    col="codebook_size", hue="num_quantizers",
                    marker="o", facet_kws=dict(sharey=False), height=4, aspect=1.1)
    g.set_titles("K={col_name}"); g.set_axis_labels("latent_grid_dim", "acc")
    plt.suptitle("Accuracy vs latent_grid_dim (hue=num_quantizers)", y=1.02)
    plt.tight_layout(); plt.savefig(os.path.join(args.outdir, "03_acc_vs_gdim_by_codebook.png"), dpi=150, bbox_inches="tight"); plt.close()

# Box by codebook faceted by latent_dim
g = sns.catplot(data=df_ok, kind="box", x="codebook_size", y="acc",
                col="latent_dim", col_wrap=4, height=3.6, aspect=1.0)
g.set_titles("latent_dim={col_name}"); g.set_axis_labels("codebook_size", "acc")
plt.suptitle("acc across codebook_size (by latent_dim)", y=1.02)
plt.tight_layout(); plt.savefig(os.path.join(args.outdir, "04_box_acc_by_codebook_faceted_ldim.png"), dpi=150, bbox_inches="tight"); plt.close()

# Acc vs CUDA mem
plt.figure(figsize=(8,6))
ax = sns.scatterplot(data=df_ok, x="cuda_peak_mem_MB", y="acc", hue="codebook_size", style="num_quantizers", s=80)
ax.set_title("Accuracy vs CUDA peak memory"); ax.set_xlabel("CUDA peak memory (MB)"); ax.set_ylabel("Accuracy")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left"); plt.tight_layout()
plt.savefig(os.path.join(args.outdir, "05_acc_vs_cuda_peak.png"), dpi=150); plt.close()

# Acc vs MSE
plt.figure(figsize=(8,6))
ax = sns.scatterplot(data=df_ok, x="mse", y="acc", hue="latent_grid_dim", palette="viridis", s=80)
ax.set_title("acc vs mse"); ax.set_xlabel("MSE"); ax.set_ylabel("Accuracy")
plt.legend(title="latent_grid_dim", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout(); plt.savefig(os.path.join(args.outdir, "06_acc_vs_mse.png"), dpi=150); plt.close()

# Perplexity vs acc
if "perpl_mean" in df_ok.columns and df_ok["perpl_mean"].notna().any():
    plt.figure(figsize=(8,6))
    ax = sns.scatterplot(data=df_ok, x="perpl_mean", y="acc", hue="codebook_size", style="latent_dim", s=80)
    ax.set_title("acc vs codebook perplexity (mean)"); ax.set_xlabel("perpl_mean"); ax.set_ylabel("acc")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left"); plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "07_acc_vs_perpl_mean.png"), dpi=150); plt.close()

# Status counts
plt.figure(figsize=(6,5))
ax = sns.countplot(data=df, x="status", order=df["status"].value_counts().index)
ax.set_title("Run status counts"); ax.set_xlabel("status"); ax.set_ylabel("count")
plt.tight_layout(); plt.savefig(os.path.join(args.outdir, "08_status_counts.png"), dpi=150); plt.close()

# Pairplot
subset_cols = ["acc","mse","commit","cuda_peak_mem_MB","latent_grid_dim","latent_dim","codebook_size","num_quantizers","hidden_channels","width_mult_per_stage"]
subset = df_ok[subset_cols].dropna()
if len(subset) > 2:
    g = sns.pairplot(subset, corner=True, diag_kind="hist", plot_kws=dict(s=20, edgecolor="none", alpha=0.7))
    g.fig.suptitle("Pairwise relationships", y=1.02)
    plt.tight_layout(); g.savefig(os.path.join(args.outdir, "09_pairplot_metrics_hparams.png"), dpi=150); plt.close()

# Acc by width_mult & hidden_channels
plt.figure(figsize=(8,6))
ax = sns.boxplot(data=df_ok, x="width_mult_per_stage", y="acc", hue="hidden_channels")
ax.set_title("acc by width_mult_per_stage & hidden_channels")
ax.set_xlabel("width_mult_per_stage"); ax.set_ylabel("acc")
plt.legend(title="hidden_channels", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout(); plt.savefig(os.path.join(args.outdir, "10_acc_by_width_mult_hidden_channels.png"), dpi=150); plt.close()

print(f"Saved visualizations to: {os.path.abspath(args.outdir)}")