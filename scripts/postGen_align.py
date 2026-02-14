import os
import sys
import pickle
import argparse
import numpy as np
import h5py
import torch
from sklearn.decomposition import PCA

sys.path.append('/n/home03/ahmadazim/WORKING/genGen/DiffuGene/src')
from DiffuGene.utils.file_utils import read_bim_file

# read arguments
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int)
parser.add_argument("--chrNo", type=int)
args = parser.parse_args()
seed = args.seed
chrNo = args.chrNo

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

N = 8000
gen_calls_path = f'/n/home03/ahmadazim/WORKING/genGen/UKBVQVAE/genomic_data/generated_samples/synObs_decoded_unrelWhite_allchr_AE128z_8192_genBatch1/chr{chrNo}_calls.pt'
gen_logits_path = f'/n/home03/ahmadazim/WORKING/genGen/UKBVQVAE/genomic_data/generated_samples/synObs_decoded_unrelWhite_allchr_AE128z_8192_genBatch1/chr{chrNo}_logits.pt'
orig_calls_h5_path = f'/n/home03/ahmadazim/WORKING/genGen/UKBVQVAE/genomic_data/ae_h5/chr{chrNo}/batch00001.h5'

with h5py.File(orig_calls_h5_path, "r") as f:
    X = f["X"][:N]  # (N,L)
calls_orig = np.asarray(X, dtype=np.float32)
print(f"Original calls shape: {calls_orig.shape}")

calls_gen = torch.load(gen_calls_path, map_location="cpu").numpy().astype(np.float32, copy=False)[:N]
print(f"Generated calls shape: {calls_gen.shape}")

logits_gen = torch.load(gen_logits_path, map_location="cpu").numpy().astype(np.float32, copy=False)[:N]
print(f"Generated logits shape: {logits_gen.shape}")

X_o = calls_orig.astype(float)  # [max_rows, L]
X_g = calls_gen.astype(float)   # [max_rows, L]

def _parse_ld_blocks(det_file: str, min_snps: int = 20, max_blocks: int = 5) -> list[list[str]]:
    blocks: list[list[str]] = []
    try:
        with open(det_file, 'r') as fh:
            _ = fh.readline()
            for line in fh:
                parts = line.strip().split()
                if len(parts) < 6:
                    continue
                try:
                    nsnps = int(parts[4])
                except ValueError:
                    continue
                snp_ids = parts[5].split('|')
                if nsnps >= min_snps and len(snp_ids) == nsnps:
                    blocks.append(snp_ids)
                if len(blocks) >= max_blocks:
                    break
    except Exception as e:
        print(f"[LD] Failed to read det file: {e}")
        return []
    return blocks

def calc_maf_ivw_mse(maf_orig, maf_gen, n, eps=1e-8):
    var = (maf_orig * (1.0 - maf_orig)) / (2.0 * n)  # approx Var(\hat p)
    w = 1.0 / (var + eps)
    return np.mean(w * (maf_gen - maf_orig) ** 2)

def weighted_r2(y, yhat, w, eps=1e-12):
    w = w / (w.sum() + eps)
    ybar = np.sum(w * y)
    ss_res = np.sum(w * (y - yhat) ** 2)
    ss_tot = np.sum(w * (y - ybar) ** 2)
    return 1.0 - ss_res / (ss_tot + eps)

def maf_weighted_r2(maf_orig, maf_gen, eps=1e-6, mode="inv_var"):
    w = 1.0 / (maf_orig * (1.0 - maf_orig) + eps)
    return weighted_r2(maf_orig, maf_gen, w)

def mean_binomial_nll(Y, P, eps=1e-12):
    P = np.clip(P, eps, 1 - eps)
    return -np.mean(Y * np.log(P) + (2 - Y) * np.log(1 - P))

def loss_fn(X_o, X_g_corr, X_g_round, X_o_tr = None, chr_no = chrNo):
    maf_gen = X_g_corr.mean(axis=0) / 2.0
    maf_gen_round = X_g_round.mean(axis=0) / 2.0
    maf_orig = X_o.mean(axis=0) / 2.0

    # collect R2
    r2_unrounded = maf_weighted_r2(maf_orig, maf_gen)
    r2_rounded = maf_weighted_r2(maf_orig, maf_gen_round)

    # collect MAF MSE
    # maf_ivw_mse = calc_maf_ivw_mse(maf_orig, maf_gen, X_o.shape[0])
    # maf_ivw_mse_round = calc_maf_ivw_mse(maf_orig, maf_gen_round, X_o.shape[0])

    eps = 1e-6
    maf_log_mse = np.sqrt(np.mean(np.log((maf_gen + eps) / (maf_orig + eps))**2))
    maf_log_mse_round = np.sqrt(np.mean(np.log((maf_gen_round + eps) / (maf_orig + eps))**2))

    if X_o_tr is None:
        # we are evaluating IS so X_o_tr is X_o
        X_o_tr = X_o
    # phat_tr = X_o_tr.mean(0, keepdims=True) / 2.0
    # nll = None  # eval_generated_nll(X_g_corr,  phat_tr)
    # nll_round = mean_binomial_nll(X_g_round, phat_tr)
    nll = mean_binomial_nll(X_o_tr, X_g_corr.mean(0, keepdims=True) / 2.0)
    nll_round = mean_binomial_nll(X_o_tr, X_g_round.mean(0, keepdims=True) / 2.0)

    ld_block_dir = "/n/home03/ahmadazim/WORKING/genGen/UKB6PC/genomic_data/haploblocks/"
    bim_path = '/n/home03/ahmadazim/WORKING/genGen/UKBVQVAE/genomic_data/geneticBinary/ukb_allchr_unrel_britishWhite.bim'
    import glob as _glob
    det_files = _glob.glob(os.path.join(ld_block_dir, f"*chr{chr_no}_blocks.blocks.det"))
    if det_files and os.path.exists(bim_path):
        det_file = det_files[0]
        blocks = _parse_ld_blocks(det_file, min_snps=50, max_blocks=100)
        
        bim_chr = read_bim_file(bim_path, chr_no)
        snp_ids_chr = bim_chr["SNP"].astype(str).tolist()
        id_to_index = {snp: idx for idx, snp in enumerate(snp_ids_chr)}
        block_indices: list[list[int]] = []
        for snp_list in blocks:
            idxs = [id_to_index[s] for s in snp_list if s in id_to_index]
            if len(idxs) >= 2:
                block_indices.append(idxs)
        max_rows = min(2000, X_o.shape[0])
        block_true_rows = [X_o[:max_rows, idxs] for idxs in block_indices]
        block_recon_rows = [X_g_corr[:max_rows, idxs] for idxs in block_indices]
        block_recon_rows_round = [X_g_round[:max_rows, idxs] for idxs in block_indices]

        diffs = []
        diffs_round = []
        for i in range(len(block_true_rows)):
            # measure frobenius norm of the difference between the two matrices: unrounded 
            xt = block_true_rows[i].astype(np.float32)
            xr = block_recon_rows[i].astype(np.float32)
            xt = xt - xt.mean(0, keepdims=True)
            xr = xr - xr.mean(0, keepdims=True)
            cov_t = (xt.T @ xt) / max(1, xt.shape[0] - 1)
            cov_r = (xr.T @ xr) / max(1, xr.shape[0] - 1)
            std_t = np.sqrt(np.clip(np.diag(cov_t), 1e-6, None))
            std_r = np.sqrt(np.clip(np.diag(cov_r), 1e-6, None))
            corr_t = cov_t / (std_t[:, None] * std_t[None, :])
            corr_r = cov_r / (std_r[:, None] * std_r[None, :])
            diff = np.linalg.norm(corr_t - corr_r)
            diffs.append(diff)
            
            # measure frobenius norm of the difference between the two matrices: rounded 
            xt = block_true_rows[i].astype(np.float32)
            xr = block_recon_rows_round[i].astype(np.float32)
            xt = xt - xt.mean(0, keepdims=True)
            xr = xr - xr.mean(0, keepdims=True)
            cov_t = (xt.T @ xt) / max(1, xt.shape[0] - 1)
            cov_r = (xr.T @ xr) / max(1, xr.shape[0] - 1)
            std_t = np.sqrt(np.clip(np.diag(cov_t), 1e-6, None))
            std_r = np.sqrt(np.clip(np.diag(cov_r), 1e-6, None))
            corr_t = cov_t / (std_t[:, None] * std_t[None, :])
            corr_r = cov_r / (std_r[:, None] * std_r[None, :])
            diff_round = np.linalg.norm(corr_t - corr_r)
            diffs_round.append(diff_round)
    return r2_unrounded, r2_rounded, maf_log_mse, maf_log_mse_round, np.mean(diffs), np.mean(diffs_round), nll, nll_round


k_grid = [50, 100, 250, 500, 600, 750, 850, 900, 950, 1000, 1050, 1100, 1200, 1500, 2000, 3000, 4000, 5000, 6000]
k_max = max(k_grid)

rng = np.random.default_rng(seed)
n = X_o.shape[0]
perm = rng.permutation(n)
n_tr = int(0.8 * n)
tr_idx = perm[:n_tr]
te_idx = perm[n_tr:]
print(tr_idx.shape, te_idx.shape)


mean_o = X_o[tr_idx].mean(axis=0, keepdims=True)          # [1, L]
std_o = X_o.std(axis=0, keepdims=True)            # [1, L]
std_o[std_o == 0] = 1.0                           # avoid /0

X_o_norm = (X_o - mean_o) / std_o                 # [max_rows, L]
X_g_norm = (X_g - mean_o) / std_o                 # [max_rows, L]

pca_o = PCA(n_components=k_max).fit(X_o_norm[tr_idx])
pca_g = PCA(n_components=k_max).fit(X_g_norm[tr_idx])

# Z_o = pca_o.transform(X_o_norm)
Z_g = pca_g.transform(X_g_norm)
Z_g_te = pca_g.transform(X_g_norm[te_idx])
Z_g_tr = pca_g.transform(X_g_norm[tr_idx])

W_o = pca_o.components_
W_g = pca_g.components_

losses = {}
losses_insample = {}
R_mats = {}

for k in k_grid:
    M = W_g[:k, :] @ W_o[:k, :].T
    try:
        U, _, Vt = np.linalg.svd(M)
        R = U @ Vt
        R_mats[k] = R
    except Exception as e:
        print(f"Error in SVD for k={k}: {e}")
        continue

    Z_g_corr_k = Z_g_te[:, :k] @ R.T
    Z_g_corr = np.zeros_like(Z_g_te)
    Z_g_corr[:, :k] = Z_g_corr_k
    X_g_norm_corr = pca_o.inverse_transform(Z_g_corr)

    X_g_corr = X_g_norm_corr * std_o + mean_o           # [max_rows, L]
    X_g_round = np.clip(np.round(X_g_corr), 0, 2)

    losses[k] = loss_fn(X_o[te_idx], X_g_corr, X_g_round, X_o[tr_idx], chr_no = chrNo)

    Z_g_corr_k = Z_g_tr[:, :k] @ R.T
    Z_g_corr = np.zeros_like(Z_g_tr)
    Z_g_corr[:, :k] = Z_g_corr_k
    X_g_norm_corr = pca_o.inverse_transform(Z_g_corr)

    X_g_corr = X_g_norm_corr * std_o + mean_o           # [max_rows, L]
    X_g_round = np.clip(np.round(X_g_corr), 0, 2)

    losses_insample[k] = loss_fn(X_o[tr_idx], X_g_corr, X_g_round, chr_no = chrNo)

res = {}
res['losses'] = losses
res['losses_insample'] = losses_insample
res['pca_g'] = pca_g
res['pca_o'] = pca_o
res['mean_o'] = mean_o
res['std_o'] = std_o
res['R_mats'] = R_mats
res['tr_idx'] = tr_idx
res['te_idx'] = te_idx
save_path = f'/n/home03/ahmadazim/WORKING/genGen/UKBVQVAE/models/PCalign/test/chr{chrNo}_seed{seed}.pkl'
with open(save_path, 'wb') as f:
    pickle.dump(res, f)
print(f"Saved results to {save_path}")