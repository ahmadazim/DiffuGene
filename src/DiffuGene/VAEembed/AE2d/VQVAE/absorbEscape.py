# ar_models.py
# ============================================================
# Definitions only. Modular, concise, and covers:
#  - GRU AR variants (uni/bi, with pos emb + residuals)
#  - MaskedLM Transformer refiner
#  - DosageCorrection CNN
#  - Datasets for AR / MLM / CNN
#  - Generic training utilities (per-task loss hooks)
#  - Absorb & Escape utilities (single/batch/fast)
# ============================================================

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, Literal, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


# ----------------------------
# Models
# ----------------------------

class BlockARSimple(nn.Module):
    """
    Simple unidirectional GRU AR over tokens {0,1,2,...,V-1}
    Forward: x -> logits for each position (teacher-forced)
    """
    def __init__(self, n_cat: int = 3, emb_dim: int = 16, hidden_dim: int = 64):
        super().__init__()
        self.embed = nn.Embedding(n_cat, emb_dim)
        self.rnn1 = nn.GRU(emb_dim, hidden_dim, batch_first=True)
        self.ln1  = nn.LayerNorm(hidden_dim)
        self.rnn2 = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.ln2  = nn.LayerNorm(hidden_dim)
        self.out  = nn.Linear(hidden_dim, n_cat)

    def forward(self, x: torch.LongTensor) -> torch.Tensor:
        e = self.embed(x)               # [B,L,E]
        h1, _ = self.rnn1(e)
        h1 = self.ln1(h1)
        h2, _ = self.rnn2(h1)
        h2 = self.ln2(h2) + h1
        return self.out(h2)             # [B,L,V]


class BlockARPosRes(nn.Module):
    """
    GRU stack w/ token+position embeddings, residual projection, LayerNorm.
    Set bidirectional=False to match earlier variant.
    """
    def __init__(self,
                 block_len: int,
                 n_cat: int = 3,
                 emb_dim: int = 64,
                 hidden_dim: int = 256,
                 nlayers: int = 2,
                 bidirectional: bool = False):
        super().__init__()
        self.pos_embed = nn.Embedding(block_len, emb_dim)
        self.tok_embed = nn.Embedding(n_cat, emb_dim)
        self.rnns, self.norms, self.res_projs = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()

        in_dim = emb_dim
        for _ in range(nlayers):
            if bidirectional:
                assert hidden_dim % 2 == 0, "hidden_dim must be even for bidirectional GRU"
                gru_h = hidden_dim // 2
                rnn = nn.GRU(in_dim, gru_h, batch_first=True, bidirectional=True)
                out_dim = gru_h * 2
            else:
                rnn = nn.GRU(in_dim, hidden_dim, batch_first=True)
                out_dim = hidden_dim
            self.rnns.append(rnn)
            self.norms.append(nn.LayerNorm(out_dim))
            self.res_projs.append(nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity())
            in_dim = out_dim

        self.out = nn.Linear(in_dim, n_cat)

    def forward(self, x: torch.LongTensor) -> torch.Tensor:
        B, L = x.size()
        pos = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)
        h = self.tok_embed(x) + self.pos_embed(pos)
        for rnn, ln, proj in zip(self.rnns, self.norms, self.res_projs):
            res, _ = rnn(h)
            h = ln(res + proj(h))
        return self.out(h)  # [B,L,V]


class MaskedLMTransformer(nn.Module):
    """
    MLM-style refiner that conditions on:
      - token inputs (rounded calls or [MASK])
      - positional embeddings
      - a residual channel from continuous recon - rounded
    Forward returns logits over vocab at each site.
    """
    def __init__(self,
                 vocab_size: int = 4,
                 d_model: int = 64,
                 nhead: int = 4,
                 num_layers: int = 3,
                 dim_ff: int = 256,
                 max_len: int = 4096):
        super().__init__()
        self.tok_emb    = nn.Embedding(vocab_size, d_model)
        self.pos_emb    = nn.Parameter(torch.randn(1, max_len, d_model))
        self.resid_proj = nn.Linear(1, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                               dim_feedforward=dim_ff, activation='gelu', batch_first=False)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.lm_head  = nn.Linear(d_model, vocab_size)

    def forward(self,
                input_ids: torch.LongTensor,   # [B, L]
                recon_seq: torch.FloatTensor,  # [B, L] in [0,2]
                x_round:   torch.LongTensor    # [B, L]
                ) -> torch.Tensor:
        B, L = input_ids.size()
        tok = self.tok_emb(input_ids)                 # [B,L,D]
        pos = self.pos_emb[:, :L, :]                  # [1,L,D]
        resid = (recon_seq - x_round.float()).unsqueeze(-1)  # [B,L,1]
        resid = self.resid_proj(resid)                # [B,L,D]
        x = (tok + pos + resid).transpose(0, 1)       # [L,B,D]
        h = self.encoder(x).transpose(0, 1)           # [B,L,D]
        return self.lm_head(h)                        # [B,L,V]


class DosageCorrectionCNN(nn.Module):
    """
    1D CNN over continuous recon_seq to predict categorical dosage {0,1,2}.
    """
    def __init__(self,
                 in_channels: int = 1,
                 hidden_channels: int = 64,
                 num_layers: int = 3,
                 kernel_size: int = 3,
                 num_classes: int = 3):
        super().__init__()
        layers = [nn.Conv1d(in_channels, hidden_channels, kernel_size, padding=kernel_size // 2), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers += [nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=kernel_size // 2), nn.ReLU()]
        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Conv1d(hidden_channels, num_classes, kernel_size=1)

    def forward(self, recon_seq: torch.FloatTensor) -> torch.Tensor:
        x = recon_seq.unsqueeze(1)                    # [B,1,L]
        feat = self.feature_extractor(x)              # [B,C,L]
        logits = self.classifier(feat).permute(0, 2, 1)  # [B,L,V]
        return logits


# ----------------------------
# Datasets
# ----------------------------

class ARShiftDataset(Dataset):
    """
    Next-token AR training:
      inputs: seq[:, :-1], targets: seq[:, 1:]
    """
    def __init__(self, full_block: np.ndarray):
        x = torch.as_tensor(full_block, dtype=torch.long)
        self.inp = x[:, :-1]
        self.tgt = x[:,  1:]

    def __len__(self) -> int: return self.inp.size(0)
    def __getitem__(self, idx) -> Tuple[torch.LongTensor, torch.LongTensor]:
        return self.inp[idx], self.tgt[idx]


class SNPRefineDataset(Dataset):
    """
    For MaskedLMTransformer:
      - Create labels only at masked locations (else -100).
      - Mask set = union(low-confidence proxy, random subset).
    """
    def __init__(self,
                 true_matrix: np.ndarray,     # [N,L] ints {0,1,2}
                 recon_matrix: np.ndarray,    # [N,L] floats in [0,2]
                 low_conf_thresh: float = 0.4,
                 mask_token_id: int = 3,
                 p_random: float = 0.10):
        self.true  = torch.as_tensor(true_matrix, dtype=torch.long)
        self.recon = torch.as_tensor(recon_matrix, dtype=torch.float)
        self.thr   = low_conf_thresh
        self.mask_id = mask_token_id
        self.p = p_random

    def __len__(self) -> int: return self.true.size(0)

    def __getitem__(self, idx):
        true_seq  = self.true[idx]               # [L]
        recon_seq = self.recon[idx]              # [L]
        L = true_seq.size(0)

        x_round = torch.clamp(torch.round(recon_seq), 0, 2).long()
        conf = 1.0 - 2.0 * torch.abs(recon_seq - x_round.float())   # 1..-1
        mask_low = conf < self.thr
        num_rand = max(1, int(self.p * L))
        rand_idx = torch.randperm(L)[:num_rand]
        mask_rand = torch.zeros(L, dtype=torch.bool); mask_rand[rand_idx] = True
        mask = mask_low | mask_rand

        input_ids = x_round.clone(); input_ids[mask] = self.mask_id
        labels = true_seq.clone();   labels[~mask] = -100

        return input_ids, recon_seq, x_round, labels


class CorrectionDataset(Dataset):
    """
    For DosageCorrectionCNN:
      input: recon_seq float [0,2], label: true_seq {0,1,2}
    """
    def __init__(self, true_mat: np.ndarray, recon_mat: np.ndarray):
        self.true  = torch.as_tensor(true_mat, dtype=torch.long)
        self.recon = torch.as_tensor(recon_mat, dtype=torch.float)

    def __len__(self) -> int: return self.true.size(0)
    def __getitem__(self, idx): return self.recon[idx], self.true[idx]


# ----------------------------
# Task registry / factories
# ----------------------------

TaskName = Literal["ar_simple", "ar_pos", "ar_bi", "mlm_refine", "cnn_correct"]

@dataclass
class ModelSpec:
    name: TaskName
    kwargs: Dict[str, Any]


def build_model(spec: ModelSpec) -> nn.Module:
    if spec.name == "ar_simple":
        return BlockARSimple(**spec.kwargs)
    if spec.name == "ar_pos":
        return BlockARPosRes(bidirectional=False, **spec.kwargs)
    if spec.name == "ar_bi":
        return BlockARPosRes(bidirectional=True, **spec.kwargs)
    if spec.name == "mlm_refine":
        return MaskedLMTransformer(**spec.kwargs)
    if spec.name == "cnn_correct":
        return DosageCorrectionCNN(**spec.kwargs)
    raise ValueError(f"Unknown model spec: {spec.name}")


def build_dataset(task: TaskName, **kwargs) -> Dataset:
    if task in ("ar_simple", "ar_pos", "ar_bi"):
        # expects: full_block=np.ndarray [N,L] of ints
        return ARShiftDataset(full_block=kwargs["full_block"])
    if task == "mlm_refine":
        return SNPRefineDataset(**kwargs)
    if task == "cnn_correct":
        return CorrectionDataset(**kwargs)
    raise ValueError(f"Unknown task: {task}")


# ----------------------------
# Training hooks (loss & batch → model args)
# ----------------------------

class LossHook(nn.Module):
    """Maps a batch to (logits, target) and computes loss."""
    def __init__(self, task: TaskName, ignore_index: int = -100):
        super().__init__()
        self.task = task
        self.ignore_index = ignore_index
        self.ce_all = nn.CrossEntropyLoss()
        self.ce_masked = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, model: nn.Module, batch: Iterable[torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if self.task in ("ar_simple", "ar_pos", "ar_bi"):
            xb, yb = batch                                   # [B,L-1], [B,L-1]
            logits = model(xb)                               # [B,L-1,V]
            loss = self.ce_all(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
            return loss, {"logits": logits, "targets": yb}

        if self.task == "mlm_refine":
            input_ids, recon_seq, x_round, labels = batch    # shapes [B,L]
            logits = model(input_ids, recon_seq, x_round)    # [B,L,V]
            loss = self.ce_masked(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            return loss, {"logits": logits, "targets": labels}

        if self.task == "cnn_correct":
            recon_seq, true_seq = batch
            logits = model(recon_seq)                        # [B,L,V]
            loss = self.ce_all(logits.reshape(-1, logits.size(-1)), true_seq.reshape(-1))
            return loss, {"logits": logits, "targets": true_seq}

        raise ValueError(f"Unknown task: {self.task}")


# ----------------------------
# AR sampling helpers
# ----------------------------

@torch.no_grad()
def ar_sample_greedy(model: nn.Module,
                     start_token_dist: torch.Tensor,
                     L: int,
                     device: torch.device,
                     mask_token: Optional[int] = None) -> torch.LongTensor:
    """
    Generate one sequence of length L from an AR model that accepts full prefix.
    start_token_dist: [V] probabilities for x_0
    """
    x = torch.empty(L, dtype=torch.long, device=device)
    x[0] = torch.multinomial(start_token_dist.to(device), 1).item()
    for j in range(1, L):
        if mask_token is None:
            inp = x[:j].unsqueeze(0)                # [1, j]
        else:
            # feed masked-at-j sequence if model expects full-length input
            masked = torch.full((L,), fill_value=mask_token, device=device, dtype=torch.long)
            masked[:j] = x[:j]
            inp = masked.unsqueeze(0)               # [1, L]
        logits = model(inp)                         # [..., V]
        next_logits = logits[0, j if mask_token is not None else j-1]
        x[j] = next_logits.argmax(-1)
    return x


# ----------------------------
# Absorb & Escape utilities
# ----------------------------

def absorb_escape_block(x_hat_block: np.ndarray,
                        model: nn.Module,
                        marginal: torch.Tensor,
                        max_iters: int = 10,
                        device: Optional[torch.device] = None) -> np.ndarray:
    """
    Original per-block Absorb & Escape: resample full block until convergence.
    x_hat_block: [L] continuous; rounds to init discrete guess.
    """
    if device is None:
        device = next(model.parameters()).device
    x_round = np.round(x_hat_block).astype(int)
    L = len(x_round)

    for _ in range(max_iters):
        new_seg: list[int] = []
        for j in range(L):
            if j == 0:
                probs = marginal.to(device)                         # [V]
            else:
                hist = torch.tensor(new_seg, dtype=torch.long, device=device).unsqueeze(0)  # [1,j]
                with torch.no_grad():
                    logits = model(hist)                            # [1,j,V]
                    probs = F.softmax(logits[0, -1], dim=-1)        # [V]
            xj = torch.multinomial(probs, 1).item()
            new_seg.append(xj)
        new_arr = np.array(new_seg, dtype=int)
        if np.array_equal(new_arr, x_round):
            break
        x_round = new_arr
    return x_round


def absorb_escape_block_batch(x_hat_block_batch: np.ndarray,
                              model: nn.Module,
                              marginal: torch.Tensor,
                              max_iters: int = 10,
                              device: Optional[torch.device] = None) -> np.ndarray:
    """
    Batched Absorb & Escape across B samples.
    """
    if device is None:
        device = next(model.parameters()).device
    x_round = np.round(x_hat_block_batch).astype(int)   # [B,L]
    x_round_t = torch.tensor(x_round, dtype=torch.long, device=device)
    B, L = x_round_t.shape
    marginal = marginal.to(device)

    for _ in range(max_iters):
        new_seg = torch.empty((B, L), dtype=torch.long, device=device)
        for j in range(L):
            if j == 0:
                probs = marginal.unsqueeze(0).expand(B, -1)                 # [B,V]
            else:
                hist = new_seg[:, :j]                                       # [B,j]
                with torch.no_grad():
                    logits = model(hist)                                     # [B,j,V]
                    probs  = F.softmax(logits[:, -1, :], dim=-1)            # [B,V]
            xj = torch.multinomial(probs, num_samples=1).squeeze(1)         # [B]
            new_seg[:, j] = xj
        if torch.equal(new_seg, x_round_t):
            break
        x_round_t = new_seg.clone()

    return x_round_t.cpu().numpy()


def fast_absorb_escape(x0: torch.LongTensor,
                       p_dm: torch.Tensor,
                       ar_model: nn.Module,
                       T_absorb: float,
                       device: torch.device,
                       mask_token: int) -> torch.LongTensor:
    """
    Selective, windowed A&E guided by DM confidence p_dm.
    x0: [L] initial discrete calls, p_dm: [L] site confidence proxy.
    ar_model: accepts length-L masked sequence where position j is [MASK].
    """
    L = x0.size(0)
    x = x0.clone().to(device)
    i = 0
    while i < L:
        if i < L - 1 and p_dm[i] < T_absorb:
            j = i + 1
            x_prime = x.clone()
            first = True
            while True:
                masked_seq = x_prime.clone()
                masked_seq[j] = mask_token
                inp = masked_seq.unsqueeze(0)
                with torch.no_grad():
                    logits = ar_model(inp)                  # [1,L,V]
                    probs  = F.softmax(logits[0, j], dim=-1)
                new_token = torch.multinomial(probs, 1).item()
                x_prime[j] = new_token
                p_ar_j = probs[new_token]
                if j == L - 1:
                    break
                if (p_ar_j <= p_dm[j]) and (not first or p_ar_j <= p_dm[j]):
                    break
                first = False
                j += 1
            x[i: j+1] = x_prime[i: j+1]
            i = j + 1
        else:
            i += 1
    return x


def fast_absorb_escape_batch(x0_batch: np.ndarray,
                             p_dm_batch: np.ndarray,
                             ar_model: nn.Module,
                             T_absorb: float,
                             device: torch.device,
                             mask_token: Optional[int] = None) -> np.ndarray:
    """
    Vectorized wrapper over fast_absorb_escape.
    If mask_token is None, uses last embedding index of ar_model (common for [MASK]).
    """
    if mask_token is None:
        # heuristic: assume tok_embed exists
        mask_token = int(getattr(ar_model, "tok_embed").num_embeddings) - 1
    B, _ = x0_batch.shape
    out = np.zeros_like(x0_batch, dtype=np.int64)
    for b in range(B):
        x0 = torch.as_tensor(x0_batch[b], dtype=torch.long, device=device)
        p  = torch.as_tensor(p_dm_batch[b], dtype=torch.float, device=device)
        out[b] = fast_absorb_escape(x0, p, ar_model, T_absorb, device, mask_token).cpu().numpy()
    return out


# ----------------------------
# Lightweight trainer (definition only)
# ----------------------------

@dataclass
class TrainConfig:
    lr: float = 3e-4
    weight_decay: float = 0.0
    ignore_index: int = -100


class Stepper:
    """
    Stateless helpers to perform one optimization step or eval step given a LossHook.
    """
    def __init__(self, task: TaskName, ignore_index: int = -100):
        self.hook = LossHook(task, ignore_index)

    def train_step(self,
                   model: nn.Module,
                   batch: Tuple[torch.Tensor, ...],
                   optim: torch.optim.Optimizer) -> Dict[str, Any]:
        model.train()
        loss, out = self.hook(model, batch)
        optim.zero_grad()
        loss.backward()
        optim.step()
        out["loss"] = float(loss.detach().cpu())
        return out

    @torch.no_grad()
    def eval_step(self, model: nn.Module, batch: Tuple[torch.Tensor, ...]) -> Dict[str, Any]:
        model.eval()
        loss, out = self.hook(model, batch)
        out["loss"] = float(loss.detach().cpu())
        return out


def make_optimizer(model: nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)


# ----------------------------
# Tiny accuracy helpers (optional)
# ----------------------------

@torch.no_grad()
def masked_token_accuracy(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100) -> float:
    """
    logits: [B,L,V], labels: [B,L] with -100 ignored
    """
    preds = logits.argmax(-1)
    mask = labels != ignore_index
    if mask.sum().item() == 0:
        return float("nan")
    return (preds[mask] == labels[mask]).float().mean().item()


@torch.no_grad()
def full_seq_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """
    logits: [B,L,V], targets: [B,L], exact-token accuracy over all sites
    """
    preds = logits.argmax(-1)
    return (preds == targets).float().mean().item()