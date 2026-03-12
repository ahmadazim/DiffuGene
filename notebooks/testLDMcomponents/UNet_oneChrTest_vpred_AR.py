import os
import sys
import argparse
from typing import Tuple
import numpy as np
import h5py
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from diffusers import UNet2DConditionModel, DDPMScheduler, DDIMScheduler
import re
import matplotlib.pyplot as plt
from timm.utils import ModelEmaV3
from torch.optim.lr_scheduler import CosineAnnealingLR

base_channels: int = 256

# Make project src importable
sys.path.append('/n/home03/ahmadazim/WORKING/genGen/DiffuGene/src')
from DiffuGene.VAEembed.ae import GenotypeAutoencoder, VAEConfig, local_ld_penalty
from DiffuGene.utils.file_utils import read_bim_file


def load_ae_from_checkpoint(ae_ckpt_path: str, device: torch.device) -> Tuple[GenotypeAutoencoder, VAEConfig]:
    """
    Load a trained AE checkpoint that includes a 'config' and a 'model_state'.
    """
    payload = torch.load(ae_ckpt_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"AE checkpoint must be a dict payload: {ae_ckpt_path}")
    cfg_dict = payload.get("config")
    # Prefer last_state_dict; fall back to model_state, then best_state_dict if needed
    # state = payload.get("last_state_dict") or payload.get("model_state") or payload.get("best_state_dict")
    state = payload.get("model_state")
    if cfg_dict is None or state is None:
        raise KeyError(f"AE checkpoint missing 'config' or a valid state dict ('last_state_dict'/'model_state'/'best_state_dict'): {ae_ckpt_path}")
    cfg = VAEConfig(**cfg_dict)
    ae = GenotypeAutoencoder(
        input_length=cfg.input_length,
        K1=cfg.K1,
        K2=cfg.K2,
        C=cfg.C,
        embed_dim=cfg.embed_dim,
    )
    incompat = ae.load_state_dict(state, strict=True)
    if getattr(incompat, "missing_keys", None):
        print(f"[AE] Missing keys (expected for new heads or buffers): {incompat.missing_keys}")
    if getattr(incompat, "unexpected_keys", None):
        print(f"[AE] Unexpected keys present in checkpoint: {incompat.unexpected_keys}")
    ae.to(device).eval()
    for p in ae.parameters():
        p.requires_grad = False
    return ae, cfg


class LatentDiscreteViTEncoder(nn.Module):
    """
    ViT-style encoder over AE latents z ∈ R^{CxHxW} that produces a small set
    of discrete latents (slots), relaxed with Gumbel-Softmax, and a continuous
    conditioning vector for the UNet.
    """
    def __init__(
        self,
        in_channels: int,
        H: int,
        W: int,
        num_slots: int = 8,
        codebook_size: int = 64,
        code_dim: int = 32,
        d_model: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.H = int(H)
        self.W = int(W)
        self.num_slots = int(num_slots)
        self.codebook_size = int(codebook_size)
        self.code_dim = int(code_dim)
        self.token_proj = nn.Linear(self.in_channels, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.H * self.W, d_model))

        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            batch_first=True,
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers)

        # Predict logits for num_slots categorical variables, each with codebook_size categories.
        self.slot_proj = nn.Linear(d_model, self.num_slots * self.codebook_size)

        # Shared codebook; each category has an embedding in R^{code_dim}.
        self.codebook = nn.Embedding(self.codebook_size, self.code_dim)

        # Cond vector dimension that will be fed into the UNet FiLM
        self.cond_dim = self.num_slots * self.code_dim

    def forward(
        self,
        z: torch.Tensor,
        tau: float = 1.0,
        hard: bool = True,
    ) -> dict[str, torch.Tensor]:
        """
        z: [B, C, H, W] AE latents (in z-space, NOT genotypes).
        Returns:
            {
              "logits":      [B, num_slots, codebook_size],
              "soft_onehot": [B, num_slots, codebook_size],
              "cond_vec":    [B, num_slots * code_dim],
            }
        """
        B, C, H, W = z.shape
        assert C == self.in_channels, f"Expected C={self.in_channels}, got {C}"
        assert H == self.H and W == self.W, f"Expected H,W={self.H,self.W}, got {H,W}"

        # Flatten spatial dims -> tokens: [B, L, C]
        tokens = z.view(B, C, H * W).permute(0, 2, 1)  # [B, L, C]
        h = self.token_proj(tokens)  # [B, L, d_model]
        h = h + self.pos_embed  # broadcast over batch
        h = self.transformer(h)  # [B, L, d_model]

        # Pool tokens (simple mean pooling here)
        h_pool = h.mean(dim=1)  # [B, d_model]

        # Slot logits: [B, num_slots, codebook_size]
        logits = self.slot_proj(h_pool).view(
            B, self.num_slots, self.codebook_size
        )

        # Gumbel-Softmax relaxation for discrete slots
        soft_onehot = F.gumbel_softmax(
            logits,
            tau=tau,
            hard=hard,
            dim=-1,
        )  # [B, num_slots, codebook_size]

        # Map each slot's categorical distribution through the codebook
        # codebook.weight: [codebook_size, code_dim]
        # soft_onehot:     [B, num_slots, codebook_size]
        # => discrete_embs: [B, num_slots, code_dim]
        discrete_embs = torch.matmul(soft_onehot, self.codebook.weight)

        # Flatten all slots into a single conditioning vector
        cond_vec = discrete_embs.reshape(B, self.cond_dim)  # [B, num_slots * code_dim]

        return {
            "logits": logits,
            "soft_onehot": soft_onehot,
            "cond_vec": cond_vec,
        }
        
    @torch.no_grad()
    def codes_to_cond_vec(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Map integer codes [B, num_slots] -> cond_vec [B, num_slots * code_dim]
        using the same codebook used in forward().
        """
        B, S = codes.shape
        assert S == self.num_slots, f"Expected num_slots={self.num_slots}, got {S}"
        # clamp in case of any stray indices
        codes = codes.clamp(0, self.codebook_size - 1)
        # [B, num_slots, code_dim]
        discrete_embs = self.codebook(codes)
        # [B, num_slots * code_dim]
        cond_vec = discrete_embs.reshape(B, self.cond_dim)
        return cond_vec

class LatentUNET2D(nn.Module):
    """
    Latent UNet wrapper using Diffusers UNet2DConditionModel and
    cross-attention conditioning on a discrete latent vector.
    """
    def __init__(
        self,
        input_channels: int = 64,
        output_channels: int = 64,
        base_channels: int = 256,
        sample_size: int = 16,
        cond_dim: int = 0,
    ) -> None:
        super().__init__()
        self.base_channels = int(base_channels)
        self.cond_dim = int(cond_dim)

        # Project z-latents into UNet channel space
        self.input_proj = nn.Conv2d(
            input_channels,
            self.base_channels,
            kernel_size=3,
            padding=1,
        )

        # Cross-attention embedding space for discrete latents
        self.cross_attention_dim = 128
        if self.cond_dim > 0:
            self.cond_emb = nn.Sequential(
                nn.Linear(self.cond_dim, self.base_channels),
                nn.SiLU(),
                nn.Linear(self.base_channels, self.cross_attention_dim),
            )
            # Learnable null condition embedding for classifier-free guidance
            self.null_cond_emb = nn.Parameter(torch.randn(1, self.cross_attention_dim))
        else:
            self.cond_emb = None
            self.null_cond_emb = None

        # 16 -> 8 -> 4 -> 2 (3 downsamples), with a cross-attn mid block
        self.unet = UNet2DConditionModel(
            sample_size=sample_size,
            in_channels=self.base_channels,
            out_channels=self.base_channels,
            layers_per_block=2,
            block_out_channels=[
                self.base_channels,
                2 * self.base_channels,
                3 * self.base_channels,
            ],
            down_block_types=[
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
            ],
            up_block_types=[
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ],
            mid_block_type="UNetMidBlock2DCrossAttn",
            cross_attention_dim=self.cross_attention_dim,
            attention_head_dim=64,
        )

        # Project back to latent channel dimension
        self.output_proj = nn.Conv2d(
            self.base_channels,
            output_channels,
            kernel_size=3,
            padding=1,
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond_vec: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        x:        [B, C, H, W]   noisy latents
        t:        [B]            timesteps
        cond_vec: [B, cond_dim]  discrete-latent embedding (optional)
        """
        h = self.input_proj(x)  # [B, base_channels, H, W]

        if self.cond_emb is not None and cond_vec is not None:
            # Map cond_vec -> cross-attn space, add sequence dim
            cond_tokens = self.cond_emb(cond_vec).unsqueeze(1)  # [B,1,cross_attention_dim]
        else:
            cond_tokens = None

        y = self.unet(h, t, encoder_hidden_states=cond_tokens).sample
        return self.output_proj(y)


class LatentARPrior(nn.Module):
    """
    Simple autoregressive prior over discrete slots.
    Models p(s_1, ..., s_L) with a causal Transformer over slot indices.
    """
    def __init__(
        self,
        num_slots: int,
        codebook_size: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.num_slots = int(num_slots)
        self.codebook_size = int(codebook_size)
        self.hidden_dim = int(hidden_dim)

        self.token_embed = nn.Embedding(self.codebook_size, self.hidden_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_slots, self.hidden_dim))

        encoder_layer = TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=num_heads,
            batch_first=True,
        )

        self.transformer = TransformerEncoder(encoder_layer, num_layers)
        self.out_proj = nn.Linear(self.hidden_dim, self.codebook_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens: [B, num_slots] integer indices in [0, codebook_size)
        Returns:
            logits: [B, num_slots, codebook_size]
        """
        B, L = tokens.shape
        assert L == self.num_slots, f"Expected num_slots={self.num_slots}, got {L}"

        x = self.token_embed(tokens)  # [B, L, hidden_dim]
        x = x + self.pos_embed[:, :L, :]

        # Causal mask so position t can't attend to > t
        causal_mask = torch.triu(
            torch.ones(L, L, device=tokens.device, dtype=torch.bool),
            diagonal=1,
        )
        h = self.transformer(x, mask=causal_mask)  # [B, L, hidden_dim]
        logits = self.out_proj(h)  # [B, L, codebook_size]
        return logits

    @torch.no_grad()
    def sample(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Autoregressively sample slot indices.
        Returns:
            samples: [B, num_slots] integer indices
        """
        B, L = batch_size, self.num_slots
        tokens = torch.zeros(B, L, dtype=torch.long, device=device)
        for t in range(L):
            logits = self.forward(tokens)              # [B, L, vocab]
            step_logits = logits[:, t, :]              # [B, vocab]
            probs = F.softmax(step_logits, dim=-1)
            tokens[:, t] = torch.multinomial(probs, num_samples=1).squeeze(-1)
        return tokens

def create_memmap(path: str, shape: Tuple[int, int, int, int]) -> np.memmap:
    mm = np.memmap(path, dtype='float32', mode='w+', shape=shape)
    return mm


def save_shape_file(memmap_path: str, shape: Tuple[int, ...]) -> None:
    shape_file = memmap_path.replace('.npy', '_shape.txt')
    with open(shape_file, 'w') as f:
        f.write(','.join(map(str, shape)))


def compute_channel_stats_memmap(memmap_path: str, shape: Tuple[int, int, int, int], chunk_size: int = 512) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute per-channel mean/std over (N,H,W) by iterating the memmap to avoid peak memory.
    """
    N, C, H, W = shape
    mm = np.memmap(memmap_path, dtype='float32', mode='r', shape=shape)
    sum_c = np.zeros((C,), dtype=np.float64)
    sumsq_c = np.zeros((C,), dtype=np.float64)
    total = N * H * W
    for s in range(0, N, chunk_size):
        e = min(N, s + chunk_size)
        arr = np.asarray(mm[s:e])  # (b,C,H,W)
        sum_c += arr.sum(axis=(0, 2, 3))
        sumsq_c += (arr ** 2).sum(axis=(0, 2, 3))
    del mm
    mu = sum_c / float(total)
    ex2 = sumsq_c / float(total)
    var = np.maximum(ex2 - mu * mu, 1e-12)
    sd = np.sqrt(var)
    return torch.from_numpy(mu.astype('float32')), torch.from_numpy(sd.astype('float32'))


class MemmapLatentDataset(Dataset):
    def __init__(self, memmap_path: str, shape: Tuple[int, int, int, int]) -> None:
        self.memmap_path = memmap_path
        self.shape = shape
        self.mm = np.memmap(memmap_path, dtype='float32', mode='r', shape=shape)
    def __len__(self) -> int:
        return self.shape[0]
    def __getitem__(self, idx: int) -> torch.Tensor:
        x = self.mm[idx]  # (C,H,W)
        return torch.from_numpy(np.array(x, copy=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Temporary UNet training on chr22 16x16 latents using VAE")
    parser.add_argument("--ae-model-path", type=str, required=True, help="Path to trained AE checkpoint (.pt)")
    parser.add_argument("--h5-batch-path", type=str, required=True, help="Path to one H5 batch file, e.g., .../vqvae_h5_cache/chr22/batch00001.h5")
    parser.add_argument("--memmap-out", type=str, required=True, help="Output memmap file, e.g., /path/to/latents_chr22_batch00001_memmap.npy")
    parser.add_argument("--model-out-path", type=str, required=True, help="Where to save the trained UNet")
    parser.add_argument("--encode-batch-size", type=int, default=128)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--max-batches", type=int, default=8, help="Max number of sequential H5 batches to include starting from the provided one")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--active-channels", type=int, default=64, help="Number of active channels to train on")
    parser.add_argument("--global-sigma-batches", type=int, default=4, help="Batches to estimate global sigma from")
    parser.add_argument(
        "--use-channel-norm",
        action="store_true",
        help="If set, apply per-channel mean/std normalization to latents and save stats.",
    )
    parser.add_argument(
        "--use-decoded-losses",
        action="store_true",
        help="If set, add AE-decoded CE/MAF/LD losses on top of epsilon MSE.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Load AE
    ae, cfg = load_ae_from_checkpoint(args.ae_model_path, device=device)

    # 2) Encode up to max-batches sequential H5 files starting from the provided one
    def _enumerate_h5_batch_paths(start_path: str, max_batches: int) -> list[str]:
        d = os.path.dirname(start_path)
        m = re.search(r"batch(\d{5})\.h5$", os.path.basename(start_path))
        if not m:
            raise ValueError(f"Cannot parse batch index from {start_path}")
        start_idx = int(m.group(1))
        paths: list[str] = []
        for i in range(start_idx, start_idx + int(max_batches)):
            p = os.path.join(d, f"batch{i:05d}.h5")
            if os.path.exists(p):
                paths.append(p)
            else:
                break
        if not paths:
            raise FileNotFoundError(f"No H5 batch files found starting at {start_path}")
        return paths

    def _encode_h5s_to_memmap(ae: GenotypeAutoencoder,
                              h5_paths: list[str],
                              memmap_out: str,
                              device: torch.device,
                              encode_batch_size: int = 128) -> tuple[str, tuple[int, int, int, int]]:
        # First pass: determine total samples
        total_N = 0
        for p in h5_paths:
            with h5py.File(p, 'r') as f:
                X = f['X']
                total_N += int(X.shape[0])
        # Use latent channels, since ae.forward returns z in latent space
        C, M = int(ae.latent_channels), int(ae.M2D)
        shape = (total_N, C, M, M)
        mm = create_memmap(memmap_out, shape)
        # Second pass: encode sequentially
        offset = 0
        for p in h5_paths:
            print(f"Encoding {p} -> memmap rows [{offset}, ...)")
            with h5py.File(p, 'r') as f:
                X = f['X']
                N = int(X.shape[0])
                for s in range(0, N, encode_batch_size):
                    e = min(N, s + encode_batch_size)
                    xb = torch.from_numpy(X[s:e].astype('int64'))
                    xb = xb.to(device, non_blocking=(device.type == 'cuda'))
                    with torch.no_grad():
                        logits, z = ae(xb)
                        z_cpu = z.detach().to('cpu').float().numpy()
                    mm[offset + s:offset + e] = z_cpu
                    del xb, logits, z, z_cpu
                    # del xb, logits, z, mu_z, logvar_z, mask, z_cpu
                    # if device.type == 'cuda':
                    #     torch.cuda.empty_cache()
            offset += N
        del mm
        save_shape_file(memmap_out, shape)
        return memmap_out, shape

    h5_paths = _enumerate_h5_batch_paths(args.h5_batch_path, int(args.max_batches))
    print(f"Encoding {len(h5_paths)} H5 batches to memmap: {args.memmap_out}")
    memmap_path, mm_shape = _encode_h5s_to_memmap(
        ae=ae,
        h5_paths=h5_paths,
        memmap_out=args.memmap_out,
        device=device,
        encode_batch_size=int(args.encode_batch_size),
    )

    # Infer latent shape and choose UNet sample size
    N_total, C, H, W = mm_shape
    unet_sample_size = H
    print(f"UNET SAMPLE SIZE: {unet_sample_size}")

    # Pre-training diagnostic: per-channel spatial variance (up to 2000 samples)
    try:
        print(f"[DIAGNOSTICS] Computing per-channel spatial variance (up to 2000 samples) from memmap: {args.memmap_out}")
        mm = np.memmap(memmap_path, dtype='float32', mode='r', shape=mm_shape)
        n_diag = min(2000, N_total)
        arr_diag = np.asarray(mm[:n_diag])  # (n_diag, C, H, W)
        del mm
        var_maps = arr_diag.var(axis=0)  # (C, H, W)
        del arr_diag
        vmax = float(var_maps.max())
        out_base_diag = os.path.splitext(args.model_out_path)[0]
        ncols = min(8, C)
        nrows = int(math.ceil(C / ncols)) if ncols > 0 else 1
        fig, axes = plt.subplots(nrows, ncols, figsize=(2 * ncols, 2 * nrows))
        axes = np.array(axes).reshape(nrows, ncols)
        for c in range(C):
            r, col = divmod(c, ncols)
            ax = axes[r, col]
            ax.imshow(var_maps[c], cmap='viridis', vmin=0.0, vmax=vmax)
            ax.set_title(f'ch {c}')
            ax.set_xticks([]); ax.set_yticks([])
        plt.tight_layout()
        fig_path_varmaps = f"{out_base_diag}_pretrain_latent_channel_variance.png"
        plt.savefig(fig_path_varmaps, dpi=150)
        plt.close(fig)
        print(f"[DIAGNOSTICS] Saved pre-training latent channel variance grid: {fig_path_varmaps}")
    except Exception as diag_ex:
        print(f"[DIAGNOSTICS] Skipping variance grid due to error: {diag_ex}")
    
    # Pre-training diagnostic: per-channel spatial mean (up to 2000 samples)
    try:
        print(f"[DIAGNOSTICS] Computing per-channel spatial mean (up to 2000 samples) from memmap: {args.memmap_out}")
        mm = np.memmap(memmap_path, dtype='float32', mode='r', shape=mm_shape)
        n_diag = min(2000, N_total)
        arr_diag = np.asarray(mm[:n_diag])  # (n_diag, C, H, W)
        del mm
        mean_maps = arr_diag.mean(axis=0)  # (C, H, W)
        del arr_diag
        vmax = float(mean_maps.max())
        out_base_diag = os.path.splitext(args.model_out_path)[0]
        ncols = min(8, C)
        nrows = int(math.ceil(C / ncols)) if ncols > 0 else 1
        fig, axes = plt.subplots(nrows, ncols, figsize=(2 * ncols, 2 * nrows))
        axes = np.array(axes).reshape(nrows, ncols)
        for c in range(C):
            r, col = divmod(c, ncols)
            ax = axes[r, col]
            ax.imshow(mean_maps[c], cmap='viridis', vmin=0.0, vmax=vmax)
            ax.set_title(f'ch {c}')
            ax.set_xticks([]); ax.set_yticks([])
        # hide any unused axes
        for idx in range(C, nrows * ncols):
            r, col = divmod(idx, ncols)
            axes[r, col].axis('off')
        plt.tight_layout()
        fig_path_meanmaps = f"{out_base_diag}_pretrain_latent_channel_mean.png"
        plt.savefig(fig_path_meanmaps, dpi=150)
        plt.close(fig)
        print(f"[DIAGNOSTICS] Saved pre-training latent channel mean grid: {fig_path_meanmaps}")
    except Exception as diag_ex:
        print(f"[DIAGNOSTICS] Skipping mean grid due to error: {diag_ex}")

    # 3) Optional per-channel normalization over the memmap
    print(f"Computing per-channel normalization over the memmap: {memmap_path}")
    if args.use_channel_norm:
        mu, sd = compute_channel_stats_memmap(memmap_path, mm_shape, chunk_size=max(256, args.train_batch_size))
        sd = torch.clamp(sd, min=1e-6)
    else:
        # Identity stats (no normalization)
        _, C, _, _ = mm_shape
        mu = torch.zeros(C, dtype=torch.float32)
        sd = torch.ones(C, dtype=torch.float32)

    # 4) Build dataset/loader from memmap
    print(f"Building dataset/loader from memmap: {memmap_path}")
    ds = MemmapLatentDataset(memmap_path, mm_shape)
    train_loader = DataLoader(ds, batch_size=int(args.train_batch_size), shuffle=True, num_workers=0, pin_memory=(device.type == "cuda"))

    # 4.5) Build discrete-latent ViT encoder for conditioning
    print(f"Building LatentDiscreteViTEncoder for C={C}, H={H}, W={W}")
    cond_slots = 8
    cond_codebook_size = 64
    cond_code_dim = 32
    disc_encoder = LatentDiscreteViTEncoder(
        in_channels=C,
        H=H,
        W=W,
        num_slots=cond_slots,
        codebook_size=cond_codebook_size,
        code_dim=cond_code_dim,
        d_model=128,
        num_layers=2,
        num_heads=4,
    ).to(device)
    cond_dim = disc_encoder.cond_dim

    # Stage-2 autoregressive prior over discrete slots
    print(f"Building LatentARPrior with num_slots={cond_slots}, vocab={cond_codebook_size}")
    ar_prior = LatentARPrior(
        num_slots=cond_slots,
        codebook_size=cond_codebook_size,
        hidden_dim=128,
        num_layers=2,
        num_heads=4,
    ).to(device)

    # 5) Build UNet for HxW latents with C channels (use sample_size=H) and conditioning
    print(f"Building UNet with sample_size={unet_sample_size} for latent shape C={C}, H={H}, W={W} and cond_dim={cond_dim}")
    model = LatentUNET2D(
        input_channels=C,
        output_channels=C,
        base_channels=base_channels,
        sample_size=unet_sample_size,
        cond_dim=cond_dim,
    ).to(device)

    # 6) DDPM scheduler and optimizer
    print(f"Building DDPM scheduler and optimizer")
    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        # beta_schedule="linear",
        beta_schedule="squaredcos_cap_v2",
        beta_start=1e-4,
        beta_end=0.02,
        clip_sample=True,
        clip_sample_range=10.0,
    )
    try:
        # scheduler.config.prediction_type = "epsilon"
        scheduler.config.prediction_type = "v_prediction"
        print(f"Set prediction type to v_prediction")
    except Exception:
        pass
    scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(device)
    scheduler.betas = scheduler.betas.to(device)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) 
        + list(disc_encoder.parameters())
        + list(ar_prior.parameters()), 
        lr=float(args.lr), 
        betas=(0.9, 0.99), 
        weight_decay=1e-4,
    )
    
    # Simple LR cooldown over epochs
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=int(args.epochs), eta_min=0.0)
    print(f"[LR] Using CosineAnnealingLR with T_max={int(args.epochs)} eta_min=0.0")

    # EMA on CPU
    ema_src = model
    for p in ema_src.parameters():
        p.data = p.data.float()
    ema = ModelEmaV3(ema_src, decay=0.99, device='cpu')
    for p in ema.module.parameters():
        p.data = p.data.float()

    # 7) If a checkpoint exists, load it to resume; then train for specified epochs
    if os.path.exists(args.model_out_path):
        print(f"[RESUME] Found checkpoint at {args.model_out_path}. Loading weights to continue training.")
        try:
            ckpt = torch.load(args.model_out_path, map_location="cpu")
            w = ckpt.get("weights")
            w_enc = ckpt.get("disc_encoder")
            w_ar = ckpt.get("ar_prior")
            e = ckpt.get("ema")
            if w is not None:
                model.load_state_dict(w)
            if w_enc is not None:
                disc_encoder.load_state_dict(w_enc)
            if w_ar is not None:
                ar_prior.load_state_dict(w_ar)
            if e is not None:
                ema.load_state_dict(e)
        except Exception as ex:
            print(f"[RESUME] Failed to load checkpoint ({ex}); starting from scratch.")
    
    # # special case hardcoded
    # hardcoded_ckpt_path = "/n/home03/ahmadazim/WORKING/genGen/UKBVQVAE/models/diffusion_test/UNET_regLat/UNET_chr22_regLat_veryhighbetaKL_vpred.pth"
    # ckpt = torch.load(hardcoded_ckpt_path, map_location="cpu")
    # model.load_state_dict(ckpt.get("weights"), strict=False)
    # ema.load_state_dict(ckpt.get("ema"), strict=False)

    # Always continue training
    print(f"Training for {args.epochs} epochs on this single memmap batch")
    model.train()
    mu_dev = mu.view(1, C, 1, 1).to(device)
    sd_dev = sd.view(1, C, 1, 1).to(device)
    
    # Weight for AR prior loss relative to diffusion loss
    ar_loss_weight = 5.0
    p_uncond = 0.1
    for epoch in range(1, int(args.epochs) + 1):
        total_loss = 0.0
        total_dm_loss = 0.0
        total_ar_loss = 0.0
        count = 0
        for x_cpu in train_loader:
            optimizer.zero_grad(set_to_none=True)
            z_clean = x_cpu.to(device, non_blocking=(device.type == "cuda")).float()
            disc_out = disc_encoder(z_clean)
            cond_vec = disc_out["cond_vec"]
            
            probs = F.softmax(disc_out["logits"], dim=-1)
            slot_ent = -(probs * probs.log()).sum(-1).mean(0)
            if count == 0:
                print("slot entropies:", slot_ent)

            # Stage-2 AR prior loss (next-slot prediction)
            with torch.no_grad():
                # hard assignments from encoder
                slot_idx = disc_out["logits"].argmax(dim=-1)  # [B, num_slots]

            B, L = slot_idx.shape

            # Input to AR (shift right and prepend BOS=0)
            bos_token = 0
            ar_input = torch.full(
                (B, L), bos_token,
                dtype=torch.long,
                device=slot_idx.device,
            )
            # positions 1..L-1 see previous true token (BOS is 0)

            ar_input[:, 1:] = slot_idx[:, :-1]
            # Targets are the true tokens at positions 0..L-1 (BOS is 0)
            ar_logits = ar_prior(ar_input)  # [B, L, codebook_size]

            # Compute CE only on positions 0..L-1 (all), but now each logit_t
            # only sees tokens < t+1 (because of the causal mask + BOS shift)
            ar_ce = F.cross_entropy(
                ar_logits.view(-1, cond_codebook_size),
                slot_idx.view(-1),
            )

            x0 = z_clean.to(memory_format=torch.channels_last).float()
            if args.use_channel_norm:
                x0 = (x0 - mu_dev) / sd_dev

            B = x0.size(0)
            t = torch.randint(
                0,
                scheduler.config.num_train_timesteps,
                (B,),
                device=device,
                dtype=torch.long,
            )

            noise = torch.randn_like(x0)
            x_noisy = scheduler.add_noise(x0, noise, t)

            # v-target: sqrt(alpha_bar) * eps - sqrt(1 - alpha_bar) * x0
            alpha_bar = scheduler.alphas_cumprod[t].view(B, 1, 1, 1).to(device)  # [B,1,1,1]
            sqrt_ab = torch.sqrt(alpha_bar)
            sqrt_1mab = torch.sqrt(1.0 - alpha_bar)
            v_true = sqrt_ab * noise - sqrt_1mab * x0

            # ------------------------------------------------------------
            # Classifier-free guidance training on discrete latents
            #   - e_cond: encoder_hidden_states for conditional branch
            #   - e_null: learned null embedding for unconditional branch
            #   - double batch: [uncond; cond], single UNet forward
            # ------------------------------------------------------------
            if model.cond_emb is not None and model.null_cond_emb is not None:
                # 1) Embed cond_vec into cross-attention space
                e_cond = model.cond_emb(cond_vec).unsqueeze(1)  # [B,1,cross_attention_dim]
                hidden_dim = e_cond.size(-1)
                e_null = model.null_cond_emb.unsqueeze(0).expand(B, 1, hidden_dim)  # [B,1,Hdim]
                if e_null.dtype != e_cond.dtype:
                    e_null = e_null.to(dtype=e_cond.dtype)

                # 2) Randomly drop some examples to null (unconditional) embedding
                drop_mask = (torch.rand(B, device=device) < p_uncond)  # [B]
                e_drop = e_cond.clone()
                e_drop[drop_mask] = e_null[drop_mask]

                # 3) Build double-batch: first half uses e_drop, second half uses full e_cond
                x_in = torch.cat([x_noisy, x_noisy], dim=0)      # [2B,C,H,W]
                t_in = torch.cat([t, t], dim=0)                  # [2B]
                emb_in = torch.cat([e_drop, e_cond], dim=0)      # [2B,1,Hdim]

                h_in = model.input_proj(x_in)
                out = model.unet(h_in, t_in, encoder_hidden_states=emb_in).sample
                out = model.output_proj(out)                     # [2B,C,H,W]

                v_uncond, v_cond = out.chunk(2, dim=0)           # each [B,C,H,W]

                # 4) v-space MSE for both branches
                diff_sq_un = (v_uncond - v_true).pow(2)
                diff_sq_co = (v_cond   - v_true).pow(2)
                dm_loss_un = diff_sq_un.mean()
                dm_loss_co = diff_sq_co.mean()
                dm_loss = 0.5 * (dm_loss_un + dm_loss_co)
            else:
                # Fallback: no CFG if conditioning is somehow disabled
                v_pred = model(x_noisy, t, cond_vec=None)
                dm_loss = (v_pred - v_true).pow(2).mean()

            entropy_loss = -0.001 * slot_ent.mean()  # currently not used

            loss = dm_loss + ar_loss_weight * ar_ce

            # # Recover clean latents on masked channels via v-parameterization:
            # # x0 = sqrt(alpha_bar) * x_t - sqrt(1 - alpha_bar) * v
            # x0_pred = sqrt_ab * x_noisy - sqrt_1mab * v_pred

            # # Map to latent units (z-space), respecting whether channel norm is enabled
            # if args.use_channel_norm:
            #     z_pred = x0_pred * sd_dev + mu_dev
            # else:
            #     z_pred = x0_pred
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Update EMA
            ema.update(ema_src)

            total_loss += float(loss.item()) * B
            total_dm_loss += float(dm_loss.item()) * B
            total_ar_loss += float(ar_ce.item()) * B
            count += B
        
        curr_lr = optimizer.param_groups[0].get("lr", float(args.lr))
        print(f"[Epoch {epoch}/{args.epochs}] loss={total_loss / max(1, count):.6f} | dm_loss={total_dm_loss / max(1, count):.6f} | ar_loss={total_ar_loss / max(1, count):.6f} | lr={curr_lr:.6e}")
        lr_scheduler.step()

    # 8) Save final model and normalization stats
    print(f"Saving final model and normalization stats: {args.model_out_path}")
    os.makedirs(os.path.dirname(args.model_out_path), exist_ok=True)
    torch.save(
        {
            "weights": model.state_dict(),
            "disc_encoder": disc_encoder.state_dict(),
            "ar_prior": ar_prior.state_dict(),
            "ema": ema.state_dict(),
            "input_channels": C,
            "sample_size": (H, W),
            "channel_mean": mu,  # CPU tensors
            "channel_std": sd,
        },
        args.model_out_path,
    )
    print(f"[DONE] Saved UNet to {args.model_out_path}")

    # 9) Diagnostics: generate latents and compare distributions vs original
    def _std_normal_pdf(x: np.ndarray) -> np.ndarray:
        return (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * x * x)

    @torch.no_grad()
    def _sample_ddim_ar_cfg(
        model: LatentUNET2D,
        disc_encoder: LatentDiscreteViTEncoder,
        ar_prior: LatentARPrior,
        num_samples: int,
        num_steps: int,
        guidance_scale: float,
        device: torch.device,
    ) -> torch.Tensor:
        """
        DDIM sampling with:
          - discrete codes sampled from the AR prior
          - codes -> cond_vec via encoder codebook
          - classifier-free guidance w.r.t. learned null vs AR codes
        """
        sched = DDIMScheduler(
            num_train_timesteps=1000,
            beta_schedule="squaredcos_cap_v2",
            beta_start=1e-4,
            beta_end=0.02,
            clip_sample=True,
            clip_sample_range=10.0,
        )
        sched.config.prediction_type = "v_prediction"
        sched.set_timesteps(num_inference_steps=num_steps, device=device)

        # 1) Sample discrete codes from AR prior
        codes = ar_prior.sample(num_samples, device=device)  # [B, num_slots]

        # 2) Map codes -> cond_vec using the same codebook as training
        cond_vec = disc_encoder.codes_to_cond_vec(codes).to(device)  # [B, cond_dim]

        # 3) Build unconditional (null) cond tokens for CFG
        if model.cond_emb is not None and model.null_cond_emb is not None:
            e_cond = model.cond_emb(cond_vec).unsqueeze(1)  # [B,1,Hdim]
            hidden_dim = e_cond.size(-1)
            e_null = model.null_cond_emb.unsqueeze(0).expand(num_samples, 1, hidden_dim)
            if e_null.dtype != e_cond.dtype:
                e_null = e_null.to(dtype=e_cond.dtype)
        else:
            e_cond = None
            e_null = None

        # 4) DDIM loop with classifier-free guidance
        x = torch.randn(num_samples, C, H, W, device=device).to(memory_format=torch.channels_last)
        model.eval()

        for t in sched.timesteps:
            t_vec = torch.full((num_samples,), t, device=device, dtype=torch.long)

            if e_cond is not None and e_null is not None:
                # Double-batch: [uncond; cond]
                x_in = torch.cat([x, x], dim=0)          # [2B,C,H,W]
                t_in = torch.cat([t_vec, t_vec], dim=0)  # [2B]
                emb_in = torch.cat([e_null, e_cond], dim=0)  # [2B,1,Hdim]
                h_in = model.input_proj(x_in)
                out = model.unet(h_in, t_in, encoder_hidden_states=emb_in).sample
                out = model.output_proj(out)

                v_uncond, v_cond = out.chunk(2, dim=0)  # [B,C,H,W] each
                v = v_uncond + guidance_scale * (v_cond - v_uncond)
            else:
                # Fallback: purely conditional without CFG
                v = model(x, t_vec, cond_vec=cond_vec)

            step = sched.step(model_output=v, timestep=t, sample=x)
            x = step.prev_sample
        return x

    def _hist_probs(x: np.ndarray, bins: int, lo: float, hi: float) -> np.ndarray:
        hist, _ = np.histogram(x, bins=bins, range=(lo, hi))
        p = hist.astype(np.float64)
        p += 1e-12
        p /= p.sum()
        return p

    def _js_div(p: np.ndarray, q: np.ndarray) -> float:
        m = 0.5 * (p + q)
        p = np.maximum(p, 1e-12)
        q = np.maximum(q, 1e-12)
        m = np.maximum(m, 1e-12)
        return 0.5 * (np.sum(p * (np.log(p) - np.log(m))) + np.sum(q * (np.log(q) - np.log(m))))

    # Load first N_plot original latents from memmap
    N_plot = 2000
    mm = np.memmap(memmap_path, dtype='float32', mode='r', shape=mm_shape)
    n_use = min(N_plot, mm_shape[0])
    orig_lat = np.asarray(mm[:n_use])  # (n_use, 64, 16, 16)
    del mm

    # Generate same number of samples in normalized space, then de-normalize
    # Use EMA weights for generation/evaluation
    # Choose evaluation weights (EMA or last)
    eval_model = LatentUNET2D(
        input_channels=C,
        output_channels=C,
        base_channels=base_channels,
        sample_size=unet_sample_size,
        cond_dim=cond_dim,
    ).to(device)
    eval_model.load_state_dict(ema.module.state_dict())
    
    gen_norm = _sample_ddim_ar_cfg(
        eval_model,
        disc_encoder,
        ar_prior,
        num_samples=n_use,
        num_steps=50,
        guidance_scale=5.0,
        device=device,
    )
    gen_norm_cpu = gen_norm.detach().to("cpu").float().numpy()
    if args.use_channel_norm:
        mu_np = mu.view(1, C, 1, 1).cpu().numpy()
        sd_np = sd.view(1, C, 1, 1).cpu().numpy()
        gen_lat = gen_norm_cpu * sd_np + mu_np
    else:
        gen_lat = gen_norm_cpu

    # Flatten (active channels only)
    orig_flat = orig_lat.reshape(n_use, -1)
    gen_flat = gen_lat.reshape(n_use, -1)
    D = orig_flat.shape[1]

    rng = np.random.default_rng(42)
    dims = rng.choice(D, size=min(D, 4096), replace=False)

    # JS divergences per selected dim
    js_vals = []
    for d in dims:
        xo = orig_flat[:, d]
        xg = gen_flat[:, d]
        lo = float(min(xo.min(), xg.min()))
        hi = float(max(xo.max(), xg.max()))
        if lo == hi:
            js_vals.append(0.0)
            continue
        p = _hist_probs(xo, bins=50, lo=lo, hi=hi)
        q = _hist_probs(xg, bins=50, lo=lo, hi=hi)
        js_vals.append(_js_div(p, q))
    js_vals = np.array(js_vals, dtype=np.float64)
    worst_js_idx = np.argsort(js_vals)[::-1]

    out_base = os.path.splitext(args.model_out_path)[0]

    # Plot worst 20 dims by JS
    k_plot = min(20, len(dims))
    fig, axes = plt.subplots(4, 5, figsize=(20, 10))
    axes = axes.ravel()
    for i in range(k_plot):
        d = int(dims[worst_js_idx[i]])
        ax = axes[i]
        ax.hist(orig_flat[:, d], bins=50, alpha=0.5, label='orig', density=True)
        ax.hist(gen_flat[:, d], bins=50, alpha=0.5, label='gen', density=True)
        ax.set_title(f'dim {d} | JS={js_vals[worst_js_idx[i]]:.4f}')
        ax.legend()
    plt.tight_layout()
    fig_path_js = f"{out_base}_js_worst.png"
    plt.savefig(fig_path_js, dpi=150)
    plt.close(fig)

    # Plot best 20 dims by JS (closest matches)
    best_js_idx = np.argsort(js_vals)  # ascending
    fig, axes = plt.subplots(4, 5, figsize=(20, 10))
    axes = axes.ravel()
    for i in range(k_plot):
        d = int(dims[best_js_idx[i]])
        ax = axes[i]
        ax.hist(orig_flat[:, d], bins=50, alpha=0.5, label='orig', density=True)
        ax.hist(gen_flat[:, d], bins=50, alpha=0.5, label='gen', density=True)
        ax.set_title(f'dim {d} | JS={js_vals[best_js_idx[i]]:.4f}')
        ax.legend()
    plt.tight_layout()
    fig_path_js_best = f"{out_base}_js_best.png"
    plt.savefig(fig_path_js_best, dpi=150)
    plt.close(fig)

    # Plot a few hardcoded dimensions, overlaying orig vs gen
    hardcoded_dims = [15290, 6128, 12272, 25546130, 14834]
    dims_in_range = [int(d) for d in hardcoded_dims if 0 <= int(d) < D]
    if len(dims_in_range) > 0:
        k_hard = min(20, len(dims_in_range))
        rows = 4
        cols = 5
        fig, axes = plt.subplots(rows, cols, figsize=(20, 10))
        axes = axes.ravel()
        for i in range(k_hard):
            d = dims_in_range[i]
            ax = axes[i]
            ax.hist(orig_flat[:, d], bins=50, alpha=0.5, label='orig', density=True)
            ax.hist(gen_flat[:, d], bins=50, alpha=0.5, label='gen', density=True)
            ax.set_title(f'dim {d}')
            ax.legend()
        # If fewer than rows*cols, hide the remaining axes
        for j in range(k_hard, rows * cols):
            axes[j].axis('off')
        plt.tight_layout()
        fig_path_js_hard = f"{out_base}_js_hardcoded.png"
        plt.savefig(fig_path_js_hard, dpi=150)
        plt.close(fig)
    else:
        fig_path_js_hard = None

    # KL to standard normal for generated dims (parametric)
    mu_g = gen_flat.mean(axis=0)
    sd_g = gen_flat.std(axis=0) + 1e-12
    kl_to_std = 0.5 * (np.log(1.0 / (sd_g ** 2)) + (sd_g ** 2 + mu_g ** 2) - 1.0)
    order_std = np.argsort(kl_to_std)[::-1]

    fig, axes = plt.subplots(4, 5, figsize=(20, 10))
    axes = axes.ravel()
    for i in range(k_plot):
        d = int(order_std[i])
        ax = axes[i]
        data = gen_flat[:, d]
        ax.hist(data, bins=50, alpha=0.6, density=True, label='gen dim')
        x_min, x_max = float(data.min()), float(data.max())
        xs = np.linspace(x_min, x_max, 200)
        ax.plot(xs, _std_normal_pdf(xs), label='N(0,1)')
        ax.set_title(f'dim {d} | KL_to_std={kl_to_std[d]:.4f}')
        ax.legend()
    plt.tight_layout()
    fig_path_std = f"{out_base}_kl_to_std.png"
    plt.savefig(fig_path_std, dpi=150)
    plt.close(fig)

    # Lowest and highest variance dims in generated latents
    var_all = gen_flat.var(axis=0)
    var_sorted = np.argsort(var_all)
    k_each = min(10, len(var_sorted) // 2)
    dims_var = list(var_sorted[:k_each]) + list(var_sorted[-k_each:])
    titles = (["low-var"] * k_each) + (["high-var"] * k_each)
    fig, axes = plt.subplots(4, 5, figsize=(20, 10))
    axes = axes.ravel()
    for i in range(min(20, len(dims_var))):
        d = int(dims_var[i])
        ax = axes[i]
        data = gen_flat[:, d]
        ax.hist(data, bins=50, alpha=0.6, density=True, label='gen dim')
        x_min, x_max = float(data.min()), float(data.max())
        xs = np.linspace(x_min, x_max, 200)
        ax.plot(xs, _std_normal_pdf(xs), label='N(0,1)')
        ax.set_title(f'{titles[i]} dim {d}\nvar={var_all[d]:.5f}')
        ax.legend()
    plt.tight_layout()
    fig_path_var = f"{out_base}_var_extremes.png"
    plt.savefig(fig_path_var, dpi=150)
    plt.close(fig)

    # Decode latents to genotype calls for LD/MAF diagnostics
    print("[DIAGNOSTICS] Decoding to genotype calls for LD/MAF...")
    z_orig_t = torch.from_numpy(orig_lat[:n_use]).to(device=device, dtype=torch.float32)
    z_gen_t  = torch.from_numpy(gen_lat[:n_use]).to(device=device, dtype=torch.float32)
    ae.eval()
    with torch.no_grad():
        logits_o = ae.decode(z_orig_t)  # [B,3,L]
        logits_g = ae.decode(z_gen_t)
        calls_o = logits_o.argmax(dim=1)  # [B,L]
        calls_g = logits_g.argmax(dim=1)
    # MAF scatter
    maf_o = calls_o.float().mean(dim=0).cpu().numpy() / 2.0
    maf_g = calls_g.float().mean(dim=0).cpu().numpy() / 2.0
    plt.figure(figsize=(5,5))
    plt.scatter(maf_o, maf_g, alpha=0.3, s=4)
    lim0, lim1 = 0.0, 0.5
    plt.plot([lim0, lim1], [lim0, lim1], color='red', linestyle='--', alpha=0.7)
    plt.title('MAF: generated vs original')
    plt.xlabel('Original MAF'); plt.ylabel('Generated MAF')
    plt.tight_layout()
    fig_path_maf = f"{out_base}_maf_scatter.png"
    plt.savefig(fig_path_maf, dpi=150)
    plt.close()

    # LD blocks heatmaps (if block definitions available)
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

    def _plot_ld_blocks_heatmaps(block_true_rows: list[np.ndarray],
                                 block_recon_rows: list[np.ndarray],
                                 title: str,
                                 out_path: str) -> None:
        num_blocks = len(block_true_rows)
        if num_blocks == 0:
            return
        fig, axes = plt.subplots(num_blocks, 2, figsize=(6, 2.5 * num_blocks))
        if num_blocks == 1:
            axes = np.array([axes])
        vmin, vmax = -1.0, 1.0
        for idx, (Xt, Xr) in enumerate(zip(block_true_rows, block_recon_rows)):
            xt = Xt.astype(np.float32)
            xr = Xr.astype(np.float32)
            xt = xt - xt.mean(0, keepdims=True)
            xr = xr - xr.mean(0, keepdims=True)
            cov_t = (xt.T @ xt) / max(1, xt.shape[0] - 1)
            cov_r = (xr.T @ xr) / max(1, xr.shape[0] - 1)
            std_t = np.sqrt(np.clip(np.diag(cov_t), 1e-6, None))
            std_r = np.sqrt(np.clip(np.diag(cov_r), 1e-6, None))
            corr_t = cov_t / (std_t[:, None] * std_t[None, :])
            corr_r = cov_r / (std_r[:, None] * std_r[None, :])
            ax_t = axes[idx, 0]; ax_r = axes[idx, 1]
            ax_t.imshow(corr_t, vmin=vmin, vmax=vmax, cmap='coolwarm')
            ax_r.imshow(corr_r, vmin=vmin, vmax=vmax, cmap='coolwarm')
            ax_t.set_title(f'Block {idx + 1} Original')
            ax_r.set_title(f'Block {idx + 1} Generated')
            for ax in (ax_t, ax_r):
                ax.set_xticks([]); ax.set_yticks([])
        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close(fig)

    ld_block_dir = "/n/home03/ahmadazim/WORKING/genGen/UKB6PC/genomic_data//haploblocks/"
    bim_path = '/n/home03/ahmadazim/WORKING/genGen/UKBVQVAE/genomic_data/geneticBinary/ukb_allchr_unrel_britishWhite.bim'
    chr_no = 1 if H == 32 else 22
    import glob as _glob
    det_files = _glob.glob(os.path.join(ld_block_dir, f"*chr{chr_no}_blocks.blocks.det"))
    if det_files and os.path.exists(bim_path):
        det_file = det_files[0]
        blocks = _parse_ld_blocks(det_file, min_snps=20, max_blocks=5)
        try:
            bim_chr = read_bim_file(bim_path, chr_no)
            snp_ids_chr = bim_chr["SNP"].astype(str).tolist()
            id_to_index = {snp: idx for idx, snp in enumerate(snp_ids_chr)}
            block_indices: list[list[int]] = []
            for snp_list in blocks:
                idxs = [id_to_index[s] for s in snp_list if s in id_to_index]
                if len(idxs) >= 2:
                    block_indices.append(idxs)
            max_rows = min(2000, calls_o.size(0))
            block_true_rows = [calls_o[:max_rows, idxs].cpu().numpy() for idxs in block_indices]
            block_recon_rows = [calls_g[:max_rows, idxs].cpu().numpy() for idxs in block_indices]
            ld_out_path = f"{out_base}_ld_blocks.png"
            _plot_ld_blocks_heatmaps(block_true_rows, block_recon_rows, f"Chr{chr_no} LD blocks", ld_out_path)
            print(f"[DIAGNOSTICS] Saved LD blocks heatmaps: {ld_out_path}")
        except Exception as e:
            print(f"[LD] Skipping LD block plots due to error: {e}")
    else:
        print("[LD] LD block det or BIM file not found; skipping LD heatmaps.")

    if fig_path_js_hard is not None:
        print(f"[DIAGNOSTICS] Saved: {fig_path_js}, {fig_path_js_best}, {fig_path_js_hard}, {fig_path_std}, {fig_path_var}, {fig_path_maf}")
    else:
        print(f"[DIAGNOSTICS] Saved: {fig_path_js}, {fig_path_js_best}, {fig_path_std}, {fig_path_var}, {fig_path_maf}")


if __name__ == "__main__":
    main()

 