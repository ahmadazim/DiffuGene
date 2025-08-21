import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .distribution import DiagonalGaussianDistribution

def _gn_groups(ch: int) -> int:
    return min(32, max(1, ch // 8))

def morton_index_to_xy(index: int, n: int) -> tuple[int, int]:
    """
    Convert a Morton (Z-order) index into (x,y) on a 2^n x 2^n grid.
    This is used to generate a deterministic space-filling ordering.
    """
    x = 0
    y = 0
    for i in range(n):
        # pick bits: bit 0 -> x, bit 1 -> y, bit 2 -> x, etc.
        x |= ((index >> (2 * i)) & 1) << i
        y |= ((index >> (2 * i + 1)) & 1) << i
    return x, y

class FourierPos(nn.Module):
    """
    Fourier features for continuous positions:
      inputs: start_norm, length_norm ∈ [0,1], shapes (B,N,1)
      outputs: concat([sin,cos] for start & length) with n_freq bands → (B,N,4*n_freq)
    """
    def __init__(self, n_freq: int = 8):
        super().__init__()
        freqs = 2.0 ** torch.arange(n_freq, dtype=torch.float32)
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, start_norm: torch.Tensor, length_norm: torch.Tensor) -> torch.Tensor:
        s = start_norm * self.freqs  # (B,N,n_freq)
        l = length_norm * self.freqs
        feat = torch.cat([
            torch.sin(2 * math.pi * s),
            torch.cos(2 * math.pi * s),
            torch.sin(2 * math.pi * l),
            torch.cos(2 * math.pi * l)
        ], dim=-1)  # (B,N,4*n_freq)
        return feat

class JointBlockEmbedder(nn.Module):
    """
    Encoder with deterministic Z-order curve + learned jitter.
      1) token MLP: (B,N,D) → (B,N,E) with FiLM (chr/start/length)
      2) compute deterministic base coords (u0,v0) via Z-order curve
      3) predict jitters Δ(u,v) via jitter_mlp → (u0+Δu,v0+Δv)
      4) bilinearly splat token features into grid → (B,E,H,W)
      5) optional small 2D conv → 1×1 conv → (B,latent_channels,H,W)
    No huge padding; grid_size is arbitrary.
    """
    def __init__(
        self,
        n_blocks: int,
        block_emb_dim: int = 6,
        pos_emb_dim: int   = 16,
        grid_size: tuple   = (64,64),
        latent_channels: int = 128,
        stem_channels: int = 64,
        jitter_mlp: nn.Module | None = None,
    ):
        super().__init__()
        H_target, W_fixed = grid_size
        self.H_target = H_target
        self.W_fixed  = W_fixed
        self.n_blocks = n_blocks
        E = stem_channels

        # content MLP (D → E); keep PCs mostly intact
        self.content_mlp = nn.Sequential(
            nn.Linear(block_emb_dim, pos_emb_dim),
            nn.SiLU(inplace=True),
            nn.Linear(pos_emb_dim, E),
        )
        self.content_skip = nn.Linear(block_emb_dim, E)

        # FiLM MLP (chr_emb + Fourier features → 2E)
        chr_emb_dim = 8
        self.chr_embedding = nn.Embedding(23, chr_emb_dim)  # indices 1..22 used
        self.fourier_pos = FourierPos(n_freq=8)
        n_fourier = 4 * self.fourier_pos.freqs.numel()
        film_in_dim = chr_emb_dim + n_fourier
        self.film_mlp = nn.Sequential(
            nn.Linear(film_in_dim, pos_emb_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(pos_emb_dim, 2 * E),
        )
        nn.init.zeros_(self.film_mlp[-1].weight)
        nn.init.zeros_(self.film_mlp[-1].bias)

        # jitter MLP shared with decoder; if none, create one here.
        if jitter_mlp is None:
            self.jitter_mlp = nn.Sequential(
                nn.Linear(film_in_dim, pos_emb_dim),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(pos_emb_dim, 2),
            )
            nn.init.zeros_(self.jitter_mlp[-1].weight)
            nn.init.zeros_(self.jitter_mlp[-1].bias)
        else:
            self.jitter_mlp = jitter_mlp

        # # jitter scale: move at most one cell in normalized coords
        # self.jitter_scale = 1.0 / max(H_target, W_fixed)
        self.jitter_scale = 1.5 / max(H_target, W_fixed)
        self.register_buffer("jitter_factor", torch.tensor(1.0), persistent=False)

        # Precompute base coords (u0,v0) for each of the n_blocks tokens
        # using Z-order curve on a 2^n × 2^n grid with n = ceil(log2(max(H,W))).
        n = math.ceil(math.log2(max(H_target, W_fixed)))
        two_n = 1 << n
        max_idx = (1 << (2 * n)) - 1
        base = torch.zeros(n_blocks, 2, dtype=torch.float32)
        for i in range(n_blocks):
            # linearize index → 0..max_idx
            p = i / (n_blocks - 1) if n_blocks > 1 else 0.0
            hil_idx = int(p * max_idx + 0.5)
            x0, y0 = morton_index_to_xy(hil_idx, n)
            # normalized coords in [0,1]
            u0 = x0 / (two_n - 1)
            v0 = y0 / (two_n - 1)
            base[i, 0] = u0
            base[i, 1] = v0
        self.register_buffer("base_coords", base, persistent=False)

        self.refine = nn.Sequential(
            nn.Conv2d(2*E + 1, E, kernel_size=3, padding=1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(E, E, kernel_size=3, padding=1, bias=True),
        )
        # small 1×1 expand–mix–project before latent head
        self.pre_latent = nn.Sequential(
            nn.Conv2d(E, 2*E, kernel_size=1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(2*E, E, kernel_size=1, bias=True),
        )
        # 1×1 conv to latent channels
        self.to_latent = nn.Conv2d(E, latent_channels, kernel_size=1, bias=True)

    def forward(self, block_embs: torch.Tensor, spans: torch.Tensor) -> torch.Tensor:
        B, N, _ = block_embs.shape

        # 1) token content: (B,N,E)
        c = self.content_mlp(block_embs) + self.content_skip(block_embs)

        # 2) FiLM: compute scale & shift per token
        chr_idx, start_norm, length_norm = spans.chunk(3, dim=-1)
        chr_emb = self.chr_embedding(chr_idx.long().squeeze(-1))  # (B,N,chr_emb_dim)
        pos_enc = self.fourier_pos(start_norm, length_norm)       # (B,N,4*n_freq)
        film_input = torch.cat([chr_emb, pos_enc], dim=-1)        # (B,N,film_in_dim)
        delt = self.film_mlp(film_input)                          # (B,N,2E)
        gamma, beta = delt.chunk(2, dim=-1)
        gamma = 0.5 * torch.tanh(gamma)  # small scale deviations
        x = (1.0 + gamma) * c + beta     # (B,N,E)

        # 3) base coords + jitter
        device = x.device
        coords = self.base_coords[:N].to(device)                  # (N,2)
        coords = coords.unsqueeze(0).expand(B, N, 2)              # (B,N,2)
        delta = self.jitter_mlp(film_input)                       # (B,N,2)
        delta = torch.tanh(delta) * (self.jitter_scale * self.jitter_factor)
        uv = coords + delta                                       # (B,N,2)
        uv = torch.clamp(uv, 0.0, 1.0)

        # 4) bilinear scatter into grid
        H = self.H_target
        W = self.W_fixed
        du = uv[..., 0] * (W - 1)                                 # (B,N)
        dv = uv[..., 1] * (H - 1)
        j0 = torch.floor(du).long()
        i0 = torch.floor(dv).long()
        j1 = torch.clamp(j0 + 1, max=W - 1)
        i1 = torch.clamp(i0 + 1, max=H - 1)
        w00 = (i1.float() - dv) * (j1.float() - du)               # (B,N)
        w10 = (i1.float() - dv) * (du - j0.float())
        w01 = (dv - i0.float()) * (j1.float() - du)
        w11 = (dv - i0.float()) * (du - j0.float())

        f_i = x.permute(0, 2, 1).contiguous()                     # (B,E,N)
        # Flatten grid for scatter: (H*W)
        F_flat = torch.zeros(B, f_i.shape[1], H * W, device=device)
        M_flat = torch.zeros(B, 1, H * W, device=device)
        # compute linear indices
        idx00 = (i0 * W + j0)                                     # (B,N)
        idx10 = (i0 * W + j1)
        idx01 = (i1 * W + j0)
        idx11 = (i1 * W + j1)
        for weight, idx in ((w00, idx00), (w10, idx10), (w01, idx01), (w11, idx11)):
            w_exp  = weight.unsqueeze(1)                          # (B,1,N)
            src    = f_i * w_exp                                  # (B,E,N)
            F_flat.scatter_add_(2, idx.unsqueeze(1).expand_as(src), src)
            M_flat.scatter_add_(2, idx.unsqueeze(1).expand(B, 1, N), weight.unsqueeze(1))
        # reshape to (B,E,H,W)
        F_grid = F_flat.view(B, f_i.shape[1], H, W)
        M_grid = M_flat.view(B, 1, H, W)
        
        # build inputs for refiner: mean, sum, and mass
        mu = F_grid / (M_grid + 1e-8)
        x_in = torch.cat([mu, F_grid, M_grid], dim=1)             # (B, 2E+1, H, W)
        x_grid = self.refine(x_in)                                # (B,E,H,W)
        
        # extra 1×1 mixing prior to latent projection
        x_grid = self.pre_latent(x_grid)

        # 5) project to latent channels
        x_latent = self.to_latent(x_grid)                        # (B,C,H,W)
        return x_latent

class JointBlockDecoder(nn.Module):
    """
    Decoder using the same Z-order coords and jitter for bilinear readout.
      1) convert z (B,C2,H,W) → mid channels (E) via 1×1
      2) compute coords (u0+Δu, v0+Δv)
      3) bilinearly read from grid → (B,N,E)
      4) MLP E→block_emb_dim to reconstruct original 6-D block PCs.
    """
    def __init__(
        self,
        n_blocks: int,
        grid_size: tuple   = (64,64),
        block_emb_dim: int = 6,
        pos_emb_dim: int   = 16,
        latent_channels: int = 128,
        stem_channels: int = 64,
        jitter_mlp: nn.Module | None = None,
    ):
        super().__init__()
        H_target, W_fixed = grid_size
        self.H_target = H_target
        self.W_fixed  = W_fixed
        self.n_blocks = n_blocks
        E = stem_channels
        C2 = latent_channels // 2  # sampled latent channels

        # 1×1 conv to reduce latent_channels/2 → E
        self.from_latent = nn.Conv2d(C2, E, kernel_size=1, bias=True)

        # reuse jitter_mlp or create if none
        if jitter_mlp is None:
            # same dimensions as encoder
            chr_emb_dim = 8
            n_freq = 8
            n_fourier = 4 * n_freq
            film_in_dim = chr_emb_dim + n_fourier
            self.jitter_mlp = nn.Sequential(
                nn.Linear(film_in_dim, pos_emb_dim),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(pos_emb_dim, 2),
            )
            nn.init.zeros_(self.jitter_mlp[-1].weight)
            nn.init.zeros_(self.jitter_mlp[-1].bias)
        else:
            self.jitter_mlp = jitter_mlp

        # self.jitter_scale = 1.0 / max(H_target, W_fixed)
        self.jitter_scale = 1.5 / max(H_target, W_fixed)
        self.register_buffer("jitter_factor", torch.tensor(1.0), persistent=False)

        # Precompute base coords as in encoder
        n = math.ceil(math.log2(max(H_target, W_fixed)))
        two_n = 1 << n
        max_idx = (1 << (2 * n)) - 1
        base = torch.zeros(n_blocks, 2, dtype=torch.float32)
        for i in range(n_blocks):
            p = i / (n_blocks - 1) if n_blocks > 1 else 0.0
            hil_idx = int(p * max_idx + 0.5)
            x0, y0 = morton_index_to_xy(hil_idx, n)
            u0 = x0 / (two_n - 1)
            v0 = y0 / (two_n - 1)
            base[i, 0] = u0
            base[i, 1] = v0
        self.register_buffer("base_coords", base, persistent=False)

        # Reuse Fourier and chr embedding for jitter MLP
        self.chr_embedding = nn.Embedding(23, 8)  # match encoder
        self.fourier_pos  = FourierPos(n_freq=8)

        # final linear to original PC dimension
        self.cell_unproj = nn.Linear(E, block_emb_dim)

    def forward(self, z: torch.Tensor, spans: torch.Tensor) -> torch.Tensor:
        """
        z:     (B, latent_channels//2, H, W)
        spans: (B, N, 3) with (chr_idx, start_norm, length_norm)
        returns: (B,N,block_emb_dim)
        """
        B, C2, H, W = z.shape
        N = spans.size(1)

        # 1) convert latent z to E channels
        x = self.from_latent(z)                              # (B,E,H,W)

        # 2) coords + jitter
        coords = self.base_coords[:N].to(z.device).unsqueeze(0).expand(B, N, 2)
        chr_idx, start_norm, length_norm = spans.chunk(3, dim=-1)
        chr_emb = self.chr_embedding(chr_idx.long().squeeze(-1))
        pos_enc = self.fourier_pos(start_norm, length_norm)
        film_input = torch.cat([chr_emb, pos_enc], dim=-1)
        delta = self.jitter_mlp(film_input)
        delta = torch.tanh(delta) * (self.jitter_scale * self.jitter_factor)
        uv = coords + delta
        uv = torch.clamp(uv, 0.0, 1.0)

        du = uv[..., 0] * (self.W_fixed - 1)
        dv = uv[..., 1] * (self.H_target - 1)
        j0 = torch.floor(du).long()
        i0 = torch.floor(dv).long()
        j1 = torch.clamp(j0 + 1, max=self.W_fixed - 1)
        i1 = torch.clamp(i0 + 1, max=self.H_target - 1)
        w00 = (i1.float() - dv) * (j1.float() - du)
        w10 = (i1.float() - dv) * (du - j0.float())
        w01 = (dv - i0.float()) * (j1.float() - du)
        w11 = (dv - i0.float()) * (du - j0.float())

        # flatten grid for gather
        F_flat = x.view(B, x.shape[1], -1)                   # (B,E,H*W)
        g = torch.zeros(B, x.shape[1], N, device=z.device)   # (B,E,N)
        # compute linear indices
        idx00 = (i0 * self.W_fixed + j0)
        idx10 = (i0 * self.W_fixed + j1)
        idx01 = (i1 * self.W_fixed + j0)
        idx11 = (i1 * self.W_fixed + j1)
        for weight, idx in ((w00, idx00), (w10, idx10), (w01, idx01), (w11, idx11)):
            w_exp = weight.unsqueeze(1)                     # (B,1,N)
            gathered = F_flat.gather(2, idx.unsqueeze(1).expand(-1, x.shape[1], -1))
            g += gathered * w_exp                           # accumulate weighted features

        g = g.permute(0, 2, 1).contiguous()                # (B,N,E)

        # 3) project to original block dimension (6 PCs)
        recon = self.cell_unproj(g)                         # (B,N,block_emb_dim)
        return recon

class SNPVAE(nn.Module):
    """
    VAE wrapper using Z-order + jitter encoder/decoder.
    """
    def __init__(
        self,
        n_blocks: int,
        grid_size: tuple = (64, 64),
        block_emb_dim: int = 6,
        pos_emb_dim: int = 16,
        latent_channels: int = 128,
        stem_channels: int = 64,
    ):
        super().__init__()
        # define a shared jitter MLP for coords
        chr_emb_dim = 8
        n_freq = 8
        n_fourier = 4 * n_freq
        film_in_dim = chr_emb_dim + n_fourier
        jitter_mlp = nn.Sequential(
            nn.Linear(film_in_dim, pos_emb_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(pos_emb_dim, 2),
        )
        nn.init.zeros_(jitter_mlp[-1].weight)
        nn.init.zeros_(jitter_mlp[-1].bias)

        self.joint_embedder = JointBlockEmbedder(
            n_blocks        = n_blocks,
            block_emb_dim   = block_emb_dim,
            pos_emb_dim     = pos_emb_dim,
            grid_size       = grid_size,
            latent_channels = latent_channels,
            stem_channels   = stem_channels,
            jitter_mlp      = jitter_mlp,
        )
        self.joint_decoder = JointBlockDecoder(
            n_blocks        = n_blocks,
            grid_size       = grid_size,
            block_emb_dim   = block_emb_dim,
            pos_emb_dim     = pos_emb_dim,
            latent_channels = latent_channels,
            stem_channels   = stem_channels,
            jitter_mlp      = jitter_mlp,
        )

    def encode(self, block_embs, spans):
        x = self.joint_embedder(block_embs, spans)
        dist = DiagonalGaussianDistribution(x)
        z = dist.sample()  # (B,C2,H,W)
        return z, dist

    def decode(self, z, spans):
        return self.joint_decoder(z, spans)

    def forward(self, block_embs, spans):
        z, dist = self.encode(block_embs, spans)
        recon = self.decode(z, spans)
        return recon, dist