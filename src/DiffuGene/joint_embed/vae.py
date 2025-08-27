import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .distribution import DiagonalGaussianDistribution


class JointBlockEmbedder(nn.Module):
    """
    Strategy:
      Input: (B, N=37185, D=6)
      1) Content MLP: (B,N,6) -> (B,N,E=16)
      2) FiLM pos-modulation using chr embedding + continuous positions
      3) Pre-pool local mixer (depthwise+pointwise Conv1d)
      4) Shrink length via AdaptiveAvgPool1d: N -> target_len = 2^floor(log2(N)) (32768)
      5) Conv1d stride-2 ladder (ds=3): 2^15x2^4 -> 2^14x2^5 -> 2^13x2^6 -> 2^12x2^7
      6) Reshape to (B, 128, 64, 64)
    """
    def __init__(self,
                 n_blocks: int,
                 block_emb_dim: int = 6,
                 pos_emb_dim: int   = 16,
                 grid_size: tuple   = (64, 64),
                 latent_channels: int = 128):
        super().__init__()
        H, W = grid_size
        self.H, self.W = H, W
        self.n_blocks = n_blocks

        # Geometry
        self.hw = H * W                      # 4096 = 2^12
        m = int(math.floor(math.log2(n_blocks)))
        self.target_len = 1 << m             # 32768 = 2^15
        assert self.target_len >= self.hw, "target_len must be >= H*W"
        self.ds = int(math.log2(self.target_len) - math.log2(self.hw))   # 15 - 12 = 3
        assert (1 << self.ds) * self.hw == self.target_len
        # Base channels E so that E * 2^ds = latent_channels
        E = latent_channels // (2 ** self.ds)  # 128 / 8 = 16
        assert E * (2 ** self.ds) == latent_channels
        self.E = E

        # Chromosome embedding
        chr_emb_dim = 8
        self.chr_embedding = nn.Embedding(num_embeddings=23, embedding_dim=chr_emb_dim)  # 0..22; 1..22 used

        # Content MLP: D -> E
        self.content_mlp = nn.Sequential(
            nn.Linear(block_emb_dim, pos_emb_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(pos_emb_dim, E),
        )

        # FiLM MLP: [chr_emb, start_norm, length_norm] -> (gamma, beta) in R^{2E}
        in_dim = chr_emb_dim + 2
        self.film_mlp = nn.Sequential(
            nn.Linear(in_dim, pos_emb_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(pos_emb_dim, 2 * E),
        )

        # Pre-pool local mixer: depthwise + pointwise Conv1d
        self.prepool_mixer = nn.Sequential(
            nn.Conv1d(E, E, kernel_size=5, padding=2, groups=E, bias=False),
            nn.GELU(),
            nn.Conv1d(E, E, kernel_size=1, bias=False),
            nn.GELU(),
        )

        # Strided Conv1d ladder: ds steps, halve length, double channels each time
        convs = []
        in_ch = E
        for _ in range(self.ds):  # 3 steps: 16->32->64->128
            out_ch = in_ch * 2
            convs.append(nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm1d(out_ch),
                nn.LeakyReLU(0.2, inplace=True),
            ))
            in_ch = out_ch
        assert in_ch == latent_channels, f"Expected final channels {latent_channels}, got {in_ch}"
        self.convs = nn.ModuleList(convs)

    def forward(self, block_embs: torch.Tensor, spans: torch.Tensor) -> torch.Tensor:
        """
        block_embs: (B, N, D=6)
        spans:      (B, N, 3) where 3 = (chr_idx, start_norm, length_norm)
        returns:    (B, 128, 64, 64)
        """
        B, N, _ = block_embs.shape
        H, W    = self.H, self.W

        # 1) Content MLP -> (B, N, E)
        c = self.content_mlp(block_embs)

        # 2) FiLM
        chr_idx, start_norm, length_norm = spans.chunk(3, dim=-1)  # each (B, N, 1)
        chr_emb = self.chr_embedding(chr_idx.long().squeeze(-1))   # (B, N, chr_emb_dim)
        film_input = torch.cat([chr_emb, start_norm, length_norm], dim=-1)  # (B, N, chr_emb_dim+2)
        gamma, beta = self.film_mlp(film_input).chunk(2, dim=-1)   # each (B, N, E)
        x = gamma * c + beta                                       # (B, N, E)

        # 3) Pre-pool local mixer
        x = x.permute(0, 2, 1)                                     # (B, E, N)
        x = self.prepool_mixer(x)                                  # (B, E, N)

        # 4) Shrink length: AdaptiveAvgPool1d N -> target_len
        x = F.adaptive_avg_pool1d(x, self.target_len)              # (B, E, 32768)

        # 5) Downsample ladder to (B, 128, 4096)
        for conv in self.convs:
            x = conv(x)

        # 6) Reshape to 2D grid
        return x.view(B, -1, H, W)                                 # (B, 128, 64, 64)


class JointBlockDecoder(nn.Module):
    """
    Mirror of the embedder (channel schedule + length):
      Input z: (B, C2=latent_channels//2, 64, 64)
      1) Flatten -> (B, C2, 4096)
      2) ConvTranspose1d ds steps to reach (B, E, target_len)   [length: 4096->32768]
         Schedule: keep channels once, then halve until reaching E (e.g., 64 -> 64 -> 32 -> 16)
      3) (Minimal deviation from your bullet points) Interpolate length target_len -> N
      4) Linear E(=16) -> D(=6)
    """
    def __init__(self,
                 n_blocks: int,
                 grid_size: tuple   = (64, 64),
                 block_emb_dim: int = 6,
                 pos_emb_dim: int   = 16,
                 latent_channels: int = 128):
        super().__init__()
        H, W = grid_size
        self.H, self.W = H, W
        self.n_blocks = n_blocks

        # Geometry mirror
        self.hw = H * W                            # 4096
        m = int(math.floor(math.log2(n_blocks)))
        self.target_len = 1 << m                   # 32768
        self.ds = int(math.log2(self.target_len) - math.log2(self.hw))  # 3
        assert (1 << self.ds) * self.hw == self.target_len

        self.latent_channels = latent_channels
        self.E = latent_channels // (2 ** self.ds) # 16 when ds=3
        C2 = latent_channels // 2                  # 64

        # Deconv ladder: length x2 each step; channels: keep then halve to E
        deconvs = []
        in_ch = C2
        halving_steps = int(math.log2(max(1, C2 // self.E)))  # e.g., log2(64/16)=2
        keep_steps = self.ds - halving_steps                  # e.g., 3-2 = 1
        for i in range(self.ds):
            out_ch = in_ch if i < keep_steps else max(self.E, in_ch // 2)
            deconvs.append(nn.Sequential(
                nn.ConvTranspose1d(in_ch, out_ch,
                                   kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
                nn.BatchNorm1d(out_ch),
                nn.LeakyReLU(0.2, inplace=True),
            ))
            in_ch = out_ch
        assert in_ch == self.E, f"Decoder end channels {in_ch} != E {self.E}"
        self.deconvs = nn.ModuleList(deconvs)

        # Final projection E -> D (per position)
        self.cell_unproj = nn.Linear(self.E, block_emb_dim)

    def forward(self, z: torch.Tensor, spans: torch.Tensor) -> torch.Tensor:
        """
        z:     (B, latent_channels//2=64, H=64, W=64)
        spans: (B, N, 3)
        returns:
          recon_emb: (B, N, D=6)
        """
        B, C2, H, W = z.shape
        N = spans.size(1)

        # 1) Flatten H*W
        x = z.view(B, C2, H * W)                       # (B, 64, 4096)

        # 2) Deconv ladder to (B, E, target_len)
        for deconv in self.deconvs:
            x = deconv(x)                              # length doubles each step
        # x is (B, E=16, target_len=32768)

        # 3) Interpolate to original N (minimal deviation; required to match input length)
        if N != self.target_len:
            x = F.interpolate(x, size=N, mode='linear', align_corners=False)  # (B, 16, N)

        # 4) Project channels to 6 PCs
        x = x.permute(0, 2, 1)                         # (B, N, 16)
        x = self.cell_unproj(x)                        # (B, N, 6)
        return x


class SNPVAE(nn.Module):
    def __init__(
        self,
        n_blocks,
        grid_size=(64, 64),
        block_emb_dim=6,
        pos_emb_dim=16,
        latent_channels=128
    ):
        super().__init__()
        self.n_blocks = n_blocks

        # Encoder
        self.joint_embedder = JointBlockEmbedder(
            n_blocks        = n_blocks,
            block_emb_dim   = block_emb_dim,
            pos_emb_dim     = pos_emb_dim,
            grid_size       = grid_size,
            latent_channels = latent_channels,
        )

        # Decoder
        self.joint_decoder = JointBlockDecoder(
            n_blocks        = n_blocks,
            grid_size       = grid_size,
            block_emb_dim   = block_emb_dim,
            pos_emb_dim     = pos_emb_dim,
            latent_channels = latent_channels,
        )

    def encode(self, block_embs, spans):
        """
        block_embs: (B, N, D=6)
        spans:      (B, N, 3) where 3 = (chr_idx, start_norm, length_norm)
        returns:
          z:    (B, latent_channels//2=64, H=64, W=64)
          dist: DiagonalGaussianDistribution over z
        """
        x = self.joint_embedder(block_embs, spans)     # (B, 128, 64, 64)
        dist = DiagonalGaussianDistribution(x)         # splits 128 -> (64 mean, 64 logvar)
        z = dist.sample()                              # (B, 64, 64, 64)
        return z, dist

    def decode(self, z, spans):
        """
        z:     (B, 64, 64, 64)
        spans: (B, N, 3)
        returns:
          recon_emb: (B, N, D=6)
        """
        return self.joint_decoder(z, spans)

    def forward(self, block_embs, spans):
        z, dist = self.encode(block_embs, spans)
        recon = self.decode(z, spans)
        return recon, dist