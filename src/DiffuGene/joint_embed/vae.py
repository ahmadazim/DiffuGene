import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .distribution import DiagonalGaussianDistribution


class JointBlockEmbedder(nn.Module):
    """
    1) content MLP: (B,N,D) → (B,N,E)
    2) FiLM pos‐modulation: (B,N,E)
    3) pad/truncate seq‐len → 4*H*W
    4) permute → (B, E, 4*H*W)
    5) Conv1d(stride=2)  → (B, C/2, 2*H*W)
    6) Conv1d(stride=2)  → (B, C,   H*W)
    7) reshape → (B, C, H, W)
    """
    def __init__(self,
                 block_emb_dim: int = 3,
                 pos_emb_dim: int   = 16,
                 grid_size: tuple   = (16,16),
                 latent_channels: int = 32):
        super().__init__()
        H, W = grid_size
        self.H, self.W = H, W
        self.n_cells = H * W

        # embed‐dim is a quarter of your final channels
        E = latent_channels // 4

        # 1) content MLP: D → E
        self.content_mlp = nn.Sequential(
            nn.Linear(block_emb_dim, pos_emb_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(pos_emb_dim, E),
        )
        # 2) FiLM MLP: pos(3) → 2*E (γ,β)
        self.film_mlp = nn.Sequential(
            nn.Linear(3, pos_emb_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(pos_emb_dim, 2 * E),
        )

        # 3–6) two strided Conv1d blocks: E→C/2→C
        C2 = latent_channels // 2
        self.conv1 = nn.Sequential(
            nn.Conv1d(E,    C2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(C2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(C2, latent_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(latent_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, block_embs: torch.Tensor, spans: torch.Tensor) -> torch.Tensor:
        """
        block_embs: (B, N, D)
        spans:      (B, N, 3)
        returns:    (B, C, H, W)
        """
        B, N, D = block_embs.shape
        H, W    = self.H, self.W

        # 1) content → (B,N,E)
        c = self.content_mlp(block_embs)

        # 2) FiLM: compute (γ,β) from spans → apply
        gam_bias = self.film_mlp(spans)            # (B,N,2E)
        gamma, beta = gam_bias.chunk(2, dim=-1)   # each (B,N,E)
        x = gamma * c + beta                      # (B,N,E)

        # 3) pad/truncate sequence-length to exactly 4*H*W
        seq_len = 4 * H * W
        if N < seq_len:
            pad = x.new_zeros((B, seq_len - N, x.size(-1)))
            x = torch.cat([x, pad], dim=1)
        else:
            x = x[:, :seq_len]

        # 4) → (B, E, seq_len)
        x = x.permute(0, 2, 1)

        # 5) → (B, C/2, 2*H*W)
        x = self.conv1(x)

        # 6) → (B, C, H*W)
        x = self.conv2(x)

        # 7) reshape to 2D grid
        return x.view(B, -1, H, W)                # (B, C, H, W)



class JointBlockDecoder(nn.Module):
    """
    Mirror of the embedder, but taking in z of shape (B, C2, H, W), where
    C2 = latent_channels // 2 == the number of sampled latent channels.
    """
    def __init__(self,
                 grid_size: tuple   = (16,16),
                 block_emb_dim: int = 3,
                 pos_emb_dim: int   = 16,
                 latent_channels: int = 32):
        super().__init__()
        H, W = grid_size
        self.H, self.W = H, W
        self.n_cells  = H * W

        # match the embedder’s channel math:
        E  = latent_channels // 4   # content‐MLP hidden dim
        C2 = latent_channels // 2   # z’s channel count

        # first upsample: C2→E, length N→2N
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=C2, out_channels=E,
                               kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm1d(E),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # second upsample: E→E, length 2N→4N
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=E, out_channels=E,
                               kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm1d(E),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # project back to original block‐embedding dim
        self.cell_unproj = nn.Linear(E, block_emb_dim)

    def forward(self, z: torch.Tensor, spans: torch.Tensor) -> torch.Tensor:
        B, C2, H, W = z.shape
        N = spans.size(1)

        # fuse H×W → sequence of length H*W
        x = z.view(B, C2, H*W)

        # 1) upsample → (B, E, 2*H*W)
        x = self.deconv1(x)

        # 2) upsample → (B, E, 4*H*W)
        x = self.deconv2(x)

        # 3) → (B, 4*H*W, E)
        x = x.permute(0, 2, 1)

        # 4) pad back to original N
        seq_len = 4 * H * W
        if N < seq_len:
            x = x[:, :N, :]

        # 5) project E→D
        x = self.cell_unproj(x)

        return x    # (B, N, block_emb_dim)



class SNPVAE(nn.Module):
    def __init__(
        self,
        grid_size=(16,16),
        block_emb_dim=3,
        pos_emb_dim=16,
        latent_channels=32
    ):
        super().__init__()

        # 1) Embedder: content‐MLP + FiLM + Conv1d↓ → (B,32,16,16)
        self.joint_embedder = JointBlockEmbedder(
            block_emb_dim   = block_emb_dim,
            pos_emb_dim     = pos_emb_dim,
            grid_size       = grid_size,
            latent_channels = latent_channels,
        )

        # 2) VAE head: split 32→(16+16) for mean/logvar
        #    DiagonalGaussianDistribution expects a 32-channel map
        #    and splits it internally into two 16-ch tensors.
        
        # 3) Decoder: mirror of embedder
        #    Needs the original span→delta-MLP from the embedder
        self.joint_decoder = JointBlockDecoder(
            grid_size       = grid_size,
            block_emb_dim   = block_emb_dim,
            pos_emb_dim     = pos_emb_dim,
            latent_channels = latent_channels,
        )

    def encode(self, block_embs, spans):
        """
        block_embs: (B, N, D)
        spans:      (B, N, 3)
        returns:
          z:    (B,16,16,16)
          dist: DiagonalGaussianDistribution over z
        """
        x = self.joint_embedder(block_embs, spans)   # (B,32,16,16)
        dist = DiagonalGaussianDistribution(x)       # splits 32→16+16 internally
        z = dist.sample()                            # (B,16,16,16)
        return z, dist

    def decode(self, z, spans):
        """
        z:     (B,16,16,16)
        spans: (B, N, 3)
        returns:
          recon_emb: (B, N, D)
        """
        return self.joint_decoder(z, spans)

    def forward(self, block_embs, spans):
        z, dist = self.encode(block_embs, spans)
        recon = self.decode(z, spans)
        return recon, dist