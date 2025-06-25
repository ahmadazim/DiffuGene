import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .distribution import DiagonalGaussianDistribution


class JointBlockEmbedder(nn.Module):
    """
    1) pos‐embed each block (B,N,3)→(B,N,3)
    2) pad/truncate N→1024
    3) project 3→4 channels via an MLP
    4) reshape (B,4,32,32) and one down‐conv to (B,32,16,16)
    """
    def __init__(self, block_emb_dim=3, pos_emb_dim=16, grid_size=(32,32)):
        super().__init__()
        H, W = grid_size
        self.H, self.W = H, W
        self.n_cells = H * W
        
        # keep your existing pos-MLP
        self.pos_mlp = nn.Sequential(
            nn.Linear(3, pos_emb_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(pos_emb_dim, block_emb_dim),
        )
        # project block_emb_dim→4 so we can form a 4-channel "image"
        self.cell_proj = nn.Linear(block_emb_dim, 4)
        
        # single conv: (B,4,32,32) → (B,32,16,16)
        self.down = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
        )
    
    def forward(self, block_embs, spans):
        B, N, D = block_embs.shape
        # 1) positional add
        delta = self.pos_mlp(spans)        # (B,N,3)
        x = block_embs + delta             # (B,N,3)

        # 2) pad/truncate to 1024
        cells = self.n_cells
        if N < cells:
            pad = x.new_zeros((B, cells-N, D))
            x = torch.cat([x, pad], dim=1)
        else:
            x = x[:, :cells]  # TODO: JUST A PLACEHOLDER FOR NOW, BUT WE NEVER WANT TO SLICE LIKE THIS!

        # 3) project 3→4 channels
        x = self.cell_proj(x)   # (B,1024,4)

        # 4) reshape into 32×32 grid
        x = x.view(B, cells, 4).permute(0,2,1)        # (B,4,1024)
        x = x.view(B, 4, self.H, self.W)              # (B,4,H,W)

        # 5) one down-conv to 16×16 with 32 channels
        return self.down(x)    # (B,32,16,16)
    

class JointBlockDecoder(nn.Module):
    """
    Mirror of the new JointBlockEmbedder:
      1) z: (B,16,16,16)  → upsample to (B, 4, 32, 32)
      2) reshape → (B,1024,4)
      3) linear 4→3 → (B,1024,3)
      4) unpad → (B, N, 3)
      5) subtract positional delta
    """
    def __init__(self,
                 pos_mlp: nn.Module,
                 grid_size: tuple = (32,32),
                 block_emb_dim: int = 3):
        super().__init__()
        H, W = grid_size
        self.n_cells = H * W           # =1024
        self.pos_mlp = pos_mlp

        # 1) invert the single down‐conv: 32→16 (we split 32→16+16 in VAE)
        #    so input to decoder is 16 channels
        #    we upsample 16×16→32×32 in one step
        self.up = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=16,
                out_channels=4,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
                bias=False
            ),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # 2) project channels 4→3 for the block embeddings
        self.cell_unproj = nn.Linear(4, block_emb_dim)

    def forward(self, z: torch.Tensor, spans: torch.Tensor) -> torch.Tensor:
        """
        z:     (B,16,16,16)
        spans: (B, N, 3)
        returns recon_emb: (B, N, 3)
        """
        B, _, _, _ = z.shape

        # 1) upsample → (B,4,32,32)
        x = self.up(z)

        # 2) flatten to (B,1024,4)
        x = x.view(B, 4, self.n_cells).permute(0, 2, 1)

        # 3) linear 4→3 → (B,1024,3)
        x = self.cell_unproj(x)

        # 4) unpad back to N
        N = spans.shape[1]
        if N < self.n_cells:
            x = x[:, :N, :]  # (B, N, 3)

        # 5) subtract positional delta
        delta = self.pos_mlp(spans)         # (B, N, 3)
        recon = x - delta                   # (B, N, 3)

        return recon


class SNPVAE(nn.Module):
    def __init__(self, grid_size=(32,32), block_emb_dim=3, pos_emb_dim=16, latent_channels=32):
        super().__init__()
        # 1) joint embedder now yields (B,32,16,16)
        self.joint_embedder = JointBlockEmbedder(block_emb_dim, pos_emb_dim, grid_size)

        # 2) VAE head: split 32→(16+16) for mean/logvar
        #    → DiagonalGaussianDistribution will take exactly (B,32,16,16)
        #    and chunk it into two (B,16,16,16)
        
        # 3) single up-conv to get back to 32×32
        self.spatial_decoder = nn.Sequential(
            # 16×16→32×32, 16→4 channels
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2, inplace=True),
            # collapse to 4 channels full-res
            nn.Conv2d(8, 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # 4) reshape back to (B,4,32,32) → (B,1024,4) → (B,1024,4) permute → (B,1024,4)
        #     then subtract delta, project back to 3‐dim, and return.
        self.cell_unproj = nn.Linear(4, block_emb_dim)
        self.pos_mlp = self.joint_embedder.pos_mlp
        
        self.joint_decoder = JointBlockDecoder(self.pos_mlp, grid_size = grid_size, block_emb_dim = block_emb_dim)

    def encode(self, block_embs, spans):
        x = self.joint_embedder(block_embs, spans)     # (B,32,16,16)
        params = x                                      # (B,32,16,16)
        dist = DiagonalGaussianDistribution(params)     # will chunk → two 16‐ch maps
        z = dist.sample()                              # (B,16,16,16)
        return z, dist

    def decode(self, z, spans):
        return self.joint_decoder(z, spans)

    def forward(self, block_embs, spans):
        z, dist = self.encode(block_embs, spans)
        return self.decode(z, spans), dist
