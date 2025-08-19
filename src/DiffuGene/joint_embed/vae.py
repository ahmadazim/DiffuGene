import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .distribution import DiagonalGaussianDistribution

def _gn_groups(ch: int) -> int:
    return min(32, max(1, ch // 8))

class DownResBlock1D(nn.Module):
    """
    Residual downsampling block:
      main: Conv1d(k=3, s=2) → GN → Act → Conv1d(k=3, s=1) → GN
      skip: Conv1d(k=1, s=2)
    Output: Act(main + skip)
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv_ds = nn.Conv1d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False)
        self.gn1     = nn.GroupNorm(num_groups=_gn_groups(out_ch), num_channels=out_ch)
        self.act     = nn.SiLU(inplace=True)
        self.conv    = nn.Conv1d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn2     = nn.GroupNorm(num_groups=_gn_groups(out_ch), num_channels=out_ch)
        self.skip    = nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv_ds(x)
        y = self.gn1(y)
        y = self.act(y)
        y = self.conv(y)
        y = self.gn2(y)
        s = self.skip(x)
        return self.act(y + s)

class UpResBlock1D(nn.Module):
    """
    Residual upsampling block:
      main: ConvT1d(k=3, s=2, p=1, out_pad=1) → GN → Act → Conv1d(k=3, s=1) → GN
      skip: ConvT1d(k=1, s=2, out_pad=1)
    Output: Act(main + skip)
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.deconv   = nn.ConvTranspose1d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.gn1      = nn.GroupNorm(num_groups=_gn_groups(out_ch), num_channels=out_ch)
        self.act      = nn.SiLU(inplace=True)
        self.conv     = nn.Conv1d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn2      = nn.GroupNorm(num_groups=_gn_groups(out_ch), num_channels=out_ch)
        # k=1, s=2, out_pad=1 to exactly double length on skip
        self.skip_up  = nn.ConvTranspose1d(in_ch, out_ch, kernel_size=1, stride=2, padding=0, output_padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.deconv(x)
        y = self.gn1(y)
        y = self.act(y)
        y = self.conv(y)
        y = self.gn2(y)
        s = self.skip_up(x)
        return self.act(y + s)

class JointBlockEmbedder(nn.Module):
    """
    1) content MLP: (B,N,D) → (B,N,E)
    2) FiLM pos‐modulation: (B,N,E) using chromosome embeddings + continuous positions
    3) pad/truncate seq‐len → 4*H*W
    4) permute → (B, E, 4*H*W)
    5) Conv1d(stride=2)  → (B, C/2, 2*H*W)
    6) Conv1d(stride=2)  → (B, C,   H*W)
    7) reshape → (B, C, H, W)
    """
    def __init__(self,
                 n_blocks: int,
                 block_emb_dim: int = 3,
                 pos_emb_dim: int   = 16,
                 grid_size: tuple   = (64,64),
                 latent_channels: int = 128):
        super().__init__()
        H, W = grid_size
        self.H, self.W = H, W
        self.n_blocks = n_blocks

        # 1) compute pad_len and # downsamples
        base = 4 * H * W
        ratio = n_blocks / base
        k = max(0, math.ceil(math.log2(ratio)))   # number of stride-2 steps
        self.k = k
        pad_len = base * (2**k)

        # 2) compute total downsamples: k to reach 4HW from padded n_blocks + 2 to reach HW
        num_downsamples = k + 2
        E = latent_channels // (2**num_downsamples)
        assert E * (2**num_downsamples) == latent_channels, "latent_channels must be divisible by 2**(k+2) for a clean conv path"
        
        # Chromosome embedding dimension (hardcoded)
        chr_emb_dim = 8

        # 3) content MLP: D → E
        self.content_mlp = nn.Sequential(
            nn.Linear(block_emb_dim, pos_emb_dim),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.SiLU(inplace=True),
            nn.Dropout(p=0.05),
            nn.Linear(pos_emb_dim, E),
        )
        self.content_skip = nn.Linear(block_emb_dim, E)
        
        # 4) Chromosome embedding (1-22)
        self.chr_embedding = nn.Embedding(num_embeddings=23, embedding_dim=chr_emb_dim)  # 0-22 (0 unused, 1-22)
        
        # 5) FiLM MLP: chr_emb + continuous positions → 2*E (γ,β)
        in_dim = chr_emb_dim + 2  # chr_embed + start_norm + length_norm
        self.film_mlp = nn.Sequential(
            nn.Linear(in_dim, pos_emb_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=0.05),
            nn.Linear(pos_emb_dim, 2 * E),
        )
        # zero-init FiLM layer (start near identity)
        nn.init.zeros_(self.film_mlp[-1].weight)
        nn.init.zeros_(self.film_mlp[-1].bias)

        # 6) residual downsample blocks: E→...→C, k+2 stages (each doubles channels, stride=2)
        convs = []
        in_ch = E
        for _ in range(num_downsamples):
            out_ch = in_ch * 2
            convs.append(DownResBlock1D(in_ch, out_ch))
            in_ch = out_ch
        assert in_ch == latent_channels, f"Expected final channels {latent_channels}, got {in_ch}"
        self.convs = nn.ModuleList(convs)

    def forward(self, block_embs: torch.Tensor, spans: torch.Tensor) -> torch.Tensor:
        """
        block_embs: (B, N, D)
        spans:      (B, N, 3) where 3 = (chr_idx, start_norm, length_norm)
        returns:    (B, C, H, W)
        """
        B, N, D = block_embs.shape
        H, W    = self.H, self.W
        base = 4 * H * W
        pad_len = base * (2**self.k)

        # 1) content → (B,N,E)
        c = self.content_mlp(block_embs) + self.content_skip(block_embs)

        # 2) Split spans: (chr_idx, start_norm, length_norm)
        chr_idx, start_norm, length_norm = spans.chunk(3, dim=-1)  # each (B, N, 1)
        
        # 3) Get chromosome embedding
        chr_emb = self.chr_embedding(chr_idx.long().squeeze(-1))     # (B, N, chr_emb_dim)
        
        # 4) Concatenate chromosome embedding with continuous positions
        pos_cont = torch.cat([start_norm, length_norm], dim=-1)      # (B, N, 2)
        film_input = torch.cat([chr_emb, pos_cont], dim=-1)          # (B, N, chr_emb_dim+2)
        
        # 5) FiLM: compute (γ,β) from film_input → apply
        # gam_bias = self.film_mlp(film_input)      # (B, N, 2E)
        # gamma, beta = gam_bias.chunk(2, dim=-1)   # each (B, N, E)
        # x = gamma * c + beta                      # (B, N, E)
        delt = self.film_mlp(film_input)           # (B, N, 2E)
        delta_gamma, beta = delt.chunk(2, dim=-1)  # each (B, N, E)
        x = (1.0 + delta_gamma) * c + beta         # near-identity at init

        # 6) pad/truncate sequence-length to pad_len
        if N < pad_len:
            pad_sz = pad_len - N
            pad = x.new_zeros((B, pad_sz, x.size(-1)))
            x = torch.cat([x, pad], dim=1)

        # 7) → (B, E, pad_len)
        x = x.permute(0, 2, 1)

        # 8) → (B, C, H*W)
        for conv in self.convs:
            x = conv(x)

        # 9) reshape to 2D grid
        return x.view(B, -1, H, W)                # (B, C, H, W)



class JointBlockDecoder(nn.Module):
    """
    Mirror of the embedder, but taking in z of shape (B, C2, H, W), where
    C2 = latent_channels // 2 == the number of sampled latent channels.
    
    Uses the same dynamic scaling logic as the encoder.
    """
    def __init__(self,
                 n_blocks: int,
                 grid_size: tuple   = (64,64),
                 block_emb_dim: int = 4,
                 pos_emb_dim: int   = 16,
                 latent_channels: int = 128):
        super().__init__()
        H, W = grid_size
        self.H, self.W = H, W
        self.n_blocks = n_blocks

        # 1) compute pad_len and # downsamples (same as encoder)
        base = 4 * H * W
        ratio = n_blocks / base
        k = max(0, math.ceil(math.log2(ratio)))   # number of stride-2 steps
        self.k = k
        self.pad_len = base * (2**k)

        # 2) embed‐dims (same as encoder)
        num_downsamples = k + 2
        E = latent_channels // (2**num_downsamples)
        assert E * (2**num_downsamples) == latent_channels, "latent_channels must be divisible by 2**(k+2) for a clean conv path"
        C2 = latent_channels // 2   # z's channel count (sampled latents)

        # 3) Build transposed conv layers: reverse of encoder
        deconvs = []
        in_ch = C2
        for i in range(num_downsamples):
            if i < num_downsamples - 1:
                out_ch = in_ch // 2
            else:
                out_ch = in_ch
            deconvs.append(UpResBlock1D(in_ch, out_ch))
            in_ch = out_ch
        assert in_ch == E
        
        self.deconvs = nn.ModuleList(deconvs)

        # 4) project back to original block‐embedding dim
        self.cell_unproj = nn.Linear(E, block_emb_dim)

    def forward(self, z: torch.Tensor, spans: torch.Tensor) -> torch.Tensor:
        B, C2, H, W = z.shape
        N = spans.size(1)

        # fuse H×W → sequence of length H*W
        x = z.view(B, C2, H*W)

        # Apply transposed convolutions to reverse the encoder
        for deconv in self.deconvs:
            x = deconv(x)

        # → (B, pad_len, E)
        x = x.permute(0, 2, 1)

        # pad/truncate back to original N
        if N < self.pad_len:
            x = x[:, :N, :]

        # project E→block_emb_dim
        x = self.cell_unproj(x)

        return x    # (B, N, block_emb_dim)



class SNPVAE(nn.Module):
    def __init__(
        self,
        n_blocks,
        grid_size=(64,64),
        block_emb_dim=3,
        pos_emb_dim=16,
        latent_channels=128
    ):
        super().__init__()
        self.n_blocks = n_blocks

        # 1) Embedder: content‐MLP + FiLM + Conv1d↓ → (B, latent_channels, H, W)
        self.joint_embedder = JointBlockEmbedder(
            n_blocks        = n_blocks,
            block_emb_dim   = block_emb_dim,
            pos_emb_dim     = pos_emb_dim,
            grid_size       = grid_size,
            latent_channels = latent_channels,
        )

        # 2) VAE head: DiagonalGaussianDistribution expects a latent_channels-channel map
        #    and splits it internally into two (latent_channels//2)-ch tensors for mean/logvar.
        
        # 3) Decoder: mirror of embedder
        self.joint_decoder = JointBlockDecoder(
            n_blocks        = n_blocks,
            grid_size       = grid_size,
            block_emb_dim   = block_emb_dim,
            pos_emb_dim     = pos_emb_dim,
            latent_channels = latent_channels,
        )

    def encode(self, block_embs, spans):
        """
        block_embs: (B, N, D)
        spans:      (B, N, 3) where 3 = (chr_idx, start_norm, length_norm)
        returns:
          z:    (B, latent_channels//2, H, W)
          dist: DiagonalGaussianDistribution over z
        """
        x = self.joint_embedder(block_embs, spans)   # (B, latent_channels, H, W)
        dist = DiagonalGaussianDistribution(x)       # splits latent_channels → (latent_channels//2) + (latent_channels//2) internally
        z = dist.sample()                            # (B, latent_channels//2, H, W)
        return z, dist

    def decode(self, z, spans):
        """
        z:     (B, latent_channels//2, H, W)
        spans: (B, N, 3) where 3 = (chr_idx, start_norm, length_norm)
        returns:
          recon_emb: (B, N, D)
        """
        return self.joint_decoder(z, spans)

    def forward(self, block_embs, spans):
        z, dist = self.encode(block_embs, spans)
        recon = self.decode(z, spans)
        return recon, dist