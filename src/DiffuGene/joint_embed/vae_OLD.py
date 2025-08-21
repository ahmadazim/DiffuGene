import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .distribution import DiagonalGaussianDistribution

def _gn_groups(ch: int) -> int:
    return min(32, max(1, ch // 8))

class DownResBlock2D_H(nn.Module):
    """
    Residual downsampling block (height-only):
      main: Conv2d(k=3x3, s=(2,1)) → GN → SiLU → Conv2d(k=3x3, s=1) → GN(=0)
      skip: Conv2d(k=1x1, s=(2,1))
      out:  SiLU(main + skip)
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv_ds = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=(2,1), padding=1, bias=False)
        self.gn1     = nn.GroupNorm(_gn_groups(out_ch), out_ch)
        self.act     = nn.SiLU(inplace=True)
        self.conv    = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn2     = nn.GroupNorm(_gn_groups(out_ch), out_ch)
        self.skip    = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=(2,1), bias=False)
        nn.init.zeros_(self.gn2.weight)
        nn.init.zeros_(self.gn2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.gn1(self.conv_ds(x)))
        y = self.gn2(self.conv(y))
        s = self.skip(x)
        return self.act(y + s)

class UpResBlock2D_H(nn.Module):
    """
    Residual upsampling block (height-only):
      main: ConvT2d(k=3x3, s=(2,1), out_pad=(1,0)) → GN → SiLU → Conv2d(k=3x3, s=1) → GN(=0)
      skip: ConvT2d(k=1x1, s=(2,1), out_pad=(1,0))
      out:  SiLU(main + skip)
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.deconv   = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=3, stride=(2,1), padding=1, output_padding=(1,0), bias=False)
        self.gn1      = nn.GroupNorm(_gn_groups(out_ch), out_ch)
        self.act      = nn.SiLU(inplace=True)
        self.conv     = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn2      = nn.GroupNorm(_gn_groups(out_ch), out_ch)
        self.skip_up  = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=1, stride=(2,1), padding=0, output_padding=(1,0), bias=False)
        nn.init.zeros_(self.gn2.weight)
        nn.init.zeros_(self.gn2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.gn1(self.deconv(x)))
        y = self.gn2(self.conv(y))
        s = self.skip_up(x)
        return self.act(y + s)

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
        ], dim=-1)                    # (B,N,4*n_freq)
        return feat

class JointBlockEmbedder(nn.Module):
    """
    Fold-to-grid encoder:
      1) token MLP: (B,N,D=6) → (B,N,E) with FiLM (chr/start/length)
      2) fold to (B,E,H0,64) where H0 = ceil(N/64); pad ≤63 tokens
      3) run 2D resnet blocks that halve height only (stride (2,1)), doubling channels
      4) resize to (B,*,64,64) (down via adaptive avg pool, or up via interpolate)
      5) 1×1 conv → (B, latent_channels, 64, 64)
    """
    def __init__(self,
                 n_blocks: int,
                 block_emb_dim: int = 6,
                 pos_emb_dim: int   = 16,
                 grid_size: tuple   = (64,64),
                 latent_channels: int = 128, 
                 stem_channels: int = 8):
        super().__init__()
        H, W = grid_size
        self.H_target, self.W_fixed = H, W
        self.n_blocks = n_blocks
        E = stem_channels
        
        # Chromosome embedding dimension (hardcoded)
        chr_emb_dim = 8

        # 1) content MLP: D → E (keep PCs mostly intact, light projection)
        self.content_mlp = nn.Sequential(
            nn.Linear(block_emb_dim, pos_emb_dim),
            nn.SiLU(inplace=True),
            nn.Linear(pos_emb_dim, E),
        )
        self.content_skip = nn.Linear(block_emb_dim, E)
        
        # 2) Chromosome embedding (1-22)
        self.chr_embedding = nn.Embedding(num_embeddings=23, embedding_dim=chr_emb_dim)  # 0-22 (0 unused, 1-22)
        
        # 3) FiLM MLP: chr_emb + Fourier features(start_norm, length_norm) → 2*E (γ,β)
        self.fourier_pos = FourierPos(n_freq=8)
        n_fourier = 4 * self.fourier_pos.freqs.numel()  # 4*n_freq
        in_dim = chr_emb_dim + n_fourier

        self.film_mlp = nn.Sequential(
            nn.Linear(in_dim, pos_emb_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(pos_emb_dim, 2 * E),
        )
        # zero-init FiLM layer (start near identity)
        nn.init.zeros_(self.film_mlp[-1].weight)
        nn.init.zeros_(self.film_mlp[-1].bias)

        # downsampling (height only)
        H0_max = math.ceil(n_blocks / self.W_fixed)
        if H0_max <= self.H_target:
            num_down = 0
        else:
            num_down = math.floor(math.log2(H0_max / self.H_target))
        self.num_down = max(0, num_down)
        self.prepool_H = max(1, math.ceil(H0_max / (2 ** self.num_down)))  # will be in [64,128)
        downs = []
        ch = E
        for _ in range(self.num_down):
            out_ch = ch * 2
            downs.append(DownResBlock2D_H(ch, out_ch))
            ch = out_ch
        self.down2d = nn.ModuleList(downs)
        self.mid_channels = ch
        # 1×1 to latent channels
        self.to_latent = nn.Conv2d(self.mid_channels, latent_channels, kernel_size=1, bias=True) 

    def forward(self, block_embs: torch.Tensor, spans: torch.Tensor) -> torch.Tensor:
        """
        block_embs: (B, N, D)
        spans:      (B, N, 3) where 3 = (chr_idx, start_norm, length_norm)
        returns:    (B, C, H, W)
        """
        B, N, _ = block_embs.shape
        W = self.W_fixed

        # 1) content → (B,N,E)
        c = self.content_mlp(block_embs) + self.content_skip(block_embs)

        # 2) Split spans: (chr_idx, start_norm, length_norm)
        chr_idx, start_norm, length_norm = spans.chunk(3, dim=-1)  # each (B, N, 1)
        
        # 3) Get chromosome embedding
        chr_emb = self.chr_embedding(chr_idx.long().squeeze(-1))     # (B, N, chr_emb_dim)
        
        # 4) Concatenate chromosome embedding with Fourier features of continuous positions
        pos_enc = self.fourier_pos(start_norm, length_norm)          # (B, N, 4*n_freq)
        film_input = torch.cat([chr_emb, pos_enc], dim=-1)           # (B, N, chr_emb_dim+4*n_freq)
        
        # 5) FiLM: compute (γ,β) from film_input 
        delt = self.film_mlp(film_input) 
        delta_gamma, beta = delt.chunk(2, dim=-1)
        delta_gamma = 0.5 * torch.tanh(delta_gamma)
        x = (1.0 + delta_gamma) * c + beta

        # 6) fold to grid (pad ≤63)
        H0 = math.ceil(N / W)
        Np = H0 * W
        if Np > N:
            pad = x.new_zeros(B, Np - N, x.size(-1))
            x = torch.cat([x, pad], dim=1)        # (B, N', E)
        x = x.view(B, H0, W, x.size(-1)).permute(0, 3, 1, 2)  # (B, E, H0, 64)

        # 7) downsample height only
        for block in self.down2d:
            x = block(x)  # height //= 2, channels *= 2

        # 8) get to grid size
        curH, curW = x.shape[2], x.shape[3]
        if curH != self.H_target or curW != self.W_fixed:
            if curH >= self.H_target:
                x = F.adaptive_avg_pool2d(x, (self.H_target, self.W_fixed))
            else:
                x = F.interpolate(x, size=(self.H_target, self.W_fixed), mode="bilinear", align_corners=False)

        # 9) project to latent channels
        x = self.to_latent(x)                     # (B, latent_channels, 64, 64)
        return x



class JointBlockDecoder(nn.Module):
    """
    Mirror of the embedder, but taking in z of shape (B, C2, H, W), where
    C2 = latent_channels // 2 == the number of sampled latent channels.
    """
    def __init__(self,
                 n_blocks: int,
                 grid_size: tuple   = (64,64),
                 block_emb_dim: int = 6,
                 pos_emb_dim: int   = 16,
                 latent_channels: int = 128, 
                 stem_channels: int = 8):
        super().__init__()
        H, W = grid_size
        self.H_target, self.W_fixed = H, W
        self.n_blocks = n_blocks
        E = stem_channels

        # must mirror encoder's planning
        H0_max = math.ceil(n_blocks / self.W_fixed)
        if H0_max <= self.H_target:
            num_down = 0
        else:
            num_down = math.floor(math.log2(H0_max / self.H_target))
        self.num_up    = max(0, num_down)
        self.prepool_H = max(1, math.ceil(H0_max / (2 ** self.num_up)))

        # channel schedule
        mid_channels = E * (2 ** self.num_up)
        C2 = latent_channels // 2
        self.from_latent = nn.Conv2d(C2, mid_channels, kernel_size=1, bias=True)

        ups = []
        ch = mid_channels
        for _ in range(self.num_up):
            out_ch = ch // 2
            ups.append(UpResBlock2D_H(ch, out_ch))
            ch = out_ch
        assert ch == E, f"Decoder channel plan ended at {ch}, expected {E}"
        self.up2d = nn.ModuleList(ups)

        self.cell_unproj = nn.Linear(E, block_emb_dim)

    def forward(self, z: torch.Tensor, spans: torch.Tensor) -> torch.Tensor:
        B, C2, _, _ = z.shape
        N = spans.size(1)
        W = self.W_fixed

        # 1) map to mid channels
        x = self.from_latent(z)

        # 2) up to encoder's pre-pool height if needed
        if self.prepool_H != self.H_target:
            x = F.interpolate(x, size=(self.prepool_H, W), mode="bilinear", align_corners=False)

        # 3) height-only upsampling blocks
        for block in self.up2d:
            x = block(x)

        # 4) resize height to H0
        H0 = math.ceil(N / W)
        curH = x.size(2)
        if curH != H0:
            if curH > H0:
                x = F.adaptive_avg_pool2d(x, (H0, W))
            else:
                x = F.interpolate(x, size=(H0, W), mode="bilinear", align_corners=False)

        # 5) unfold and crop to N, then project to D
        x = x.permute(0, 2, 3, 1).contiguous().view(B, H0 * W, x.size(1))  # (B, N', E)
        x = x[:, :N, :]                                                    # remove pad tail
        x = self.cell_unproj(x)                                            # (B, N, D)
        return x



class SNPVAE(nn.Module):
    def __init__(
        self,
        n_blocks,
        grid_size=(64,64),
        block_emb_dim=3,
        pos_emb_dim=16,
        latent_channels=128,
        stem_channels=8
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
            stem_channels   = stem_channels
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
            stem_channels   = stem_channels
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