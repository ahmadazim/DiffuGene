import torch
import torch.nn as nn
from typing import Union

class FiLM2D(nn.Module):
    """
    Shared FiLM-based 1×1 Conv + GroupNorm + modulation for 2D latents.
    """
    def __init__(self, in_channels, film_dim=16, num_groups=8):
        super().__init__()
        # Initialize 1×1 conv as identity
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        with torch.no_grad():
            eye = torch.eye(in_channels).view(in_channels, in_channels, 1, 1)
            self.conv.weight.copy_(eye)
        # GroupNorm
        self.norm = nn.GroupNorm(num_groups=min(num_groups, in_channels), num_channels=in_channels)
        # FiLM embedding for chromosome ID
        self.embed = nn.Embedding(num_embeddings=22, embedding_dim=film_dim)
        self.gamma = nn.Linear(film_dim, in_channels)
        self.beta  = nn.Linear(film_dim, in_channels)
        # Initialize FiLM to identity
        nn.init.zeros_(self.gamma.weight); nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight);  nn.init.zeros_(self.beta.bias)

    def forward(self, x: torch.Tensor, chrom_id: Union[int, torch.Tensor]):
        # x: (B, C, H, W)
        x = self.conv(x)
        x = self.norm(x)
        if isinstance(chrom_id, int):
            chrom_id = torch.full((x.size(0),), chrom_id, dtype=torch.long, device=x.device)
        elif chrom_id.dim() == 0:
            chrom_id = chrom_id.expand(x.size(0))
        chrom_id = chrom_id.to(device=x.device, dtype=torch.long)

        embedding = self.embed(chrom_id)  # (B, film_dim)
        gamma = self.gamma(embedding).view(-1, x.size(1), 1, 1)
        beta  = self.beta(embedding).view(-1, x.size(1), 1, 1)
        return x * (1 + gamma) + beta


class HomogenizedAE(nn.Module):
    def __init__(self, ae_list):
        super().__init__()
        self.aes = nn.ModuleList(ae_list)
        # Assume all AEs have the same latent channel count
        latent_channels = ae_list[0].latent_channels
        self.encode_head = FiLM2D(latent_channels)
        self.decode_head = FiLM2D(latent_channels)

        # Freeze all AE parameters
        for ae in self.aes:
            for p in ae.parameters():
                p.requires_grad = False

    def forward(self, x, chrom_id):
        # x: (B, L) integer genotypes, chrom_id: scalar index for chromosome
        ae = self.aes[chrom_id]
        logits, z = ae(x)
        b = z.size(0)
        chrom_vec = torch.full((b,), int(chrom_id), dtype=torch.long, device=z.device)
        z_hom  = self.encode_head(z, chrom_vec)
        z_dec  = self.decode_head(z_hom, chrom_vec)
        logits_out = ae.decode(z_dec)
        return logits_out, z_hom


def train_shared_heads(models, dataloaders, epochs, device):
    model = HomogenizedAE(models).to(device)
    optimizer = torch.optim.AdamW(
        list(model.encode_head.parameters()) + list(model.decode_head.parameters()),
        lr=2e-3, weight_decay=0.01
    )
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for chrom_id, loader in enumerate(dataloaders):
            for batch in loader:
                batch = batch.to(device)
                logits, z_hom = model(batch, chrom_id)
                loss = F.cross_entropy(logits, batch.long())
                # optional latent MSE loss: encourage minimal distortion in latent space
                loss_lat = F.mse_loss(z_hom, model.aes[chrom_id](batch)[1])
                total = loss + 0.1 * loss_lat
                total.backward()
                optimizer.step()
                optimizer.zero_grad()
                total_loss += total.item()
        print(f"Epoch {epoch+1}: loss={total_loss:.4f}")
    return model