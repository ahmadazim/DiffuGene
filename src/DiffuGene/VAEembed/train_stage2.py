import os
import argparse
import random
import sys
import math
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

# Support both package and script execution
try:
    from DiffuGene.VAEembed.ae import GenotypeAutoencoder, VAEConfig
    from DiffuGene.VAEembed.train import H5ChromosomeDataset
    from DiffuGene.VAEembed.sharedEmbed import (
        FiLM2D,
        HomogenizedAE,
        Stage2PenaltyConfig,
        compute_shared_head_penalties,
    )
except Exception:
    this_dir = os.path.dirname(__file__)
    src_root = os.path.abspath(os.path.join(this_dir, "..", "..", "..", ".."))
    if src_root not in sys.path:
        sys.path.insert(0, src_root)
    from DiffuGene.VAEembed.ae import GenotypeAutoencoder, VAEConfig
    from DiffuGene.VAEembed.train import H5ChromosomeDataset
    from DiffuGene.VAEembed.sharedEmbed import (
        FiLM2D,
        HomogenizedAE,
        Stage2PenaltyConfig,
        compute_shared_head_penalties,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage-2 DDP training for homogenizing latents")
    parser.add_argument('--ae-checkpoints', type=str, required=True, help='Directory containing per-chromosome best.pt files (one per chromosome)')
    parser.add_argument('--h5-dir', type=str, required=True, help='Root directory for H5 training data, with subdir per chromosome')
    parser.add_argument("--chromosomes", nargs='+', type=str, default=["all"], help="Chromosomes to process or 'all'")
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--val-batch-size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-2)
    parser.add_argument('--latent-loss-weight', type=float, default=0.05, help='Weight for latent MSE penalty (homogenized vs original)')
    parser.add_argument('--tv-lambda', type=float, default=2e-2, help='Total-variation weight on shared latent tiles')
    parser.add_argument('--robust-lambda', type=float, default=100.0, help='Weight for latent robustness penalty with additive noise')
    parser.add_argument('--stable-lambda', type=float, default=3e-1, help='Weight for latent stability penalty (noise before shared encode head)')
    parser.add_argument('--latent-noise-std', type=float, default=0.05, help='Noise std for robust latent perturbations')
    parser.add_argument('--embed-noise-std', type=float, default=0.03, help='Noise std applied to pre-shared latents for stability')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--val-h5-dir', type=str, required=True, help='Root directory for H5 validation data')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--val-num-workers', type=int, default=2)
    parser.add_argument('--dist-backend', type=str, default='nccl')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no-amp', dest='amp', action='store_false', help='Disable automatic mixed precision (enabled by default)')
    parser.set_defaults(amp=True)
    return parser.parse_args()


def load_ae_model(checkpoint_path: str, device: torch.device) -> Tuple[GenotypeAutoencoder, Dict[str, Any]]:
    """Load a single chromosome AE from checkpoint and freeze its params."""
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    cfg_dict = ckpt['config']
    cfg = VAEConfig(**cfg_dict)
    model = GenotypeAutoencoder(
        input_length=cfg.input_length,
        K1=cfg.K1,
        K2=cfg.K2,
        C=cfg.C,
        embed_dim=cfg.embed_dim,
    )
    model.load_state_dict(ckpt['model_state'])
    model.to(device)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model, cfg_dict


def build_dataloaders(
    h5_dir: str,
    chromosomes: List[int],
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    device: torch.device,
) -> List[DataLoader]:
    pin_memory = device.type == 'cuda'
    persistent = num_workers > 0
    loaders: List[DataLoader] = []
    for chrom in chromosomes:
        ds = H5ChromosomeDataset(h5_dir, chrom, load_num_batches=None)
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent,
        )
        loaders.append(loader)
    return loaders


class SharedHeadModel(nn.Module):
    """Shared encode/decode heads applied to latent representations."""

    def __init__(self, latent_channels: int) -> None:
        super().__init__()
        self.encode_head = FiLM2D(latent_channels)
        self.decode_head = FiLM2D(latent_channels)

    def forward(self, z: torch.Tensor, chrom_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        chrom_vec = torch.full((z.size(0),), int(chrom_idx), dtype=torch.long, device=z.device)
        z_hom = self.encode_head(z, chrom_vec)
        z_dec = self.decode_head(z_hom, chrom_vec)
        return z_hom, z_dec


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_chromosomes(spec: List[str]) -> List[int]:
    if len(spec) == 1 and spec[0].lower() == "all":
        return list(range(1, 23))
    return [int(x) for x in spec]


def load_local_autoencoders(
    ckpt_dir: str,
    chromosomes: List[int],
    device: torch.device,
) -> Dict[int, GenotypeAutoencoder]:
    ae_map: Dict[int, GenotypeAutoencoder] = {}
    for chrom in chromosomes:
        ckpt_path = os.path.join(ckpt_dir, f'ae_chr{chrom}.pt')
        ae, _ = load_ae_model(ckpt_path, device)
        ae_map[chrom] = ae
    return ae_map


def compute_steps_per_epoch(loaders: List[DataLoader], device: torch.device) -> int:
    local_steps = sum(len(loader) for loader in loaders) if loaders else 0
    steps_tensor = torch.tensor([local_steps], device=device, dtype=torch.long)
    if is_distributed():
        dist.all_reduce(steps_tensor, op=dist.ReduceOp.MAX)
    return int(steps_tensor.item())


def build_epoch_iterator(loaders: List[DataLoader]):
    iterators = [iter(loader) for loader in loaders]
    order: List[int] = []
    cursor = 0

    while True:
        if not loaders:
            yield None, None
            continue
        if cursor >= len(order):
            order = list(range(len(loaders)))
            random.shuffle(order)
            cursor = 0
        local_idx = order[cursor]
        cursor += 1
        try:
            batch = next(iterators[local_idx])
        except StopIteration:
            iterators[local_idx] = iter(loaders[local_idx])
            batch = next(iterators[local_idx])
        yield local_idx, batch


def unwrap_model(model: nn.Module) -> SharedHeadModel:
    if isinstance(model, DDP):
        return model.module  # type: ignore[return-value]
    return model  # type: ignore[return-value]


def extract_head_state(model: nn.Module) -> Dict[str, Dict[str, torch.Tensor]]:
    module = unwrap_model(model)
    encode_state = {k: v.detach().cpu().clone() for k, v in module.encode_head.state_dict().items()}
    decode_state = {k: v.detach().cpu().clone() for k, v in module.decode_head.state_dict().items()}
    return {"encode": encode_state, "decode": decode_state}


def load_existing_shared_heads_state(
    model: SharedHeadModel,
    chromosomes: List[int],
    ckpt_dir: str,
) -> Optional[str]:
    existing_paths = [
        os.path.join(ckpt_dir, f"ae_chr{chrom}_homog.pt")
        for chrom in chromosomes
    ]
    existing_paths = [path for path in existing_paths if os.path.isfile(path)]
    if not existing_paths:
        return None

    existing_paths.sort(key=os.path.getmtime, reverse=True)
    selected_path = existing_paths[0]

    payload = torch.load(selected_path, map_location="cpu")
    state_dict = payload.get("model_state") if isinstance(payload, dict) else None
    if state_dict is None:
        state_dict = payload if isinstance(payload, dict) else {}

    head_state = {
        k: v for k, v in state_dict.items()
        if k.startswith("encode_head.") or k.startswith("decode_head.")
    }

    if not head_state:
        return None

    incompat = model.load_state_dict(head_state, strict=False)
    if getattr(incompat, "missing_keys", []) or getattr(incompat, "unexpected_keys", []):
        missing = ", ".join(incompat.missing_keys)
        unexpected = ", ".join(incompat.unexpected_keys)
        warning = (
            f"[Warning] Issues loading shared head weights from {selected_path}: "
            f"missing_keys=[{missing}] unexpected_keys=[{unexpected}]"
        )
        print(warning)

    return selected_path


def train_one_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    train_loaders: List[DataLoader],
    local_chromosomes: List[int],
    ae_map: Dict[int, GenotypeAutoencoder],
    steps_per_epoch: int,
    device: torch.device,
    args: argparse.Namespace,
    penalty_cfg: Stage2PenaltyConfig,
) -> Tuple[float, float, float, float, float, float, float, int]:
    model.train()
    iterator = build_epoch_iterator(train_loaders)
    optimizer.zero_grad(set_to_none=True)

    total_loss = 0.0
    total_ce = 0.0
    total_latent = 0.0
    total_tv = 0.0
    total_robust = 0.0
    total_stable = 0.0
    total_stable_norm = 0.0
    total_steps = 0

    for _ in range(steps_per_epoch):
        local_idx, batch = next(iterator)
        if local_idx is None or batch is None:
            break
        chrom = local_chromosomes[local_idx]
        chrom_embed_idx = chrom - 1
        ae = ae_map[chrom]

        batch = batch.to(device, non_blocking=True)
        with torch.no_grad():
            _, z_orig = ae(batch)
        z_orig = z_orig.detach()

        with autocast(enabled=args.amp):
            z_hom, z_dec = model(z_orig, chrom_embed_idx)
            logits = ae.decode(z_dec)
            ce_loss = F.cross_entropy(logits, batch.long())
            latent_loss = F.mse_loss(z_hom, z_orig)
            stage2_loss, penalty_metrics = compute_shared_head_penalties(
                ae=ae,
                shared_forward=lambda latent: model(latent, chrom_embed_idx),
                z_input=z_orig,
                z_dec=z_dec,
                x_batch=batch,
                penalty_cfg=penalty_cfg,
            )
            loss = ce_loss + args.latent_loss_weight * latent_loss + stage2_loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        total_loss += float(loss.detach().item())
        total_ce += float(ce_loss.detach().item())
        total_latent += float(latent_loss.detach().item())
        total_tv += float(penalty_metrics["tv"].item())
        total_robust += float(penalty_metrics["robust"].item())
        total_stable += float(penalty_metrics["stable"].item())
        total_stable_norm += float(penalty_metrics["stable_norm"].item())
        total_steps += 1

    return (
        total_loss,
        total_ce,
        total_latent,
        total_tv,
        total_robust,
        total_stable,
        total_stable_norm,
        total_steps,
    )


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loaders: List[DataLoader],
    local_chromosomes: List[int],
    ae_map: Dict[int, GenotypeAutoencoder],
    device: torch.device,
    args: argparse.Namespace,
) -> Tuple[float, float, float]:
    model.eval()
    sum_ce = 0.0
    sum_correct = 0.0
    total_sites = 0.0

    for local_idx, chrom in enumerate(local_chromosomes):
        loader = val_loaders[local_idx]
        ae = ae_map[chrom]
        chrom_embed_idx = chrom - 1

        for batch in loader:
            batch = batch.to(device, non_blocking=True)
            with torch.no_grad():
                _, z_orig = ae(batch)
            with autocast(enabled=args.amp):
                z_hom, z_dec = model(z_orig, chrom_embed_idx)
                logits = ae.decode(z_dec)
                ce = F.cross_entropy(logits, batch.long(), reduction="sum")
                pred = logits.argmax(dim=1)
            sum_ce += float(ce.item())
            sum_correct += float((pred == batch).sum().item())
            total_sites += float(batch.numel())

    return sum_ce, sum_correct, total_sites


def reduce_tensor(tensor: torch.Tensor, op: dist.ReduceOp = dist.ReduceOp.SUM) -> torch.Tensor:
    if is_distributed():
        dist.all_reduce(tensor, op=op)
    return tensor


def save_homogenized_models(
    encode_state: Dict[str, torch.Tensor],
    decode_state: Dict[str, torch.Tensor],
    chromosomes: List[int],
    ckpt_dir: str,
    epoch: int,
    mean_ce: float,
    mean_acc: float,
    args: argparse.Namespace,
    best_epoch: int,
    best_mean_ce: float,
    best_mean_acc: float,
) -> None:
    cpu_device = torch.device('cpu')
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    best_epoch_report = int(best_epoch) if best_epoch != -1 else int(epoch)

    for chrom in chromosomes:
        ckpt_path = os.path.join(ckpt_dir, f'ae_chr{chrom}.pt')
        ae, cfg_dict = load_ae_model(ckpt_path, cpu_device)
        single_model = HomogenizedAE([ae]).to(cpu_device)
        single_model.encode_head.load_state_dict(encode_state)
        single_model.decode_head.load_state_dict(decode_state)
        for p in single_model.parameters():
            p.requires_grad = False

        base, _ = os.path.splitext(ckpt_path)
        out_path = f"{base}_homog.pt"
        payload = {
            'model_state': single_model.state_dict(),
            'meta': {
                'chromosome': int(chrom),
                'source_checkpoint': ckpt_path,
                'config': cfg_dict,
                'stage2': {
                    'current_epoch': int(epoch),
                    'current_mean_val_ce': float(mean_ce),
                    'current_mean_val_acc': float(mean_acc),
                    'best_epoch': best_epoch_report,
                    'best_mean_val_ce': float(best_mean_ce),
                    'best_mean_val_acc': float(best_mean_acc),
                    'train_epochs': int(args.epochs),
                    'batch_size': int(args.batch_size),
                    'val_batch_size': int(args.val_batch_size),
                    'lr': float(args.lr),
                    'weight_decay': float(args.weight_decay),
                    'latent_loss_weight': float(args.latent_loss_weight),
                    'tv_lambda': float(args.tv_lambda),
                    'robust_lambda': float(args.robust_lambda),
                    'stable_lambda': float(args.stable_lambda),
                    'latent_noise_std': float(args.latent_noise_std),
                    'embed_noise_std': float(args.embed_noise_std),
                    'timestamp': timestamp,
                },
            },
        }
        torch.save(payload, out_path)
        print(f"[Stage2][Save] Epoch {epoch} saved homogenized model for chr{chrom} to: {out_path}")


def main():
    args = parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", 0)))
    world_size_env = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", 1)))
    distributed = world_size_env > 1

    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed training requires CUDA-enabled devices.")
        if not dist.is_initialized():
            dist.init_process_group(backend=args.dist_backend)
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
    else:
        world_size = 1
        rank = 0
        if torch.cuda.is_available() and args.device.startswith('cuda'):
            device = torch.device(args.device)
        elif torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

    args.amp = bool(args.amp and device.type == 'cuda')
    torch.backends.cudnn.benchmark = device.type == 'cuda'

    seed = args.seed + rank
    set_random_seed(seed)

    penalty_cfg = Stage2PenaltyConfig(
        tv_lambda=float(args.tv_lambda),
        robust_lambda=float(args.robust_lambda),
        stable_lambda=float(args.stable_lambda),
        latent_noise_std=float(args.latent_noise_std),
        embed_noise_std=float(args.embed_noise_std),
    )

    chromosomes = resolve_chromosomes(args.chromosomes)
    if world_size > len(chromosomes):
        if rank == 0:
            raise ValueError(f"World size {world_size} exceeds number of chromosomes {len(chromosomes)}")
        else:
            raise SystemExit

    local_chromosomes = chromosomes[rank::world_size]
    if not local_chromosomes:
        raise RuntimeError(f"Rank {rank} has no chromosomes assigned. Adjust world size or chromosome list.")

    if rank == 0:
        print(
            f"[Setup] World size={world_size} | Backend={args.dist_backend} | "
            f"AMP={'enabled' if args.amp else 'disabled'}"
        )
        print(
            f"[Setup] Total chromosomes={len(chromosomes)} | Batch size={args.batch_size} | "
            f"Val batch size={args.val_batch_size}"
        )
    print(
        f"[Setup][Rank {rank}] Device={device} | Local rank={local_rank} | "
        f"Chromosomes={local_chromosomes}"
    )

    train_loaders = build_dataloaders(
        args.h5_dir,
        local_chromosomes,
        args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        device=device,
    )
    val_loaders = build_dataloaders(
        args.val_h5_dir,
        local_chromosomes,
        max(args.val_batch_size, args.batch_size),
        shuffle=False,
        num_workers=args.val_num_workers,
        device=device,
    )

    ae_map = load_local_autoencoders(args.ae_checkpoints, local_chromosomes, device)
    sample_latent = next(iter(ae_map.values())).latent_channels
    if is_distributed():
        latent_tensor = torch.tensor([sample_latent], device=device, dtype=torch.long)
        dist.all_reduce(latent_tensor, op=dist.ReduceOp.MAX)
        latent_channels = int(latent_tensor.item())
    else:
        latent_channels = sample_latent

    shared_heads = SharedHeadModel(latent_channels).to(device)
    initial_heads_path = load_existing_shared_heads_state(
        shared_heads,
        chromosomes,
        args.ae_checkpoints,
    )
    optimizer = torch.optim.AdamW(
        shared_heads.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scaler = GradScaler(enabled=args.amp)

    if world_size > 1:
        shared_heads = DDP(
            shared_heads,
            device_ids=[device.index] if device.type == 'cuda' else None,
            output_device=device.index if device.type == 'cuda' else None,
            broadcast_buffers=False,
        )

    steps_per_epoch = compute_steps_per_epoch(train_loaders, device)
    if steps_per_epoch == 0:
        raise RuntimeError("Training dataloaders yielded zero batches. Check dataset configuration.")

    best_epoch = -1
    best_val_mean_ce = math.inf
    best_val_mean_acc = 0.0
    best_heads = extract_head_state(shared_heads)

    main_process = rank == 0

    if main_process:
        if initial_heads_path:
            print(f"[Setup] Loaded shared heads initialization from {initial_heads_path}")
        else:
            print("[Setup] No existing homogenized checkpoints found; initializing shared heads from scratch.")
        if penalty_cfg.is_active():
            print(
                "[Setup] Stage-2 penalties active: "
                f"tv_lambda={penalty_cfg.tv_lambda:.3e}, "
                f"robust_lambda={penalty_cfg.robust_lambda:.3e}, "
                f"stable_lambda={penalty_cfg.stable_lambda:.3e}, "
                f"latent_noise_std={penalty_cfg.latent_noise_std:.3f}, "
                f"embed_noise_std={penalty_cfg.embed_noise_std:.3f}"
            )
        else:
            print("[Setup] Stage-2 penalties disabled (all lambdas set to 0).")

    if main_process:
        print(
            f"[Setup] Training starting with steps_per_epoch={steps_per_epoch} | "
            f"Num epochs={args.epochs}"
        )

    for epoch in range(1, args.epochs + 1):
        # print(f"[Epoch {epoch}] Training on {len(local_chromosomes)} chromosomes")
        (
            train_loss,
            train_ce,
            train_lat,
            train_tv,
            train_robust,
            train_stable,
            train_stable_norm,
            train_steps,
        ) = train_one_epoch(
            shared_heads,
            optimizer,
            scaler,
            train_loaders,
            local_chromosomes,
            ae_map,
            steps_per_epoch,
            device,
            args,
            penalty_cfg,
        )

        train_tensor = torch.tensor(
            [
                train_loss,
                train_ce,
                train_lat,
                train_tv,
                train_robust,
                train_stable,
                train_stable_norm,
                float(train_steps),
            ],
            device=device,
            dtype=torch.float64,
        )
        train_tensor = reduce_tensor(train_tensor, op=dist.ReduceOp.SUM)
        global_steps = max(1.0, train_tensor[7].item())
        mean_train_loss = train_tensor[0].item() / global_steps
        mean_train_ce = train_tensor[1].item() / global_steps
        mean_train_lat = train_tensor[2].item() / global_steps
        mean_train_tv = train_tensor[3].item() / global_steps
        mean_train_robust = train_tensor[4].item() / global_steps
        mean_train_stable = train_tensor[5].item() / global_steps
        mean_train_stable_norm = train_tensor[6].item() / global_steps

        val_ce_sum, val_correct_sum, val_sites_sum = validate(
            shared_heads,
            val_loaders,
            local_chromosomes,
            ae_map,
            device,
            args,
        )
        val_tensor = torch.tensor(
            [val_ce_sum, val_correct_sum, val_sites_sum],
            device=device,
            dtype=torch.float64,
        )
        val_tensor = reduce_tensor(val_tensor, op=dist.ReduceOp.SUM)

        total_sites = max(1.0, val_tensor[2].item())
        mean_val_ce = val_tensor[0].item() / total_sites
        mean_val_acc = val_tensor[1].item() / total_sites

        if main_process:
            log_parts = [
                f"[Epoch {epoch}/{args.epochs}] steps={int(global_steps)}",
                f"train_loss={mean_train_loss:.6f}",
                f"train_ce={mean_train_ce:.6f}",
                f"train_lat={mean_train_lat:.6f}",
            ]
            if penalty_cfg.is_active():
                log_parts.extend(
                    [
                        f"train_tv={mean_train_tv:.6e}",
                        f"train_robust={mean_train_robust:.6e}",
                        f"train_stable={mean_train_stable:.6e}",
                        f"train_stable_norm={mean_train_stable_norm:.6e}",
                    ]
                )
            print(" | ".join(log_parts))
            print(
                f"[Val][Epoch {epoch}] mean_ce={mean_val_ce:.6f} | mean_acc={mean_val_acc:.6f}"
            )

        if mean_val_ce < best_val_mean_ce:
            best_val_mean_ce = mean_val_ce
            best_val_mean_acc = mean_val_acc
            best_epoch = epoch
            best_heads = extract_head_state(shared_heads)
            if main_process:
                print(f"[Val] New best at epoch {best_epoch} (mean_ce={best_val_mean_ce:.6f})")
        elif main_process:
            print(f"[Val] No improvement (best epoch {best_epoch}, mean_ce={best_val_mean_ce:.6f})")

        if main_process:
            encode_state = best_heads["encode"]
            decode_state = best_heads["decode"]
            save_homogenized_models(
                encode_state,
                decode_state,
                chromosomes,
                args.ae_checkpoints,
                epoch,
                mean_val_ce,
                mean_val_acc,
                args,
                best_epoch,
                best_val_mean_ce,
                best_val_mean_acc,
            )

        if is_distributed():
            dist.barrier()

    if main_process:
        print(
            f"[Stage2] Training complete. Best epoch {best_epoch} with "
            f"mean_ce={best_val_mean_ce:.6f}"
        )

    if is_distributed():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()