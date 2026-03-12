from diffusers import DDPMScheduler
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import random
from abc import abstractmethod


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def conv_nd(dims, *args, **kwargs):
    """Create a 1D, 2D, or 3D convolution module."""
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def linear(*args, **kwargs):
    """Create a linear module."""
    return nn.Linear(*args, **kwargs)


def avg_pool_nd(dims, *args, **kwargs):
    """Create a 1D, 2D, or 3D average pooling module."""
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def zero_module(module):
    """Zero out the parameters of a module and return it."""
    for p in module.parameters():
        p.detach().zero_()
    return module


def normalization(channels):
    """Make a standard normalization layer."""
    return GroupNorm32(32, channels)


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with torch.enable_grad():
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads


class TimestepBlock(nn.Module):
    """Any module where forward() takes timestep embeddings as a second argument."""

    @abstractmethod
    def forward(self, x, emb):
        """Apply the module to `x` given `emb` timestep embeddings."""


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """An upsampling layer with an optional convolution."""

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest")
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """A downsampling layer with an optional convolution."""

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=1
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """Apply the block to a Tensor, conditioned on a timestep embedding."""
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    """An attention block that allows spatial positions to attend to each other."""

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        use_new_attention_order=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        self.attention = QKVAttention(self.num_heads)
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), True)

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


class QKVAttention(nn.Module):
    """A module which performs QKV attention and splits in a different order."""

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)


class SkipResBlock(TimestepBlock):
    """A residual block that handles skip connections in the up path."""

    def __init__(
        self,
        channels,
        skip_channels, 
        emb_channels,
        dropout,
        out_channels=None,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
    ):
        super().__init__()
        self.channels = channels
        self.skip_channels = skip_channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        
        # Input expects concatenated channels (main + skip)
        input_channels = channels + skip_channels

        self.in_layers = nn.Sequential(
            normalization(input_channels),
            nn.SiLU(),
            conv_nd(dims, input_channels, self.out_channels, 3, padding=1),
        )

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        # Skip connection projection for residual
        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """Apply the block to a Tensor, conditioned on a timestep embedding."""
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb):
        # x is already concatenated [main, skip]
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        
        # For residual: use only the main input (before concatenation)
        main_input = x[:, :self.channels]  # Extract main channels
        return self.skip_connection(main_input) + h


class UNet2DModelResult:
    """Container for UNet2D results to match diffusers interface."""
    def __init__(self, sample):
        self.sample = sample


class UNet2DModel(nn.Module):
    """
    Custom UNet2D implementation that matches diffusers UNet2DModel interface.
    """

    def __init__(
        self,
        sample_size,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(128, 256, 384, 512),
        down_block_types=("DownBlock2D", "AttnDownBlock2D", "DownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "UpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
        mid_block_type="UNetMidBlock2DSelfAttn",
        attention_head_dim=64,
        dropout=0.0,
        use_scale_shift_norm=True,
        time_emb_factor=4,
        **kwargs
    ):
        super().__init__()
        
        self.sample_size = sample_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.layers_per_block = layers_per_block
        self.block_out_channels = list(block_out_channels)
        self.attention_head_dim = attention_head_dim
        
        # Time embedding
        time_embed_dim = self.block_out_channels[0] * time_emb_factor
        self.time_embed = nn.Sequential(
            linear(self.block_out_channels[0], time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        # Input projection
        self.conv_in = conv_nd(2, in_channels, self.block_out_channels[0], 3, padding=1)

        # Down blocks
        self.down_blocks = nn.ModuleList([])
        down_block_res_samples = []
        
        output_channel = self.block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = self.block_out_channels[i]
            is_final_block = i == len(self.block_out_channels) - 1
            
            down_block = self._get_down_block(
                down_block_type=down_block_type,
                num_layers=layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                add_downsample=not is_final_block,
                use_scale_shift_norm=use_scale_shift_norm,
                dropout=dropout,
                attention_head_dim=attention_head_dim,
            )
            self.down_blocks.append(down_block)

        # Middle block
        self.mid_block = self._get_mid_block(
            mid_block_type=mid_block_type,
            in_channels=self.block_out_channels[-1],
            temb_channels=time_embed_dim,
            use_scale_shift_norm=use_scale_shift_norm,
            dropout=dropout,
            attention_head_dim=attention_head_dim,
        )

        # Up blocks
        self.up_blocks = nn.ModuleList([])
        reversed_block_out_channels = list(reversed(self.block_out_channels))
        
        for i, up_block_type in enumerate(up_block_types):
            is_final_block = i == len(self.block_out_channels) - 1
            prev_output_channel = reversed_block_out_channels[i]
            output_channel = reversed_block_out_channels[i] if is_final_block else reversed_block_out_channels[i + 1]
            
            # Skip channels come from corresponding down block (in correct order)
            # Up block 0 gets skip from down block 2, up block 1 gets skip from down block 1, etc.
            skip_idx = len(self.block_out_channels) - 2 - i
            skip_channels = self.block_out_channels[skip_idx] if skip_idx >= 0 else 0

            up_block = self._get_up_block(
                up_block_type=up_block_type,
                num_layers=layers_per_block + 1,
                in_channels=prev_output_channel,
                out_channels=output_channel,
                skip_channels=skip_channels,
                temb_channels=time_embed_dim,
                add_upsample=not is_final_block,
                use_scale_shift_norm=use_scale_shift_norm,
                dropout=dropout,
                attention_head_dim=attention_head_dim,
            )
            self.up_blocks.append(up_block)

        # Output projection
        self.conv_norm_out = normalization(self.block_out_channels[0])
        self.conv_act = nn.SiLU()
        self.conv_out = conv_nd(2, self.block_out_channels[0], out_channels, 3, padding=1)

    def _get_down_block(self, down_block_type, num_layers, in_channels, out_channels, 
                       temb_channels, add_downsample, use_scale_shift_norm, dropout, attention_head_dim):
        layers = []
        
        for i in range(num_layers):
            layers.append(
                ResBlock(
                    channels=in_channels if i == 0 else out_channels,
                    emb_channels=temb_channels,
                    dropout=dropout,
                    out_channels=out_channels,
                    use_scale_shift_norm=use_scale_shift_norm,
                    dims=2,
                )
            )
            
            if "Attn" in down_block_type:
                layers.append(
                    AttentionBlock(
                        channels=out_channels,
                        num_head_channels=attention_head_dim,
                    )
                )
        
        if add_downsample:
            layers.append(
                Downsample(
                    channels=out_channels,
                    use_conv=True,
                    dims=2,
                )
            )
        
        return TimestepEmbedSequential(*layers)

    def _get_up_block(self, up_block_type, num_layers, in_channels, out_channels,
                     skip_channels, temb_channels, add_upsample, use_scale_shift_norm,
                     dropout, attention_head_dim):
        layers = []
        
        # Add upsampling at the beginning to match skip resolution
        if add_upsample:
            layers.append(
                Upsample(
                    channels=in_channels,
                    use_conv=True,
                    dims=2,
                )
            )
        
        # Create ResBlocks - first one handles skip connection
        for i in range(num_layers):
            if i == 0:
                # First layer: uses skip connection
                main_channels = in_channels
                layers.append(
                    SkipResBlock(
                        channels=main_channels,
                        skip_channels=skip_channels,
                        emb_channels=temb_channels,
                        dropout=dropout,
                        out_channels=out_channels,
                        use_scale_shift_norm=use_scale_shift_norm,
                        dims=2,
                    )
                )
            else:
                # Subsequent layers: regular ResBlock, no skip
                layers.append(
                    ResBlock(
                        channels=out_channels,
                        emb_channels=temb_channels,
                        dropout=dropout,
                        out_channels=out_channels,
                        use_scale_shift_norm=use_scale_shift_norm,
                        dims=2,
                    )
                )
            
            if "Attn" in up_block_type:
                layers.append(
                    AttentionBlock(
                        channels=out_channels,
                        num_head_channels=attention_head_dim,
                    )
                )
        
        return TimestepEmbedSequential(*layers)

    def _get_mid_block(self, mid_block_type, in_channels, temb_channels, 
                      use_scale_shift_norm, dropout, attention_head_dim):
        return TimestepEmbedSequential(
            ResBlock(
                channels=in_channels,
                emb_channels=temb_channels,
                dropout=dropout,
                use_scale_shift_norm=use_scale_shift_norm,
                dims=2,
            ),
            AttentionBlock(
                channels=in_channels,
                num_head_channels=attention_head_dim,
            ),
            ResBlock(
                channels=in_channels,
                emb_channels=temb_channels,
                dropout=dropout,
                use_scale_shift_norm=use_scale_shift_norm,
                dims=2,
            ),
        )

    def forward(self, sample, timestep):
        """
        Args:
            sample: (B, C, H, W) tensor
            timestep: (B,) tensor
        Returns:
            UNet2DModelResult with .sample attribute
        """
        # Time embedding
        t_emb = timestep_embedding(timestep, self.block_out_channels[0])
        emb = self.time_embed(t_emb)

        # Input projection
        sample = self.conv_in(sample)
        
        # Down blocks - collect features for skip connections  
        down_block_res_samples = []
        # print(f"Initial sample: {sample.shape}")
        
        for i, down_block in enumerate(self.down_blocks):
            # Process all ResBlocks and Attention first
            for layer in down_block:
                if isinstance(layer, ResBlock):
                    sample = layer(sample, emb)
                elif isinstance(layer, AttentionBlock):
                    sample = layer(sample)
            
            # Save feature for skip connection ONLY if there's downsampling (not final block)
            has_downsample = any(isinstance(layer, Downsample) for layer in down_block)
            if has_downsample:
                down_block_res_samples.append(sample)
                # print(f"Down block {i}: saved skip {sample.shape}")
            
            # Then do downsampling if present
            for layer in down_block:
                if isinstance(layer, Downsample):
                    sample = layer(sample)
                    # print(f"Down block {i}: after downsample {sample.shape}")

        # Middle block
        sample = self.mid_block(sample, emb)

        # Up blocks with skip connections (reverse order)
        # print(f"Middle block output: {sample.shape}")
        # print(f"Skip samples available: {[x.shape for x in down_block_res_samples]}")
        
        for i, up_block in enumerate(self.up_blocks):
            # print(f"\nUp block {i} start: {sample.shape}")
            is_final_up_block = i == len(self.up_blocks) - 1
            
            for layer in up_block:
                if isinstance(layer, Upsample):
                    sample = layer(sample)
                    # print(f"Up block {i}: after upsample {sample.shape}")
                elif isinstance(layer, SkipResBlock):
                    # Concatenate skip connection
                    if down_block_res_samples:
                        res_sample = down_block_res_samples.pop()
                        # print(f"Up block {i}: concatenating {sample.shape} + {res_sample.shape}")
                        sample = torch.cat([sample, res_sample], dim=1)
                    sample = layer(sample, emb)
                    # print(f"Up block {i}: after SkipResBlock {sample.shape}")
                elif isinstance(layer, ResBlock):
                    sample = layer(sample, emb)
                    # print(f"Up block {i}: after ResBlock {sample.shape}")
                elif isinstance(layer, AttentionBlock):
                    sample = layer(sample)
                    # print(f"Up block {i}: after Attention {sample.shape}")

        # Output projection
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return UNet2DModelResult(sample)




class SinusoidalEmbeddings(nn.Module):
    def __init__(self, time_steps:int, embed_dim: int):
        super().__init__()
        position = torch.arange(time_steps).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim))
        embeddings = torch.zeros(time_steps, embed_dim, requires_grad=False)
        embeddings[:, 0::2] = torch.sin(position * div)
        embeddings[:, 1::2] = torch.cos(position * div)
        self.register_buffer('embeddings', embeddings)

    def forward(self, x, t):
        return self.embeddings[t]


class LatentUNET2D(nn.Module):
    """
    2D UNet for 16x16x16 latent embeddings using custom UNet2DModel.
    Input/output: (B,16,16,16)
    """
    def __init__(
        self,
        input_channels: int = 16,
        output_channels: int = 16,
        time_steps: int = 1000,
        layers_per_block: int = 2,
    ):
        super().__init__()
        
        # initial channel expansion 16 -> 128
        self.input_proj = nn.Conv2d(input_channels, 128, kernel_size=3, padding=1)

        self.unet = UNet2DModel(
            sample_size=(16, 16),
            in_channels=128,
            out_channels=128,
            layers_per_block=layers_per_block,
            block_out_channels=[128, 256, 384, 512],
            down_block_types=[
                 "DownBlock2D",         # spatial:16×16→8×8,   channels:128→256
                 "AttnDownBlock2D",     # spatial:8×8→4×4,     channels:256→384
                 "DownBlock2D",         # spatial:4×4→2×2,     channels:384→512
                 "AttnDownBlock2D",     # spatial:2×2→1×1,     channels:512→512
             ],
             up_block_types=[
                 "AttnUpBlock2D",       # spatial:1×1→2×2,     channels:512→384
                 "UpBlock2D",           # spatial:2×2→4×4,     channels:384→256
                 "AttnUpBlock2D",       # spatial:4×4→8×8,     channels:256→128
                 "UpBlock2D"            # spatial:8×8→16×16,   channels:128→128
             ],
             mid_block_type="UNetMidBlock2DSelfAttn",
             attention_head_dim=64,
        )
        # final channel contraction 128 -> 16
        self.output_proj = nn.Conv2d(128, output_channels, kernel_size=3, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        x: (B,16,16,16)
        t: (B,)
        """
        # expand channels, run through UNet, then project back
        x = self.input_proj(x)
        x = self.unet(x, t).sample
        x = self.output_proj(x)
        return x


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def noise_pred_loss(eps_pred: torch.Tensor,
                    eps_true: torch.Tensor,
                    t: torch.Tensor,
                    scheduler: DDPMScheduler,
                    simplified_loss: bool = True) -> torch.Tensor:
    # fetch β_t and cumulative ᾱ_t, broadcast to match (B,C,…)
    beta_t    = scheduler.betas[t].view(-1, *([1] * (eps_true.dim()-1)))
    alpha_bar = scheduler.alphas_cumprod[t].view(-1, *([1] * (eps_true.dim()-1)))
    # instantaneous α_t = 1 - β_t, and σ_t^2 = β_t
    alpha_inst = 1 - beta_t
    sigma2_t   = beta_t

    # DDPM loss weight: β_t^2 / (σ_t^2 * α_t * (1 - ᾱ_t))
    if simplified_loss: 
        weight = 1.0
    else:
        weight = beta_t.pow(2) / (sigma2_t * alpha_inst * (1 - alpha_bar))
    return (weight * (eps_pred - eps_true).pow(2)).mean()
