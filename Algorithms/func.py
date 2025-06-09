import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Callable, List
from functools import reduce
from operator import mul
from torch.nn.functional import scaled_dot_product_attention

def get_timestep_embedding(time_embed_dim: int, t: torch.Tensor) -> torch.Tensor:
    """
    Create sinusoidal timestep embeddings. d d
    """
    half_dim = time_embed_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float32) * -emb)
    emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if time_embed_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1))
    return emb

class ConvNd(nn.Module):
    """
    N-dimensional convolution layer with 1×1 kernel.
    """
    def __init__(self, in_c: int, out_c: int, ndim: int):
        super().__init__()
        ConvNd = getattr(nn, f'Conv{ndim}d')
        self.conv = ConvNd(in_c, out_c, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class FNOBlockNd(nn.Module):
    """
    Single FNO block: inline N‑dimensional spectral conv + 1×1 Conv bypass + activation.
    """
    def __init__(self, in_c: int, out_c: int, modes: List[int]):
        super().__init__()
        self.in_c, self.out_c = in_c, out_c
        self.modes = modes
        self.ndim = len(modes)
        # initialize complex spectral weights
        scale = 1.0 / (in_c * out_c)
        w_shape = (in_c, out_c, *modes)
        init = torch.randn(*w_shape, dtype=torch.cfloat)
        self.weight = nn.Parameter(init * 2 * scale - scale)
        # 1×1 convolution bypass
        self.bypass = ConvNd(in_c, out_c, self.ndim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, *spatial)
        dims = tuple(range(-self.ndim, 0))
        # forward FFT
        x_fft = torch.fft.rfftn(x, dim=dims, norm='ortho')
        # trim to modes
        slices = [slice(None), slice(None)] + [slice(0, m) for m in self.modes]
        x_fft = x_fft[tuple(slices)]
        # einsum: "b i a b..., i o a b... -> b o a b..."
        letters = [chr(ord('k') + i) for i in range(self.ndim)]
        sub_in  = 'bi' + ''.join(letters)
        sub_w   = 'io' + ''.join(letters)
        sub_out = 'bo' + ''.join(letters)
        eq = f"{sub_in}, {sub_w} -> {sub_out}"
        out_fft = torch.einsum(eq, x_fft, self.weight)
        # inverse FFT
        spatial = x.shape[-self.ndim:]
        x_spec = torch.fft.irfftn(out_fft, s=spatial, dim=dims, norm='ortho')
        return x_spec + self.bypass(x)

class NDAttention(nn.Module):
    """
    N-dimensional self-attention block with residual + LayerNorm.
    
    Args:
        in_channels: number of input channels (C)
        num_heads:   number of attention heads
        dropout:     attention & residual dropout
    """
    def __init__(self, in_channels: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.in_channels = in_channels
        self.num_heads   = num_heads

        # multi-head self-attention, batch_first so input is (B, S, C)
        self.attn = nn.MultiheadAttention(
            embed_dim=in_channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.norm    = nn.LayerNorm(in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, d1, d2, ..., dN)
        returns: same shape
        """
        B, C, *spatial = x.shape
        # flatten spatial dims to sequence length S
        S = reduce(mul, spatial, 1)
        # (B, C, S) -> (B, S, C)
        x_flat = x.view(B, C, S).permute(0, 2, 1)

        # self-attention
        # attn_out, _ = self.attn(x_flat, x_flat, x_flat)   # (B, S, C)
        attn_out = scaled_dot_product_attention(
            x_flat, x_flat, x_flat,
            attn_mask=None,
            is_causal=False,
            dropout_p=0.1 if self.training else 0.0,
        )

        # residual + dropout + norm
        out = x_flat + self.dropout(attn_out)             # (B, S, C)
        out = self.norm(out)                              # (B, S, C)

        # back to (B, C, *spatial)
        out = out.permute(0, 2, 1).view(B, C, *spatial)
        return out