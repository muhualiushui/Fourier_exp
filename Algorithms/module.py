import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, List
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from .func import FNOBlockNd, NDAttention


class FNOnd(nn.Module):
    """
    N-dimensional FNO model.
    modes: list specifying the number of Fourier modes per dimension.
    """
    def __init__(self,
                 in_c: int,
                 out_c: int,
                 modes: List[int],
                 width: int,
                 n_blocks: int = 4,
                 loss_fn: Callable = nn.MSELoss(),
                 activation: Callable = nn.GELU()):
        super().__init__()
        ConvNd = getattr(nn, f'Conv{len(modes)}d')
        self.lift = ConvNd(in_c, width, kernel_size=1)
        # Shared blocks for all channels
        self.blocks = nn.ModuleList([
            FNOBlockNd(width, width, modes)
            for _ in range(n_blocks)
        ])
        self.proj = ConvNd(width, out_c, kernel_size=1)
        self.act = activation
        # Loss functions remain the same
        self.loss_fn = loss_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.lift(x)
        x_branch = x0
        for blk in self.blocks:
            x_branch = blk(x_branch)
            x_branch = self.act(x_branch)
        return self.proj(x_branch)

    def cal_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        outputs = self.forward(x)
        return self.loss_fn(outputs, y)
    


class ATTFNOnd(nn.Module):
    """
    N-dimensional ATT + FNO model.
    modes: list specifying the number of Fourier modes per dimension.
    """
    def __init__(self,
                 in_c: int,
                 out_c: int,
                 modes: List[int],
                 width: int,
                 n_blocks: int = 4,
                 loss_fn: Callable = nn.MSELoss(),
                 activation: Callable = nn.GELU()):
        super().__init__()
        ConvNd = getattr(nn, f'Conv{len(modes)}d')
        self.lift = ConvNd(in_c, width, kernel_size=1)
        # Shared blocks for all channels
        self.blocks = nn.ModuleList([
            FNOBlockNd(width, width, modes)
            for _ in range(n_blocks)
        ])
        self.Attention = NDAttention(in_c, num_heads=1, dropout=0.1)
        # send Attention block to a dedicated GPU
        self.attn_device = torch.device('cuda:7')

        self.proj = ConvNd(width, out_c, kernel_size=1)
        self.act = activation
        # Loss functions remain the same
        self.loss_fn = loss_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # move input to attention device, apply attention, then move back
        self.Attention.to(self.attn_device)
        x_device = x.device
        x_att = self.Attention(x.to(self.attn_device)).to(x_device)
        x0 = self.lift(x_att)
        x_branch = x0
        for blk in self.blocks:
            x_branch = blk(x_branch)
            x_branch = self.act(x_branch)
        return self.proj(x_branch)

    def cal_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        outputs = self.forward(x)
        return self.loss_fn(outputs, y)