import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Callable, List

class DiceCELoss(nn.Module):
    def __init__(self, ce_weight: float = 0.5, smooth: float = 1e-5):
        super().__init__()
        self.ce_weight = ce_weight
        self.smooth = smooth
        self.ce = nn.CrossEntropyLoss()

    def dice_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred: [B, C, H, W] logits; target: [B, C, H, W] one-hot
        probs = torch.softmax(pred, dim=1)
        intersection = torch.sum(probs * target, dim=(2, 3))
        union = torch.sum(probs + target, dim=(2, 3))
        dice_score = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice_score.mean()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred: logits [B, C, H, W]
        target: one-hot [B, C, H, W]
        """
        # CrossEntropyLoss expects class indices
        target_indices = target.argmax(dim=1)  # [B, H, W]
        ce_loss = self.ce(pred, target_indices)
        dice = self.dice_loss(pred, target)
        return dice + self.ce_weight * ce_loss
    

