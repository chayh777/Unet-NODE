from __future__ import annotations

import torch
from torch import nn


class DiceBCELoss(nn.Module):
    """
    Binary segmentation loss: BCEWithLogitsLoss + Dice loss.

    Expects:
      logits: [B, 1, H, W] (or any [B, C, ...] binary mask logits)
      targets: same shape, values in {0,1} (float or int)
    """

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = float(smooth)
        self._bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.shape != targets.shape:
            raise ValueError(f"logits and targets must have same shape, got {logits.shape} vs {targets.shape}")

        logits = logits.float()
        targets = targets.float()

        bce = self._bce(logits, targets)

        probs = torch.sigmoid(logits)
        # Reduce over all non-batch dims and average over batch.
        reduce_dims = tuple(range(1, probs.dim()))
        intersection = (probs * targets).sum(dim=reduce_dims)
        denom = probs.sum(dim=reduce_dims) + targets.sum(dim=reduce_dims)
        dice = (2.0 * intersection + self.smooth) / (denom + self.smooth)
        dice_loss = (1.0 - dice).mean()

        return bce + dice_loss

