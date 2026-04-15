from __future__ import annotations

import torch


def _to_binary_predictions(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    probs = torch.sigmoid(logits.float())
    return (probs >= threshold).to(dtype=torch.float32)


def _to_binary_targets(targets: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    # Be permissive: many datasets provide {0,1} floats, but some may provide 0/255 or probabilities.
    return (targets.float() >= threshold).to(dtype=torch.float32)


def compute_binary_dice(logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    if logits.shape != targets.shape:
        raise ValueError(f"logits and targets must have same shape, got {logits.shape} vs {targets.shape}")

    preds = _to_binary_predictions(logits)
    targs = _to_binary_targets(targets)

    reduce_dims = tuple(range(1, preds.dim()))
    intersection = (preds * targs).sum(dim=reduce_dims)
    denom = preds.sum(dim=reduce_dims) + targs.sum(dim=reduce_dims)
    dice = (2.0 * intersection + float(smooth)) / (denom + float(smooth))
    return dice.mean()


def compute_binary_iou(logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    if logits.shape != targets.shape:
        raise ValueError(f"logits and targets must have same shape, got {logits.shape} vs {targets.shape}")

    preds = _to_binary_predictions(logits)
    targs = _to_binary_targets(targets)

    reduce_dims = tuple(range(1, preds.dim()))
    intersection = (preds * targs).sum(dim=reduce_dims)
    union = preds.sum(dim=reduce_dims) + targs.sum(dim=reduce_dims) - intersection
    iou = (intersection + float(smooth)) / (union + float(smooth))
    return iou.mean()

