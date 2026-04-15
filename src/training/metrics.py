from __future__ import annotations

import torch


def _to_binary_predictions(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    probs = torch.sigmoid(logits.float())
    # Important: treat logits==0 (prob==0.5) as negative at the default threshold.
    return (probs > threshold).to(dtype=torch.float32)


def _to_binary_targets(targets: torch.Tensor) -> torch.Tensor:
    """
    Normalize common binary encodings to {0,1} floats.

    Supports:
      - {0,1}
      - {0,255}
      - any 0/positive mask-like encoding (via normalization + thresholding)
    """
    t = targets.float()
    if t.numel() == 0:
        return t
    maxv = t.detach().max()
    if float(maxv) > 1.0:
        t = t / maxv.clamp(min=1.0)
    return (t >= 0.5).to(dtype=torch.float32)


def compute_binary_dice(
    logits: torch.Tensor,
    targets: torch.Tensor,
    smooth: float = 1.0,
    threshold: float = 0.5,
) -> torch.Tensor:
    if logits.shape != targets.shape:
        raise ValueError(f"logits and targets must have same shape, got {logits.shape} vs {targets.shape}")

    preds = _to_binary_predictions(logits, threshold=threshold)
    targs = _to_binary_targets(targets)

    reduce_dims = tuple(range(1, preds.dim()))
    intersection = (preds * targs).sum(dim=reduce_dims)
    denom = preds.sum(dim=reduce_dims) + targs.sum(dim=reduce_dims)
    dice = (2.0 * intersection + float(smooth)) / (denom + float(smooth))
    return dice.mean()


def compute_binary_iou(
    logits: torch.Tensor,
    targets: torch.Tensor,
    smooth: float = 1.0,
    threshold: float = 0.5,
) -> torch.Tensor:
    if logits.shape != targets.shape:
        raise ValueError(f"logits and targets must have same shape, got {logits.shape} vs {targets.shape}")

    preds = _to_binary_predictions(logits, threshold=threshold)
    targs = _to_binary_targets(targets)

    reduce_dims = tuple(range(1, preds.dim()))
    intersection = (preds * targs).sum(dim=reduce_dims)
    union = preds.sum(dim=reduce_dims) + targs.sum(dim=reduce_dims) - intersection
    iou = (intersection + float(smooth)) / (union + float(smooth))
    return iou.mean()
