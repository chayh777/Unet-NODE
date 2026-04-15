from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from src.training.losses import DiceBCELoss
from src.training.metrics import compute_binary_dice, compute_binary_iou


@dataclass(frozen=True)
class EpochMetrics:
    epoch: int
    train_loss: float
    val_loss: float
    val_dice: float
    val_iou: float


def save_history(rows: list[EpochMetrics], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["epoch", "train_loss", "val_loss", "val_dice", "val_iou"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    return path


def save_metrics_json(metrics: dict[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    serializable: dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, float) and not math.isfinite(value):
            # JSON has no portable representation for inf/NaN.
            serializable[key] = None
            continue
        if isinstance(value, Path):
            serializable[key] = str(value)
        else:
            serializable[key] = value

    with path.open("w", encoding="utf-8") as handle:
        json.dump(serializable, handle, indent=2)

    return path


def _extract_logits(model_output: Any) -> torch.Tensor:
    if hasattr(model_output, "logits"):
        logits = getattr(model_output, "logits")
        if not isinstance(logits, torch.Tensor):
            raise TypeError("Model output .logits must be a torch.Tensor.")
        return logits
    if isinstance(model_output, torch.Tensor):
        return model_output
    raise TypeError("Model output must be a torch.Tensor or expose a .logits torch.Tensor attribute.")


def _validate_binary_logits(logits: torch.Tensor) -> None:
    """
    This engine is for binary segmentation. It expects either:
      - logits [B, 1, H, W]
      - logits [B, H, W]
    """
    if logits.dim() == 4:
        if logits.shape[1] != 1:
            raise ValueError(
                "Binary engine expects logits with channel dimension == 1 for 4D logits; "
                f"got logits shape {tuple(logits.shape)}."
            )
        return
    if logits.dim() == 3:
        return
    raise ValueError(
        "Binary engine expects logits of shape [B,1,H,W] or [B,H,W]; "
        f"got logits shape {tuple(logits.shape)}."
    )


def _align_binary_targets_or_raise(logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """
    Align common binary mask encodings to match binary logits shape.

    Shared loss/metrics expect logits/targets to have identical shapes.
    """
    _validate_binary_logits(logits)

    if masks.dim() == 2:
        # Single sample mask without a batch dimension.
        masks = masks.unsqueeze(0)

    if logits.dim() == 4:
        # logits: [B,1,H,W]
        if masks.dim() == 3:
            # masks: [B,H,W] -> [B,1,H,W]
            if masks.shape[0] != logits.shape[0] or masks.shape[1:] != logits.shape[2:]:
                raise ValueError(
                    "Cannot align masks to logits: expected masks shape [B,H,W] matching logits [B,1,H,W], "
                    f"got logits={tuple(logits.shape)} masks={tuple(masks.shape)}."
                )
            return masks.unsqueeze(1)
        if masks.dim() == 4:
            if masks.shape != logits.shape:
                raise ValueError(
                    "Masks must match logits shape for binary training. "
                    f"Got logits={tuple(logits.shape)} masks={tuple(masks.shape)}."
                )
            return masks
        raise ValueError(
            "Cannot align masks to 4D logits. "
            f"Expected masks [B,H,W] or [B,1,H,W], got {tuple(masks.shape)}."
        )

    # logits: [B,H,W]
    if masks.dim() == 4 and masks.shape[1] == 1:
        masks = masks.squeeze(1)
    if masks.dim() != 3 or masks.shape != logits.shape:
        raise ValueError(
            "Masks must match logits shape for binary training. "
            f"Got logits={tuple(logits.shape)} masks={tuple(masks.shape)}."
        )
    return masks


def run_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: Any | None = None,
    device: str | torch.device = "cpu",
) -> dict[str, float]:
    criterion = DiceBCELoss()
    training = optimizer is not None
    device_t = torch.device(device)

    if training:
        model.train()
    else:
        model.eval()

    loss_sum = 0.0
    dice_sum = 0.0
    iou_sum = 0.0
    num_samples = 0

    for batch in loader:
        if not isinstance(batch, dict):
            raise TypeError(
                "Batch must be a dict-like mapping with keys 'image' and 'mask'. "
                f"Got batch type {type(batch)!r}."
            )
        if "image" not in batch or "mask" not in batch:
            missing = [k for k in ("image", "mask") if k not in batch]
            raise KeyError(
                "Batch is missing required keys for training engine: "
                + ", ".join(missing)
                + ". Expected keys include at least: image, mask."
            )

        images = batch["image"].float().to(device_t)
        masks = batch["mask"].float().to(device_t)
        if images.dim() < 3:
            raise ValueError(f"Images must include a batch dimension; got shape {tuple(images.shape)}.")
        batch_size = int(images.shape[0])
        if batch_size <= 0:
            continue

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            model_output = model(images)
            logits = _extract_logits(model_output)
            _validate_binary_logits(logits)
            masks = _align_binary_targets_or_raise(logits, masks)

            loss = criterion(logits, masks)

            if training:
                loss.backward()
                optimizer.step()

        with torch.no_grad():
            dice = compute_binary_dice(logits, masks)
            iou = compute_binary_iou(logits, masks)

        # Weight by sample count so smaller last batches don't skew metrics.
        loss_sum += float(loss.item()) * batch_size
        dice_sum += float(dice.item()) * batch_size
        iou_sum += float(iou.item()) * batch_size
        num_samples += batch_size

    denom = max(1, num_samples)
    return {
        "loss": loss_sum / denom,
        "dice": dice_sum / denom,
        "iou": iou_sum / denom,
    }


def fit(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    optimizer: Any,
    epochs: int,
    patience: int,
    output_dir: str | Path,
    device: str | torch.device = "cpu",
) -> Path | None:
    output_dir_p = Path(output_dir)
    output_dir_p.mkdir(parents=True, exist_ok=True)

    device_t = torch.device(device)
    model.to(device_t)

    history: list[EpochMetrics] = []
    best_dice: float | None = None
    best_path = output_dir_p / "best.pt"
    best_saved = False
    stale_epochs = 0

    for epoch in range(1, int(epochs) + 1):
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device_t,
        )
        with torch.no_grad():
            val_metrics = run_epoch(
                model=model,
                loader=val_loader,
                optimizer=None,
                device=device_t,
            )

        history.append(
            EpochMetrics(
                epoch=epoch,
                train_loss=float(train_metrics["loss"]),
                val_loss=float(val_metrics["loss"]),
                val_dice=float(val_metrics["dice"]),
                val_iou=float(val_metrics["iou"]),
            )
        )

        epoch_val_dice = float(val_metrics["dice"])
        improved = math.isfinite(epoch_val_dice) and (best_dice is None or epoch_val_dice > best_dice)
        if improved:
            best_dice = epoch_val_dice
            best_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), best_path)
            best_saved = True
            stale_epochs = 0
        else:
            stale_epochs += 1

        if (not improved) and stale_epochs >= int(patience):
            break

    save_history(history, output_dir_p / "history.csv")
    save_metrics_json(
        {
            "best_val_dice": best_dice if best_saved else None,
            "epochs_ran": len(history),
            "best_checkpoint": str(best_path) if best_saved else None,
        },
        output_dir_p / "metrics.json",
    )

    return best_path if best_saved else None
