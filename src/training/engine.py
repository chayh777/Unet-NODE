from __future__ import annotations

import csv
import json
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


def _align_binary_targets(logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """
    Shared loss/metrics expect logits/targets to have identical shapes.

    Our dataset historically returns masks as [B,H,W] (see ISIC2018Dataset),
    while binary segmentation models commonly output [B,1,H,W].
    """
    if logits.shape == masks.shape:
        return masks

    if (
        logits.dim() == 4
        and logits.shape[1] == 1
        and masks.dim() == 3
        and masks.shape[0] == logits.shape[0]
        and masks.shape[1:] == logits.shape[2:]
    ):
        return masks.unsqueeze(1)

    return masks


def run_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer | None = None,
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
    num_batches = 0

    for batch in loader:
        images = batch["image"].float().to(device_t)
        masks = batch["mask"].float().to(device_t)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            model_output = model(images)
            logits = _extract_logits(model_output)
            masks = _align_binary_targets(logits, masks)

            loss = criterion(logits, masks)

            if training:
                loss.backward()
                optimizer.step()

        with torch.no_grad():
            dice = compute_binary_dice(logits, masks)
            iou = compute_binary_iou(logits, masks)

        loss_sum += float(loss.item())
        dice_sum += float(dice.item())
        iou_sum += float(iou.item())
        num_batches += 1

    denom = max(1, num_batches)
    return {
        "loss": loss_sum / denom,
        "dice": dice_sum / denom,
        "iou": iou_sum / denom,
    }


def fit(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    patience: int,
    output_dir: str | Path,
    device: str | torch.device = "cpu",
) -> Path:
    output_dir_p = Path(output_dir)
    output_dir_p.mkdir(parents=True, exist_ok=True)

    device_t = torch.device(device)
    model.to(device_t)

    history: list[EpochMetrics] = []
    best_dice = float("-inf")
    best_path = output_dir_p / "best.pt"
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

        improved = float(val_metrics["dice"]) > best_dice
        if improved:
            best_dice = float(val_metrics["dice"])
            best_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), best_path)
            stale_epochs = 0
        else:
            stale_epochs += 1

        if (not improved) and stale_epochs >= int(patience):
            break

    save_history(history, output_dir_p / "history.csv")
    save_metrics_json(
        {
            "best_val_dice": best_dice,
            "epochs_ran": len(history),
            "best_checkpoint": str(best_path),
        },
        output_dir_p / "metrics.json",
    )

    return best_path
