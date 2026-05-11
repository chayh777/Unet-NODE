from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

CLASS_COLORS = {
    0: (0, 0, 0),
    1: (255, 215, 0),
}


def _get_plotting_libs():
    mpl_config_dir = Path.cwd() / ".matplotlib"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def load_model_for_group(checkpoint_path: Path, config: dict[str, Any], group: str):
    from src.models.segmentation_model import build_segmentation_model
    from src.experiments.low_data_runner import resolve_group_adapter

    adapter_type = resolve_group_adapter(group)
    model = build_segmentation_model(
        encoder_name=config["model"]["encoder_name"],
        encoder_weights=None,
        in_channels=int(config["model"]["in_channels"]),
        num_classes=int(config["model"]["num_classes"]),
        adapter_type=adapter_type,
        bottleneck_channels=int(config["model"]["bottleneck_channels"]),
        adapter_hidden_channels=int(config["adapter"]["hidden_channels"]),
        freeze_encoder=False,
        node_steps=int(config["node"]["steps"]),
        node_step_size=float(config["node"]["step_size"]),
        adapter_init="default",
    )
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def run_inference(model, loader: DataLoader, device: str = "cpu") -> list[dict[str, Any]]:
    device_t = torch.device(device)
    results = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].float().to(device_t)
            logits = model(images).logits
            preds = (torch.sigmoid(logits) > 0.5).squeeze(1).cpu().numpy().astype(np.uint8)
            for i, sample_id in enumerate(batch["sample_id"]):
                results.append({
                    "sample_id": sample_id,
                    "prediction": preds[i],
                })
    return results


def compute_sample_dice(predictions: np.ndarray, ground_truth: np.ndarray) -> float:
    intersection = np.sum((predictions == 1) & (ground_truth == 1))
    union = np.sum(predictions == 1) + np.sum(ground_truth == 1)
    if union == 0:
        return 1.0
    return 2.0 * intersection / union


def select_top_variance_samples(
    results_by_group: dict[str, list[dict[str, Any]]],
    ground_truth_by_id: dict[str, np.ndarray],
    top_n: int = 3,
) -> list[str]:
    sample_ids = set()
    for group_results in results_by_group.values():
        for r in group_results:
            sample_ids.add(r["sample_id"])

    variances = []
    for sample_id in sample_ids:
        gt = ground_truth_by_id.get(sample_id)
        if gt is None:
            continue
        dice_scores = []
        for group_results in results_by_group.values():
            for r in group_results:
                if r["sample_id"] == sample_id:
                    dice_scores.append(compute_sample_dice(r["prediction"], gt))
                    break
        if len(dice_scores) == 3:
            mean = sum(dice_scores) / 3
            variance = sum((d - mean) ** 2 for d in dice_scores) / 3
            variances.append((variance, sample_id))

    variances.sort(reverse=True)
    return [sid for _, sid in variances[:top_n]]


def render_grid(
    sample_ids: list[str],
    image_by_id: dict[str, np.ndarray],
    gt_by_id: dict[str, np.ndarray],
    pred_by_group: dict[str, dict[str, np.ndarray]],
    output_path: Path,
    alpha: float = 0.7,
    dpi: int = 150,
) -> None:
    plt = _get_plotting_libs()
    n_cols = 5
    n_rows = len(sample_ids)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    column_labels = ["Image", "Ground Truth", "Group A", "Group B", "Group C"]
    for col_idx, label in enumerate(column_labels):
        axes[0, col_idx].set_title(label, fontsize=14, fontweight="bold")

    for row_idx, sample_id in enumerate(sample_ids):
        img = image_by_id.get(sample_id)
        if img is not None:
            axes[row_idx, 0].imshow(img)

        gt = gt_by_id.get(sample_id)
        if gt is not None:
            colored = _colorize_mask(gt)
            axes[row_idx, 1].imshow(colored)

        groups = ["A", "B", "C"]
        for col_idx, group in enumerate(groups, start=2):
            pred = pred_by_group.get(group, {}).get(sample_id)
            if pred is not None and img is not None:
                overlay = img.copy()
                colored_pred = _colorize_mask(pred)
                mask_bool = pred > 0
                for c in range(3):
                    overlay[:, :, c] = np.where(
                        mask_bool,
                        (1 - alpha) * overlay[:, :, c] + alpha * colored_pred[:, :, c],
                        overlay[:, :, c],
                    )
                axes[row_idx, col_idx].imshow(overlay)
            elif pred is not None:
                colored = _colorize_mask(pred, alpha=alpha)
                axes[row_idx, col_idx].imshow(colored)

        for col_idx in range(n_cols):
            axes[row_idx, col_idx].axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def _colorize_mask(mask: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    h, w = mask.shape
    color_map = np.zeros((h, w, 3), dtype=np.float32)
    for class_id, rgb in CLASS_COLORS.items():
        color_map[mask == class_id] = rgb
    color_map /= 255.0
    if alpha < 1.0:
        color_map = color_map * alpha
    return color_map