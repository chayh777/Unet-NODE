from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader


def _get_plotting_libs():
    mpl_config_dir = Path.cwd() / ".matplotlib"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def compute_sample_dice(prediction: np.ndarray, ground_truth: np.ndarray) -> float:
    intersection = np.sum((prediction == 1) & (ground_truth == 1))
    union = np.sum(prediction == 1) + np.sum(ground_truth == 1)
    if union == 0:
        return 1.0
    return 2.0 * intersection / union


def compute_sample_iou(prediction: np.ndarray, ground_truth: np.ndarray) -> float:
    intersection = np.sum((prediction == 1) & (ground_truth == 1))
    union = np.sum((prediction == 1) | (ground_truth == 1))
    if union == 0:
        return 1.0
    return intersection / union


def add_gaussian_noise(images: torch.Tensor, sigma: float) -> torch.Tensor:
    return images + torch.randn_like(images) * sigma


def run_noisy_inference(
    model: torch.nn.Module,
    loader: DataLoader,
    sigma: float,
    device: str = "cpu",
) -> list[dict[str, Any]]:
    device_t = torch.device(device)
    results = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].float()
            if sigma > 0:
                images = add_gaussian_noise(images, sigma)
            images = images.to(device_t)
            logits = model(images).logits
            preds = (torch.sigmoid(logits) > 0.5).squeeze(1).cpu().numpy().astype(np.uint8)
            for i, sample_id in enumerate(batch["sample_id"]):
                results.append({
                    "sample_id": sample_id,
                    "prediction": preds[i],
                })
    return results


def aggregate_metrics(
    results: list[dict[str, Any]],
    ground_truth_by_id: dict[str, np.ndarray],
) -> dict[str, float]:
    dice_scores = []
    iou_scores = []
    for r in results:
        gt = ground_truth_by_id.get(r["sample_id"])
        if gt is None:
            continue
        dice = compute_sample_dice(r["prediction"], gt)
        iou = compute_sample_iou(r["prediction"], gt)
        dice_scores.append(dice)
        iou_scores.append(iou)

    if not dice_scores:
        return {"mean_dice": 0.0, "std_dice": 0.0, "mean_iou": 0.0, "std_iou": 0.0, "num_samples": 0}

    return {
        "mean_dice": float(np.mean(dice_scores)),
        "std_dice": float(np.std(dice_scores)),
        "mean_iou": float(np.mean(iou_scores)),
        "std_iou": float(np.std(iou_scores)),
        "num_samples": len(dice_scores),
    }


def save_robustness_metrics(
    all_metrics: list[dict[str, Any]],
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["group", "sigma", "mean_dice", "std_dice", "mean_iou", "std_iou", "num_samples"]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_metrics:
            writer.writerow(row)
    return output_path


def plot_decay_curve(
    metrics_df,
    metric: str,
    output_path: Path,
    title: str,
    ylabel: str,
) -> None:
    plt = _get_plotting_libs()
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = {"A": "#7a7a7a", "B": "#377eb8", "C": "#e15759"}
    for group in ["A", "B", "C"]:
        group_data = metrics_df[metrics_df["group"] == group].sort_values("sigma")
        if group_data.empty:
            continue
        ax.errorbar(
            group_data["sigma"],
            group_data[f"mean_{metric}"],
            yerr=group_data[f"std_{metric}"],
            marker="o",
            label=f"Group {group}",
            color=colors.get(group, "#4f83cc"),
            capsize=4,
        )

    ax.set_title(title)
    ax.set_xlabel("Noise Level (σ)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(title="Group")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
