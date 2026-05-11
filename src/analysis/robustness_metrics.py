from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
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
            logits = model(images)
            if hasattr(logits, "logits"):
                logits = logits.logits
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


def run_robustness_experiment(
    config: dict[str, Any],
    artifacts_dir: Path,
    groups: list[str] = ["A", "B", "C"],
    noise_levels: list[float] = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3],
) -> Path:
    from src.data.isic2018 import ISIC2018Dataset
    from src.models.segmentation_model import build_segmentation_model
    from src.experiments.low_data_runner import resolve_group_adapter

    device = "cuda" if torch.cuda.is_available() else "cpu"

    val_dataset = ISIC2018Dataset(
        images_dir=config["paths"]["val_images_dir"],
        masks_dir=config["paths"]["val_masks_dir"],
        image_size=config["data"]["image_size"],
        class_values={"background": 0, "lesion": 1},
        sample_ids=None,
    )
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    ground_truth_by_id = {}
    for idx in range(len(val_dataset)):
        item = val_dataset[idx]
        ground_truth_by_id[item["sample_id"]] = item["mask"].numpy().astype(np.uint8)

    all_metrics = []
    for group in groups:
        checkpoint_path = artifacts_dir / f"group_{group.lower()}" / "best.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

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
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state)
        model.eval()
        model.to(device)

        for sigma in noise_levels:
            results = run_noisy_inference(model, val_loader, sigma, device=device)
            agg = aggregate_metrics(results, ground_truth_by_id)
            all_metrics.append({
                "group": group,
                "sigma": sigma,
                **agg,
            })

    output_dir = artifacts_dir / "robustness"
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_csv = output_dir / "robustness_metrics.csv"
    save_robustness_metrics(all_metrics, metrics_csv)
    metrics_df = pd.read_csv(metrics_csv)

    plot_decay_curve(
        metrics_df,
        metric="dice",
        output_path=output_dir / "dice_decay_curve.png",
        title="DICE vs Noise Level",
        ylabel="Dice",
    )
    plot_decay_curve(
        metrics_df,
        metric="iou",
        output_path=output_dir / "iou_decay_curve.png",
        title="IoU vs Noise Level",
        ylabel="IoU",
    )

    return output_dir
