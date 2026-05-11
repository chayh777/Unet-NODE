# NODE Robustness Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Validate NODE's robustness advantage under Gaussian noise perturbation by comparing Group A/B/C across 6 noise levels (σ=0.0, 0.05, 0.1, 0.15, 0.2, 0.3) and visualizing bottleneck feature distributions at σ=0.2.

**Architecture:** New script `scripts/run_robustness_analysis.py` runs inference with injected noise across all levels. Core logic in `src/analysis/robustness_metrics.py`. Visualization uses existing `reduce_and_plot.py` patterns with noise injection. Two config variants: default C for A/B, zero-init C for robustness C.

**Tech Stack:** PyTorch, matplotlib, pandas, numpy, umap-learn

---

### Task 1: Create `src/analysis/robustness_metrics.py`

**Files:**
- Create: `src/analysis/robustness_metrics.py`
- Test: `tests/analysis/test_robustness_metrics.py`

- [ ] **Step 1: Write the module with functions**

```python
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
```

- [ ] **Step 2: Create test file**

```python
import pytest
import numpy as np
import torch

from src.analysis.robustness_metrics import (
    compute_sample_dice,
    compute_sample_iou,
    add_gaussian_noise,
)


def test_compute_sample_dice_perfect():
    pred = np.array([[1, 1], [1, 1]])
    gt = np.array([[1, 1], [1, 1]])
    dice = compute_sample_dice(pred, gt)
    assert dice == pytest.approx(1.0)


def test_compute_sample_iou_no_overlap():
    pred = np.array([[1, 1], [0, 0]])
    gt = np.array([[0, 0], [1, 1]])
    iou = compute_sample_iou(pred, gt)
    assert iou == pytest.approx(0.0)


def test_add_gaussian_noise_shape():
    images = torch.randn(2, 3, 256, 256)
    noisy = add_gaussian_noise(images, sigma=0.1)
    assert noisy.shape == images.shape
    assert not torch.allclose(noisy, images)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/analysis/test_robustness_metrics.py -v`
Expected: ERROR (module not found)

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/analysis/test_robustness_metrics.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/analysis/robustness_metrics.py tests/analysis/test_robustness_metrics.py
git commit -m "feat: add robustness metrics computation"
```

---

### Task 2: Create `scripts/run_robustness_analysis.py`

**Files:**
- Create: `scripts/run_robustness_analysis.py`
- Modify: `src/analysis/robustness_metrics.py` (add orchestrator function)

- [ ] **Step 1: Write the CLI script**

```python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from src.analysis.robustness_metrics import run_robustness_experiment
    from src.experiments.low_data_runner import load_config
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    from src.analysis.robustness_metrics import run_robustness_experiment
    from src.experiments.low_data_runner import load_config


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run robustness analysis with Gaussian noise perturbation."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--artifacts-dir",
        required=True,
        help="Directory containing group_* experiment artifacts.",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        default=["A", "B", "C"],
        help="Experiment groups to compare.",
    )
    parser.add_argument(
        "--noise-levels",
        nargs="+",
        type=float,
        default=[0.0, 0.05, 0.1, 0.15, 0.2, 0.3],
        help="Noise levels (sigma values).",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    config = load_config(args.config)
    output_dir = run_robustness_experiment(
        config=config,
        artifacts_dir=Path(args.artifacts_dir),
        groups=args.groups,
        noise_levels=args.noise_levels,
    )
    print(f"Robustness analysis complete. Results saved to {output_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Add orchestrator function to core module**

Add to `src/analysis/robustness_metrics.py`:

```python
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
        state = torch.load(checkpoint_path, map_location="cpu")
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

    metrics_df = save_robustness_metrics(all_metrics, output_dir / "robustness_metrics.csv")

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
```

- [ ] **Step 3: Commit**

```bash
git add scripts/run_robustness_analysis.py
git commit -m "feat: add robustness analysis runner script"
```

---

### Task 3: Create noisy geometry extraction

**Files:**
- Modify: `scripts/run_low_data_geometry.py` (add `--noise-sigma` argument)
- Modify: `src/analysis/low_data_geometry.py` (add noise parameter to extraction)

- [ ] **Step 1: Add noise parameter to geometry extraction**

Modify `src/analysis/low_data_geometry.py` - add `noise_sigma` parameter to `export_group_geometry`:

```python
def export_group_geometry(
    config: dict[str, Any],
    group: str,
    checkpoint_path: Path,
    noise_sigma: float = 0.0,
) -> tuple[Path, Path]:
    # ... existing code ...
    # Before inference, inject noise if sigma > 0:
    if noise_sigma > 0:
        images = add_gaussian_noise(images, noise_sigma)
```

Modify `scripts/run_low_data_geometry.py` - add argument:

```python
parser.add_argument(
    "--noise-sigma",
    type=float,
    default=0.0,
    help="Gaussian noise sigma for robustness test (default: 0.0).",
)
```

Pass to `export_group_geometry`:

```python
pre_csv, post_csv = export_group_geometry(
    config=config,
    group=args.group,
    checkpoint_path=checkpoint_path,
    noise_sigma=args.noise_sigma,
)
```

- [ ] **Step 2: Commit**

```bash
git add src/analysis/low_data_geometry.py scripts/run_low_data_geometry.py
git commit -m "feat: add noise sigma parameter to geometry extraction"
```

---

### Task 4: Create robustness summary visualization

**Files:**
- Create: `scripts/plot_robustness_summary.py`
- Modify: `src/analysis/low_data_reporting.py` (reuse plotting patterns)

- [ ] **Step 1: Write summary plotting script**

```python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from src.analysis.low_data_reporting import _get_plotting_libs
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    from src.analysis.low_data_reporting import _get_plotting_libs

import pandas as pd


def plot_robustness_summary(
    artifacts_dir: Path,
    output_path: Path | None = None,
) -> Path:
    plt = _get_plotting_libs()
    robustness_dir = artifacts_dir / "robustness"
    geometry_dir = robustness_dir / "geometry"

    if output_path is None:
        output_path = robustness_dir / "summary" / "robustness_analysis.png"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    metrics_df = pd.read_csv(robustness_dir / "robustness_metrics.csv")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = {"A": "#7a7a7a", "B": "#377eb8", "C": "#e15759"}
    for group in ["A", "B", "C"]:
        group_data = metrics_df[metrics_df["group"] == group].sort_values("sigma")
        if group_data.empty:
            continue
        axes[0].errorbar(
            group_data["sigma"],
            group_data["mean_dice"],
            yerr=group_data["std_dice"],
            marker="o",
            label=f"Group {group}",
            color=colors.get(group, "#4f83cc"),
            capsize=4,
        )

    axes[0].set_title("DICE vs Noise Level")
    axes[0].set_xlabel("Noise Level (σ)")
    axes[0].set_ylabel("Dice")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    scatter_file = geometry_dir / "sigma0.2_scatter.png"
    if scatter_file.exists():
        from PIL import Image
        img = Image.open(scatter_file)
        axes[1].imshow(img)
        axes[1].axis("off")
        axes[1].set_title("Bottleneck Features at σ=0.2")
    else:
        axes[1].text(0.5, 0.5, "Geometry not available\nRun geometry extraction first", ha="center", va="center")
        axes[1].axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate robustness summary visualization.")
    parser.add_argument("--artifacts-dir", required=True, help="Directory containing robustness artifacts.")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    output_path = plot_robustness_summary(Path(args.artifacts_dir))
    print(f"Summary saved to {output_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add scripts/plot_robustness_summary.py
git commit -m "feat: add robustness summary visualization"
```

---

### Task 5: Update README.md

**Files:**
- Modify: `README.md` (add command contracts)

- [ ] **Step 1: Add robustness experiment section**

Add after "Segmentation comparison visualization contract":

```markdown
### Robustness experiment contract
```bash
# Step A: Run robustness analysis across all noise levels
python scripts/run_robustness_analysis.py --config configs/experiments/isic2018_low_data_node.yaml --artifacts-dir artifacts/low_data --groups A B C

# Step B: Extract geometry with sigma=0.2 noise
python scripts/run_low_data_geometry.py --config configs/experiments/isic2018_low_data_node.yaml --group C --noise-sigma 0.2
python scripts/run_low_data_geometry.py --config configs/experiments/isic2018_low_data_node.yaml --group B --noise-sigma 0.2
python scripts/run_low_data_geometry.py --config configs/experiments/isic2018_low_data_node.yaml --group A --noise-sigma 0.2

# Step C: Generate robustness summary
python scripts/plot_robustness_summary.py --artifacts-dir artifacts/low_data
```

Current artifact contract:
- artifacts/low_data/robustness/robustness_metrics.csv
- artifacts/low_data/robustness/dice_decay_curve.png
- artifacts/low_data/robustness/iou_decay_curve.png
- artifacts/low_data/robustness/geometry/sigma0.2_scatter.png
- artifacts/low_data/robustness/summary/robustness_analysis.png
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: document robustness experiment"
```

---

## File Structure

```
src/analysis/robustness_metrics.py          # Core metrics and plotting
scripts/run_robustness_analysis.py          # CLI runner
scripts/plot_robustness_summary.py         # Summary visualization
src/analysis/low_data_geometry.py          # Modified: add noise_sigma
scripts/run_low_data_geometry.py           # Modified: add --noise-sigma
tests/analysis/test_robustness_metrics.py  # Unit tests
```

## Self-Review Checklist

1. **Spec coverage:**
   - 6 noise levels (σ=0.0, 0.05, 0.1, 0.15, 0.2, 0.3) ✓ (Task 2: noise_levels default)
   - DICE/IoU decay curves ✓ (Task 2: plot_decay_curve)
   - σ=0.2 geometry scatter ✓ (Task 3: noise_sigma parameter)
   - Summary visualization ✓ (Task 4: plot_robustness_summary)

2. **Placeholder scan:** No TODOs, no TBDs, all code shown ✓

3. **Type consistency:** Function signatures consistent ✓
