# Segmentation Comparison Visualization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate a 3-row × 5-column grid visualization comparing segmentation results across Group A/B/C. Rows are validation samples with highest DICE variance; columns are: Image | Ground Truth | Group A | Group B | Group C. Masks use per-class unique colors with alpha=0.7 overlay.

**Architecture:** New script `scripts/plot_segmentation_compare.py` drives the workflow. Core logic lives in `src/analysis/segmentation_compare.py` which: (1) loads validation data, (2) runs inference for each group, (3) computes per-sample DICE, (4) selects top-3 by variance, (5) renders the grid visualization. Follows existing patterns in `src/analysis/low_data_reporting.py`.

**Tech Stack:** PyTorch, matplotlib, PIL, pandas, numpy

---

### Task 1: Create `src/analysis/segmentation_compare.py`

**Files:**
- Create: `src/analysis/segmentation_compare.py`
- Modify: `configs/experiments/isic2018_low_data_node.yaml` (add optional config keys)
- Test: `tests/analysis/test_segmentation_compare.py`

- [ ] **Step 1: Write skeleton with imports and function signatures**

```python
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
```

- [ ] **Step 2: Create test file**

```python
import pytest
import numpy as np
from pathlib import Path

from src.analysis.segmentation_compare import (
    compute_sample_dice,
    _colorize_mask,
)


def test_compute_sample_dice_perfect():
    pred = np.array([[1, 1], [1, 1]])
    gt = np.array([[1, 1], [1, 1]])
    dice = compute_sample_dice(pred, gt)
    assert dice == pytest.approx(1.0)


def test_compute_sample_dice_partial():
    pred = np.array([[1, 1], [0, 0]])
    gt = np.array([[1, 1], [1, 1]])
    dice = compute_sample_dice(pred, gt)
    assert dice == pytest.approx(0.5)


def test_colorize_mask_shape():
    mask = np.array([[0, 1], [1, 0]])
    colored = _colorize_mask(mask)
    assert colored.shape == (2, 2, 3)
    assert colored.dtype == np.float32
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/analysis/test_segmentation_compare.py -v`
Expected: ERROR (module not found)

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/analysis/test_segmentation_compare.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/analysis/segmentation_compare.py tests/analysis/test_segmentation_compare.py
git commit -m "feat: add segmentation comparison visualization core"
```

---

### Task 2: Create `scripts/plot_segmentation_compare.py`

**Files:**
- Create: `scripts/plot_segmentation_compare.py`
- Modify: `src/analysis/segmentation_compare.py` (add main orchestrator function)

- [ ] **Step 1: Write script scaffold**

```python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from src.analysis.segmentation_compare import (
        generate_segmentation_comparison,
    )
    from src.experiments.low_data_runner import load_config
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    from src.analysis.segmentation_compare import (
        generate_segmentation_comparison,
    )
    from src.experiments.low_data_runner import load_config


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate segmentation comparison grid for low-data experiment groups."
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
        help="Experiment groups to compare, for example: A B C",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=3,
        help="Number of samples to display (default: 3).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.7,
        help="Mask overlay alpha (default: 0.7).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Output DPI (default: 150).",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    config = load_config(args.config)
    output_dir = generate_segmentation_comparison(
        config=config,
        artifacts_dir=Path(args.artifacts_dir),
        groups=args.groups,
        num_samples=args.num_samples,
        alpha=args.alpha,
        dpi=args.dpi,
    )
    print(f"Saved to {output_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Add `generate_segmentation_comparison` function to core module**

Add to `src/analysis/segmentation_compare.py`:

```python
def generate_segmentation_comparison(
    config: dict[str, Any],
    artifacts_dir: str | Path,
    groups: list[str] = ["A", "B", "C"],
    num_samples: int = 3,
    alpha: float = 0.7,
    dpi: int = 150,
) -> Path:
    from src.data.isic2018 import ISIC2018Dataset

    artifacts_dir = Path(artifacts_dir)
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
    image_by_id = {}
    for idx in range(len(val_dataset)):
        item = val_dataset[idx]
        sample_id = item["sample_id"]
        img_np = (item["image"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        mask_np = item["mask"].numpy().astype(np.uint8)
        image_by_id[sample_id] = img_np
        ground_truth_by_id[sample_id] = mask_np

    results_by_group = {}
    pred_by_group = {}
    for group in groups:
        checkpoint_path = artifacts_dir / f"group_{group.lower()}" / "best.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
        model = load_model_for_group(checkpoint_path, config, group)
        model.to(device)
        results = run_inference(model, val_loader, device=device)
        results_by_group[group] = results
        pred_by_group[group] = {r["sample_id"]: r["prediction"] for r in results}

    selected_ids = select_top_variance_samples(results_by_group, ground_truth_by_id, top_n=num_samples)

    output_dir = artifacts_dir / "summary"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "segmentation_compare.png"

    render_grid(
        sample_ids=selected_ids,
        image_by_id=image_by_id,
        gt_by_id=ground_truth_by_id,
        pred_by_group=pred_by_group,
        output_path=output_path,
        alpha=alpha,
        dpi=dpi,
    )

    return output_dir
```

- [ ] **Step 3: Create tests directory and test file**

```bash
mkdir -p tests/analysis
```

```python
import pytest
from pathlib import Path
from src.analysis.segmentation_compare import generate_segmentation_comparison
from src.experiments.low_data_runner import load_config


def test_generate_requires_checkpoint(tmp_path, monkeypatch):
    config_path = Path("configs/experiments/isic2018_low_data_node.yaml")
    if not config_path.exists():
        pytest.skip("Config not found")

    config = load_config(config_path)

    with pytest.raises(FileNotFoundError):
        generate_segmentation_comparison(
            config=config,
            artifacts_dir=tmp_path,
            groups=["A"],
            num_samples=1,
        )
```

- [ ] **Step 4: Commit**

```bash
git add scripts/plot_segmentation_compare.py
git commit -m "feat: add segmentation comparison visualization script"
```

---

### Task 3: Update README.md documentation

**Files:**
- Modify: `README.md` (add command contract)

- [ ] **Step 1: Add command to README.md**

Add after existing summary reporting contract:

```markdown
### Segmentation comparison visualization contract
```bash
python scripts/plot_segmentation_compare.py --config configs/experiments/isic2018_low_data_node.yaml --artifacts-dir artifacts/low_data --groups A B C
```

Current artifact contract:
- artifacts/low_data/summary/segmentation_compare.png
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: document segmentation comparison visualization"
```

---

## File Structure

```
src/analysis/segmentation_compare.py  # Core visualization logic
scripts/plot_segmentation_compare.py  # CLI script
tests/analysis/test_segmentation_compare.py  # Unit tests
```

## Self-Review Checklist

1. **Spec coverage:** All design requirements mapped:
   - 3 rows by 5 columns grid ✓ (Task 2: `render_grid`)
   - DICE variance sample selection ✓ (Task 1: `select_top_variance_samples`)
   - Per-class unique colors ✓ (Task 1: `CLASS_COLORS`, `_colorize_mask`)
   - Alpha overlay ✓ (Task 1: `alpha` parameter in `render_grid`)
   - Ground Truth column ✓ (Task 1: column index 1)
   - Group A/B/C columns ✓ (Task 2: loop over groups starting at col 2)

2. **Placeholder scan:** No TODOs, no TBDs, all function bodies shown ✓

3. **Type consistency:** Function signatures consistent across tasks ✓
