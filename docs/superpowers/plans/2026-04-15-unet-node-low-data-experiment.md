# U-Net NODE Low-Data Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first-round low-data ISIC2018 experiment that compares frozen-encoder pretrained U-Net under three bottleneck settings: no adapter, conv adapter, and NODE adapter.

**Architecture:** Keep the ImageNet-pretrained encoder frozen, keep the decoder trainable, and inject the experiment variable only at the encoder bottleneck. Build one reusable segmentation model wrapper that can switch among `none`, `conv`, and `node` adapters via config, plus a fixed 10% split runner, metric logging, checkpoint selection, and optional bottleneck feature export for later geometry analysis.

**Tech Stack:** Python, PyTorch, timm, PyYAML, pandas, matplotlib, pytest

---

## Planned File Structure

- Create: `configs/experiments/isic2018_low_data_node.yaml`
  Purpose: round-one config for data paths, split seed, model policy, optimizer, and artifacts.
- Modify: `src/data/isic2018.py`
  Purpose: support fixed subset loading without duplicating dataset logic.
- Create: `src/data/splits.py`
  Purpose: deterministic ratio-based subset sampling and manifest export.
- Create: `src/models/adapters.py`
  Purpose: define `IdentityAdapter` and `ConvBottleneckAdapter`.
- Create: `src/models/node_adapter.py`
  Purpose: define the round-one NODE bottleneck module.
- Create: `src/models/segmentation_model.py`
  Purpose: compose frozen encoder, selected adapter, trainable decoder, and logits head.
- Create: `src/training/losses.py`
  Purpose: define `DiceBCELoss`.
- Create: `src/training/metrics.py`
  Purpose: define Dice and IoU metrics.
- Create: `src/training/engine.py`
  Purpose: train/validate loops, early stopping, and artifact export.
- Create: `src/experiments/low_data_runner.py`
  Purpose: config-driven A/B/C experiment orchestration.
- Create: `scripts/run_low_data_experiment.py`
  Purpose: CLI entrypoint.
- Create: `tests/test_splits.py`
  Purpose: verify deterministic split generation.
- Create: `tests/test_adapters.py`
  Purpose: verify adapter shape contract.
- Create: `tests/test_losses_metrics.py`
  Purpose: verify loss and metric calculations.
- Create: `tests/test_low_data_runner.py`
  Purpose: verify group-to-adapter mapping and encoder freezing.
- Modify: `README.md`
  Purpose: add round-one experiment commands and artifact expectations.

### Task 1: Add The Experiment Contract

**Files:**
- Create: `configs/experiments/isic2018_low_data_node.yaml`
- Modify: `README.md`

- [ ] **Step 1: Add the low-data experiment section to the README**

```markdown
## Low-data NODE experiment

First-round validation compares three groups on ISIC2018 with a frozen ImageNet-pretrained encoder and a fixed 10% training subset:

- Group A: decoder only
- Group B: decoder + conv adapter
- Group C: decoder + NODE adapter

Run one group:

```bash
python scripts/run_low_data_experiment.py --config configs/experiments/isic2018_low_data_node.yaml --group A
```

Expected artifacts:

- artifacts/low_data/group_a/best.pt
- artifacts/low_data/group_a/history.csv
- artifacts/low_data/group_a/metrics.json
- artifacts/low_data/group_b/best.pt
- artifacts/low_data/group_c/best.pt
- artifacts/low_data/splits/train_seed42_ratio10.csv
```
```

- [ ] **Step 2: Add the config file**

```yaml
seed: 42

paths:
  train_images_dir: data/train/images
  train_masks_dir: data/train/labels
  val_images_dir: data/validation/images
  val_masks_dir: data/validation/labels
  artifacts_dir: artifacts/low_data

data:
  image_size: 256
  train_ratio: 0.1
  num_workers: 4
  pin_memory: true

model:
  encoder_name: resnet34
  encoder_weights: imagenet
  in_channels: 3
  num_classes: 1
  bottleneck_channels: 512
  freeze_encoder: true

adapter:
  type: node
  hidden_channels: 512

node:
  solver: euler
  steps: 4
  step_size: 0.25

train:
  batch_size: 8
  epochs: 50
  learning_rate: 0.0003
  weight_decay: 0.0001
  early_stopping_patience: 8
```
```

- [ ] **Step 3: Verify the new contract exists**

Run: `rg "run_low_data_experiment|isic2018_low_data_node|train_seed42_ratio10" README.md configs/experiments/isic2018_low_data_node.yaml`
Expected: all three phrases are found

- [ ] **Step 4: Commit**

```bash
git add README.md configs/experiments/isic2018_low_data_node.yaml
git commit -m "docs: scaffold low-data node experiment contract"
```

### Task 2: Build Deterministic Low-Data Split Utilities

**Files:**
- Create: `src/data/splits.py`
- Create: `tests/test_splits.py`

- [ ] **Step 1: Write the failing split test**

```python
from src.data.splits import build_ratio_subset


def test_build_ratio_subset_is_deterministic():
    sample_ids = [f"ISIC_{index:04d}" for index in range(20)]

    first = build_ratio_subset(sample_ids, ratio=0.1, seed=42)
    second = build_ratio_subset(sample_ids, ratio=0.1, seed=42)

    assert first == second
    assert len(first) == 2
```
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_splits.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement the split helpers**

```python
from __future__ import annotations

import random
from pathlib import Path

import pandas as pd


def build_ratio_subset(sample_ids: list[str], ratio: float, seed: int) -> list[str]:
    if not 0 < ratio <= 1:
        raise ValueError(f"ratio must be in (0, 1], got {ratio}")
    if not sample_ids:
        raise ValueError("sample_ids must not be empty")

    rng = random.Random(seed)
    ordered = sorted(sample_ids)
    target_count = max(1, int(round(len(ordered) * ratio)))
    return sorted(rng.sample(ordered, target_count))


def save_split_manifest(sample_ids: list[str], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"sample_id": sample_ids}).to_csv(path, index=False)
    return path
```
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_splits.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/data/splits.py tests/test_splits.py
git commit -m "feat: add deterministic low-data split helpers"
```

### Task 3: Extend The Dataset For Fixed Subset Loading

**Files:**
- Modify: `src/data/isic2018.py`
- Modify: `tests/test_isic_dataset.py`

- [ ] **Step 1: Add the failing subset test**

```python
from pathlib import Path

import numpy as np
from PIL import Image

from src.data.isic2018 import ISIC2018Dataset


def test_isic_dataset_can_filter_by_sample_ids(tmp_path: Path):
    images_dir = tmp_path / "images"
    masks_dir = tmp_path / "masks"
    images_dir.mkdir()
    masks_dir.mkdir()

    for sample_id in ["ISIC_0001", "ISIC_0002"]:
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        mask = np.zeros((8, 8), dtype=np.uint8)
        Image.fromarray(image).save(images_dir / f"{sample_id}.jpg")
        Image.fromarray(mask).save(masks_dir / f"{sample_id}.png")

    dataset = ISIC2018Dataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        image_size=8,
        class_values={"background": 0, "lesion": 1},
        sample_ids=["ISIC_0002"],
    )

    assert len(dataset) == 1
    assert dataset[0]["sample_id"] == "ISIC_0002"
```
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_isic_dataset.py::test_isic_dataset_can_filter_by_sample_ids -v`
Expected: FAIL with `TypeError`

- [ ] **Step 3: Implement subset-aware dataset loading**

```python
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class ISIC2018Dataset(Dataset):
    def __init__(self, images_dir, masks_dir, image_size, class_values, sample_ids=None):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.image_size = image_size
        self.class_values = class_values
        selected_ids = set(sample_ids) if sample_ids is not None else None

        image_paths = sorted(self.images_dir.glob("*.jpg"))
        if selected_ids is not None:
            image_paths = [path for path in image_paths if path.stem in selected_ids]
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        sample_id = image_path.stem
        mask_path = self.masks_dir / f"{sample_id}.png"

        image = Image.open(image_path).convert("RGB").resize((self.image_size, self.image_size))
        mask = Image.open(mask_path).convert("L").resize((self.image_size, self.image_size))

        image_np = np.asarray(image, dtype=np.float32) / 255.0
        mask_np = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.float32)

        return {
            "sample_id": sample_id,
            "image": torch.from_numpy(image_np).permute(2, 0, 1),
            "mask": torch.from_numpy(mask_np).unsqueeze(0),
        }
```
```

- [ ] **Step 4: Run the dataset tests**

Run: `pytest tests/test_isic_dataset.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/data/isic2018.py tests/test_isic_dataset.py
git commit -m "feat: add fixed subset support to isic dataset"
```

### Task 4: Implement The Adapter Modules

**Files:**
- Create: `src/models/adapters.py`
- Create: `src/models/node_adapter.py`
- Create: `tests/test_adapters.py`

- [ ] **Step 1: Write the failing adapter tests**

```python
import torch

from src.models.adapters import ConvBottleneckAdapter, IdentityAdapter
from src.models.node_adapter import NODEAdapter


def test_identity_adapter_preserves_shape():
    x = torch.randn(2, 32, 8, 8)
    y = IdentityAdapter()(x)
    assert tuple(y.shape) == (2, 32, 8, 8)


def test_conv_adapter_preserves_shape():
    x = torch.randn(2, 32, 8, 8)
    y = ConvBottleneckAdapter(channels=32, hidden_channels=32)(x)
    assert tuple(y.shape) == (2, 32, 8, 8)


def test_node_adapter_preserves_shape():
    x = torch.randn(2, 32, 8, 8)
    y = NODEAdapter(channels=32, hidden_channels=32, steps=4, step_size=0.25)(x)
    assert tuple(y.shape) == (2, 32, 8, 8)
```
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_adapters.py -v`
Expected: FAIL with missing modules

- [ ] **Step 3: Implement the identity and conv adapters**

```python
from __future__ import annotations

import torch
import torch.nn as nn


class IdentityAdapter(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class ConvBottleneckAdapter(nn.Module):
    def __init__(self, channels: int, hidden_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
```
```

- [ ] **Step 4: Implement the round-one NODE adapter**

```python
from __future__ import annotations

import torch
import torch.nn as nn


class ODEFunction(nn.Module):
    def __init__(self, channels: int, hidden_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NODEAdapter(nn.Module):
    def __init__(self, channels: int, hidden_channels: int, steps: int, step_size: float):
        super().__init__()
        self.func = ODEFunction(channels=channels, hidden_channels=hidden_channels)
        self.steps = steps
        self.step_size = step_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x
        for _ in range(self.steps):
            z = z + self.step_size * self.func(z)
        return z
```
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest tests/test_adapters.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/models/adapters.py src/models/node_adapter.py tests/test_adapters.py
git commit -m "feat: add conv and node bottleneck adapters"
```

### Task 5: Build The Configurable Frozen-Encoder Segmentation Model

**Files:**
- Create: `src/models/segmentation_model.py`
- Create: `tests/test_low_data_runner.py`

- [ ] **Step 1: Write the failing model test**

```python
from src.models.segmentation_model import build_segmentation_model


def test_build_segmentation_model_freezes_encoder():
    model = build_segmentation_model(
        encoder_name="resnet18",
        encoder_weights=None,
        in_channels=3,
        num_classes=1,
        adapter_type="node",
        bottleneck_channels=512,
        adapter_hidden_channels=512,
        freeze_encoder=True,
        node_steps=4,
        node_step_size=0.25,
    )

    encoder_flags = [p.requires_grad for name, p in model.named_parameters() if "encoder" in name]
    assert encoder_flags
    assert all(flag is False for flag in encoder_flags)
```
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_low_data_runner.py::test_build_segmentation_model_freezes_encoder -v`
Expected: FAIL with missing module or function

- [ ] **Step 3: Implement the segmentation model**

```python
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from src.models.adapters import ConvBottleneckAdapter, IdentityAdapter
from src.models.node_adapter import NODEAdapter


class SegmentationModel(nn.Module):
    def __init__(
        self,
        encoder_name: str,
        encoder_weights,
        in_channels: int,
        num_classes: int,
        adapter_type: str,
        bottleneck_channels: int,
        adapter_hidden_channels: int,
        freeze_encoder: bool,
        node_steps: int,
        node_step_size: float,
    ):
        super().__init__()
        self.encoder = timm.create_model(
            encoder_name,
            pretrained=encoder_weights == "imagenet",
            in_chans=in_channels,
            features_only=True,
        )
        encoder_channels = self.encoder.feature_info.channels()
        self.proj = nn.Conv2d(encoder_channels[-1], bottleneck_channels, kernel_size=1)

        if adapter_type == "none":
            self.adapter = IdentityAdapter()
        elif adapter_type == "conv":
            self.adapter = ConvBottleneckAdapter(bottleneck_channels, adapter_hidden_channels)
        elif adapter_type == "node":
            self.adapter = NODEAdapter(
                channels=bottleneck_channels,
                hidden_channels=adapter_hidden_channels,
                steps=node_steps,
                step_size=node_step_size,
            )
        else:
            raise ValueError(f"Unsupported adapter_type: {adapter_type}")

        self.decoder = nn.Sequential(
            nn.Conv2d(bottleneck_channels, bottleneck_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck_channels // 2, bottleneck_channels // 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(bottleneck_channels // 4, num_classes, kernel_size=1)

        if freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.encoder(x)
        bottleneck = self.proj(features[-1])
        adapted = self.adapter(bottleneck)
        decoded = self.decoder(adapted)
        logits = self.head(
            F.interpolate(decoded, size=x.shape[-2:], mode="bilinear", align_corners=False)
        )
        return {"logits": logits, "bottleneck": bottleneck, "adapted_bottleneck": adapted}


def build_segmentation_model(**kwargs) -> SegmentationModel:
    return SegmentationModel(**kwargs)
```
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_low_data_runner.py::test_build_segmentation_model_freezes_encoder -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/models/segmentation_model.py tests/test_low_data_runner.py
git commit -m "feat: add configurable frozen-encoder segmentation model"
```

### Task 6: Add The Shared Loss And Metric Modules

**Files:**
- Create: `src/training/losses.py`
- Create: `src/training/metrics.py`
- Create: `tests/test_losses_metrics.py`

- [ ] **Step 1: Write the failing loss and metric tests**

```python
import torch

from src.training.losses import DiceBCELoss
from src.training.metrics import compute_binary_dice, compute_binary_iou


def test_dice_bce_loss_returns_scalar():
    logits = torch.zeros(2, 1, 4, 4)
    targets = torch.zeros(2, 1, 4, 4)
    loss = DiceBCELoss()(logits, targets)
    assert loss.ndim == 0


def test_binary_metrics_reach_one_for_perfect_prediction():
    logits = torch.full((1, 1, 2, 2), 10.0)
    targets = torch.ones(1, 1, 2, 2)
    assert float(compute_binary_dice(logits, targets)) == 1.0
    assert float(compute_binary_iou(logits, targets)) == 1.0
```
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_losses_metrics.py -v`
Expected: FAIL with missing modules

- [ ] **Step 3: Implement the loss module**

```python
from __future__ import annotations

import torch
import torch.nn as nn


class DiceBCELoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum(dim=(1, 2, 3))
        denominator = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        dice_loss = 1.0 - ((2.0 * intersection + self.smooth) / (denominator + self.smooth))
        return self.bce(logits, targets) + dice_loss.mean()
```
```

- [ ] **Step 4: Implement the metric module**

```python
from __future__ import annotations

import torch


def _to_binary_predictions(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    return (torch.sigmoid(logits) >= threshold).float()


def compute_binary_dice(logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    preds = _to_binary_predictions(logits)
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    denominator = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    return ((2.0 * intersection + smooth) / (denominator + smooth)).mean()


def compute_binary_iou(logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    preds = _to_binary_predictions(logits)
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) - intersection
    return ((intersection + smooth) / (union + smooth)).mean()
```
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest tests/test_losses_metrics.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/training/losses.py src/training/metrics.py tests/test_losses_metrics.py
git commit -m "feat: add low-data segmentation loss and metrics"
```

### Task 7: Implement The Training Engine

**Files:**
- Create: `src/training/engine.py`

- [ ] **Step 1: Implement the history helpers**

```python
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json

import pandas as pd


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    val_loss: float
    val_dice: float
    val_iou: float


def save_history(rows: list[EpochMetrics], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([asdict(row) for row in rows]).to_csv(path, index=False)
    return path


def save_metrics_json(metrics: dict, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    return path
```
```

- [ ] **Step 2: Implement the train/validate loop**

```python
from __future__ import annotations

from pathlib import Path

import torch

from src.training.losses import DiceBCELoss
from src.training.metrics import compute_binary_dice, compute_binary_iou


def run_epoch(model, loader, optimizer=None, device="cpu"):
    criterion = DiceBCELoss()
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    total_steps = 0

    for batch in loader:
        images = batch["image"].float().to(device)
        masks = batch["mask"].float().to(device)

        if is_training:
            optimizer.zero_grad()

        outputs = model(images)
        logits = outputs["logits"]
        loss = criterion(logits, masks)

        if is_training:
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item())
        total_dice += float(compute_binary_dice(logits.detach(), masks).item())
        total_iou += float(compute_binary_iou(logits.detach(), masks).item())
        total_steps += 1

    return {
        "loss": total_loss / max(1, total_steps),
        "dice": total_dice / max(1, total_steps),
        "iou": total_iou / max(1, total_steps),
    }


def fit(model, train_loader, val_loader, optimizer, epochs, patience, output_dir, device="cpu"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    history = []
    best_dice = float("-inf")
    best_path = output_dir / "best.pt"
    stale_epochs = 0

    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer=optimizer, device=device)
        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, optimizer=None, device=device)

        row = EpochMetrics(
            epoch=epoch,
            train_loss=train_metrics["loss"],
            val_loss=val_metrics["loss"],
            val_dice=val_metrics["dice"],
            val_iou=val_metrics["iou"],
        )
        history.append(row)

        if row.val_dice > best_dice:
            best_dice = row.val_dice
            stale_epochs = 0
            torch.save(model.state_dict(), best_path)
        else:
            stale_epochs += 1

        if stale_epochs >= patience:
            break

    save_history(history, output_dir / "history.csv")
    save_metrics_json(
        {
            "best_val_dice": best_dice,
            "epochs_ran": len(history),
            "best_checkpoint": str(best_path),
        },
        output_dir / "metrics.json",
    )
    return best_path
```
```

- [ ] **Step 3: Run a smoke import check**

Run: `python -c "from src.training.engine import fit; print(callable(fit))"`
Expected: `True`

- [ ] **Step 4: Commit**

```bash
git add src/training/engine.py
git commit -m "feat: add low-data experiment training engine"
```

### Task 8: Implement The Config-Driven Experiment Runner

**Files:**
- Create: `src/experiments/low_data_runner.py`
- Create: `scripts/run_low_data_experiment.py`
- Modify: `tests/test_low_data_runner.py`

- [ ] **Step 1: Add the failing runner test**

```python
from src.experiments.low_data_runner import resolve_group_adapter


def test_resolve_group_adapter_maps_groups_to_expected_types():
    assert resolve_group_adapter("A") == "none"
    assert resolve_group_adapter("B") == "conv"
    assert resolve_group_adapter("C") == "node"
```
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_low_data_runner.py::test_resolve_group_adapter_maps_groups_to_expected_types -v`
Expected: FAIL with missing module or function

- [ ] **Step 3: Implement the runner**

```python
from __future__ import annotations

from pathlib import Path

import torch
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.data.isic2018 import ISIC2018Dataset
from src.data.splits import build_ratio_subset, save_split_manifest
from src.models.segmentation_model import build_segmentation_model
from src.training.engine import fit


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_group_adapter(group: str) -> str:
    mapping = {"A": "none", "B": "conv", "C": "node"}
    if group not in mapping:
        raise ValueError(f"Unsupported group: {group}")
    return mapping[group]


def run_group(config_path: str, group: str):
    config = load_config(config_path)
    adapter_type = resolve_group_adapter(group)

    full_train_dataset = ISIC2018Dataset(
        images_dir=config["paths"]["train_images_dir"],
        masks_dir=config["paths"]["train_masks_dir"],
        image_size=config["data"]["image_size"],
        class_values={"background": 0, "lesion": 1},
    )
    selected_ids = build_ratio_subset(
        [path.stem for path in full_train_dataset.image_paths],
        ratio=config["data"]["train_ratio"],
        seed=config["seed"],
    )

    split_path = Path(config["paths"]["artifacts_dir"]) / "splits" / "train_seed42_ratio10.csv"
    save_split_manifest(selected_ids, split_path)

    train_dataset = ISIC2018Dataset(
        images_dir=config["paths"]["train_images_dir"],
        masks_dir=config["paths"]["train_masks_dir"],
        image_size=config["data"]["image_size"],
        class_values={"background": 0, "lesion": 1},
        sample_ids=selected_ids,
    )
    val_dataset = ISIC2018Dataset(
        images_dir=config["paths"]["val_images_dir"],
        masks_dir=config["paths"]["val_masks_dir"],
        image_size=config["data"]["image_size"],
        class_values={"background": 0, "lesion": 1},
    )

    train_loader = DataLoader(train_dataset, batch_size=config["train"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["train"]["batch_size"], shuffle=False)

    model = build_segmentation_model(
        encoder_name=config["model"]["encoder_name"],
        encoder_weights=config["model"]["encoder_weights"],
        in_channels=config["model"]["in_channels"],
        num_classes=config["model"]["num_classes"],
        adapter_type=adapter_type,
        bottleneck_channels=config["model"]["bottleneck_channels"],
        adapter_hidden_channels=config["adapter"]["hidden_channels"],
        freeze_encoder=config["model"]["freeze_encoder"],
        node_steps=config["node"]["steps"],
        node_step_size=config["node"]["step_size"],
    )

    optimizer = AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=config["train"]["learning_rate"],
        weight_decay=config["train"]["weight_decay"],
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    output_dir = Path(config["paths"]["artifacts_dir"]) / f"group_{group.lower()}"
    return fit(model, train_loader, val_loader, optimizer, config["train"]["epochs"], config["train"]["early_stopping_patience"], output_dir, device=device)
```
```

- [ ] **Step 4: Add the CLI script**

```python
import argparse

from src.experiments.low_data_runner import run_group


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--group", required=True, choices=["A", "B", "C"])
    args = parser.parse_args()
    run_group(args.config, args.group)


if __name__ == "__main__":
    main()
```
```

- [ ] **Step 5: Run the runner tests**

Run: `pytest tests/test_low_data_runner.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/experiments/low_data_runner.py scripts/run_low_data_experiment.py tests/test_low_data_runner.py
git commit -m "feat: add config-driven low-data node experiment runner"
```

### Task 9: Verify The Full Plan End To End

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Run the focused tests**

Run: `pytest tests/test_splits.py tests/test_isic_dataset.py tests/test_adapters.py tests/test_losses_metrics.py tests/test_low_data_runner.py -v`
Expected: PASS

- [ ] **Step 2: Run the full test suite**

Run: `pytest -q`
Expected: PASS

- [ ] **Step 3: Verify the CLI import path**

Run: `python -c "from scripts.run_low_data_experiment import main; print(callable(main))"`
Expected: `True`

- [ ] **Step 4: Run one dry experiment group**

Run: `python scripts/run_low_data_experiment.py --config configs/experiments/isic2018_low_data_node.yaml --group A`
Expected:
- `artifacts/low_data/splits/train_seed42_ratio10.csv` exists
- `artifacts/low_data/group_a/best.pt` exists
- `artifacts/low_data/group_a/history.csv` exists
- `artifacts/low_data/group_a/metrics.json` exists

- [ ] **Step 5: Commit**

```bash
git add README.md artifacts/low_data
git commit -m "feat: verify first-round low-data node experiment pipeline"
```

## Self-Review

- Spec coverage: this plan covers the round-one design only: fixed ISIC2018 10% split, frozen pretrained encoder, trainable decoder, and three bottleneck settings `none/conv/node`. It also includes the required outputs: checkpoint, history, metrics, and split manifest.
- Placeholder scan: no `TODO`, `TBD`, or "implement later" placeholders remain. Each code step includes concrete file paths, code blocks, commands, and expected outcomes.
- Type consistency: the plan consistently uses `build_ratio_subset`, `IdentityAdapter`, `ConvBottleneckAdapter`, `NODEAdapter`, `build_segmentation_model`, `DiceBCELoss`, `fit`, `resolve_group_adapter`, and `run_group`.
