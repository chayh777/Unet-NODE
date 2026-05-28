# Stage 1 Experiment Infrastructure And GlaS Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make low-data experiments cheap enough to scale by adding configurable checkpoint saving, training-time logging, and first-class GlaS dataset support for the frozen-encoder adapter experiments.

**Architecture:** Keep the current low-data runner as the main orchestration entrypoint, but remove its ISIC-only assumptions by introducing a small dataset factory layer and a dedicated `GlaSDataset`. Extend the shared training engine with a configurable checkpoint policy and wall-clock timing so every later experiment variant, including future comparison methods, inherits the same artifact and efficiency reporting behavior.

**Tech Stack:** Python, PyTorch, pandas-free dataset IO, YAML configs, pytest, existing `scripts/run_low_data_experiment.py` CLI, existing artifact layout under `artifacts/`.

---

## Scope And Decomposition

This plan intentionally covers only the first two approved priorities:

1. checkpoint `.pt` saving control plus training-time logging
2. first extra dataset support via `GlaS`

External comparison methods are intentionally **not** included here because the shortlist is not fixed yet. That needs a follow-up plan once the exact methods are chosen; otherwise we would introduce placeholders, which this plan forbids.

## File Structure

- Modify: `src/training/engine.py`
  - Add configurable checkpoint saving policy and training duration metrics to the shared fit loop.
- Modify: `src/experiments/low_data_runner.py`
  - Replace hard-coded ISIC dataset wiring with dataset-factory wiring and pass save-policy options into the engine.
- Create: `src/data/glas.py`
  - Implement `GlaSDataset` with the same sample contract as `ISIC2018Dataset`.
- Create: `src/data/factory.py`
  - Centralize dataset-name validation, class mapping, and train/val dataset construction for low-data experiments.
- Modify: `scripts/run_low_data_experiment.py`
  - Keep the CLI stable but ensure help text reflects multi-dataset low-data experiments.
- Create: `configs/experiments/glas_low_data_conv_b_base.yaml`
  - First GlaS baseline config for group B.
- Create: `configs/experiments/glas_low_data_node_c_base.yaml`
  - First GlaS NODE config for group C.
- Modify: `src/analysis/report_visualization.py`
  - Preserve compatibility with richer metrics JSON and optionally surface training duration in CSV summaries.
- Create: `tests/test_glas_dataset.py`
  - Unit tests for the new dataset loader.
- Create: `tests/test_data_factory.py`
  - Unit tests for dataset-name resolution and low-data dataset creation.
- Modify: `tests/test_low_data_runner.py`
  - Add regression tests for save policy and dataset factory wiring.
- Modify: `tests/test_report_visualization.py`
  - Add a regression test proving extra timing fields in `metrics.json` do not break report summarization.
- Modify: `tests/test_low_data_clis.py`
  - Add CLI help/wiring coverage for a non-ISIC config path.

## Output Contract

After this plan is implemented, each low-data run must produce:

- `history.csv`
- `metrics.json`
- optionally `best.pt`, depending on config

The updated `metrics.json` must include:

- `best_val_dice`
- `epochs_ran`
- `best_checkpoint`
- `best_epoch`
- `duration_sec`
- `avg_epoch_sec`
- `checkpoint_saved`

The updated low-data config contract must support:

```yaml
data:
  dataset_name: isic2018  # or glas
  image_size: 256
  train_ratio: 0.1
  num_workers: 0
  pin_memory: false
train:
  batch_size: 4
  epochs: 40
  learning_rate: 0.0001
  weight_decay: 0.0001
  early_stopping_patience: 10
  save_best_checkpoint: true
```

## Task 1: Make Checkpoint Saving Configurable And Record Training Time

**Files:**
- Modify: `src/training/engine.py`
- Modify: `src/experiments/low_data_runner.py`
- Modify: `tests/test_low_data_runner.py`

- [ ] **Step 1: Add a failing runner test for `save_best_checkpoint: false` and timing passthrough**

Append to `tests/test_low_data_runner.py`:

```python
def test_run_group_passes_save_best_checkpoint_flag_to_engine(tmp_path, monkeypatch):
    artifacts_dir = tmp_path / "artifacts"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "seed: 7",
                "paths:",
                "  train_images_dir: data/train/images",
                "  train_masks_dir: data/train/labels",
                "  val_images_dir: data/val/images",
                "  val_masks_dir: data/val/labels",
                f"  artifacts_dir: {artifacts_dir.as_posix()}",
                "data:",
                "  dataset_name: isic2018",
                "  image_size: 256",
                "  train_ratio: 0.1",
                "train:",
                "  batch_size: 2",
                "  epochs: 3",
                "  learning_rate: 0.001",
                "  weight_decay: 0.01",
                "  early_stopping_patience: 5",
                "  save_best_checkpoint: false",
                "model:",
                "  encoder_name: resnet18",
                "  encoder_weights: null",
                "  in_channels: 3",
                "  num_classes: 1",
                "  bottleneck_channels: 16",
                "  freeze_encoder: true",
                "adapter:",
                "  hidden_channels: 8",
                "node:",
                "  steps: 4",
                "  step_size: 0.25",
                "  solver: euler",
                "",
            ]
        ),
        encoding="utf-8",
    )

    calls: dict[str, object] = {}

    fake_factory = ModuleType("src.data.factory")

    class _FakeDataset:
        def __init__(self, names):
            self.image_paths = [type("_P", (), {"stem": name})() for name in names]

    def build_low_data_datasets(config):
        calls["dataset_name"] = config["data"]["dataset_name"]
        return {
            "full_train_dataset": _FakeDataset(["img1", "img2", "img3"]),
            "train_dataset": _FakeDataset(["img2", "img3"]),
            "val_dataset": _FakeDataset(["val1"]),
            "selected_ids": ["img2", "img3"],
        }

    fake_factory.build_low_data_datasets = build_low_data_datasets

    fake_models = ModuleType("src.models.segmentation_model")

    class _FakeParam:
        def __init__(self, requires_grad: bool) -> None:
            self.requires_grad = requires_grad

    class _FakeModel:
        def __init__(self):
            self._params = [_FakeParam(True)]

        def parameters(self):
            return list(self._params)

        def to(self, device):
            return self

    fake_models.build_segmentation_model = lambda **kwargs: _FakeModel()

    fake_engine = ModuleType("src.training.engine")

    def fit(model, train_loader, val_loader, optimizer, epochs, patience, output_dir, device="cpu", save_best_checkpoint=True):
        calls["fit"] = {
            "epochs": epochs,
            "patience": patience,
            "save_best_checkpoint": save_best_checkpoint,
            "output_dir": str(output_dir),
        }
        return None

    fake_engine.fit = fit

    fake_splits = ModuleType("src.data.splits")
    fake_splits.save_split_manifest = lambda sample_ids, output_path: None

    monkeypatch.setitem(sys.modules, "src.data.factory", fake_factory)
    monkeypatch.setitem(sys.modules, "src.models.segmentation_model", fake_models)
    monkeypatch.setitem(sys.modules, "src.training.engine", fake_engine)
    monkeypatch.setitem(sys.modules, "src.data.splits", fake_splits)

    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.optim, "AdamW", lambda params, lr, weight_decay: object())

    class _FakeDataLoader:
        def __init__(self, dataset, batch_size, shuffle, **kwargs):
            self.dataset = dataset

    monkeypatch.setattr(torch.utils.data, "DataLoader", _FakeDataLoader)

    from src.experiments.low_data_runner import run_group

    run_group(config_path, "B")

    assert calls["dataset_name"] == "isic2018"
    assert calls["fit"]["save_best_checkpoint"] is False
```

- [ ] **Step 2: Run the new test to verify it fails**

Run:

```bash
python -m pytest tests/test_low_data_runner.py::test_run_group_passes_save_best_checkpoint_flag_to_engine -q
```

Expected: FAIL because `src.data.factory` does not exist and `fit()` does not accept `save_best_checkpoint`.

- [ ] **Step 3: Add a failing engine test for `metrics.json` timing fields and optional checkpoint omission**

Create `tests/test_training_engine.py` with:

```python
from __future__ import annotations

import json
from pathlib import Path

import torch


class _TinySegModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return x[:, :1] * self.scale


def _batch():
    image = torch.ones((1, 3, 4, 4), dtype=torch.float32)
    mask = torch.ones((1, 4, 4), dtype=torch.long)
    return {"image": image, "mask": mask}


def test_fit_can_skip_best_checkpoint_and_still_write_timing_metrics(tmp_path: Path):
    from src.training.engine import fit

    model = _TinySegModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    train_loader = [_batch()]
    val_loader = [_batch()]

    best_path = fit(
        model,
        train_loader,
        val_loader,
        optimizer,
        epochs=2,
        patience=2,
        output_dir=tmp_path,
        device="cpu",
        save_best_checkpoint=False,
    )

    assert best_path is None
    assert not (tmp_path / "best.pt").exists()

    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["checkpoint_saved"] is False
    assert metrics["best_checkpoint"] is None
    assert metrics["best_epoch"] == 1
    assert metrics["duration_sec"] >= 0.0
    assert metrics["avg_epoch_sec"] >= 0.0
```

- [ ] **Step 4: Run the engine test to verify it fails**

Run:

```bash
python -m pytest tests/test_training_engine.py::test_fit_can_skip_best_checkpoint_and_still_write_timing_metrics -q
```

Expected: FAIL with `TypeError` because `fit()` does not accept `save_best_checkpoint`.

- [ ] **Step 5: Implement minimal engine changes**

Update `src/training/engine.py`:

```python
from __future__ import annotations

import csv
import json
import math
import time
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


def fit(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    optimizer: Any,
    epochs: int,
    patience: int,
    output_dir: str | Path,
    device: str | torch.device = "cpu",
    save_best_checkpoint: bool = True,
) -> Path | None:
    output_dir_p = Path(output_dir)
    output_dir_p.mkdir(parents=True, exist_ok=True)

    device_t = torch.device(device)
    model.to(device_t)

    history: list[EpochMetrics] = []
    best_dice: float | None = None
    best_epoch: int | None = None
    best_path = output_dir_p / "best.pt"
    best_saved = False
    stale_epochs = 0
    start_time = time.perf_counter()

    for epoch in range(1, int(epochs) + 1):
        train_metrics = run_epoch(model=model, loader=train_loader, optimizer=optimizer, device=device_t)
        with torch.no_grad():
            val_metrics = run_epoch(model=model, loader=val_loader, optimizer=None, device=device_t)

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
            best_epoch = epoch
            if save_best_checkpoint:
                best_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), best_path)
                best_saved = True
            stale_epochs = 0
        else:
            stale_epochs += 1

        if (not improved) and stale_epochs >= int(patience):
            break

    duration_sec = time.perf_counter() - start_time
    epochs_ran = len(history)

    save_history(history, output_dir_p / "history.csv")
    save_metrics_json(
        {
            "best_val_dice": best_dice,
            "best_epoch": best_epoch,
            "epochs_ran": epochs_ran,
            "best_checkpoint": str(best_path) if best_saved else None,
            "checkpoint_saved": best_saved,
            "duration_sec": duration_sec,
            "avg_epoch_sec": duration_sec / max(1, epochs_ran),
        },
        output_dir_p / "metrics.json",
    )

    return best_path if best_saved else None
```

- [ ] **Step 6: Implement minimal runner changes**

Update the call site in `src/experiments/low_data_runner.py`:

```python
save_best_checkpoint = bool(config["train"].get("save_best_checkpoint", True))

return fit(
    model,
    train_loader,
    val_loader,
    optimizer,
    int(config["train"]["epochs"]),
    int(config["train"]["early_stopping_patience"]),
    output_dir,
    device=device,
    save_best_checkpoint=save_best_checkpoint,
)
```

- [ ] **Step 7: Run the focused tests to verify they pass**

Run:

```bash
python -m pytest tests/test_training_engine.py::test_fit_can_skip_best_checkpoint_and_still_write_timing_metrics tests/test_low_data_runner.py::test_run_group_passes_save_best_checkpoint_flag_to_engine -q
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/training/engine.py src/experiments/low_data_runner.py tests/test_training_engine.py tests/test_low_data_runner.py
git commit -m "feat: add configurable checkpoint saving and timing metrics"
```

## Task 2: Introduce A Dataset Factory So Low-Data Runs Are Not ISIC-Hardcoded

**Files:**
- Create: `src/data/factory.py`
- Modify: `src/experiments/low_data_runner.py`
- Create: `tests/test_data_factory.py`
- Modify: `tests/test_low_data_runner.py`

- [ ] **Step 1: Add failing tests for dataset-name resolution**

Create `tests/test_data_factory.py` with:

```python
from __future__ import annotations

from pathlib import Path

import pytest


def test_resolve_dataset_spec_supports_isic2018():
    from src.data.factory import resolve_dataset_spec

    spec = resolve_dataset_spec(
        {
            "data": {"dataset_name": "isic2018", "image_size": 256},
            "paths": {
                "train_images_dir": "train/images",
                "train_masks_dir": "train/masks",
                "val_images_dir": "val/images",
                "val_masks_dir": "val/masks",
            },
        }
    )

    assert spec.dataset_name == "isic2018"
    assert spec.class_values == {"background": 0, "lesion": 1}
    assert spec.train_images_dir == Path("train/images")


def test_resolve_dataset_spec_rejects_unknown_dataset_name():
    from src.data.factory import resolve_dataset_spec

    with pytest.raises(ValueError, match="dataset_name"):
        resolve_dataset_spec(
            {
                "data": {"dataset_name": "unknown_ds", "image_size": 256},
                "paths": {
                    "train_images_dir": "train/images",
                    "train_masks_dir": "train/masks",
                    "val_images_dir": "val/images",
                    "val_masks_dir": "val/masks",
                },
            }
        )
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
python -m pytest tests/test_data_factory.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'src.data.factory'`.

- [ ] **Step 3: Implement dataset-spec resolution**

Create `src/data/factory.py` with:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DatasetSpec:
    dataset_name: str
    image_size: int
    train_images_dir: Path
    train_masks_dir: Path
    val_images_dir: Path
    val_masks_dir: Path
    class_values: dict[str, int]


def resolve_dataset_spec(config: dict[str, Any]) -> DatasetSpec:
    dataset_name = str(config["data"].get("dataset_name", "isic2018")).lower()
    if dataset_name not in {"isic2018", "glas"}:
        raise ValueError(
            "config.data.dataset_name must be one of ['isic2018', 'glas']; "
            f"got {dataset_name!r}."
        )

    class_values = {"background": 0, "lesion": 1}
    if dataset_name == "glas":
        class_values = {"background": 0, "gland": 1}

    return DatasetSpec(
        dataset_name=dataset_name,
        image_size=int(config["data"]["image_size"]),
        train_images_dir=Path(config["paths"]["train_images_dir"]),
        train_masks_dir=Path(config["paths"]["train_masks_dir"]),
        val_images_dir=Path(config["paths"]["val_images_dir"]),
        val_masks_dir=Path(config["paths"]["val_masks_dir"]),
        class_values=class_values,
    )
```

- [ ] **Step 4: Add a failing test for dataset construction**

Append to `tests/test_data_factory.py`:

```python
def test_build_low_data_datasets_builds_selected_train_and_val(monkeypatch):
    from src.data.factory import build_low_data_datasets

    calls: dict[str, object] = {}

    class _FakeDataset:
        def __init__(self, *, images_dir, masks_dir, image_size, class_values, sample_ids=None):
            calls.setdefault("datasets", []).append(
                {
                    "images_dir": str(images_dir),
                    "masks_dir": str(masks_dir),
                    "image_size": image_size,
                    "class_values": dict(class_values),
                    "sample_ids": None if sample_ids is None else list(sample_ids),
                }
            )
            stems = ["a", "b", "c"] if sample_ids is None else list(sample_ids)
            self.image_paths = [type("_P", (), {"stem": stem})() for stem in stems]

    monkeypatch.setattr("src.data.factory.ISIC2018Dataset", _FakeDataset)
    monkeypatch.setattr("src.data.factory.build_ratio_subset", lambda sample_ids, ratio, seed: ["b"])

    built = build_low_data_datasets(
        {
            "seed": 5,
            "data": {"dataset_name": "isic2018", "image_size": 256, "train_ratio": 0.1},
            "paths": {
                "train_images_dir": "train/images",
                "train_masks_dir": "train/masks",
                "val_images_dir": "val/images",
                "val_masks_dir": "val/masks",
            },
        }
    )

    assert built["selected_ids"] == ["b"]
    assert len(calls["datasets"]) == 3
```

- [ ] **Step 5: Run the focused tests to verify they fail**

Run:

```bash
python -m pytest tests/test_data_factory.py::test_build_low_data_datasets_builds_selected_train_and_val -q
```

Expected: FAIL because `build_low_data_datasets` does not exist.

- [ ] **Step 6: Implement dataset construction in the factory**

Extend `src/data/factory.py`:

```python
from src.data.isic2018 import ISIC2018Dataset
from src.data.splits import build_ratio_subset

try:
    from src.data.glas import GlaSDataset
except ModuleNotFoundError:
    GlaSDataset = None


def _dataset_cls_for_name(dataset_name: str):
    if dataset_name == "isic2018":
        return ISIC2018Dataset
    if dataset_name == "glas":
        if GlaSDataset is None:
            raise RuntimeError("GlaSDataset is unavailable.")
        return GlaSDataset
    raise ValueError(f"Unsupported dataset_name: {dataset_name!r}")


def build_low_data_datasets(config: dict[str, Any]) -> dict[str, Any]:
    spec = resolve_dataset_spec(config)
    dataset_cls = _dataset_cls_for_name(spec.dataset_name)

    full_train_dataset = dataset_cls(
        images_dir=spec.train_images_dir,
        masks_dir=spec.train_masks_dir,
        image_size=spec.image_size,
        class_values=spec.class_values,
        sample_ids=None,
    )
    selected_ids = build_ratio_subset(
        [path.stem for path in full_train_dataset.image_paths],
        ratio=float(config["data"]["train_ratio"]),
        seed=int(config["seed"]),
    )
    train_dataset = dataset_cls(
        images_dir=spec.train_images_dir,
        masks_dir=spec.train_masks_dir,
        image_size=spec.image_size,
        class_values=spec.class_values,
        sample_ids=selected_ids,
    )
    val_dataset = dataset_cls(
        images_dir=spec.val_images_dir,
        masks_dir=spec.val_masks_dir,
        image_size=spec.image_size,
        class_values=spec.class_values,
        sample_ids=None,
    )
    return {
        "spec": spec,
        "full_train_dataset": full_train_dataset,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "selected_ids": selected_ids,
    }
```

- [ ] **Step 7: Switch the runner to the factory**

Replace the dataset block in `src/experiments/low_data_runner.py` with:

```python
from src.data.factory import build_low_data_datasets
from src.data.splits import save_split_manifest
from src.models.segmentation_model import build_segmentation_model
from src.training.engine import fit

datasets = build_low_data_datasets(config)
spec = datasets["spec"]
full_train_dataset = datasets["full_train_dataset"]
train_dataset = datasets["train_dataset"]
val_dataset = datasets["val_dataset"]
selected_ids = datasets["selected_ids"]
```

- [ ] **Step 8: Run the focused tests to verify they pass**

Run:

```bash
python -m pytest tests/test_data_factory.py tests/test_low_data_runner.py::test_run_group_passes_save_best_checkpoint_flag_to_engine -q
```

Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add src/data/factory.py src/experiments/low_data_runner.py tests/test_data_factory.py tests/test_low_data_runner.py
git commit -m "feat: add low-data dataset factory"
```

## Task 3: Add A First-Class GlaS Dataset Loader

**Files:**
- Create: `src/data/glas.py`
- Create: `tests/test_glas_dataset.py`
- Modify: `tests/test_data_factory.py`

- [ ] **Step 1: Add failing dataset tests for GlaS**

Create `tests/test_glas_dataset.py` with:

```python
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def _write_pair(images_dir: Path, masks_dir: Path, sample_id: str) -> None:
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:6, 2:6] = 255
    Image.fromarray(image).save(images_dir / f"{sample_id}.png")
    Image.fromarray(mask).save(masks_dir / f"{sample_id}_anno.bmp")


def test_glas_dataset_returns_expected_sample(tmp_path: Path):
    from src.data.glas import GlaSDataset

    images_dir = tmp_path / "images"
    masks_dir = tmp_path / "masks"
    _write_pair(images_dir, masks_dir, "train_001")

    dataset = GlaSDataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        image_size=16,
        class_values={"background": 0, "gland": 1},
    )

    sample = dataset[0]
    assert sample["sample_id"] == "train_001"
    assert tuple(sample["image"].shape) == (3, 16, 16)
    assert tuple(sample["mask"].shape) == (16, 16)
    assert int(sample["mask"].max()) == 1


def test_glas_dataset_supports_sample_id_filter(tmp_path: Path):
    from src.data.glas import GlaSDataset

    images_dir = tmp_path / "images"
    masks_dir = tmp_path / "masks"
    _write_pair(images_dir, masks_dir, "train_001")
    _write_pair(images_dir, masks_dir, "train_002")

    dataset = GlaSDataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        image_size=16,
        class_values={"background": 0, "gland": 1},
        sample_ids=["train_002"],
    )

    assert len(dataset) == 1
    assert dataset[0]["sample_id"] == "train_002"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
python -m pytest tests/test_glas_dataset.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'src.data.glas'`.

- [ ] **Step 3: Implement `GlaSDataset` with the same sample contract as ISIC**

Create `src/data/glas.py` with:

```python
from __future__ import annotations

from collections.abc import Set
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class GlaSDataset(Dataset):
    def __init__(self, images_dir, masks_dir, image_size, class_values, sample_ids=None):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.image_size = int(image_size)
        self.class_values = class_values

        required_names = {"background", "gland"}
        provided_names = set(class_values.keys())
        if provided_names != required_names:
            raise ValueError(
                f"GlaSDataset requires class_values to be exactly {sorted(required_names)}; "
                f"got {sorted(provided_names)}"
            )
        if class_values.get("background") != 0 or class_values.get("gland") != 1:
            raise ValueError(
                "GlaSDataset currently requires 'background': 0 and 'gland': 1 mappings."
            )

        all_image_paths = sorted(list(self.images_dir.glob("*.bmp")) + list(self.images_dir.glob("*.png")))
        if sample_ids is None:
            self.image_paths = all_image_paths
            return

        if isinstance(sample_ids, (str, bytes)):
            requested_ids = [sample_ids]
        elif isinstance(sample_ids, Set):
            requested_ids = sorted(sample_ids)
        else:
            requested_ids = list(sample_ids)

        seen = set()
        deduped_ids = []
        for sample_id in requested_ids:
            if sample_id not in seen:
                seen.add(sample_id)
                deduped_ids.append(sample_id)

        by_stem = {path.stem: path for path in all_image_paths}
        missing = [sample_id for sample_id in deduped_ids if sample_id not in by_stem]
        if missing:
            raise ValueError(
                "Requested sample_ids are missing from images_dir: " + ", ".join(missing)
            )
        self.image_paths = [by_stem[sample_id] for sample_id in deduped_ids]

    def _resolve_mask_path(self, sample_id: str) -> Path:
        candidates = [
            self.masks_dir / f"{sample_id}_anno.bmp",
            self.masks_dir / f"{sample_id}.png",
            self.masks_dir / f"{sample_id}.bmp",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            f"Missing mask for sample {sample_id}. Checked: "
            + ", ".join(str(path) for path in candidates)
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        sample_id = image_path.stem
        mask_path = self._resolve_mask_path(sample_id)

        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB").resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        with Image.open(mask_path) as mask_file:
            mask = mask_file.convert("L").resize((self.image_size, self.image_size), Image.Resampling.NEAREST)

        image_np = np.asarray(image, dtype=np.float32) / 255.0
        mask_np = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8)

        class_presence = {
            class_name: bool((mask_np == class_value).any())
            for class_name, class_value in self.class_values.items()
        }

        return {
            "sample_id": sample_id,
            "image": torch.from_numpy(image_np).permute(2, 0, 1),
            "mask": torch.from_numpy(mask_np).long(),
            "class_presence": class_presence,
        }
```

- [ ] **Step 4: Add a failing factory integration test for GlaS**

Append to `tests/test_data_factory.py`:

```python
def test_build_low_data_datasets_uses_glas_dataset(monkeypatch):
    from src.data.factory import build_low_data_datasets

    calls: list[dict[str, object]] = []

    class _FakeGlaS:
        def __init__(self, *, images_dir, masks_dir, image_size, class_values, sample_ids=None):
            calls.append(
                {
                    "images_dir": str(images_dir),
                    "masks_dir": str(masks_dir),
                    "class_values": dict(class_values),
                    "sample_ids": None if sample_ids is None else list(sample_ids),
                }
            )
            stems = ["g1", "g2"] if sample_ids is None else list(sample_ids)
            self.image_paths = [type("_P", (), {"stem": stem})() for stem in stems]

    monkeypatch.setattr("src.data.factory.GlaSDataset", _FakeGlaS)
    monkeypatch.setattr("src.data.factory.build_ratio_subset", lambda sample_ids, ratio, seed: ["g2"])

    built = build_low_data_datasets(
        {
            "seed": 11,
            "data": {"dataset_name": "glas", "image_size": 256, "train_ratio": 0.5},
            "paths": {
                "train_images_dir": "glas/train/images",
                "train_masks_dir": "glas/train/masks",
                "val_images_dir": "glas/val/images",
                "val_masks_dir": "glas/val/masks",
            },
        }
    )

    assert built["selected_ids"] == ["g2"]
    assert calls[0]["class_values"] == {"background": 0, "gland": 1}
```

- [ ] **Step 5: Run the focused tests to verify they fail**

Run:

```bash
python -m pytest tests/test_data_factory.py::test_build_low_data_datasets_uses_glas_dataset -q
```

Expected: FAIL because `src.data.factory` does not yet import `GlaSDataset`.

- [ ] **Step 6: Wire GlaS into the factory and rerun**

Run:

```bash
python -m pytest tests/test_glas_dataset.py tests/test_data_factory.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/data/glas.py src/data/factory.py tests/test_glas_dataset.py tests/test_data_factory.py
git commit -m "feat: add GlaS dataset support"
```

## Task 4: Add GlaS Low-Data Experiment Configs And CLI Coverage

**Files:**
- Create: `configs/experiments/glas_low_data_conv_b_base.yaml`
- Create: `configs/experiments/glas_low_data_node_c_base.yaml`
- Modify: `scripts/run_low_data_experiment.py`
- Modify: `tests/test_low_data_clis.py`

- [ ] **Step 1: Add a failing CLI wiring test using a GlaS config**

Append to `tests/test_low_data_clis.py`:

```python
def test_run_low_data_experiment_cli_handles_glas_config(tmp_path: Path, monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    experiments_dir = src_dir / "experiments"

    _ensure_package("src", src_dir)
    _ensure_package("src.experiments", experiments_dir)

    calls: dict[str, object] = {}
    runner_module = ModuleType("src.experiments.low_data_runner")

    def run_group(config_path, group):
        calls["run_group"] = {"config_path": Path(config_path), "group": group}
        return tmp_path / "artifacts" / "group_c" / "best.pt"

    runner_module.run_group = run_group
    monkeypatch.setitem(sys.modules, "src.experiments.low_data_runner", runner_module)

    module = _load_script_module("scripts.run_low_data_experiment", "scripts/run_low_data_experiment.py")
    glas_config = tmp_path / "glas_config.yaml"
    glas_config.write_text("seed: 1\n", encoding="utf-8")

    monkeypatch.setattr(
        sys,
        "argv",
        ["run_low_data_experiment.py", "--config", str(glas_config), "--group", "C"],
    )

    module.main()

    assert calls["run_group"] == {"config_path": glas_config, "group": "C"}
```

- [ ] **Step 2: Run the CLI test to verify it passes before config work**

Run:

```bash
python -m pytest tests/test_low_data_clis.py::test_run_low_data_experiment_cli_handles_glas_config -q
```

Expected: PASS. This confirms the current CLI already supports arbitrary config paths and we only need better config coverage, not a CLI redesign.

- [ ] **Step 3: Add config parse tests for GlaS experiment files**

Append to `tests/test_low_data_runner.py`:

```python
def test_glas_low_data_configs_parse_with_expected_dataset_name():
    from src.experiments.low_data_runner import load_config

    expected = {
        "configs/experiments/glas_low_data_conv_b_base.yaml": {"dataset_name": "glas", "group": "B"},
        "configs/experiments/glas_low_data_node_c_base.yaml": {"dataset_name": "glas", "group": "C"},
    }

    for path, values in expected.items():
        config = load_config(path)
        assert config["data"]["dataset_name"] == values["dataset_name"]
        assert config["experiment"]["group"] == values["group"]
```

- [ ] **Step 4: Run the config test to verify it fails**

Run:

```bash
python -m pytest tests/test_low_data_runner.py::test_glas_low_data_configs_parse_with_expected_dataset_name -q
```

Expected: FAIL because the config files do not exist yet.

- [ ] **Step 5: Create the first GlaS configs**

Create `configs/experiments/glas_low_data_conv_b_base.yaml`:

```yaml
seed: 42
experiment:
  name: glas_low_data_conv_b_base
  group: B
paths:
  train_images_dir: data/glas/train/images
  train_masks_dir: data/glas/train/masks
  val_images_dir: data/glas/val/images
  val_masks_dir: data/glas/val/masks
  artifacts_dir: artifacts/glas_low_data/conv_b_base_seed42
data:
  dataset_name: glas
  image_size: 256
  train_ratio: 0.1
  num_workers: 0
  pin_memory: false
train:
  batch_size: 4
  epochs: 40
  learning_rate: 0.0001
  weight_decay: 0.0001
  early_stopping_patience: 10
  save_best_checkpoint: false
model:
  encoder_name: resnet18
  encoder_weights: imagenet
  in_channels: 3
  num_classes: 1
  bottleneck_channels: 512
  freeze_encoder: true
adapter:
  hidden_channels: 256
  init: default
node:
  steps: 4
  step_size: 0.25
  solver: euler
```

Create `configs/experiments/glas_low_data_node_c_base.yaml`:

```yaml
seed: 42
experiment:
  name: glas_low_data_node_c_base
  group: C
paths:
  train_images_dir: data/glas/train/images
  train_masks_dir: data/glas/train/masks
  val_images_dir: data/glas/val/images
  val_masks_dir: data/glas/val/masks
  artifacts_dir: artifacts/glas_low_data/node_c_base_seed42
data:
  dataset_name: glas
  image_size: 256
  train_ratio: 0.1
  num_workers: 0
  pin_memory: false
train:
  batch_size: 4
  epochs: 40
  learning_rate: 0.0001
  weight_decay: 0.0001
  early_stopping_patience: 10
  save_best_checkpoint: false
model:
  encoder_name: resnet18
  encoder_weights: imagenet
  in_channels: 3
  num_classes: 1
  bottleneck_channels: 512
  freeze_encoder: true
adapter:
  hidden_channels: 256
  init: default
node:
  steps: 4
  step_size: 0.25
  solver: euler
```

- [ ] **Step 6: Update CLI help text to reflect multi-dataset low-data experiments**

In `scripts/run_low_data_experiment.py`, adjust the parser description to:

```python
parser = argparse.ArgumentParser(
    description="Run config-driven low-data segmentation experiments (groups A/B/C) across supported datasets."
)
```

- [ ] **Step 7: Run the focused tests to verify they pass**

Run:

```bash
python -m pytest tests/test_low_data_runner.py::test_glas_low_data_configs_parse_with_expected_dataset_name tests/test_low_data_clis.py::test_run_low_data_experiment_cli_handles_glas_config -q
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add configs/experiments/glas_low_data_conv_b_base.yaml configs/experiments/glas_low_data_node_c_base.yaml scripts/run_low_data_experiment.py tests/test_low_data_runner.py tests/test_low_data_clis.py
git commit -m "feat: add initial GlaS low-data experiment configs"
```

## Task 5: Keep Reporting Compatible With New Timing Metrics

**Files:**
- Modify: `src/analysis/report_visualization.py`
- Modify: `tests/test_report_visualization.py`

- [ ] **Step 1: Add a failing regression test for richer metrics JSON**

Append to `tests/test_report_visualization.py`:

```python
def test_summarize_run_ignores_extra_timing_metrics(tmp_path: Path) -> None:
    from src.analysis.report_visualization import summarize_run

    root = tmp_path / "exp" / "group_c"
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"epoch": 1, "train_loss": 1.0, "val_loss": 0.9, "val_dice": 0.70, "val_iou": 0.55},
            {"epoch": 2, "train_loss": 0.8, "val_loss": 0.7, "val_dice": 0.74, "val_iou": 0.58},
        ]
    ).to_csv(root / "history.csv", index=False)
    (root / "metrics.json").write_text(
        json.dumps(
            {
                "best_val_dice": 0.74,
                "best_epoch": 2,
                "epochs_ran": 2,
                "best_checkpoint": None,
                "checkpoint_saved": False,
                "duration_sec": 12.5,
                "avg_epoch_sec": 6.25,
            }
        ),
        encoding="utf-8",
    )

    row = summarize_run(root=root, method="Method", run="exp", seed=0)

    assert row["best_dice"] == 0.74
    assert row["epochs_ran"] == 2
```

- [ ] **Step 2: Run the test to verify current behavior**

Run:

```bash
python -m pytest tests/test_report_visualization.py::test_summarize_run_ignores_extra_timing_metrics -q
```

Expected: PASS. This is a lock-in test confirming richer metrics JSON remains backward-compatible.

- [ ] **Step 3: Extend report summaries to carry timing when present**

Update `src/analysis/report_visualization.py` inside `summarize_run()`:

```python
    duration_sec = metrics.get("duration_sec")
    avg_epoch_sec = metrics.get("avg_epoch_sec")

    return {
        "method": method,
        "run": run,
        "seed": seed,
        "group": root.name,
        "best_dice": best_dice,
        "best_iou": float(best_row["val_iou"]),
        "best_epoch": int(best_row["epoch"]),
        "final_dice": float(final_row["val_dice"]),
        "peak_final_gap": best_dice - float(final_row["val_dice"]),
        "epochs_ran": int(metrics.get("epochs_ran", len(history))),
        "duration_sec": None if duration_sec is None else float(duration_sec),
        "avg_epoch_sec": None if avg_epoch_sec is None else float(avg_epoch_sec),
        "root": str(root),
    }
```

Update the empty-table columns and grouped summary:

```python
            "duration_sec",
            "avg_epoch_sec",
```

and:

```python
            duration_sec_mean=("duration_sec", "mean"),
            avg_epoch_sec_mean=("avg_epoch_sec", "mean"),
```

- [ ] **Step 4: Run focused reporting tests**

Run:

```bash
python -m pytest tests/test_report_visualization.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/analysis/report_visualization.py tests/test_report_visualization.py
git commit -m "feat: carry timing metrics through report summaries"
```

## Task 6: Full Verification And Stage-1 Handoff

**Files:**
- No source edits in this task.

- [ ] **Step 1: Run the focused dataset and runner suite**

Run:

```bash
python -m pytest tests/test_training_engine.py tests/test_glas_dataset.py tests/test_data_factory.py tests/test_low_data_runner.py tests/test_low_data_clis.py tests/test_report_visualization.py -q
```

Expected: PASS.

- [ ] **Step 2: Run the full test suite**

Run:

```bash
python -m pytest -q
```

Expected: PASS with only any previously accepted environment warnings.

- [ ] **Step 3: Check CLI help text**

Run:

```bash
python scripts/run_low_data_experiment.py --help
```

Expected: exit code 0 and description text mentions supported datasets.

- [ ] **Step 4: Smoke-check the new configs parse**

Run:

```bash
python - <<'PY'
from src.experiments.low_data_runner import load_config
for path in [
    "configs/experiments/glas_low_data_conv_b_base.yaml",
    "configs/experiments/glas_low_data_node_c_base.yaml",
]:
    cfg = load_config(path)
    print(path, cfg["data"]["dataset_name"], cfg["train"]["save_best_checkpoint"])
PY
```

Expected output:

```text
configs/experiments/glas_low_data_conv_b_base.yaml glas False
configs/experiments/glas_low_data_node_c_base.yaml glas False
```

- [ ] **Step 5: Document the stage-2 decision gate in the commit message or PR body**

Include this exact note in the final integration summary:

```text
Stage 1 stops after checkpoint/timing infrastructure and first GlaS support.
The next plan must choose external comparison methods before any solver/regularizer tuning or multi-seed expansion.
```

## Recommended Execution Order After This Plan

After implementation, the experimental progression should be:

1. run quick ISIC sanity checks with `save_best_checkpoint: false`
2. run first GlaS baseline and NODE configs
3. inspect Dice, timing, and training stability
4. only then write the follow-up plan for external comparison methods

## Self-Review

- Spec coverage: This plan covers the approved stage-1 scope: checkpoint `.pt` control, training-time logging, and one new dataset (`GlaS`). It intentionally excludes external comparison methods and later tuning because those are not yet concretely specified.
- Placeholder scan: No `TODO`, `TBD`, “similar to above”, or content-free “add tests” steps remain.
- Type consistency: The plan consistently uses `save_best_checkpoint`, `duration_sec`, `avg_epoch_sec`, `best_epoch`, `DatasetSpec`, `resolve_dataset_spec`, `build_low_data_datasets`, and `GlaSDataset` across tests and implementation steps.
