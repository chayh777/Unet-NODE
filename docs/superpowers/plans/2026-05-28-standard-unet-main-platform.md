# Standard U-Net Main Platform Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate the active experiment platform from the current no-skip lightweight decoder to a true skip-connected standard U-Net while preserving bottleneck/output adapter experiments, training scripts, and report semantics.

**Architecture:** Keep the existing encoder + adapter abstractions, but replace the current single-feature tiny decoder with a U-Net-style decoder that consumes the full encoder feature pyramid. Preserve the current `SegmentationModelOutput` contract so training, geometry export, and report code keep working, while introducing an explicit architecture switch so legacy runs remain reproducible during migration.

**Tech Stack:** PyTorch, timm `features_only` encoders, YAML experiment configs, pytest

---

## File Structure

### New files
- `src/models/unet_decoder.py`
  - Focused decoder building blocks for the new standard U-Net path: upsample block, skip fusion block, and final decoder wrapper.
- `tests/test_unet_decoder.py`
  - Decoder-focused shape and skip-path tests that do not depend on the full experiment runner.
- `configs/experiments/isic2018_low_data_node_standard_unet.yaml`
  - Canonical ISIC standard-U-Net baseline config to avoid mutating every old config before migration is validated.
- `configs/experiments/glas_low_data_node_standard_unet.yaml`
  - Canonical GlaS standard-U-Net baseline config.

### Modified files
- `src/models/segmentation_model.py`
  - Add architecture selection, expose encoder feature pyramid to the decoder, preserve bottleneck/output adapter hooks, and keep the output contract stable.
- `src/experiments/low_data_runner.py`
  - Validate and resolve the new architecture field, keep defaults explicit, and preserve backward-compatible config loading.
- `tests/test_segmentation_model.py`
  - Replace “plain U-Net” assumptions that currently point at the legacy decoder with tests for the standard-U-Net path.
- `tests/test_low_data_runner.py`
  - Validate config parsing for the new architecture field and the new standard-U-Net configs.
- `docs/experiments/2026-05-28-glas-experiment-matrix.md`
  - Update the matrix to reflect that future GlaS comparison/tuning runs must use the standard-U-Net platform.

### Existing files to reference while implementing
- `src/models/adapters.py`
- `src/models/node_adapter.py`
- `scripts/run_low_data_experiment.py`

---

### Task 1: Lock the migration target with failing model tests

**Files:**
- Modify: `tests/test_segmentation_model.py`
- Create: `tests/test_unet_decoder.py`
- Test: `tests/test_segmentation_model.py`, `tests/test_unet_decoder.py`

- [ ] **Step 1: Write the failing decoder-level shape test**

Add this new file:

```python
from importlib import util
from pathlib import Path
from types import ModuleType
import hashlib
import sys

import torch


def _ensure_package(name: str, package_path: Path) -> None:
    if name in sys.modules:
        return
    pkg = ModuleType(name)
    pkg.__path__ = [str(package_path)]  # type: ignore[attr-defined]
    sys.modules[name] = pkg


def _unique_package_name(base_path: Path) -> str:
    digest = hashlib.sha1(str(base_path).encode("utf-8")).hexdigest()[:10]
    return f"_task_std_unet_{digest}"


def _load_models_module(module_filename: str, fqname: str):
    module_path = Path(__file__).resolve().parents[1] / "src" / "models" / module_filename
    spec = util.spec_from_file_location(fqname, module_path)
    assert spec is not None
    module = util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[fqname] = module
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


_MODELS_DIR = Path(__file__).resolve().parents[1] / "src" / "models"
_PKG = _unique_package_name(_MODELS_DIR)
_ensure_package(_PKG, _MODELS_DIR)
_decoder_module = _load_models_module("unet_decoder.py", f"{_PKG}.unet_decoder")
StandardUNetDecoder = _decoder_module.StandardUNetDecoder


def test_standard_unet_decoder_uses_full_feature_pyramid_and_restores_stride_2_resolution():
    decoder = StandardUNetDecoder(
        encoder_channels=[64, 128, 256, 512],
        bottleneck_channels=128,
        output_channels=32,
    )
    features = [
        torch.randn(2, 64, 32, 32),
        torch.randn(2, 128, 16, 16),
        torch.randn(2, 256, 8, 8),
        torch.randn(2, 512, 4, 4),
    ]

    decoded = decoder(features[-1], skip_features=features[:-1][::-1])

    assert decoded.shape == (2, 32, 32, 32)
```

- [ ] **Step 2: Write the failing full-model standard-U-Net tests**

Append these tests to `tests/test_segmentation_model.py`:

```python
def test_standard_unet_plain_path_restores_input_resolution_and_emits_skip_aware_decoder_output():
    model = build_segmentation_model(
        encoder_name="resnet18",
        encoder_weights=None,
        in_channels=3,
        num_classes=1,
        adapter_type="none",
        adapter_placement="bottleneck",
        bottleneck_channels=64,
        adapter_hidden_channels=32,
        freeze_encoder=True,
        node_steps=4,
        node_step_size=0.25,
        architecture="standard_unet",
    )
    x = torch.randn(2, 3, 64, 64)

    output = model(x)

    assert output.logits.shape == (2, 1, 64, 64)
    assert output.bottleneck.shape == (2, 64, 4, 4)
    assert output.adapted_bottleneck.shape == output.bottleneck.shape
    assert output.output_adapter_activation is None


def test_standard_unet_output_side_adapter_attaches_after_skip_decoder():
    model = build_segmentation_model(
        encoder_name="resnet18",
        encoder_weights=None,
        in_channels=3,
        num_classes=2,
        adapter_type="conv",
        adapter_placement="output",
        bottleneck_channels=64,
        adapter_hidden_channels=32,
        freeze_encoder=True,
        node_steps=4,
        node_step_size=0.25,
        architecture="standard_unet",
    )
    x = torch.randn(2, 3, 64, 64)

    output = model(x)

    assert output.logits.shape == (2, 2, 64, 64)
    assert output.output_adapter_activation is not None
    assert output.output_adapter_activation.shape[-2:] == (32, 32)
```

- [ ] **Step 3: Run tests to verify they fail**

Run:

```bash
pytest tests/test_unet_decoder.py tests/test_segmentation_model.py -q
```

Expected:

```text
FAIL tests/test_unet_decoder.py::test_standard_unet_decoder_uses_full_feature_pyramid_and_restores_stride_2_resolution
FAIL tests/test_segmentation_model.py::test_standard_unet_plain_path_restores_input_resolution_and_emits_skip_aware_decoder_output
```

- [ ] **Step 4: Commit the failing tests**

```bash
git add tests/test_unet_decoder.py tests/test_segmentation_model.py
git commit -m "test: define standard unet migration expectations"
```

---

### Task 2: Implement a reusable standard U-Net decoder

**Files:**
- Create: `src/models/unet_decoder.py`
- Test: `tests/test_unet_decoder.py`

- [ ] **Step 1: Write the minimal decoder implementation**

Create `src/models/unet_decoder.py` with:

```python
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class UNetUpBlock(nn.Module):
    def __init__(self, *, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.fuse(x)


class StandardUNetDecoder(nn.Module):
    def __init__(
        self,
        *,
        encoder_channels: list[int],
        bottleneck_channels: int,
        output_channels: int,
    ) -> None:
        super().__init__()
        if len(encoder_channels) < 2:
            raise ValueError("StandardUNetDecoder requires at least two encoder stages.")

        skip_channels = list(encoder_channels[:-1])[::-1]
        block_out_channels = [max(bottleneck_channels // 2, output_channels), max(bottleneck_channels // 4, output_channels), output_channels]
        block_out_channels = block_out_channels[: len(skip_channels)]

        blocks = []
        in_channels = bottleneck_channels
        for idx, skip_ch in enumerate(skip_channels):
            out_ch = block_out_channels[idx]
            blocks.append(
                UNetUpBlock(
                    in_channels=in_channels,
                    skip_channels=skip_ch,
                    out_channels=out_ch,
                )
            )
            in_channels = out_ch
        self.blocks = nn.ModuleList(blocks)

    def forward(self, bottleneck: torch.Tensor, *, skip_features: list[torch.Tensor]) -> torch.Tensor:
        if len(skip_features) != len(self.blocks):
            raise ValueError(
                f"Expected {len(self.blocks)} skip features, got {len(skip_features)}."
            )
        x = bottleneck
        for block, skip in zip(self.blocks, skip_features):
            x = block(x, skip)
        return x
```

- [ ] **Step 2: Run decoder tests**

Run:

```bash
pytest tests/test_unet_decoder.py -q
```

Expected:

```text
1 passed
```

- [ ] **Step 3: Commit the decoder module**

```bash
git add src/models/unet_decoder.py tests/test_unet_decoder.py
git commit -m "feat: add reusable standard unet decoder"
```

---

### Task 3: Wire the segmentation model to the standard-U-Net path

**Files:**
- Modify: `src/models/segmentation_model.py`
- Test: `tests/test_segmentation_model.py`

- [ ] **Step 1: Extend the model signature with architecture selection**

Update the type aliases and constructor signature in `src/models/segmentation_model.py`:

```python
AdapterType = Literal["none", "conv", "node"]
AdapterPlacement = Literal["bottleneck", "output"]
ModelArchitecture = Literal["legacy_no_skip", "standard_unet"]
```

And in `SegmentationModel.__init__`:

```python
        adapter_placement: AdapterPlacement = "bottleneck",
        node_solver: str = "euler",
        adapter_init: AdapterInit = "default",
        architecture: ModelArchitecture = "standard_unet",
```

Add validation:

```python
        allowed_architectures = {"legacy_no_skip", "standard_unet"}
        if architecture not in allowed_architectures:
            raise ValueError(
                f"architecture must be one of {allowed_architectures}; got {architecture}"
            )
        self.architecture = architecture
```

- [ ] **Step 2: Replace the hard-coded tiny decoder with a branchable decoder path**

Import the new decoder:

```python
from .unet_decoder import StandardUNetDecoder
```

Replace the existing decoder construction block with:

```python
        self.bottleneck_proj = nn.Conv2d(
            encoder_channels[-1], bottleneck_channels, kernel_size=1
        )

        mid1 = bottleneck_channels // 2
        mid2 = bottleneck_channels // 4
        if mid1 <= 0 or mid2 <= 0:
            raise ValueError(
                "bottleneck_channels must be >= 4 so decoder channels stay positive."
            )

        if architecture == "legacy_no_skip":
            decoder_output_channels = mid2
            self.decoder = nn.Sequential(
                nn.Conv2d(bottleneck_channels, mid1, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid1, mid2, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )
        else:
            decoder_output_channels = encoder_channels[0]
            self.decoder = StandardUNetDecoder(
                encoder_channels=encoder_channels,
                bottleneck_channels=bottleneck_channels,
                output_channels=decoder_output_channels,
            )
```

Keep output adapter creation but use `decoder_output_channels` instead of `mid2`:

```python
            self.output_adapter = self._build_adapter(
                adapter_type=adapter_type,
                channels=decoder_output_channels,
                hidden_channels=adapter_hidden_channels,
                node_steps=node_steps,
                node_step_size=node_step_size,
                node_solver=node_solver,
                adapter_init=adapter_init,
            )
```

Update the head:

```python
        self.head = nn.Conv2d(decoder_output_channels, num_classes, kernel_size=1)
```

- [ ] **Step 3: Update the forward pass to consume the full feature pyramid**

Replace the current decoder portion in `forward` with:

```python
        features = self.encoder(x)
        bottleneck = self.bottleneck_proj(features[-1])
        adapted_bottleneck = self.adapter(bottleneck)

        if self.architecture == "legacy_no_skip":
            decoded = self.decoder(adapted_bottleneck)
        else:
            decoded = self.decoder(
                adapted_bottleneck,
                skip_features=list(features[:-1])[::-1],
            )

        decoded = self.output_adapter(decoded)
```

- [ ] **Step 4: Run model tests**

Run:

```bash
pytest tests/test_segmentation_model.py -q
```

Expected:

```text
all tests pass
```

- [ ] **Step 5: Commit the model migration**

```bash
git add src/models/segmentation_model.py tests/test_segmentation_model.py
git commit -m "feat: add standard unet architecture path"
```

---

### Task 4: Make the experiment runner understand architecture and add standard-U-Net configs

**Files:**
- Modify: `src/experiments/low_data_runner.py`
- Modify: `tests/test_low_data_runner.py`
- Create: `configs/experiments/isic2018_low_data_node_standard_unet.yaml`
- Create: `configs/experiments/glas_low_data_node_standard_unet.yaml`

- [ ] **Step 1: Add architecture resolution and validation**

In `src/experiments/low_data_runner.py`, add:

```python
def _resolve_model_architecture(config: dict[str, Any]) -> str:
    model = config.get("model", {})
    if not isinstance(model, dict):
        return "standard_unet"
    value = model.get("architecture", "standard_unet")
    if value not in {"standard_unet", "legacy_no_skip"}:
        raise ValueError(
            "config.model.architecture must be one of ['standard_unet', 'legacy_no_skip']; "
            f"got {value!r}."
        )
    return str(value)
```

And pass it into `build_segmentation_model(...)`:

```python
        architecture=_resolve_model_architecture(config),
```

- [ ] **Step 2: Add regression tests for the new config field**

Append to `tests/test_low_data_runner.py`:

```python
def test_resolve_model_architecture_defaults_to_standard_unet():
    module = _load_low_data_runner_module()
    assert module._resolve_model_architecture({}) == "standard_unet"


def test_resolve_model_architecture_rejects_unknown_value():
    module = _load_low_data_runner_module()
    try:
        module._resolve_model_architecture({"model": {"architecture": "weird"}})
    except ValueError as exc:
        assert "config.model.architecture" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unknown architecture.")
```

- [ ] **Step 3: Add canonical standard-U-Net experiment configs**

Create `configs/experiments/isic2018_low_data_node_standard_unet.yaml`:

```yaml
seed: 42

paths:
  train_images_dir: data/train/images
  train_masks_dir: data/train/labels
  val_images_dir: data/validation/images
  val_masks_dir: data/validation/labels
  artifacts_dir: artifacts/isic2018_standard_unet

data:
  dataset_name: isic2018
  image_size: 256
  train_ratio: 0.1
  num_workers: 0

train:
  batch_size: 8
  epochs: 50
  learning_rate: 0.0003
  weight_decay: 0.0001
  early_stopping_patience: 8
  save_best_checkpoint: false

model:
  architecture: standard_unet
  encoder_name: resnet34
  encoder_weights: imagenet
  in_channels: 3
  num_classes: 1
  bottleneck_channels: 256
  freeze_encoder: true

adapter:
  hidden_channels: 256

node:
  steps: 8
  step_size: 0.125
```

Create `configs/experiments/glas_low_data_node_standard_unet.yaml`:

```yaml
seed: 42

paths:
  train_images_dir: data/glas/train/images
  train_masks_dir: data/glas/train/masks
  val_images_dir: data/glas/val/images
  val_masks_dir: data/glas/val/masks
  artifacts_dir: artifacts/glas_standard_unet

data:
  dataset_name: glas
  image_size: 256
  train_ratio: 1.0
  num_workers: 0

train:
  batch_size: 4
  epochs: 50
  learning_rate: 0.0003
  weight_decay: 0.0001
  early_stopping_patience: 8
  save_best_checkpoint: false

model:
  architecture: standard_unet
  encoder_name: resnet34
  encoder_weights: imagenet
  in_channels: 3
  num_classes: 1
  bottleneck_channels: 256
  freeze_encoder: true

adapter:
  hidden_channels: 256

node:
  steps: 8
  step_size: 0.125
```

- [ ] **Step 4: Run runner/config tests**

Run:

```bash
pytest tests/test_low_data_runner.py -q
```

Expected:

```text
all tests pass
```

- [ ] **Step 5: Commit the config/runner work**

```bash
git add src/experiments/low_data_runner.py tests/test_low_data_runner.py configs/experiments/isic2018_low_data_node_standard_unet.yaml configs/experiments/glas_low_data_node_standard_unet.yaml
git commit -m "feat: add standard unet experiment configs"
```

---

### Task 5: Update experiment documentation and promote the standard-U-Net platform

**Files:**
- Modify: `docs/experiments/2026-05-28-glas-experiment-matrix.md`
- Modify: `tests/test_low_data_runner.py`
- Test: `tests/test_low_data_runner.py`, `tests/test_segmentation_model.py`, `tests/test_unet_decoder.py`

- [ ] **Step 1: Update the GlaS matrix doc so future runs use the new platform**

In `docs/experiments/2026-05-28-glas-experiment-matrix.md`, add a short migration note near the top:

```md
> Migration note (2026-05-28): GlaS comparison and tuning runs should use the `standard_unet` architecture path. The previous no-skip decoder remains available only for reproducing legacy controlled-study results and should not be treated as the main U-Net baseline.
```

- [ ] **Step 2: Add smoke tests that the new configs parse with the standard architecture**

Append to `tests/test_low_data_runner.py`:

```python
def test_standard_unet_configs_parse_with_expected_architecture():
    module = _load_low_data_runner_module()
    for path in [
        "configs/experiments/isic2018_low_data_node_standard_unet.yaml",
        "configs/experiments/glas_low_data_node_standard_unet.yaml",
    ]:
        cfg = module.load_config(path)
        assert cfg["model"]["architecture"] == "standard_unet"
```

- [ ] **Step 3: Run the focused migration suite**

Run:

```bash
pytest tests/test_unet_decoder.py tests/test_segmentation_model.py tests/test_low_data_runner.py -q
```

Expected:

```text
all tests pass
```

- [ ] **Step 4: Commit the documentation and smoke coverage**

```bash
git add docs/experiments/2026-05-28-glas-experiment-matrix.md tests/test_low_data_runner.py
git commit -m "docs: promote standard unet as main experiment platform"
```

---

### Task 6: End-to-end verification before implementation handoff

**Files:**
- Verify only: `src/models/segmentation_model.py`, `src/models/unet_decoder.py`, `src/experiments/low_data_runner.py`, `configs/experiments/isic2018_low_data_node_standard_unet.yaml`, `configs/experiments/glas_low_data_node_standard_unet.yaml`

- [ ] **Step 1: Run the full model/runner test suite**

Run:

```bash
pytest tests/test_segmentation_model.py tests/test_unet_decoder.py tests/test_low_data_runner.py tests/test_training_engine.py tests/test_report_visualization.py -q
```

Expected:

```text
all tests pass
```

- [ ] **Step 2: Smoke-check both canonical configs through the CLI**

Run:

```bash
python - <<'PY'
from src.experiments.low_data_runner import load_config
for path in [
    "configs/experiments/isic2018_low_data_node_standard_unet.yaml",
    "configs/experiments/glas_low_data_node_standard_unet.yaml",
]:
    cfg = load_config(path)
    print(path, cfg["model"]["architecture"], cfg["paths"]["artifacts_dir"])
PY
```

Expected:

```text
configs/experiments/isic2018_low_data_node_standard_unet.yaml standard_unet artifacts/isic2018_standard_unet
configs/experiments/glas_low_data_node_standard_unet.yaml standard_unet artifacts/glas_standard_unet
```

- [ ] **Step 3: Final commit**

```bash
git add src/models/segmentation_model.py src/models/unet_decoder.py src/experiments/low_data_runner.py tests/test_segmentation_model.py tests/test_unet_decoder.py tests/test_low_data_runner.py docs/experiments/2026-05-28-glas-experiment-matrix.md configs/experiments/isic2018_low_data_node_standard_unet.yaml configs/experiments/glas_low_data_node_standard_unet.yaml
git commit -m "feat: migrate experiments to standard unet platform"
```

---

## Self-Review

### Spec coverage
- Standard U-Net as the unified main platform: covered by Tasks 2-4.
- Preserve bottleneck/output adapter experiments: covered by Task 3.
- Keep runner/config/report interfaces stable enough for migration: covered by Tasks 3-4.
- Make GlaS/ISIC future experiments use the new platform: covered by Tasks 4-5.

### Placeholder scan
- No `TODO`, `TBD`, or “implement later” placeholders remain.
- Each code-changing step contains concrete code blocks or exact config contents.

### Type consistency
- `architecture` is consistently named `standard_unet` / `legacy_no_skip`.
- Decoder API is consistently `StandardUNetDecoder(...).forward(bottleneck, skip_features=...)`.
- The model output contract remains `SegmentationModelOutput`.

