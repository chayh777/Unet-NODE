# NODE Zero-Last Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a config-driven `zero_last_layer` initialization option for NODE and conv bottleneck adapters, then define the first NODE improvement and ablation experiment matrix.

**Architecture:** Keep the current A/B/C runner contract unchanged. Add initialization support at the bottleneck block/model boundary, pass it through the low-data config, and create focused experiment configs that isolate NODE initialization, integration strength, and conv-adapter control effects.

**Tech Stack:** Python 3.10/3.11, PyTorch, YAML configs, pytest, existing `src.models`, `src.experiments`, and `scripts` modules.

---

## File Structure

- Modify `src/models/adapters.py`: define adapter initialization modes and apply zero initialization to the final `1x1 Conv2d` of the bottleneck block.
- Modify `src/models/node_adapter.py`: accept the initialization mode and pass it into the NODE vector-field block.
- Modify `src/models/segmentation_model.py`: expose `adapter_init` from `build_segmentation_model` to conv and NODE adapters.
- Modify `src/experiments/low_data_runner.py`: validate optional `adapter.init` and pass it to model construction.
- Modify `src/analysis/low_data_geometry.py`: pass optional `adapter.init` when reconstructing the trained model for geometry export.
- Modify `tests/test_adapters.py`: test zero-last initialization behavior directly on conv and NODE adapters.
- Modify `tests/test_low_data_runner.py`: test config plumbing from YAML to `build_segmentation_model`.
- Modify `tests/test_low_data_geometry.py`: test geometry reconstruction passes `adapter_init`.
- Add `configs/experiments/isic2018_low_data_node_c_zero_last.yaml`: main NODE zero-last run.
- Add `configs/experiments/isic2018_low_data_node_c_zero_last_small_step.yaml`: zero-last plus smaller step.
- Add `configs/experiments/isic2018_low_data_node_c_fine_integration.yaml`: smoother NODE integration without zero-last.
- Add `configs/experiments/isic2018_low_data_node_c_zero_last_fine_integration.yaml`: zero-last plus smoother integration.
- Add `configs/experiments/isic2018_low_data_conv_b_zero_last.yaml`: conv adapter zero-last control.
- Add `configs/experiments/isic2018_low_data_node_c_zero_last_steps1.yaml`: single-step NODE mechanism ablation.
- Modify `docs/experiments/2026-04-17-node-followup-matrix.md`: mark `C-base-locked` as already run and add the zero-last experiment rows.
- Modify `src/analysis/low_data_reporting.py`: add final-epoch Dice and peak-final gap to summary metrics.
- Modify `tests/test_low_data_reporting.py`: test the new summary metrics.

## Task 1: Adapter Zero-Last Initialization

**Files:**
- Modify: `src/models/adapters.py`
- Modify: `src/models/node_adapter.py`
- Test: `tests/test_adapters.py`

- [ ] **Step 1: Add failing adapter tests**

Add these tests to `tests/test_adapters.py`:

```python
def test_conv_adapter_zero_last_layer_initializes_final_conv_to_zero():
    import torch.nn as nn

    model = ConvBottleneckAdapter(
        channels=32,
        hidden_channels=16,
        init="zero_last_layer",
    )
    convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]

    assert len(convs) == 2
    assert torch.count_nonzero(convs[0].weight).item() > 0
    assert torch.count_nonzero(convs[-1].weight).item() == 0
    assert convs[-1].bias is not None
    assert torch.count_nonzero(convs[-1].bias).item() == 0


def test_node_adapter_zero_last_layer_initializes_vector_field_final_conv_to_zero():
    import torch.nn as nn

    model = NODEAdapter(
        channels=32,
        hidden_channels=16,
        steps=4,
        step_size=0.25,
        init="zero_last_layer",
    )
    convs = [m for m in model.func.modules() if isinstance(m, nn.Conv2d)]

    assert len(convs) == 2
    assert torch.count_nonzero(convs[0].weight).item() > 0
    assert torch.count_nonzero(convs[-1].weight).item() == 0
    assert convs[-1].bias is not None
    assert torch.count_nonzero(convs[-1].bias).item() == 0


def test_node_adapter_zero_last_layer_starts_as_identity_flow():
    x = torch.randn(2, 32, 8, 8)
    model = NODEAdapter(
        channels=32,
        hidden_channels=16,
        steps=4,
        step_size=0.25,
        init="zero_last_layer",
    )

    y = model(x)

    assert torch.allclose(y, x, atol=1e-6)
```

- [ ] **Step 2: Run adapter tests and verify failure**

Run:

```bash
python -m pytest tests/test_adapters.py -q
```

Expected: FAIL because `ConvBottleneckAdapter` and `NODEAdapter` do not accept `init`.

- [ ] **Step 3: Implement initialization support in `src/models/adapters.py`**

Change `src/models/adapters.py` to include this implementation:

```python
from typing import Literal

import torch.nn as nn


AdapterInit = Literal["default", "zero_last_layer"]


def _zero_last_conv(block: nn.Sequential) -> None:
    convs = [module for module in block.modules() if isinstance(module, nn.Conv2d)]
    if not convs:
        raise ValueError("Cannot apply zero_last_layer initialization without Conv2d layers.")
    last_conv = convs[-1]
    nn.init.zeros_(last_conv.weight)
    if last_conv.bias is not None:
        nn.init.zeros_(last_conv.bias)


def build_conv_bottleneck_block(
    channels: int,
    hidden_channels: int,
    init: AdapterInit = "default",
) -> nn.Sequential:
    block = nn.Sequential(
        nn.Conv2d(channels, hidden_channels, kernel_size=1),
        # Stateless BN avoids running-mean/var drift when the block is used recurrently (NODE).
        nn.BatchNorm2d(hidden_channels, track_running_stats=False),
        nn.ReLU(inplace=True),
        nn.Conv2d(hidden_channels, channels, kernel_size=1),
    )
    if init == "default":
        return block
    if init == "zero_last_layer":
        _zero_last_conv(block)
        return block
    raise ValueError(f"Unknown adapter init: {init!r}. Expected 'default' or 'zero_last_layer'.")
```

Update `ConvBottleneckAdapter` in the same file:

```python
class ConvBottleneckAdapter(nn.Module):
    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        init: AdapterInit = "default",
    ) -> None:
        super().__init__()
        self.net = build_conv_bottleneck_block(
            channels=channels,
            hidden_channels=hidden_channels,
            init=init,
        )
```

- [ ] **Step 4: Implement initialization support in `src/models/node_adapter.py`**

Update imports and constructors:

```python
from .adapters import AdapterInit, build_conv_bottleneck_block


class ODEFunction(nn.Module):
    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        init: AdapterInit = "default",
    ) -> None:
        super().__init__()
        self.net = build_conv_bottleneck_block(
            channels=channels,
            hidden_channels=hidden_channels,
            init=init,
        )
```

Update `NODEAdapter`:

```python
class NODEAdapter(nn.Module):
    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        steps: int,
        step_size: float,
        init: AdapterInit = "default",
    ) -> None:
        super().__init__()
        self.func = ODEFunction(
            channels=channels,
            hidden_channels=hidden_channels,
            init=init,
        )
        self.steps = steps
        self.step_size = step_size
```

- [ ] **Step 5: Run adapter tests and verify pass**

Run:

```bash
python -m pytest tests/test_adapters.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/models/adapters.py src/models/node_adapter.py tests/test_adapters.py
git commit -m "feat: add zero-last adapter initialization"
```

## Task 2: Model and Runner Config Plumbing

**Files:**
- Modify: `src/models/segmentation_model.py`
- Modify: `src/experiments/low_data_runner.py`
- Test: `tests/test_low_data_runner.py`

- [ ] **Step 1: Add failing model construction test**

Add this test after `test_build_segmentation_model_freezes_encoder` in `tests/test_low_data_runner.py`:

```python
def test_build_segmentation_model_passes_zero_last_init_to_node_adapter():
    import torch.nn as nn

    model = build_segmentation_model(
        encoder_name="resnet18",
        encoder_weights=None,
        in_channels=3,
        num_classes=1,
        adapter_type="node",
        bottleneck_channels=32,
        adapter_hidden_channels=16,
        freeze_encoder=True,
        node_steps=4,
        node_step_size=0.25,
        adapter_init="zero_last_layer",
    )
    convs = [m for m in model.adapter.func.modules() if isinstance(m, nn.Conv2d)]

    assert len(convs) == 2
    assert torch.count_nonzero(convs[-1].weight).item() == 0
    assert convs[-1].bias is not None
    assert torch.count_nonzero(convs[-1].bias).item() == 0
```

- [ ] **Step 2: Add failing runner plumbing assertion**

In `test_run_group_writes_split_manifest_and_uses_group_adapter`, add this YAML line under `adapter:`:

```python
"  init: zero_last_layer",
```

Add this assertion near the existing `build_segmentation_model` assertions:

```python
assert calls["build_segmentation_model"]["adapter_init"] == "zero_last_layer"
```

- [ ] **Step 3: Run runner tests and verify failure**

Run:

```bash
python -m pytest tests/test_low_data_runner.py -q
```

Expected: FAIL because `build_segmentation_model` does not accept `adapter_init` and `run_group` does not pass it.

- [ ] **Step 4: Update `src/models/segmentation_model.py`**

Import the init type:

```python
from .adapters import AdapterInit, ConvBottleneckAdapter, IdentityAdapter
```

Add the keyword argument to `SegmentationModel.__init__`:

```python
        adapter_init: AdapterInit = "default",
```

Pass it into conv and NODE adapters:

```python
        elif adapter_type == "conv":
            self.adapter = ConvBottleneckAdapter(
                channels=bottleneck_channels,
                hidden_channels=adapter_hidden_channels,
                init=adapter_init,
            )
        elif adapter_type == "node":
            self.adapter = NODEAdapter(
                channels=bottleneck_channels,
                hidden_channels=adapter_hidden_channels,
                steps=node_steps,
                step_size=node_step_size,
                init=adapter_init,
            )
```

Keep `build_segmentation_model(**kwargs)` unchanged so tests and callers use the same forwarding pattern.

- [ ] **Step 5: Update `src/experiments/low_data_runner.py` validation and construction**

Add this helper near `_validate_low_data_config`:

```python
def _resolve_adapter_init(config: dict[str, Any]) -> str:
    adapter = config.get("adapter", {})
    if not isinstance(adapter, dict):
        return "default"
    value = adapter.get("init", "default")
    if value not in {"default", "zero_last_layer"}:
        raise ValueError(
            "config.adapter.init must be one of ['default', 'zero_last_layer']; "
            f"got {value!r}."
        )
    return str(value)
```

In `run_group`, compute:

```python
    adapter_init = _resolve_adapter_init(config)
```

Pass it to `build_segmentation_model`:

```python
        adapter_init=adapter_init,
```

- [ ] **Step 6: Add invalid-config test**

Add this test to `tests/test_low_data_runner.py`:

```python
def test_run_group_rejects_unknown_adapter_init(tmp_path):
    artifacts_dir = tmp_path / "artifacts"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "seed: 42",
                "paths:",
                "  train_images_dir: data/train/images",
                "  train_masks_dir: data/train/labels",
                "  val_images_dir: data/val/images",
                "  val_masks_dir: data/val/labels",
                f"  artifacts_dir: {artifacts_dir.as_posix()}",
                "data:",
                "  image_size: 256",
                "  train_ratio: 0.1",
                "train:",
                "  batch_size: 2",
                "  epochs: 1",
                "  learning_rate: 0.001",
                "  weight_decay: 0.01",
                "  early_stopping_patience: 1",
                "model:",
                "  encoder_name: resnet18",
                "  encoder_weights: null",
                "  in_channels: 3",
                "  num_classes: 1",
                "  bottleneck_channels: 16",
                "  freeze_encoder: true",
                "adapter:",
                "  hidden_channels: 8",
                "  init: unsupported",
                "node:",
                "  steps: 4",
                "  step_size: 0.25",
                "",
            ]
        ),
        encoding="utf-8",
    )

    from src.experiments.low_data_runner import run_group

    try:
        run_group(config_path, "C")
        assert False, "Expected invalid adapter.init to raise ValueError"
    except ValueError as exc:
        assert "adapter.init" in str(exc)
        assert "zero_last_layer" in str(exc)
```

- [ ] **Step 7: Run runner tests and verify pass**

Run:

```bash
python -m pytest tests/test_low_data_runner.py -q
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/models/segmentation_model.py src/experiments/low_data_runner.py tests/test_low_data_runner.py
git commit -m "feat: wire adapter initialization through low-data runner"
```

## Task 3: Geometry Reconstruction Config Plumbing

**Files:**
- Modify: `src/analysis/low_data_geometry.py`
- Test: `tests/test_low_data_geometry.py`

- [ ] **Step 1: Locate the existing geometry model-construction test**

Run:

```bash
rg -n "build_segmentation_model|export_group_geometry" tests/test_low_data_geometry.py
```

Expected: output includes the tests that monkeypatch geometry export dependencies.

- [ ] **Step 2: Add failing assertion for geometry `adapter_init`**

In the geometry export test that captures `build_segmentation_model(**kwargs)`, add this to the test config:

```python
    config = _make_valid_geometry_config(tmp_path)
    config["adapter"]["init"] = "zero_last_layer"
```

Add this assertion after export runs:

```python
assert calls["build_segmentation_model"]["adapter_init"] == "zero_last_layer"
```

- [ ] **Step 3: Run geometry tests and verify failure**

Run:

```bash
python -m pytest tests/test_low_data_geometry.py -q
```

Expected: FAIL because geometry export reconstructs the model without `adapter_init`.

- [ ] **Step 4: Update `src/analysis/low_data_geometry.py`**

Import the runner helper:

```python
from src.experiments.low_data_runner import (
    _is_known_windows_dataloader_worker_permission_error,
    _resolve_adapter_init,
    resolve_group_adapter,
)
```

Pass `adapter_init` to `build_segmentation_model`:

```python
        adapter_init=_resolve_adapter_init(config),
```

- [ ] **Step 5: Run geometry tests and verify pass**

Run:

```bash
python -m pytest tests/test_low_data_geometry.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/analysis/low_data_geometry.py tests/test_low_data_geometry.py
git commit -m "fix: preserve adapter initialization in geometry export"
```

## Task 4: Experiment Configs for NODE Improvement and Ablation

**Files:**
- Add: `configs/experiments/isic2018_low_data_node_c_zero_last.yaml`
- Add: `configs/experiments/isic2018_low_data_node_c_zero_last_small_step.yaml`
- Add: `configs/experiments/isic2018_low_data_node_c_fine_integration.yaml`
- Add: `configs/experiments/isic2018_low_data_node_c_zero_last_fine_integration.yaml`
- Add: `configs/experiments/isic2018_low_data_conv_b_zero_last.yaml`
- Add: `configs/experiments/isic2018_low_data_node_c_zero_last_steps1.yaml`
- Test: `tests/test_low_data_runner.py`

- [ ] **Step 1: Add config smoke test**

Add this test to `tests/test_low_data_runner.py`:

```python
def test_node_followup_configs_parse_and_use_expected_adapter_init():
    from src.experiments.low_data_runner import load_config

    expected = {
        "configs/experiments/isic2018_low_data_node_c_zero_last.yaml": {
            "artifacts_dir": "artifacts/low_data_followup/c_zero_last",
            "init": "zero_last_layer",
            "steps": 4,
            "step_size": 0.25,
        },
        "configs/experiments/isic2018_low_data_node_c_zero_last_small_step.yaml": {
            "artifacts_dir": "artifacts/low_data_followup/c_zero_last_small_step",
            "init": "zero_last_layer",
            "steps": 4,
            "step_size": 0.125,
        },
        "configs/experiments/isic2018_low_data_node_c_fine_integration.yaml": {
            "artifacts_dir": "artifacts/low_data_followup/c_fine_integration",
            "init": "default",
            "steps": 8,
            "step_size": 0.125,
        },
        "configs/experiments/isic2018_low_data_node_c_zero_last_fine_integration.yaml": {
            "artifacts_dir": "artifacts/low_data_followup/c_zero_last_fine_integration",
            "init": "zero_last_layer",
            "steps": 8,
            "step_size": 0.125,
        },
        "configs/experiments/isic2018_low_data_conv_b_zero_last.yaml": {
            "artifacts_dir": "artifacts/low_data_followup/b_zero_last",
            "init": "zero_last_layer",
            "steps": 4,
            "step_size": 0.25,
        },
        "configs/experiments/isic2018_low_data_node_c_zero_last_steps1.yaml": {
            "artifacts_dir": "artifacts/low_data_followup/c_zero_last_steps1",
            "init": "zero_last_layer",
            "steps": 1,
            "step_size": 1.0,
        },
    }

    for path, values in expected.items():
        config = load_config(path)
        assert config["paths"]["artifacts_dir"] == values["artifacts_dir"]
        assert config["adapter"].get("init", "default") == values["init"]
        assert int(config["node"]["steps"]) == values["steps"]
        assert float(config["node"]["step_size"]) == values["step_size"]
```

- [ ] **Step 2: Run config smoke test and verify failure**

Run:

```bash
python -m pytest tests/test_low_data_runner.py::test_node_followup_configs_parse_and_use_expected_adapter_init -q
```

Expected: FAIL because the new config files do not exist.

- [ ] **Step 3: Add `C-zero-last` config**

Create `configs/experiments/isic2018_low_data_node_c_zero_last.yaml`:

```yaml
seed: 42
paths:
  train_images_dir: data/train/images
  train_masks_dir: data/train/labels
  val_images_dir: data/validation/images
  val_masks_dir: data/validation/labels
  artifacts_dir: artifacts/low_data_followup/c_zero_last
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
  init: zero_last_layer
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
geometry:
  include_classes:
    - background
    - lesion
  min_mask_pixels: 1
geometry_plot:
  pca_components: 8
  umap_neighbors: 15
  umap_min_dist: 0.1
  random_state: 42
  alpha: 0.7
  point_size: 18
  dpi: 150
```

- [ ] **Step 4: Add `C-zero-last-small-step` config**

Create `configs/experiments/isic2018_low_data_node_c_zero_last_small_step.yaml`:

```yaml
seed: 42
paths:
  train_images_dir: data/train/images
  train_masks_dir: data/train/labels
  val_images_dir: data/validation/images
  val_masks_dir: data/validation/labels
  artifacts_dir: artifacts/low_data_followup/c_zero_last_small_step
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
  init: zero_last_layer
node:
  solver: euler
  steps: 4
  step_size: 0.125
train:
  batch_size: 8
  epochs: 50
  learning_rate: 0.0003
  weight_decay: 0.0001
  early_stopping_patience: 8
geometry:
  include_classes:
    - background
    - lesion
  min_mask_pixels: 1
geometry_plot:
  pca_components: 8
  umap_neighbors: 15
  umap_min_dist: 0.1
  random_state: 42
  alpha: 0.7
  point_size: 18
  dpi: 150
```

- [ ] **Step 5: Add `C-fine-integration` config**

Create `configs/experiments/isic2018_low_data_node_c_fine_integration.yaml`:

```yaml
seed: 42
paths:
  train_images_dir: data/train/images
  train_masks_dir: data/train/labels
  val_images_dir: data/validation/images
  val_masks_dir: data/validation/labels
  artifacts_dir: artifacts/low_data_followup/c_fine_integration
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
  init: default
node:
  solver: euler
  steps: 8
  step_size: 0.125
train:
  batch_size: 8
  epochs: 50
  learning_rate: 0.0003
  weight_decay: 0.0001
  early_stopping_patience: 8
geometry:
  include_classes:
    - background
    - lesion
  min_mask_pixels: 1
geometry_plot:
  pca_components: 8
  umap_neighbors: 15
  umap_min_dist: 0.1
  random_state: 42
  alpha: 0.7
  point_size: 18
  dpi: 150
```

- [ ] **Step 6: Add `C-zero-last-fine-integration` config**

Create `configs/experiments/isic2018_low_data_node_c_zero_last_fine_integration.yaml`:

```yaml
seed: 42
paths:
  train_images_dir: data/train/images
  train_masks_dir: data/train/labels
  val_images_dir: data/validation/images
  val_masks_dir: data/validation/labels
  artifacts_dir: artifacts/low_data_followup/c_zero_last_fine_integration
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
  init: zero_last_layer
node:
  solver: euler
  steps: 8
  step_size: 0.125
train:
  batch_size: 8
  epochs: 50
  learning_rate: 0.0003
  weight_decay: 0.0001
  early_stopping_patience: 8
geometry:
  include_classes:
    - background
    - lesion
  min_mask_pixels: 1
geometry_plot:
  pca_components: 8
  umap_neighbors: 15
  umap_min_dist: 0.1
  random_state: 42
  alpha: 0.7
  point_size: 18
  dpi: 150
```

- [ ] **Step 7: Add `B-zero-last` config**

Create `configs/experiments/isic2018_low_data_conv_b_zero_last.yaml`:

```yaml
seed: 42
paths:
  train_images_dir: data/train/images
  train_masks_dir: data/train/labels
  val_images_dir: data/validation/images
  val_masks_dir: data/validation/labels
  artifacts_dir: artifacts/low_data_followup/b_zero_last
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
  type: conv
  hidden_channels: 512
  init: zero_last_layer
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
geometry:
  include_classes:
    - background
    - lesion
  min_mask_pixels: 1
geometry_plot:
  pca_components: 8
  umap_neighbors: 15
  umap_min_dist: 0.1
  random_state: 42
  alpha: 0.7
  point_size: 18
  dpi: 150
```

- [ ] **Step 8: Add `C-zero-last-steps1` config**

Create `configs/experiments/isic2018_low_data_node_c_zero_last_steps1.yaml`:

```yaml
seed: 42
paths:
  train_images_dir: data/train/images
  train_masks_dir: data/train/labels
  val_images_dir: data/validation/images
  val_masks_dir: data/validation/labels
  artifacts_dir: artifacts/low_data_followup/c_zero_last_steps1
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
  init: zero_last_layer
node:
  solver: euler
  steps: 1
  step_size: 1.0
train:
  batch_size: 8
  epochs: 50
  learning_rate: 0.0003
  weight_decay: 0.0001
  early_stopping_patience: 8
geometry:
  include_classes:
    - background
    - lesion
  min_mask_pixels: 1
geometry_plot:
  pca_components: 8
  umap_neighbors: 15
  umap_min_dist: 0.1
  random_state: 42
  alpha: 0.7
  point_size: 18
  dpi: 150
```

- [ ] **Step 9: Run config smoke test and verify pass**

Run:

```bash
python -m pytest tests/test_low_data_runner.py::test_node_followup_configs_parse_and_use_expected_adapter_init -q
```

Expected: PASS.

- [ ] **Step 10: Commit**

```bash
git add configs/experiments tests/test_low_data_runner.py
git commit -m "exp: add zero-last node ablation configs"
```

## Task 5: Summary Reporting Stability Metrics

**Files:**
- Modify: `src/analysis/low_data_reporting.py`
- Test: `tests/test_low_data_reporting.py`

- [ ] **Step 1: Add failing reporting test**

Add this test to `tests/test_low_data_reporting.py`:

```python
def test_collect_group_final_metrics_includes_final_dice_and_peak_final_gap(tmp_path):
    group_dir = tmp_path / "group_c"
    group_dir.mkdir(parents=True)
    (group_dir / "history.csv").write_text(
        "\n".join(
            [
                "epoch,train_loss,val_loss,val_dice,val_iou",
                "1,1.0,0.9,0.70,0.54",
                "2,0.8,0.7,0.78,0.65",
                "3,0.7,0.8,0.75,0.61",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (group_dir / "metrics.json").write_text(
        '{"best_val_dice": 0.78, "epochs_ran": 3, "best_checkpoint": "best.pt"}',
        encoding="utf-8",
    )

    from src.analysis.low_data_reporting import collect_group_final_metrics

    row = collect_group_final_metrics(tmp_path, "C")

    assert row["final_val_dice"] == 0.75
    assert row["peak_final_gap"] == 0.03
```

- [ ] **Step 2: Run reporting test and verify failure**

Run:

```bash
python -m pytest tests/test_low_data_reporting.py::test_collect_group_final_metrics_includes_final_dice_and_peak_final_gap -q
```

Expected: FAIL because `final_val_dice` and `peak_final_gap` are not returned.

- [ ] **Step 3: Update `collect_group_final_metrics`**

In `src/analysis/low_data_reporting.py`, update the return object:

```python
    final_row = history.sort_values("epoch").iloc[-1]
    final_val_dice = float(final_row["val_dice"])
    peak_final_gap = float(best_val_dice) - final_val_dice

    return {
        "group": group,
        "best_epoch": int(best_row["epoch"]),
        "best_val_dice": float(best_val_dice),
        "best_val_iou": float(best_row["val_iou"]),
        "final_val_dice": final_val_dice,
        "peak_final_gap": peak_final_gap,
        "epochs_ran": metrics.get("epochs_ran"),
        "best_checkpoint": metrics.get("best_checkpoint"),
    }
```

Update `build_final_metrics_table` columns:

```python
        columns=[
            "group",
            "best_epoch",
            "best_val_dice",
            "best_val_iou",
            "final_val_dice",
            "peak_final_gap",
            "epochs_ran",
            "best_checkpoint",
        ],
```

- [ ] **Step 4: Run reporting tests and verify pass**

Run:

```bash
python -m pytest tests/test_low_data_reporting.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/analysis/low_data_reporting.py tests/test_low_data_reporting.py
git commit -m "feat: report low-data stability metrics"
```

## Task 6: Update Follow-Up Experiment Matrix

**Files:**
- Modify: `docs/experiments/2026-04-17-node-followup-matrix.md`

- [ ] **Step 1: Update locked reference notes**

In `docs/experiments/2026-04-17-node-followup-matrix.md`, add this bullet under `## Execution notes`:

```markdown
- `C-base-locked` has already been run externally by the project owner and is treated as the locked unstable reference; do not schedule another rerun unless the code path changes.
```

- [ ] **Step 2: Replace the run table**

Replace the table with:

```markdown
| Run name | Category | Hypothesis | Config file | Code support needed | Best Dice | Best IoU | Best epoch | Final Dice | Peak-final gap | Decision | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C-base-locked | reference | Current unstable NODE reference | configs/experiments/isic2018_low_data_node_c_base_locked.yaml | no |  |  |  |  |  | locked-reference | Already run by project owner; no rerun planned. |
| C-small-step | integration | Smaller NODE step reduces overshoot | configs/experiments/isic2018_low_data_node_c_small_step.yaml | no |  |  |  |  |  |  | Existing config. |
| C-zero-last | init | Zeroing the NODE vector field last layer starts from identity flow and stabilizes training | configs/experiments/isic2018_low_data_node_c_zero_last.yaml | yes |  |  |  |  |  |  | Primary improvement run. |
| C-zero-last-small-step | init+integration | Zero-last initialization and smaller step size may combine stability benefits | configs/experiments/isic2018_low_data_node_c_zero_last_small_step.yaml | yes |  |  |  |  |  |  | First combined run to prioritize after C-zero-last. |
| C-fine-integration | integration | More steps with smaller step size gives smoother NODE evolution | configs/experiments/isic2018_low_data_node_c_fine_integration.yaml | no |  |  |  |  |  |  | Config-only integration ablation. |
| C-zero-last-fine-integration | init+integration | Identity-start flow plus smoother integration improves both best Dice and final stability | configs/experiments/isic2018_low_data_node_c_zero_last_fine_integration.yaml | yes |  |  |  |  |  |  | Run after C-zero-last-small-step if zero-last is promising. |
| B-zero-last | control | Conv adapter also benefits from zero-last initialization; checks whether gains are NODE-specific | configs/experiments/isic2018_low_data_conv_b_zero_last.yaml | yes |  |  |  |  |  |  | Run before claiming NODE-specific improvement. |
| C-zero-last-steps1 | mechanism | Single-step zero-last NODE tests whether the gain is just a residual adapter effect | configs/experiments/isic2018_low_data_node_c_zero_last_steps1.yaml | yes |  |  |  |  |  |  | Run if C-zero-last improves over reference. |
| C-diff-lr | training | NODE learns too fast relative to decoder | configs/experiments/isic2018_low_data_node_c_diff_lr.yaml | yes |  |  |  |  |  | reserved | Requires optimizer parameter-group support. |
| C-warmup | training | Decoder should learn coarse masks before NODE acts | configs/experiments/isic2018_low_data_node_c_warmup.yaml | yes |  |  |  |  |  | reserved | Requires staged training support. |
```

- [ ] **Step 3: Add decision rubric for zero-last runs**

Under `## Decision rubric`, add:

```markdown
- Promote `C-zero-last` or a combined zero-last run if best Dice exceeds Group B best Dice `0.7768298244476318` and peak-final gap is below the current Group C gap `0.019665851593017578`.
- Mark a zero-last run as "stability-only improvement" if best Dice does not exceed Group B but peak-final gap is clearly smaller than current Group C.
- Do not claim NODE-specific benefit until `C-zero-last` is compared with `B-zero-last`.
```

- [ ] **Step 4: Commit**

```bash
git add docs/experiments/2026-04-17-node-followup-matrix.md
git commit -m "docs: plan zero-last node follow-up experiments"
```

## Task 7: Full Verification

**Files:**
- No source files modified in this task.

- [ ] **Step 1: Run focused model and runner tests**

Run:

```bash
python -m pytest tests/test_adapters.py tests/test_low_data_runner.py tests/test_low_data_geometry.py tests/test_low_data_reporting.py -q
```

Expected: PASS.

- [ ] **Step 2: Run full suite**

Run:

```bash
python -m pytest -q
```

Expected: PASS with the existing Jupyter path warning acceptable.

- [ ] **Step 3: Smoke-check runnable commands without launching long training**

Run:

```bash
python scripts/run_low_data_experiment.py --help
python scripts/run_low_data_geometry.py --help
python scripts/plot_low_data_summary.py --help
```

Expected: each command exits with code 0 and prints argparse help.

- [ ] **Step 4: Commit final verification note if any generated docs changed**

If only source, tests, configs, and matrix docs changed, no additional commit is needed after the task-level commits.

## Execution Order

1. Task 1: Adapter Zero-Last Initialization
2. Task 2: Model and Runner Config Plumbing
3. Task 3: Geometry Reconstruction Config Plumbing
4. Task 4: Experiment Configs for NODE Improvement and Ablation
5. Task 5: Summary Reporting Stability Metrics
6. Task 6: Update Follow-Up Experiment Matrix
7. Task 7: Full Verification

## Post-Implementation Experiment Commands

Run the first priority experiments in this order:

```bash
python scripts/run_low_data_experiment.py --config configs/experiments/isic2018_low_data_node_c_zero_last.yaml --group C
python scripts/run_low_data_experiment.py --config configs/experiments/isic2018_low_data_node_c_zero_last_small_step.yaml --group C
python scripts/run_low_data_experiment.py --config configs/experiments/isic2018_low_data_node_c_small_step.yaml --group C
python scripts/run_low_data_experiment.py --config configs/experiments/isic2018_low_data_node_c_fine_integration.yaml --group C
python scripts/run_low_data_experiment.py --config configs/experiments/isic2018_low_data_conv_b_zero_last.yaml --group B
python scripts/run_low_data_experiment.py --config configs/experiments/isic2018_low_data_node_c_zero_last_steps1.yaml --group C
```

After each run, export geometry when `best.pt` exists:

```bash
python scripts/run_low_data_geometry.py --config configs/experiments/isic2018_low_data_node_c_zero_last.yaml --group C
```

Use the matching config and group for each completed run.

## Self-Review

- Spec coverage: The plan covers zero-last NODE initialization, B-zero-last control, small-step/fine-integration ablations, follow-up matrix updates, and stability metrics.
- Placeholder scan: The plan contains concrete file paths, concrete code snippets, concrete test commands, and concrete expected outcomes.
- Type consistency: The initialization value is consistently named `zero_last_layer`; the new model keyword is consistently named `adapter_init`; YAML uses `adapter.init`.
