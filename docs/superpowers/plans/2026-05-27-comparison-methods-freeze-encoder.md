# Freeze-Encoder Comparison Methods Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the next comparison-method stage for the low-data experiments by formalizing frozen-encoder baselines around three comparison lines: plain U-Net, output-side conv-adapter U-Net, and output-side NODE-style U-Net.

**Architecture:** Reuse the current training/analysis pipeline and preserve the frozen-encoder experiment contract. Treat the existing Group A path as the plain U-Net baseline, then extend the shared segmentation model with an explicit adapter placement concept so output-side adapter variants can be implemented without forking the whole training stack. Keep the new NODE comparison inside the same training budget and framework rather than attempting a full paper reproduction.

**Tech Stack:** Python, PyTorch, YAML configs, pytest, existing `scripts/run_low_data_experiment.py` CLI, existing low-data reporting and report visualization utilities.

---

## Scope And Decomposition

This plan covers only the agreed comparison-method stage:

1. formalize existing frozen-encoder Group A as the plain U-Net baseline in reporting and experiment configs
2. add output-side conv-adapter U-Net under the same frozen-encoder protocol
3. add output-side NODE-style U-Net under the same frozen-encoder protocol

This plan intentionally does **not** cover:

- full external paper reproduction of a published NODE-UNet
- solver/regularizer tuning
- multi-seed expansion
- new datasets beyond the already-prepared ISIC and GlaS plumbing

## File Structure

- Modify: `src/models/segmentation_model.py`
  - Add adapter placement support and the minimal output-side adapter wiring.
- Modify: `src/experiments/low_data_runner.py`
  - Extend config validation and model-construction wiring for comparison-method configs.
- Modify: `src/analysis/report_visualization.py`
  - Teach summary/report code about the new method names and ordering.
- Modify: `tests/test_low_data_runner.py`
  - Add config parsing and runner-wiring tests for output-side comparison configs.
- Modify: `tests/test_report_visualization.py`
  - Add summary-order coverage for the new methods.
- Modify: `tests/test_adapters.py`
  - Add focused behavior tests if output-side placement reuses but extends adapter expectations.
- Create: `tests/test_segmentation_model.py`
  - Unit tests for output-side placement behavior and shape preservation.
- Create: `configs/experiments/isic2018_low_data_output_conv_b.yaml`
  - Frozen-encoder output-side conv-adapter comparison config.
- Create: `configs/experiments/isic2018_low_data_output_node_c.yaml`
  - Frozen-encoder output-side NODE comparison config.
- Create: `configs/experiments/glas_low_data_output_conv_b.yaml`
  - GlaS output-side conv-adapter comparison config.
- Create: `configs/experiments/glas_low_data_output_node_c.yaml`
  - GlaS output-side NODE comparison config.
- Modify: `docs/experiments/2026-04-17-node-followup-matrix.md`
  - Record where these comparison methods sit in the experiment queue and what question each answers.

## Output Contract

After this plan is implemented:

- existing Group A remains the frozen-encoder plain U-Net baseline
- model configs may specify adapter placement in addition to adapter type
- output-side comparison configs can be run through the existing low-data CLI unchanged
- reporting can summarize and order:
  - plain U-Net
  - bottleneck conv adapter
  - output-side conv adapter
  - bottleneck NODE adapter
  - output-side NODE comparison

The new config contract should support:

```yaml
adapter:
  type: conv   # or node
  placement: output  # or bottleneck
  hidden_channels: 512
  init: default
```

If `adapter.placement` is omitted, behavior must remain backward-compatible with the current bottleneck placement.

## Task 1: Lock In The Baseline Vocabulary And Config Contract

**Files:**
- Modify: `src/experiments/low_data_runner.py`
- Modify: `tests/test_low_data_runner.py`

- [ ] **Step 1: Add a failing config-parse test for adapter placement defaults**

Append to `tests/test_low_data_runner.py`:

```python
def test_low_data_runner_defaults_adapter_placement_to_bottleneck(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "seed: 1",
                "paths:",
                "  train_images_dir: data/train/images",
                "  train_masks_dir: data/train/labels",
                "  val_images_dir: data/validation/images",
                "  val_masks_dir: data/validation/labels",
                "  artifacts_dir: artifacts/tmp",
                "data:",
                "  image_size: 256",
                "  train_ratio: 0.1",
                "train:",
                "  batch_size: 2",
                "  epochs: 1",
                "  learning_rate: 0.001",
                "  weight_decay: 0.0001",
                "  early_stopping_patience: 1",
                "model:",
                "  encoder_name: resnet18",
                "  encoder_weights: null",
                "  in_channels: 3",
                "  num_classes: 1",
                "  bottleneck_channels: 16",
                "  freeze_encoder: true",
                "adapter:",
                "  type: conv",
                "  hidden_channels: 8",
                "node:",
                "  steps: 4",
                "  step_size: 0.25",
                "",
            ]
        ),
        encoding="utf-8",
    )

    from src.experiments.low_data_runner import load_config

    config = load_config(config_path)
    assert config["adapter"].get("placement", "bottleneck") == "bottleneck"
```

- [ ] **Step 2: Add a failing validation test for unsupported placement values**

Append to `tests/test_low_data_runner.py`:

```python
def test_validate_low_data_config_rejects_unknown_adapter_placement():
    from src.experiments.low_data_runner import _validate_low_data_config

    config = {
        "seed": 1,
        "paths": {
            "train_images_dir": "a",
            "train_masks_dir": "b",
            "val_images_dir": "c",
            "val_masks_dir": "d",
            "artifacts_dir": "e",
        },
        "data": {"image_size": 256, "train_ratio": 0.1},
        "train": {
            "batch_size": 2,
            "epochs": 1,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "early_stopping_patience": 1,
        },
        "model": {
            "encoder_name": "resnet18",
            "encoder_weights": None,
            "in_channels": 3,
            "num_classes": 1,
            "bottleneck_channels": 16,
            "freeze_encoder": True,
        },
        "adapter": {"type": "conv", "placement": "decoder_tail", "hidden_channels": 8},
        "node": {"steps": 4, "step_size": 0.25},
    }

    with pytest.raises(ValueError, match="config.adapter.placement"):
        _validate_low_data_config(config)
```

- [ ] **Step 3: Run the two tests to verify the placement validation gap**

Run:

```bash
python -m pytest tests/test_low_data_runner.py::test_low_data_runner_defaults_adapter_placement_to_bottleneck tests/test_low_data_runner.py::test_validate_low_data_config_rejects_unknown_adapter_placement -q
```

Expected: the default test passes or is trivial, but the validation test fails because placement is not checked yet.

- [ ] **Step 4: Implement minimal placement validation**

In `src/experiments/low_data_runner.py`, extend adapter validation so:

```python
    adapter = _require_mapping(config, "adapter", "config")
    _require_keys(adapter, ["hidden_channels"], "config.adapter")

    placement = str(adapter.get("placement", "bottleneck"))
    if placement not in {"bottleneck", "output"}:
        raise ValueError(
            "config.adapter.placement must be one of ['bottleneck', 'output']; "
            f"got {placement!r}."
        )
```

- [ ] **Step 5: Run the focused validation tests**

Run:

```bash
python -m pytest tests/test_low_data_runner.py::test_low_data_runner_defaults_adapter_placement_to_bottleneck tests/test_low_data_runner.py::test_validate_low_data_config_rejects_unknown_adapter_placement -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/experiments/low_data_runner.py tests/test_low_data_runner.py
git commit -m "feat: validate adapter placement in low-data configs"
```

## Task 2: Add Output-Side Adapter Placement To The Shared Segmentation Model

**Files:**
- Modify: `src/models/segmentation_model.py`
- Create: `tests/test_segmentation_model.py`

- [ ] **Step 1: Write a failing shape-preservation test for output-side conv placement**

Create `tests/test_segmentation_model.py` with:

```python
from __future__ import annotations

import torch


def test_segmentation_model_output_side_conv_adapter_preserves_shapes():
    from src.models.segmentation_model import build_segmentation_model

    model = build_segmentation_model(
        encoder_name="resnet18",
        encoder_weights=None,
        in_channels=3,
        num_classes=1,
        adapter_type="conv",
        adapter_placement="output",
        bottleneck_channels=32,
        adapter_hidden_channels=16,
        freeze_encoder=True,
        node_steps=4,
        node_step_size=0.25,
        node_solver="euler",
        adapter_init="default",
    )

    x = torch.randn(2, 3, 64, 64)
    output = model(x)

    assert output.logits.shape == (2, 1, 64, 64)
    assert output.bottleneck.shape[1] == 32
    assert output.adapted_bottleneck.shape == output.bottleneck.shape
```

- [ ] **Step 2: Add a failing behavior test proving output-side conv placement changes the head path, not the bottleneck output contract**

Append to `tests/test_segmentation_model.py`:

```python
def test_segmentation_model_output_side_node_adapter_runs_forward():
    from src.models.segmentation_model import build_segmentation_model

    model = build_segmentation_model(
        encoder_name="resnet18",
        encoder_weights=None,
        in_channels=3,
        num_classes=1,
        adapter_type="node",
        adapter_placement="output",
        bottleneck_channels=32,
        adapter_hidden_channels=16,
        freeze_encoder=True,
        node_steps=2,
        node_step_size=0.5,
        node_solver="euler",
        adapter_init="default",
    )

    x = torch.randn(1, 3, 64, 64)
    output = model(x)

    assert output.logits.shape == (1, 1, 64, 64)
    assert output.adapted_bottleneck.shape == output.bottleneck.shape
```

- [ ] **Step 3: Run the new tests to verify they fail**

Run:

```bash
python -m pytest tests/test_segmentation_model.py -q
```

Expected: FAIL because `build_segmentation_model(...)` does not accept `adapter_placement`.

- [ ] **Step 4: Implement minimal placement-aware model wiring**

In `src/models/segmentation_model.py`:

1. extend `AdapterType`-adjacent config with:

```python
AdapterPlacement = Literal["bottleneck", "output"]
```

2. extend the constructor signature:

```python
        adapter_placement: AdapterPlacement = "bottleneck",
```

3. keep current bottleneck behavior when placement is `bottleneck`
4. when placement is `output`, instantiate the adapter against the decoder's pre-head channel count and apply it after `self.decoder(...)` but before `self.head(...)`
5. preserve the existing output contract:
   - `bottleneck` stays the encoder-projected bottleneck
   - `adapted_bottleneck` continues to reflect the bottleneck-stage adapter result
   - for output placement, set `adapted_bottleneck = bottleneck`

The minimal structure should look like:

```python
        self.adapter_placement = adapter_placement

        if adapter_placement == "bottleneck":
            self.adapter = ...
            self.output_adapter = IdentityAdapter()
        elif adapter_placement == "output":
            self.adapter = IdentityAdapter()
            self.output_adapter = ...
        else:
            raise ValueError(f"Unknown adapter_placement: {adapter_placement}")
```

and in `forward(...)`:

```python
        adapted_bottleneck = self.adapter(bottleneck)
        decoded = self.decoder(adapted_bottleneck)
        decoded = self.output_adapter(decoded)
```

- [ ] **Step 5: Run the focused model tests**

Run:

```bash
python -m pytest tests/test_segmentation_model.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/models/segmentation_model.py tests/test_segmentation_model.py
git commit -m "feat: add output-side adapter placement to segmentation model"
```

## Task 3: Pass Adapter Placement Through The Runner

**Files:**
- Modify: `src/experiments/low_data_runner.py`
- Modify: `tests/test_low_data_runner.py`

- [ ] **Step 1: Add a failing runner wiring test for output placement**

Append to `tests/test_low_data_runner.py`:

```python
def test_run_group_passes_output_adapter_placement_to_model_builder(tmp_path, monkeypatch):
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
                "  type: conv",
                "  placement: output",
                "  hidden_channels: 8",
                "node:",
                "  steps: 4",
                "  step_size: 0.25",
                "",
            ]
        ),
        encoding="utf-8",
    )

    calls: dict[str, object] = {}
```

Continue the test using the existing runner stubbing pattern already present in this file, then assert:

```python
    assert calls["build_segmentation_model"]["adapter_placement"] == "output"
```

- [ ] **Step 2: Run the single test to verify it fails**

Run:

```bash
python -m pytest tests/test_low_data_runner.py::test_run_group_passes_output_adapter_placement_to_model_builder -q
```

Expected: FAIL because the runner does not pass `adapter_placement`.

- [ ] **Step 3: Implement the runner wiring**

In `src/experiments/low_data_runner.py`, when calling `build_segmentation_model(...)`, add:

```python
        adapter_placement=str(config["adapter"].get("placement", "bottleneck")),
```

- [ ] **Step 4: Run the focused test**

Run:

```bash
python -m pytest tests/test_low_data_runner.py::test_run_group_passes_output_adapter_placement_to_model_builder -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/experiments/low_data_runner.py tests/test_low_data_runner.py
git commit -m "feat: pass adapter placement through low-data runner"
```

## Task 4: Add ISIC And GlaS Output-Side Comparison Configs

**Files:**
- Create: `configs/experiments/isic2018_low_data_output_conv_b.yaml`
- Create: `configs/experiments/isic2018_low_data_output_node_c.yaml`
- Create: `configs/experiments/glas_low_data_output_conv_b.yaml`
- Create: `configs/experiments/glas_low_data_output_node_c.yaml`
- Modify: `tests/test_low_data_runner.py`

- [ ] **Step 1: Add a failing config parse test for the four new comparison configs**

Append to `tests/test_low_data_runner.py`:

```python
def test_output_side_comparison_configs_parse_with_expected_adapter_placement():
    from src.experiments.low_data_runner import load_config

    expected = {
        "configs/experiments/isic2018_low_data_output_conv_b.yaml": {"dataset_name": "isic2018", "group": "B", "type": "conv"},
        "configs/experiments/isic2018_low_data_output_node_c.yaml": {"dataset_name": "isic2018", "group": "C", "type": "node"},
        "configs/experiments/glas_low_data_output_conv_b.yaml": {"dataset_name": "glas", "group": "B", "type": "conv"},
        "configs/experiments/glas_low_data_output_node_c.yaml": {"dataset_name": "glas", "group": "C", "type": "node"},
    }

    for path, values in expected.items():
        config = load_config(path)
        assert config["data"]["dataset_name"] == values["dataset_name"]
        assert config["experiment"]["group"] == values["group"]
        assert config["adapter"]["type"] == values["type"]
        assert config["adapter"]["placement"] == "output"
        assert config["model"]["freeze_encoder"] is True
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
python -m pytest tests/test_low_data_runner.py::test_output_side_comparison_configs_parse_with_expected_adapter_placement -q
```

Expected: FAIL because the config files do not exist yet.

- [ ] **Step 3: Create the two ISIC configs**

Create `configs/experiments/isic2018_low_data_output_conv_b.yaml`:

```yaml
seed: 42
experiment:
  name: isic2018_low_data_output_conv_b
  group: B
paths:
  train_images_dir: data/train/images
  train_masks_dir: data/train/labels
  val_images_dir: data/validation/images
  val_masks_dir: data/validation/labels
  artifacts_dir: artifacts/low_data_output/isic2018_conv_output_seed42
data:
  dataset_name: isic2018
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
  placement: output
  hidden_channels: 128
  init: default
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
  save_best_checkpoint: false
```

Create `configs/experiments/isic2018_low_data_output_node_c.yaml`:

```yaml
seed: 42
experiment:
  name: isic2018_low_data_output_node_c
  group: C
paths:
  train_images_dir: data/train/images
  train_masks_dir: data/train/labels
  val_images_dir: data/validation/images
  val_masks_dir: data/validation/labels
  artifacts_dir: artifacts/low_data_output/isic2018_node_output_seed42
data:
  dataset_name: isic2018
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
  placement: output
  hidden_channels: 128
  init: default
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
  save_best_checkpoint: false
```

- [ ] **Step 4: Create the two GlaS configs**

Create `configs/experiments/glas_low_data_output_conv_b.yaml`:

```yaml
seed: 42
experiment:
  name: glas_low_data_output_conv_b
  group: B
paths:
  train_images_dir: data/glas/train/images
  train_masks_dir: data/glas/train/masks
  val_images_dir: data/glas/val/images
  val_masks_dir: data/glas/val/masks
  artifacts_dir: artifacts/glas_low_data_output/conv_output_seed42
data:
  dataset_name: glas
  image_size: 256
  train_ratio: 0.1
  num_workers: 0
  pin_memory: false
model:
  encoder_name: resnet34
  encoder_weights: imagenet
  in_channels: 3
  num_classes: 1
  bottleneck_channels: 512
  freeze_encoder: true
adapter:
  type: conv
  placement: output
  hidden_channels: 128
  init: default
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
  save_best_checkpoint: false
```

Create `configs/experiments/glas_low_data_output_node_c.yaml`:

```yaml
seed: 42
experiment:
  name: glas_low_data_output_node_c
  group: C
paths:
  train_images_dir: data/glas/train/images
  train_masks_dir: data/glas/train/masks
  val_images_dir: data/glas/val/images
  val_masks_dir: data/glas/val/masks
  artifacts_dir: artifacts/glas_low_data_output/node_output_seed42
data:
  dataset_name: glas
  image_size: 256
  train_ratio: 0.1
  num_workers: 0
  pin_memory: false
model:
  encoder_name: resnet34
  encoder_weights: imagenet
  in_channels: 3
  num_classes: 1
  bottleneck_channels: 512
  freeze_encoder: true
adapter:
  type: node
  placement: output
  hidden_channels: 128
  init: default
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
  save_best_checkpoint: false
```

- [ ] **Step 5: Run the config parse test**

Run:

```bash
python -m pytest tests/test_low_data_runner.py::test_output_side_comparison_configs_parse_with_expected_adapter_placement -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add configs/experiments/isic2018_low_data_output_conv_b.yaml configs/experiments/isic2018_low_data_output_node_c.yaml configs/experiments/glas_low_data_output_conv_b.yaml configs/experiments/glas_low_data_output_node_c.yaml tests/test_low_data_runner.py
git commit -m "exp: add output-side comparison configs for isic and glas"
```

## Task 5: Teach Reporting About The Comparison Methods

**Files:**
- Modify: `src/analysis/report_visualization.py`
- Modify: `tests/test_report_visualization.py`

- [ ] **Step 1: Add a failing report-order test for the new comparison methods**

Append to `tests/test_report_visualization.py`:

```python
def test_method_order_places_plain_and_output_side_comparisons_explicitly():
    from src.analysis.report_visualization import _method_order

    ordered = _method_order(
        [
            "Output-Conv-U-Net",
            "B-base",
            "Plain-U-Net",
            "Output-NODE-U-Net",
        ]
    )

    assert ordered == [
        "Plain-U-Net",
        "B-base",
        "Output-Conv-U-Net",
        "Output-NODE-U-Net",
    ]
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
python -m pytest tests/test_report_visualization.py::test_method_order_places_plain_and_output_side_comparisons_explicitly -q
```

Expected: FAIL because the method order list does not include these names.

- [ ] **Step 3: Extend report method naming and ordering**

In `src/analysis/report_visualization.py`, update the method order list so it can place:

```python
_MULTISEED_ORDER = [
    "Plain-U-Net",
    "B-base",
    "Output-Conv-U-Net",
    "C-fine-steps8-default",
    "C-zero-last-steps8",
    "C-zero-last-steps16",
    "Output-NODE-U-Net",
]
```

Also keep the helper tolerant of partial subsets so old reports still work.

- [ ] **Step 4: Run the focused reporting tests**

Run:

```bash
python -m pytest tests/test_report_visualization.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/analysis/report_visualization.py tests/test_report_visualization.py
git commit -m "feat: add reporting order for comparison methods"
```

## Task 6: Document The Comparison Matrix

**Files:**
- Modify: `docs/experiments/2026-04-17-node-followup-matrix.md`

- [ ] **Step 1: Add comparison-method rows to the experiment matrix**

In `docs/experiments/2026-04-17-node-followup-matrix.md`, add rows or bullets for:

- Plain-U-Net
  - question answered: how much do any adapters help over frozen-encoder U-Net?
- Output-Conv-U-Net
  - question answered: is bottleneck placement better than output-side static adaptation?
- Output-NODE-U-Net
  - question answered: is NODE itself helping when moved to the output side under the same frozen-encoder budget?

- [ ] **Step 2: Mark the stage gate explicitly**

Add this note verbatim:

```text
Comparison-method stage precedes any solver/regularizer tuning and any multi-seed expansion.
If output-side comparisons or plain U-Net already dominate the current bottleneck NODE line, stop and reassess before further tuning.
```

- [ ] **Step 3: Commit**

```bash
git add docs/experiments/2026-04-17-node-followup-matrix.md
git commit -m "docs: record comparison-method experiment stage"
```

## Task 7: Full Verification And Handoff

**Files:**
- No source edits in this task.

- [ ] **Step 1: Run the focused suite for the comparison-method stage**

Run:

```bash
python -m pytest tests/test_segmentation_model.py tests/test_low_data_runner.py tests/test_report_visualization.py tests/test_low_data_clis.py -q
```

Expected: PASS.

- [ ] **Step 2: Run the full test suite**

Run:

```bash
python -m pytest -q
```

Expected: PASS with only any previously accepted environment warnings.

- [ ] **Step 3: Smoke-check the new configs parse**

Run:

```bash
python - <<'PY'
from src.experiments.low_data_runner import load_config
for path in [
    "configs/experiments/isic2018_low_data_output_conv_b.yaml",
    "configs/experiments/isic2018_low_data_output_node_c.yaml",
    "configs/experiments/glas_low_data_output_conv_b.yaml",
    "configs/experiments/glas_low_data_output_node_c.yaml",
]:
    cfg = load_config(path)
    print(path, cfg["data"]["dataset_name"], cfg["adapter"]["type"], cfg["adapter"]["placement"], cfg["model"]["freeze_encoder"])
PY
```

Expected output:

```text
configs/experiments/isic2018_low_data_output_conv_b.yaml isic2018 conv output True
configs/experiments/isic2018_low_data_output_node_c.yaml isic2018 node output True
configs/experiments/glas_low_data_output_conv_b.yaml glas conv output True
configs/experiments/glas_low_data_output_node_c.yaml glas node output True
```

- [ ] **Step 4: Record the decision gate in the final integration summary**

Include this exact note in the final summary:

```text
Comparison-method stage completed with frozen-encoder plain, output-conv, and output-NODE baselines.
Do not begin solver/regularizer tuning or multi-seed expansion until these comparisons are run and reviewed.
```

## Recommended Execution Order After This Plan

After implementation, the experimental progression should be:

1. treat existing Group A as the plain frozen-encoder U-Net baseline
2. run `isic2018_low_data_output_conv_b.yaml`
3. run `isic2018_low_data_output_node_c.yaml`
4. if ISIC results look promising, run the matching two GlaS configs
5. review Dice, timing, and stability before any tuning work

## Self-Review

- Spec coverage: This plan covers the exact comparison-method scope the user approved: plain U-Net, output-side adapter U-Net, and NODE-UNet-style comparison under `freeze_encoder`, while keeping prior bottleneck conv adapter as an existing baseline rather than a new method.
- Placeholder scan: No `TODO`, `TBD`, or “similar to above” placeholders remain. Each task names exact files, tests, and commands.
- Type consistency: The plan consistently uses `adapter.placement`, `adapter_type`, `adapter_placement`, `Plain-U-Net`, `Output-Conv-U-Net`, and `Output-NODE-U-Net` across model, runner, config, and report tasks.
