# Bottleneck NODE Stability Tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add stability-oriented tuning support for the bottleneck NODE main method, starting with kinetic regularization, controlled steps/solver sweeps, and experiment reporting that makes the tuning outcome easy to compare against the existing bottleneck baselines.

**Architecture:** Keep the main method definition fixed: frozen encoder, bottleneck placement, NODE adapter, and `zero_last_layer` initialization as the default starting point. Add a minimal regularization/configuration path that plugs into the existing training engine, expose the NODE states needed to compute the penalty, and add a tightly scoped experiment matrix so the first tuning round answers “does dynamics stabilization help?” before any larger structural changes such as `1x1 + NODE`, new datasets, or multi-seed expansion.

**Tech Stack:** Python, PyTorch, YAML experiment configs, pytest, existing low-data runner/reporting stack

---

## Scope Check

This plan covers one subsystem only: **bottleneck NODE stability tuning**. It deliberately excludes:

- output-side NODE tuning
- GlaS experiment rollout
- new external comparison methods
- multi-seed expansion
- large structural changes such as multi-position adapters

Those remain separate follow-up efforts. This plan is only for the first “worth continuing?” tuning pass on the main bottleneck NODE line.

## File Structure

### Existing files to modify

- `src/models/node_adapter.py`
  Responsibility: implement NODE dynamics and expose the per-step information needed for stability penalties without changing the forward API used by the segmentation model.

- `src/models/segmentation_model.py`
  Responsibility: carry extra NODE diagnostics from the adapter to the training engine while preserving current behavior for non-NODE groups.

- `src/training/engine.py`
  Responsibility: compute the task loss plus optional NODE stability regularization, log the regularized loss components, and save them to `history.csv` / `metrics.json`.

- `src/experiments/low_data_runner.py`
  Responsibility: validate new config keys, pass regularization knobs into training, and keep the bottleneck NODE tuning configs wired into the existing experiment entrypoint.

- `src/analysis/report_visualization.py`
  Responsibility: ingest optional tuning metrics such as regularization strength and best/final gap so the tuning runs remain visible in the same report flow.

- `tests/test_segmentation_model.py`
  Responsibility: assert NODE diagnostics are exposed correctly and do not affect non-NODE paths.

- `tests/test_training_engine.py`
  Responsibility: verify optional regularization is added only when configured, logged correctly, and does not break the default training path.

- `tests/test_low_data_runner.py`
  Responsibility: verify new config keys parse and route correctly.

- `tests/test_report_visualization.py`
  Responsibility: verify the reporting layer tolerates and aggregates the new metrics fields.

### New files to create

- `configs/experiments/isic2018_low_data_node_c_zero_last_kinetic.yaml`
  First stability-tuning config: zero-last bottleneck NODE plus kinetic regularization at the current default integration setting.

- `configs/experiments/isic2018_low_data_node_c_zero_last_kinetic_steps8.yaml`
  Same regularization but finer integration, used to test whether stabilization and steps interact positively.

- `configs/experiments/isic2018_low_data_node_c_zero_last_kinetic_rk4.yaml`
  Small solver follow-up, only after the kinetic penalty path exists.

- `docs/experiments/2026-05-28-bottleneck-node-stability-matrix.md`
  Human-readable experiment matrix and stopping criteria for the first tuning wave.

## Tuning Policy

The implementation should preserve these project decisions:

- Main method remains **bottleneck NODE**, not output-side NODE.
- Default initialization for tuning starts from `adapter.init: zero_last_layer`.
- First regularization target is **kinetic regularization** because it is the least invasive stability control that fits the current lightweight NODE implementation.
- Jacobian-style penalties are explicitly deferred until kinetic regularization is tested and either fails or only partially helps.
- `steps` and `solver` changes are a **small follow-up matrix**, not a broad grid search.

## Experiment Matrix To Enable

The first tuning wave should enable exactly these experiment types:

1. Existing reference:
   - `isic2018_low_data_node_c_zero_last.yaml`
2. New regularized baseline:
   - `isic2018_low_data_node_c_zero_last_kinetic.yaml`
3. Regularized + finer integration:
   - `isic2018_low_data_node_c_zero_last_kinetic_steps8.yaml`
4. Optional small solver follow-up:
   - `isic2018_low_data_node_c_zero_last_kinetic_rk4.yaml`

Stop after this wave and inspect:

- `best_val_dice`
- `best_epoch`
- `epochs_ran`
- `duration_sec`
- `avg_epoch_sec`
- `peak_final_gap` or equivalent best-vs-final behavior from history

Only continue to structural tuning if regularization plus small integration changes do not improve stability enough.

### Task 1: Add config plumbing for stability regularization

**Files:**
- Modify: `src/experiments/low_data_runner.py`
- Modify: `tests/test_low_data_runner.py`

- [ ] **Step 1: Write the failing config-parse tests**

Add tests that define the config contract for the new tuning fields. Extend `tests/test_low_data_runner.py` with assertions like:

```python
def test_stability_tuning_configs_parse_with_expected_regularization_fields():
    from src.experiments.low_data_runner import load_config

    expected = {
        "configs/experiments/isic2018_low_data_node_c_zero_last_kinetic.yaml": {
            "type": "kinetic",
            "weight": 1e-4,
            "steps": 4,
            "solver": "euler",
        },
        "configs/experiments/isic2018_low_data_node_c_zero_last_kinetic_steps8.yaml": {
            "type": "kinetic",
            "weight": 1e-4,
            "steps": 8,
            "solver": "euler",
        },
        "configs/experiments/isic2018_low_data_node_c_zero_last_kinetic_rk4.yaml": {
            "type": "kinetic",
            "weight": 1e-4,
            "steps": 4,
            "solver": "rk4",
        },
    }

    for path, values in expected.items():
        config = load_config(path)
        reg = config["regularization"]
        assert reg["type"] == values["type"]
        assert float(reg["weight"]) == values["weight"]
        assert int(config["node"]["steps"]) == values["steps"]
        assert str(config["node"]["solver"]) == values["solver"]
        assert config["adapter"]["init"] == "zero_last_layer"
```

- [ ] **Step 2: Run the targeted config test and verify it fails**

Run:

```bash
pytest tests/test_low_data_runner.py::test_stability_tuning_configs_parse_with_expected_regularization_fields -q
```

Expected: FAIL because the config files and/or fields do not exist yet.

- [ ] **Step 3: Add config validation support in the runner**

Update `src/experiments/low_data_runner.py` so the runner accepts an optional `regularization` mapping and validates it narrowly:

```python
def _resolve_regularization_config(config: dict[str, Any]) -> dict[str, Any]:
    regularization = config.get("regularization", {})
    if regularization is None:
        return {"type": "none", "weight": 0.0}
    if not isinstance(regularization, dict):
        raise ValueError(
            "config.regularization must be a mapping/dict when provided."
        )

    reg_type = str(regularization.get("type", "none"))
    if reg_type not in {"none", "kinetic"}:
        raise ValueError(
            "config.regularization.type must be one of ['none', 'kinetic']; "
            f"got {reg_type!r}."
        )

    weight = float(regularization.get("weight", 0.0))
    if weight < 0.0:
        raise ValueError(
            "config.regularization.weight must be >= 0.0; "
            f"got {weight!r}."
        )

    return {"type": reg_type, "weight": weight}
```

Also route the parsed dict into the `fit(...)` call:

```python
regularization = _resolve_regularization_config(config)

best_checkpoint_path = fit(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    epochs=int(config["train"]["epochs"]),
    patience=int(config["train"]["early_stopping_patience"]),
    output_dir=group_output_dir,
    device=device,
    save_best_checkpoint=save_best_checkpoint,
    regularization=regularization,
)
```

- [ ] **Step 4: Create the first three tuning configs**

Create `configs/experiments/isic2018_low_data_node_c_zero_last_kinetic.yaml`:

```yaml
seed: 42
paths:
  train_images_dir: data/train/images
  train_masks_dir: data/train/labels
  val_images_dir: data/validation/images
  val_masks_dir: data/validation/labels
  artifacts_dir: artifacts/low_data_tuning/c_zero_last_kinetic
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
regularization:
  type: kinetic
  weight: 0.0001
train:
  batch_size: 8
  epochs: 50
  learning_rate: 0.0003
  weight_decay: 0.0001
  early_stopping_patience: 8
  save_best_checkpoint: false
```

Create `configs/experiments/isic2018_low_data_node_c_zero_last_kinetic_steps8.yaml`:

```yaml
seed: 42
paths:
  train_images_dir: data/train/images
  train_masks_dir: data/train/labels
  val_images_dir: data/validation/images
  val_masks_dir: data/validation/labels
  artifacts_dir: artifacts/low_data_tuning/c_zero_last_kinetic_steps8
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
regularization:
  type: kinetic
  weight: 0.0001
train:
  batch_size: 8
  epochs: 50
  learning_rate: 0.0003
  weight_decay: 0.0001
  early_stopping_patience: 8
  save_best_checkpoint: false
```

Create `configs/experiments/isic2018_low_data_node_c_zero_last_kinetic_rk4.yaml`:

```yaml
seed: 42
paths:
  train_images_dir: data/train/images
  train_masks_dir: data/train/labels
  val_images_dir: data/validation/images
  val_masks_dir: data/validation/labels
  artifacts_dir: artifacts/low_data_tuning/c_zero_last_kinetic_rk4
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
  solver: rk4
  steps: 4
  step_size: 0.25
regularization:
  type: kinetic
  weight: 0.0001
train:
  batch_size: 8
  epochs: 50
  learning_rate: 0.0003
  weight_decay: 0.0001
  early_stopping_patience: 8
  save_best_checkpoint: false
```

- [ ] **Step 5: Run the targeted config test and verify it passes**

Run:

```bash
pytest tests/test_low_data_runner.py::test_stability_tuning_configs_parse_with_expected_regularization_fields -q
```

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/experiments/low_data_runner.py tests/test_low_data_runner.py configs/experiments/isic2018_low_data_node_c_zero_last_kinetic.yaml configs/experiments/isic2018_low_data_node_c_zero_last_kinetic_steps8.yaml configs/experiments/isic2018_low_data_node_c_zero_last_kinetic_rk4.yaml
git commit -m "feat: add stability tuning config plumbing"
```

### Task 2: Expose NODE dynamics diagnostics for regularization

**Files:**
- Modify: `src/models/node_adapter.py`
- Modify: `src/models/segmentation_model.py`
- Modify: `tests/test_segmentation_model.py`

- [ ] **Step 1: Write the failing model diagnostics test**

Add a test that states the required contract: NODE-based bottleneck runs expose a per-step kinetic signal, while non-NODE paths do not.

```python
def test_segmentation_model_exposes_node_diagnostics_for_bottleneck_node():
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
    x = torch.randn(2, 3, 64, 64)

    output = model(x)

    assert hasattr(output, "node_diagnostics")
    assert output.node_diagnostics is not None
    assert "kinetic_terms" in output.node_diagnostics
    assert len(output.node_diagnostics["kinetic_terms"]) == 4
```

Add the non-NODE guard test:

```python
def test_segmentation_model_non_node_paths_do_not_emit_node_diagnostics():
    model = build_segmentation_model(
        encoder_name="resnet18",
        encoder_weights=None,
        in_channels=3,
        num_classes=1,
        adapter_type="conv",
        bottleneck_channels=32,
        adapter_hidden_channels=16,
        freeze_encoder=True,
        node_steps=4,
        node_step_size=0.25,
    )
    x = torch.randn(2, 3, 64, 64)

    output = model(x)

    assert hasattr(output, "node_diagnostics")
    assert output.node_diagnostics is None
```

- [ ] **Step 2: Run the model test and verify it fails**

Run:

```bash
pytest tests/test_segmentation_model.py::test_segmentation_model_exposes_node_diagnostics_for_bottleneck_node tests/test_segmentation_model.py::test_segmentation_model_non_node_paths_do_not_emit_node_diagnostics -q
```

Expected: FAIL because the output structure does not yet expose `node_diagnostics`.

- [ ] **Step 3: Extend the NODE adapter to collect per-step kinetic terms**

Modify `src/models/node_adapter.py` to add a light-weight diagnostics path that does not change the tensor forward behavior:

```python
class NODEAdapter(nn.Module):
    def __init__(self, ...):
        super().__init__()
        ...
        self._last_diagnostics: dict[str, list[torch.Tensor]] | None = None

    @property
    def last_diagnostics(self) -> dict[str, list[torch.Tensor]] | None:
        return self._last_diagnostics

    def _store_diagnostics(self, kinetic_terms: list[torch.Tensor]) -> None:
        self._last_diagnostics = {"kinetic_terms": kinetic_terms}

    def _forward_euler(self, x: torch.Tensor) -> torch.Tensor:
        z = x
        kinetic_terms: list[torch.Tensor] = []
        for _ in range(self.steps):
            dz = self.func(z)
            kinetic_terms.append((dz.square()).mean())
            z = z + self.step_size * dz
        self._store_diagnostics(kinetic_terms)
        return z

    def _forward_rk4(self, x: torch.Tensor) -> torch.Tensor:
        z = x
        h = self.step_size
        kinetic_terms: list[torch.Tensor] = []
        for _ in range(self.steps):
            k1 = self.func(z)
            kinetic_terms.append((k1.square()).mean())
            k2 = self.func(z + 0.5 * h * k1)
            k3 = self.func(z + 0.5 * h * k2)
            k4 = self.func(z + h * k3)
            z = z + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        self._store_diagnostics(kinetic_terms)
        return z
```

- [ ] **Step 4: Thread diagnostics through the segmentation model output**

Modify `src/models/segmentation_model.py` so the output carries optional NODE diagnostics:

```python
class SegmentationModelOutput(NamedTuple):
    logits: torch.Tensor
    bottleneck: torch.Tensor
    adapted_bottleneck: torch.Tensor
    output_adapter_activation: torch.Tensor | None = None
    node_diagnostics: dict[str, list[torch.Tensor]] | None = None
```

Then in `forward(...)`:

```python
node_diagnostics = None
if hasattr(self.adapter, "last_diagnostics"):
    node_diagnostics = getattr(self.adapter, "last_diagnostics")
if node_diagnostics is None and hasattr(self.output_adapter, "last_diagnostics"):
    node_diagnostics = getattr(self.output_adapter, "last_diagnostics")

return SegmentationModel.Output(
    logits=logits,
    bottleneck=bottleneck,
    adapted_bottleneck=adapted_bottleneck,
    output_adapter_activation=output_adapter_activation,
    node_diagnostics=node_diagnostics,
)
```

- [ ] **Step 5: Run the model tests and verify they pass**

Run:

```bash
pytest tests/test_segmentation_model.py::test_segmentation_model_exposes_node_diagnostics_for_bottleneck_node tests/test_segmentation_model.py::test_segmentation_model_non_node_paths_do_not_emit_node_diagnostics -q
```

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/models/node_adapter.py src/models/segmentation_model.py tests/test_segmentation_model.py
git commit -m "feat: expose node diagnostics for stability penalties"
```

### Task 3: Add kinetic regularization to the training engine

**Files:**
- Modify: `src/training/engine.py`
- Modify: `tests/test_training_engine.py`

- [ ] **Step 1: Write the failing engine tests**

Add one test for the helper and one for the logged metrics path.

```python
def test_compute_regularization_loss_returns_zero_when_disabled():
    from src.training.engine import compute_regularization_loss

    loss = compute_regularization_loss(
        model_output=object(),
        regularization={"type": "none", "weight": 0.0},
    )

    assert torch.is_tensor(loss)
    assert float(loss.item()) == 0.0
```

```python
def test_run_epoch_includes_kinetic_regularization_when_present():
    class _Output:
        def __init__(self, logits):
            self.logits = logits
            self.node_diagnostics = {
                "kinetic_terms": [torch.tensor(2.0), torch.tensor(4.0)]
            }

    class _Model(torch.nn.Module):
        def forward(self, x):
            return _Output(torch.zeros(x.shape[0], 1, x.shape[2], x.shape[3]))

    loader = [
        {
            "image": torch.randn(2, 3, 16, 16),
            "mask": torch.zeros(2, 16, 16),
        }
    ]

    metrics = run_epoch(
        model=_Model(),
        loader=loader,
        optimizer=None,
        regularization={"type": "kinetic", "weight": 0.5},
    )

    assert "reg_loss" in metrics
    assert metrics["reg_loss"] > 0.0
    assert "task_loss" in metrics
```

- [ ] **Step 2: Run the engine tests and verify they fail**

Run:

```bash
pytest tests/test_training_engine.py::test_compute_regularization_loss_returns_zero_when_disabled tests/test_training_engine.py::test_run_epoch_includes_kinetic_regularization_when_present -q
```

Expected: FAIL because the regularization helper/path does not exist yet.

- [ ] **Step 3: Implement a narrow regularization helper**

Add to `src/training/engine.py`:

```python
def compute_regularization_loss(
    *,
    model_output: Any,
    regularization: dict[str, Any] | None,
    device: torch.device | None = None,
) -> torch.Tensor:
    if device is None:
        device = torch.device("cpu")
    if not regularization:
        return torch.tensor(0.0, device=device)

    reg_type = str(regularization.get("type", "none"))
    reg_weight = float(regularization.get("weight", 0.0))
    if reg_type == "none" or reg_weight == 0.0:
        return torch.tensor(0.0, device=device)

    if reg_type != "kinetic":
        raise ValueError(f"Unsupported regularization type: {reg_type!r}")

    diagnostics = getattr(model_output, "node_diagnostics", None)
    if not diagnostics:
        return torch.tensor(0.0, device=device)

    kinetic_terms = diagnostics.get("kinetic_terms", [])
    if not kinetic_terms:
        return torch.tensor(0.0, device=device)

    kinetic_mean = torch.stack(list(kinetic_terms)).mean()
    return reg_weight * kinetic_mean
```

- [ ] **Step 4: Thread regularization through `run_epoch` and `fit`**

Extend `run_epoch(...)`:

```python
def run_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: Any | None = None,
    device: str | torch.device = "cpu",
    regularization: dict[str, Any] | None = None,
) -> dict[str, float]:
```

Inside the loop:

```python
task_loss = criterion(logits, masks)
reg_loss = compute_regularization_loss(
    model_output=model_output,
    regularization=regularization,
    device=device_t,
)
loss = task_loss + reg_loss
```

Track separate aggregates:

```python
task_loss_sum = 0.0
reg_loss_sum = 0.0
...
task_loss_sum += float(task_loss.item()) * batch_size
reg_loss_sum += float(reg_loss.item()) * batch_size
```

Return:

```python
return {
    "loss": loss_sum / denom,
    "task_loss": task_loss_sum / denom,
    "reg_loss": reg_loss_sum / denom,
    "dice": dice_sum / denom,
    "iou": iou_sum / denom,
}
```

Extend `fit(...)`:

```python
def fit(..., save_best_checkpoint: bool = True, regularization: dict[str, Any] | None = None) -> Path | None:
```

Pass `regularization=regularization` into both `run_epoch(...)` calls.

- [ ] **Step 5: Log the extra metrics in history and metrics JSON**

Extend the epoch record to carry the additional fields:

```python
@dataclass(frozen=True)
class EpochMetrics:
    epoch: int
    train_loss: float
    train_task_loss: float
    train_reg_loss: float
    val_loss: float
    val_task_loss: float
    val_reg_loss: float
    val_dice: float
    val_iou: float
```

Update `save_history(...)` fieldnames:

```python
fieldnames = [
    "epoch",
    "train_loss",
    "train_task_loss",
    "train_reg_loss",
    "val_loss",
    "val_task_loss",
    "val_reg_loss",
    "val_dice",
    "val_iou",
]
```

When saving the final metrics dict, include:

```python
metrics = {
    ...
    "regularization_type": regularization["type"] if regularization else "none",
    "regularization_weight": float(regularization["weight"]) if regularization else 0.0,
}
```

- [ ] **Step 6: Run the engine tests and verify they pass**

Run:

```bash
pytest tests/test_training_engine.py::test_compute_regularization_loss_returns_zero_when_disabled tests/test_training_engine.py::test_run_epoch_includes_kinetic_regularization_when_present -q
```

Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/training/engine.py tests/test_training_engine.py
git commit -m "feat: add kinetic regularization to training engine"
```

### Task 4: Keep report ingestion compatible with tuning runs

**Files:**
- Modify: `src/analysis/report_visualization.py`
- Modify: `tests/test_report_visualization.py`

- [ ] **Step 1: Write the failing reporting test**

Add a test that report ingestion preserves the new tuning metadata:

```python
def test_collect_run_rows_preserves_regularization_metadata(tmp_path):
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "best_val_dice": 0.80,
                "best_epoch": 10,
                "epochs_ran": 20,
                "duration_sec": 120.0,
                "avg_epoch_sec": 6.0,
                "regularization_type": "kinetic",
                "regularization_weight": 0.0001,
            }
        ),
        encoding="utf-8",
    )

    rows = collect_run_rows([metrics_path])

    assert rows[0]["regularization_type"] == "kinetic"
    assert rows[0]["regularization_weight"] == 0.0001
```

- [ ] **Step 2: Run the reporting test and verify it fails**

Run:

```bash
pytest tests/test_report_visualization.py::test_collect_run_rows_preserves_regularization_metadata -q
```

Expected: FAIL if the current row collector drops these fields.

- [ ] **Step 3: Extend report ingestion with optional tuning metadata**

Update the run collector in `src/analysis/report_visualization.py` to keep these optional keys when present:

```python
row["regularization_type"] = metrics.get("regularization_type", "none")
row["regularization_weight"] = metrics.get("regularization_weight", 0.0)
```

If there is a method summary aggregation helper, do not force these fields into every grouped summary. Keep the summary narrow and run-level rows rich.

- [ ] **Step 4: Run the reporting test and verify it passes**

Run:

```bash
pytest tests/test_report_visualization.py::test_collect_run_rows_preserves_regularization_metadata -q
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/analysis/report_visualization.py tests/test_report_visualization.py
git commit -m "feat: preserve tuning metadata in report ingestion"
```

### Task 5: Document the first tuning matrix and verify the narrow workflow

**Files:**
- Create: `docs/experiments/2026-05-28-bottleneck-node-stability-matrix.md`
- Modify: `tests/test_low_data_runner.py`

- [ ] **Step 1: Add a config smoke test for the new matrix**

Extend `tests/test_low_data_runner.py` with a single smoke test for all three configs:

```python
def test_stability_tuning_matrix_configs_smoke_parse():
    from src.experiments.low_data_runner import load_config

    paths = [
        "configs/experiments/isic2018_low_data_node_c_zero_last_kinetic.yaml",
        "configs/experiments/isic2018_low_data_node_c_zero_last_kinetic_steps8.yaml",
        "configs/experiments/isic2018_low_data_node_c_zero_last_kinetic_rk4.yaml",
    ]

    for path in paths:
        config = load_config(path)
        assert config["adapter"]["type"] == "node"
        assert config["adapter"]["init"] == "zero_last_layer"
        assert config["train"]["save_best_checkpoint"] is False
```

- [ ] **Step 2: Run the smoke test and verify it passes**

Run:

```bash
pytest tests/test_low_data_runner.py::test_stability_tuning_matrix_configs_smoke_parse -q
```

Expected: PASS

- [ ] **Step 3: Write the experiment-matrix doc**

Create `docs/experiments/2026-05-28-bottleneck-node-stability-matrix.md` with:

```markdown
# 2026-05-28 Bottleneck NODE Stability Matrix

## Objective

Decide whether stability-oriented tuning improves the bottleneck NODE main method enough to justify later multi-seed expansion.

## Runs

1. Reference
   - `configs/experiments/isic2018_low_data_node_c_zero_last.yaml`

2. Kinetic regularization baseline
   - `configs/experiments/isic2018_low_data_node_c_zero_last_kinetic.yaml`

3. Kinetic regularization + steps8
   - `configs/experiments/isic2018_low_data_node_c_zero_last_kinetic_steps8.yaml`

4. Kinetic regularization + RK4
   - `configs/experiments/isic2018_low_data_node_c_zero_last_kinetic_rk4.yaml`

## Success criteria

- `best_val_dice` exceeds the current zero-last bottleneck NODE reference, or
- `best_val_dice` is comparable while `best_epoch`, `epochs_ran`, or best-vs-final stability improves

## Stop conditions

- If all regularized runs are worse than the reference and slower, do not continue to wider solver tuning.
- If the regularized steps8 run improves stability but not score, keep it as a candidate for GlaS before multi-seed expansion.
- If the regularized RK4 run is slower with no measurable gain, drop RK4 from the main line.
```

- [ ] **Step 4: Run the focused verification batch**

Run:

```bash
pytest tests/test_segmentation_model.py tests/test_training_engine.py tests/test_low_data_runner.py tests/test_report_visualization.py -q
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add docs/experiments/2026-05-28-bottleneck-node-stability-matrix.md tests/test_low_data_runner.py
git commit -m "docs: add bottleneck node stability tuning matrix"
```

## Final Verification

- [ ] **Step 1: Run the full targeted suite**

Run:

```bash
pytest tests/test_segmentation_model.py tests/test_training_engine.py tests/test_low_data_runner.py tests/test_report_visualization.py -q
```

Expected: PASS with all new tuning-path tests green.

- [ ] **Step 2: Run config smoke checks**

Run:

```bash
python - <<'PY'
from src.experiments.low_data_runner import load_config

paths = [
    "configs/experiments/isic2018_low_data_node_c_zero_last.yaml",
    "configs/experiments/isic2018_low_data_node_c_zero_last_kinetic.yaml",
    "configs/experiments/isic2018_low_data_node_c_zero_last_kinetic_steps8.yaml",
    "configs/experiments/isic2018_low_data_node_c_zero_last_kinetic_rk4.yaml",
]

for path in paths:
    cfg = load_config(path)
    print(
        path,
        cfg["adapter"]["init"],
        cfg["node"]["solver"],
        cfg["node"]["steps"],
        cfg.get("regularization", {}).get("type", "none"),
        cfg.get("regularization", {}).get("weight", 0.0),
    )
PY
```

Expected output shape:

```text
configs/experiments/isic2018_low_data_node_c_zero_last.yaml zero_last_layer euler 4 none 0.0
configs/experiments/isic2018_low_data_node_c_zero_last_kinetic.yaml zero_last_layer euler 4 kinetic 0.0001
...
```

- [ ] **Step 3: Run one end-to-end smoke experiment**

Run:

```bash
python scripts/run_low_data_experiment.py --config configs/experiments/isic2018_low_data_node_c_zero_last_kinetic.yaml --group C
```

Expected:

- training completes
- `artifacts/low_data_tuning/c_zero_last_kinetic/group_c/history.csv` exists
- `artifacts/low_data_tuning/c_zero_last_kinetic/group_c/metrics.json` exists
- metrics JSON includes `regularization_type`, `regularization_weight`, `duration_sec`, and `avg_epoch_sec`

## Not In This Plan

These are explicitly deferred:

- Jacobian penalty implementation
- spectral normalization / Lipschitz proxy
- `1x1 + NODE` structural tuning
- GlaS rollout for tuned configs
- multi-seed expansion for tuned configs

If the first tuning wave is promising, those should be handled in a follow-up plan, not bolted onto this one mid-execution.
