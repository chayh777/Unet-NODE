# Report Visualization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a report-ready visualization script for current low-data NODE results, split into baseline multi-seed comparison and NODE steps/init ablation figures.

**Architecture:** Add a focused `src.analysis.report_visualization` module for reading experiment artifacts, building summary tables, and saving figures. Add a thin CLI script in `scripts/` that calls the module with default artifact locations and optional output directory. Keep this separate from `low_data_reporting.py` because this report compares many independent artifact roots rather than groups under one shared root.

**Tech Stack:** Python, pandas, matplotlib Agg backend, pytest, existing artifact format (`history.csv`, `metrics.json`).

---

## File Structure

- Create `src/analysis/report_visualization.py`: artifact summarization, method/run mapping, baseline plots, ablation plots, and `write_report_visualizations`.
- Create `scripts/plot_report_results.py`: CLI entrypoint for generating all report CSVs and PNGs.
- Create `tests/test_report_visualization.py`: unit tests for summary extraction, multi-seed table construction, ablation table construction, and output generation with synthetic artifacts.
- Modify `tests/test_low_data_clis.py`: add a CLI wiring test for `scripts/plot_report_results.py`.

## Output Contract

The script writes all outputs to `artifacts/report_figures/` by default:

```text
artifacts/report_figures/multiseed_summary.csv
artifacts/report_figures/multiseed_method_summary.csv
artifacts/report_figures/steps_ablation_summary.csv
artifacts/report_figures/multiseed_best_dice.png
artifacts/report_figures/multiseed_final_dice.png
artifacts/report_figures/multiseed_delta_vs_b.png
artifacts/report_figures/steps_best_dice.png
artifacts/report_figures/steps_final_dice.png
artifacts/report_figures/steps_peak_final_gap.png
```

## Task 1: Artifact Summarization Core

**Files:**
- Create: `src/analysis/report_visualization.py`
- Test: `tests/test_report_visualization.py`

- [ ] **Step 1: Write failing tests for one-run summarization**

Create `tests/test_report_visualization.py` with:

```python
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest


def _write_run(root: Path, *, best: float, rows: list[dict[str, float | int]]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(root / "history.csv", index=False)
    (root / "metrics.json").write_text(
        json.dumps(
            {
                "best_val_dice": best,
                "epochs_ran": len(rows),
                "best_checkpoint": str(root / "best.pt"),
            }
        ),
        encoding="utf-8",
    )


def test_summarize_run_reads_best_final_and_gap(tmp_path: Path) -> None:
    from src.analysis.report_visualization import summarize_run

    root = tmp_path / "exp" / "group_c"
    _write_run(
        root,
        best=0.78,
        rows=[
            {"epoch": 2, "train_loss": 0.8, "val_loss": 0.7, "val_dice": 0.78, "val_iou": 0.65},
            {"epoch": 1, "train_loss": 1.0, "val_loss": 0.9, "val_dice": 0.70, "val_iou": 0.54},
            {"epoch": 3, "train_loss": 0.7, "val_loss": 0.8, "val_dice": 0.75, "val_iou": 0.61},
        ],
    )

    row = summarize_run(root=root, method="Method", run="exp", seed=7)

    assert row == {
        "method": "Method",
        "run": "exp",
        "seed": 7,
        "group": "group_c",
        "best_dice": 0.78,
        "best_iou": 0.65,
        "best_epoch": 2,
        "final_dice": 0.75,
        "peak_final_gap": pytest.approx(0.03),
        "epochs_ran": 3,
        "root": str(root),
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python -m pytest tests/test_report_visualization.py::test_summarize_run_reads_best_final_and_gap -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'src.analysis.report_visualization'`.

- [ ] **Step 3: Implement `summarize_run` and plotting backend setup**

Create `src/analysis/report_visualization.py` with:

```python
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd


def _get_plotting_libs():
    mpl_config_dir = Path.cwd() / ".matplotlib"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def summarize_run(
    *,
    root: str | Path,
    method: str,
    run: str,
    seed: int | None,
) -> dict[str, Any]:
    root = Path(root)
    history_path = root / "history.csv"
    metrics_path = root / "metrics.json"
    if not history_path.exists():
        raise FileNotFoundError(f"Missing history.csv at {history_path}")
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics.json at {metrics_path}")

    history = pd.read_csv(history_path).sort_values("epoch")
    if history.empty:
        raise ValueError(f"history.csv is empty at {history_path}")

    with metrics_path.open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    if not isinstance(metrics, dict):
        raise ValueError(f"metrics.json must contain an object at {metrics_path}")
    if "best_val_dice" not in metrics:
        raise ValueError(f"metrics.json missing best_val_dice at {metrics_path}")

    best_dice = float(metrics["best_val_dice"])
    best_idx = (history["val_dice"] - best_dice).abs().idxmin()
    best_row = history.loc[int(best_idx)]
    final_row = history.iloc[-1]

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
        "root": str(root),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
python -m pytest tests/test_report_visualization.py::test_summarize_run_reads_best_final_and_gap -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/analysis/report_visualization.py tests/test_report_visualization.py
git commit -m "feat: add report run summarization"
```

## Task 2: Multi-Seed Table Builder

**Files:**
- Modify: `src/analysis/report_visualization.py`
- Modify: `tests/test_report_visualization.py`

- [ ] **Step 1: Add failing tests for multi-seed collection and method summary**

Append to `tests/test_report_visualization.py`:

```python
def test_build_multiseed_tables_collects_known_methods(tmp_path: Path) -> None:
    from src.analysis.report_visualization import build_multiseed_tables

    artifacts_dir = tmp_path / "artifacts"
    _write_run(
        artifacts_dir / "low_data_multiseed" / "b_seed0" / "group_b",
        best=0.70,
        rows=[
            {"epoch": 1, "train_loss": 1.0, "val_loss": 0.9, "val_dice": 0.70, "val_iou": 0.50},
            {"epoch": 2, "train_loss": 0.9, "val_loss": 0.8, "val_dice": 0.68, "val_iou": 0.48},
        ],
    )
    _write_run(
        artifacts_dir / "low_data_multiseed" / "c_zero_last_fine_integration_seed0" / "group_c",
        best=0.75,
        rows=[
            {"epoch": 1, "train_loss": 1.0, "val_loss": 0.9, "val_dice": 0.73, "val_iou": 0.55},
            {"epoch": 2, "train_loss": 0.9, "val_loss": 0.8, "val_dice": 0.75, "val_iou": 0.57},
        ],
    )

    runs, summary = build_multiseed_tables(artifacts_dir)

    assert runs["method"].tolist() == ["B-base", "C-zero-last-steps8"]
    assert runs["seed"].tolist() == [0, 0]
    assert summary["method"].tolist() == ["B-base", "C-zero-last-steps8"]
    assert summary.loc[summary["method"] == "C-zero-last-steps8", "best_dice_mean"].iloc[0] == 0.75
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python -m pytest tests/test_report_visualization.py::test_build_multiseed_tables_collects_known_methods -q
```

Expected: FAIL because `build_multiseed_tables` does not exist.

- [ ] **Step 3: Implement multi-seed method mapping**

Append to `src/analysis/report_visualization.py`:

```python
_MULTISEED_METHODS = {
    "b_seed": "B-base",
    "c_fine_integration_seed": "C-fine-steps8-default",
    "c_zero_last_fine_integration_seed": "C-zero-last-steps8",
    "c_zero_last_steps16_seed": "C-zero-last-steps16",
}

_MULTISEED_ORDER = [
    "B-base",
    "C-fine-steps8-default",
    "C-zero-last-steps8",
    "C-zero-last-steps16",
]


def _parse_seed(run_name: str) -> int | None:
    marker = "_seed"
    if marker not in run_name:
        return None
    suffix = run_name.rsplit(marker, 1)[-1]
    return int(suffix) if suffix.isdigit() else None


def _method_for_multiseed_run(run_name: str) -> str | None:
    for prefix, method in _MULTISEED_METHODS.items():
        if run_name.startswith(prefix):
            return method
    return None


def build_multiseed_tables(artifacts_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    artifacts_dir = Path(artifacts_dir)
    roots = sorted((artifacts_dir / "low_data_multiseed").glob("*/group_*"))
    rows: list[dict[str, Any]] = []
    for root in roots:
        run_name = root.parent.name
        method = _method_for_multiseed_run(run_name)
        if method is None:
            continue
        rows.append(
            summarize_run(
                root=root,
                method=method,
                run=run_name,
                seed=_parse_seed(run_name),
            )
        )

    runs = pd.DataFrame(rows)
    if runs.empty:
        columns = [
            "method",
            "run",
            "seed",
            "group",
            "best_dice",
            "best_iou",
            "best_epoch",
            "final_dice",
            "peak_final_gap",
            "epochs_ran",
            "root",
        ]
        return pd.DataFrame(columns=columns), pd.DataFrame()

    runs["method"] = pd.Categorical(runs["method"], categories=_MULTISEED_ORDER, ordered=True)
    runs = runs.sort_values(["method", "seed"]).reset_index(drop=True)
    runs["method"] = runs["method"].astype(str)

    summary = (
        runs.groupby("method", observed=True)
        .agg(
            n=("seed", "count"),
            best_dice_mean=("best_dice", "mean"),
            best_dice_std=("best_dice", "std"),
            final_dice_mean=("final_dice", "mean"),
            final_dice_std=("final_dice", "std"),
            gap_mean=("peak_final_gap", "mean"),
            gap_std=("peak_final_gap", "std"),
            best_iou_mean=("best_iou", "mean"),
            best_epoch_mean=("best_epoch", "mean"),
            epochs_ran_mean=("epochs_ran", "mean"),
        )
        .reset_index()
    )
    summary["method"] = pd.Categorical(summary["method"], categories=_MULTISEED_ORDER, ordered=True)
    summary = summary.sort_values("method").reset_index(drop=True)
    summary["method"] = summary["method"].astype(str)
    return runs, summary
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
python -m pytest tests/test_report_visualization.py::test_build_multiseed_tables_collects_known_methods -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/analysis/report_visualization.py tests/test_report_visualization.py
git commit -m "feat: build report multiseed tables"
```

## Task 3: Steps Ablation Table Builder

**Files:**
- Modify: `src/analysis/report_visualization.py`
- Modify: `tests/test_report_visualization.py`

- [ ] **Step 1: Add failing test for steps ablation collection**

Append to `tests/test_report_visualization.py`:

```python
def test_build_steps_ablation_table_collects_default_and_zero_last(tmp_path: Path) -> None:
    from src.analysis.report_visualization import build_steps_ablation_table

    artifacts_dir = tmp_path / "artifacts"
    _write_run(
        artifacts_dir / "low_data" / "group_c",
        best=0.71,
        rows=[
            {"epoch": 1, "train_loss": 1.0, "val_loss": 0.9, "val_dice": 0.70, "val_iou": 0.50},
            {"epoch": 2, "train_loss": 0.9, "val_loss": 0.8, "val_dice": 0.71, "val_iou": 0.51},
        ],
    )
    _write_run(
        artifacts_dir / "low_data_followup" / "c_zero_last_fine_integration" / "group_c",
        best=0.76,
        rows=[
            {"epoch": 1, "train_loss": 1.0, "val_loss": 0.9, "val_dice": 0.74, "val_iou": 0.56},
            {"epoch": 2, "train_loss": 0.9, "val_loss": 0.8, "val_dice": 0.76, "val_iou": 0.58},
        ],
    )

    table = build_steps_ablation_table(artifacts_dir)

    assert table[["init", "steps", "run", "best_dice"]].to_dict("records") == [
        {"init": "default", "steps": 4, "run": "c_base", "best_dice": 0.71},
        {"init": "zero_last", "steps": 8, "run": "c_zero_last_fine_integration", "best_dice": 0.76},
    ]
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python -m pytest tests/test_report_visualization.py::test_build_steps_ablation_table_collects_default_and_zero_last -q
```

Expected: FAIL because `build_steps_ablation_table` does not exist.

- [ ] **Step 3: Implement steps ablation mapping**

Append to `src/analysis/report_visualization.py`:

```python
_STEPS_ABLATION_RUNS = [
    ("default", 2, "c_steps2_t1", "low_data_followup/c_steps2_t1/group_c"),
    ("default", 4, "c_base", "low_data/group_c"),
    ("default", 8, "c_fine_integration", "low_data_followup/c_fine_integration/group_c"),
    ("default", 16, "c_steps16_t1", "low_data_followup/c_steps16_t1/group_c"),
    ("zero_last", 2, "c_zero_last_steps2_t1", "low_data_followup/c_zero_last_steps2_t1/group_c"),
    ("zero_last", 4, "c_zero_last", "low_data_followup/c_zero_last/group_c"),
    ("zero_last", 8, "c_zero_last_fine_integration", "low_data_followup/c_zero_last_fine_integration/group_c"),
    ("zero_last", 16, "c_zero_last_steps16_t1", "low_data_followup/c_zero_last_steps16_t1/group_c"),
]


def build_steps_ablation_table(artifacts_dir: str | Path) -> pd.DataFrame:
    artifacts_dir = Path(artifacts_dir)
    rows: list[dict[str, Any]] = []
    for init, steps, run_name, relative_root in _STEPS_ABLATION_RUNS:
        root = artifacts_dir / relative_root
        if not root.exists():
            continue
        row = summarize_run(root=root, method=f"{init}-steps{steps}", run=run_name, seed=None)
        row["init"] = init
        row["steps"] = steps
        row["T"] = 1.0
        rows.append(row)

    table = pd.DataFrame(rows)
    if table.empty:
        return pd.DataFrame(
            columns=[
                "init",
                "steps",
                "T",
                "run",
                "best_dice",
                "best_iou",
                "best_epoch",
                "final_dice",
                "peak_final_gap",
                "epochs_ran",
                "root",
            ]
        )
    return table.sort_values(["init", "steps"]).reset_index(drop=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
python -m pytest tests/test_report_visualization.py::test_build_steps_ablation_table_collects_default_and_zero_last -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/analysis/report_visualization.py tests/test_report_visualization.py
git commit -m "feat: build report steps ablation table"
```

## Task 4: Baseline Multi-Seed Figures

**Files:**
- Modify: `src/analysis/report_visualization.py`
- Modify: `tests/test_report_visualization.py`

- [ ] **Step 1: Add failing test for baseline figure writing**

Append to `tests/test_report_visualization.py`:

```python
def test_save_multiseed_figures_writes_expected_pngs(tmp_path: Path) -> None:
    from src.analysis.report_visualization import save_multiseed_figures

    runs = pd.DataFrame(
        [
            {"method": "B-base", "seed": 0, "best_dice": 0.70, "final_dice": 0.68, "peak_final_gap": 0.02},
            {"method": "B-base", "seed": 1, "best_dice": 0.72, "final_dice": 0.71, "peak_final_gap": 0.01},
            {"method": "C-zero-last-steps8", "seed": 0, "best_dice": 0.75, "final_dice": 0.73, "peak_final_gap": 0.02},
            {"method": "C-zero-last-steps8", "seed": 1, "best_dice": 0.76, "final_dice": 0.74, "peak_final_gap": 0.02},
        ]
    )
    summary = pd.DataFrame(
        [
            {"method": "B-base", "best_dice_mean": 0.71, "best_dice_std": 0.014, "final_dice_mean": 0.695, "final_dice_std": 0.021},
            {"method": "C-zero-last-steps8", "best_dice_mean": 0.755, "best_dice_std": 0.007, "final_dice_mean": 0.735, "final_dice_std": 0.007},
        ]
    )

    save_multiseed_figures(runs=runs, summary=summary, output_dir=tmp_path)

    assert (tmp_path / "multiseed_best_dice.png").exists()
    assert (tmp_path / "multiseed_final_dice.png").exists()
    assert (tmp_path / "multiseed_delta_vs_b.png").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python -m pytest tests/test_report_visualization.py::test_save_multiseed_figures_writes_expected_pngs -q
```

Expected: FAIL because `save_multiseed_figures` does not exist.

- [ ] **Step 3: Implement baseline figure helpers**

Append to `src/analysis/report_visualization.py`:

```python
_COLORS = {
    "B-base": "#7a7a7a",
    "C-fine-steps8-default": "#377eb8",
    "C-zero-last-steps8": "#e15759",
    "C-zero-last-steps16": "#984ea3",
    "default": "#377eb8",
    "zero_last": "#e15759",
}


def _method_order(methods: list[str]) -> list[str]:
    return [method for method in _MULTISEED_ORDER if method in set(methods)]


def _save_bar_with_points(
    *,
    runs: pd.DataFrame,
    summary: pd.DataFrame,
    metric: str,
    mean_col: str,
    std_col: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    plt = _get_plotting_libs()
    order = _method_order([str(x) for x in summary["method"].tolist()])
    x_positions = list(range(len(order)))
    means = [float(summary.loc[summary["method"] == method, mean_col].iloc[0]) for method in order]
    stds = [float(summary.loc[summary["method"] == method, std_col].fillna(0.0).iloc[0]) for method in order]
    colors = [_COLORS.get(method, "#4f83cc") for method in order]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(x_positions, means, yerr=stds, capsize=4, color=colors, alpha=0.85)
    for idx, method in enumerate(order):
        method_rows = runs[runs["method"] == method].sort_values("seed")
        offsets = [-0.12, 0.0, 0.12, 0.24, -0.24]
        for point_idx, (_, row) in enumerate(method_rows.iterrows()):
            ax.scatter(
                idx + offsets[point_idx % len(offsets)],
                float(row[metric]),
                color="black",
                s=24,
                zorder=3,
            )

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(order, rotation=20, ha="right")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _save_delta_vs_b(runs: pd.DataFrame, output_path: Path) -> None:
    plt = _get_plotting_libs()
    wide_best = runs.pivot(index="seed", columns="method", values="best_dice")
    wide_final = runs.pivot(index="seed", columns="method", values="final_dice")
    methods = [method for method in _MULTISEED_ORDER if method != "B-base" and method in wide_best.columns]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    for method in methods:
        axes[0].plot(
            wide_best.index,
            wide_best[method] - wide_best["B-base"],
            marker="o",
            label=method,
            color=_COLORS.get(method),
        )
        axes[1].plot(
            wide_final.index,
            wide_final[method] - wide_final["B-base"],
            marker="o",
            label=method,
            color=_COLORS.get(method),
        )

    axes[0].set_title("Best Dice Delta vs B-base")
    axes[0].set_ylabel("Delta Dice")
    axes[1].set_title("Final Dice Delta vs B-base")
    for ax in axes:
        ax.axhline(0.0, color="black", linewidth=1, linestyle="--")
        ax.set_xlabel("Seed")
        ax.grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_multiseed_figures(*, runs: pd.DataFrame, summary: pd.DataFrame, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if runs.empty or summary.empty:
        return
    _save_bar_with_points(
        runs=runs,
        summary=summary,
        metric="best_dice",
        mean_col="best_dice_mean",
        std_col="best_dice_std",
        ylabel="Best Dice",
        title="Multi-Seed Best Dice",
        output_path=output_dir / "multiseed_best_dice.png",
    )
    _save_bar_with_points(
        runs=runs,
        summary=summary,
        metric="final_dice",
        mean_col="final_dice_mean",
        std_col="final_dice_std",
        ylabel="Final Dice",
        title="Multi-Seed Final Dice",
        output_path=output_dir / "multiseed_final_dice.png",
    )
    if "B-base" in set(runs["method"]):
        _save_delta_vs_b(runs, output_dir / "multiseed_delta_vs_b.png")
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
python -m pytest tests/test_report_visualization.py::test_save_multiseed_figures_writes_expected_pngs -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/analysis/report_visualization.py tests/test_report_visualization.py
git commit -m "feat: plot multiseed report figures"
```

## Task 5: Steps Ablation Figures

**Files:**
- Modify: `src/analysis/report_visualization.py`
- Modify: `tests/test_report_visualization.py`

- [ ] **Step 1: Add failing test for steps figure writing**

Append to `tests/test_report_visualization.py`:

```python
def test_save_steps_ablation_figures_writes_expected_pngs(tmp_path: Path) -> None:
    from src.analysis.report_visualization import save_steps_ablation_figures

    table = pd.DataFrame(
        [
            {"init": "default", "steps": 2, "best_dice": 0.70, "final_dice": 0.68, "peak_final_gap": 0.02},
            {"init": "default", "steps": 4, "best_dice": 0.72, "final_dice": 0.69, "peak_final_gap": 0.03},
            {"init": "zero_last", "steps": 2, "best_dice": 0.74, "final_dice": 0.73, "peak_final_gap": 0.01},
            {"init": "zero_last", "steps": 4, "best_dice": 0.75, "final_dice": 0.75, "peak_final_gap": 0.00},
        ]
    )

    save_steps_ablation_figures(table=table, output_dir=tmp_path)

    assert (tmp_path / "steps_best_dice.png").exists()
    assert (tmp_path / "steps_final_dice.png").exists()
    assert (tmp_path / "steps_peak_final_gap.png").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python -m pytest tests/test_report_visualization.py::test_save_steps_ablation_figures_writes_expected_pngs -q
```

Expected: FAIL because `save_steps_ablation_figures` does not exist.

- [ ] **Step 3: Implement steps line plot helper**

Append to `src/analysis/report_visualization.py`:

```python
def _save_steps_line_plot(
    *,
    table: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    plt = _get_plotting_libs()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for init, group in table.groupby("init", sort=False):
        ordered = group.sort_values("steps")
        ax.plot(
            ordered["steps"],
            ordered[metric],
            marker="o",
            linewidth=2,
            label=str(init),
            color=_COLORS.get(str(init), "#4f83cc"),
        )
    ax.set_title(title)
    ax.set_xlabel("NODE Euler Steps (T=1)")
    ax.set_ylabel(ylabel)
    ax.set_xticks(sorted(table["steps"].unique()))
    ax.grid(True, alpha=0.3)
    ax.legend(title="Init")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_steps_ablation_figures(*, table: pd.DataFrame, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if table.empty:
        return
    _save_steps_line_plot(
        table=table,
        metric="best_dice",
        ylabel="Best Dice",
        title="Steps Ablation: Best Dice",
        output_path=output_dir / "steps_best_dice.png",
    )
    _save_steps_line_plot(
        table=table,
        metric="final_dice",
        ylabel="Final Dice",
        title="Steps Ablation: Final Dice",
        output_path=output_dir / "steps_final_dice.png",
    )
    _save_steps_line_plot(
        table=table,
        metric="peak_final_gap",
        ylabel="Best - Final Dice",
        title="Steps Ablation: Peak-Final Gap",
        output_path=output_dir / "steps_peak_final_gap.png",
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
python -m pytest tests/test_report_visualization.py::test_save_steps_ablation_figures_writes_expected_pngs -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/analysis/report_visualization.py tests/test_report_visualization.py
git commit -m "feat: plot steps ablation report figures"
```

## Task 6: End-to-End Writer and CLI

**Files:**
- Modify: `src/analysis/report_visualization.py`
- Create: `scripts/plot_report_results.py`
- Modify: `tests/test_report_visualization.py`
- Modify: `tests/test_low_data_clis.py`

- [ ] **Step 1: Add failing end-to-end writer test**

Append to `tests/test_report_visualization.py`:

```python
def test_write_report_visualizations_writes_tables_and_figures(tmp_path: Path) -> None:
    from src.analysis.report_visualization import write_report_visualizations

    artifacts_dir = tmp_path / "artifacts"
    _write_run(
        artifacts_dir / "low_data_multiseed" / "b_seed0" / "group_b",
        best=0.70,
        rows=[
            {"epoch": 1, "train_loss": 1.0, "val_loss": 0.9, "val_dice": 0.70, "val_iou": 0.50},
            {"epoch": 2, "train_loss": 0.9, "val_loss": 0.8, "val_dice": 0.68, "val_iou": 0.48},
        ],
    )
    _write_run(
        artifacts_dir / "low_data_multiseed" / "c_zero_last_fine_integration_seed0" / "group_c",
        best=0.75,
        rows=[
            {"epoch": 1, "train_loss": 1.0, "val_loss": 0.9, "val_dice": 0.73, "val_iou": 0.55},
            {"epoch": 2, "train_loss": 0.9, "val_loss": 0.8, "val_dice": 0.75, "val_iou": 0.57},
        ],
    )
    _write_run(
        artifacts_dir / "low_data" / "group_c",
        best=0.71,
        rows=[
            {"epoch": 1, "train_loss": 1.0, "val_loss": 0.9, "val_dice": 0.70, "val_iou": 0.50},
            {"epoch": 2, "train_loss": 0.9, "val_loss": 0.8, "val_dice": 0.71, "val_iou": 0.51},
        ],
    )

    output_dir = write_report_visualizations(
        artifacts_dir=artifacts_dir,
        output_dir=tmp_path / "report",
    )

    assert output_dir == tmp_path / "report"
    assert (output_dir / "multiseed_summary.csv").exists()
    assert (output_dir / "multiseed_method_summary.csv").exists()
    assert (output_dir / "steps_ablation_summary.csv").exists()
    assert (output_dir / "multiseed_best_dice.png").exists()
    assert (output_dir / "multiseed_final_dice.png").exists()
    assert (output_dir / "multiseed_delta_vs_b.png").exists()
    assert (output_dir / "steps_best_dice.png").exists()
```

- [ ] **Step 2: Add failing CLI wiring test**

Append to `tests/test_low_data_clis.py`:

```python
def test_plot_report_results_cli_passes_args_to_writer(tmp_path: Path, monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    analysis_dir = src_dir / "analysis"

    _ensure_package("src", src_dir)
    _ensure_package("src.analysis", analysis_dir)

    calls: dict[str, object] = {}
    report_module = ModuleType("src.analysis.report_visualization")

    def write_report_visualizations(*, artifacts_dir, output_dir):
        calls["report"] = {
            "artifacts_dir": Path(artifacts_dir),
            "output_dir": Path(output_dir),
        }
        return Path(output_dir)

    report_module.write_report_visualizations = write_report_visualizations
    monkeypatch.setitem(sys.modules, "src.analysis.report_visualization", report_module)

    module = _load_script_module(
        "scripts.plot_report_results", "scripts/plot_report_results.py"
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "plot_report_results.py",
            "--artifacts-dir",
            str(tmp_path / "artifacts"),
            "--output-dir",
            str(tmp_path / "figures"),
        ],
    )

    module.main()

    assert calls["report"] == {
        "artifacts_dir": tmp_path / "artifacts",
        "output_dir": tmp_path / "figures",
    }
```

- [ ] **Step 3: Run tests to verify they fail**

Run:

```bash
python -m pytest tests/test_report_visualization.py::test_write_report_visualizations_writes_tables_and_figures tests/test_low_data_clis.py::test_plot_report_results_cli_passes_args_to_writer -q
```

Expected: FAIL because `write_report_visualizations` and `scripts/plot_report_results.py` do not exist.

- [ ] **Step 4: Implement end-to-end writer**

Append to `src/analysis/report_visualization.py`:

```python
def write_report_visualizations(
    *,
    artifacts_dir: str | Path = "artifacts",
    output_dir: str | Path = "artifacts/report_figures",
) -> Path:
    artifacts_dir = Path(artifacts_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    multiseed_runs, multiseed_summary = build_multiseed_tables(artifacts_dir)
    steps_table = build_steps_ablation_table(artifacts_dir)

    multiseed_runs.to_csv(output_dir / "multiseed_summary.csv", index=False)
    multiseed_summary.to_csv(output_dir / "multiseed_method_summary.csv", index=False)
    steps_table.to_csv(output_dir / "steps_ablation_summary.csv", index=False)

    save_multiseed_figures(
        runs=multiseed_runs,
        summary=multiseed_summary,
        output_dir=output_dir,
    )
    save_steps_ablation_figures(table=steps_table, output_dir=output_dir)
    return output_dir
```

- [ ] **Step 5: Create CLI script**

Create `scripts/plot_report_results.py`:

```python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from src.analysis.report_visualization import write_report_visualizations
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    from src.analysis.report_visualization import write_report_visualizations


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build report-ready figures for low-data NODE experiments."
    )
    parser.add_argument(
        "--artifacts-dir",
        default="artifacts",
        help="Root artifacts directory containing low_data, low_data_followup, and low_data_multiseed.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/report_figures",
        help="Directory where report CSVs and PNGs will be written.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    write_report_visualizations(
        artifacts_dir=Path(args.artifacts_dir),
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run tests to verify pass**

Run:

```bash
python -m pytest tests/test_report_visualization.py::test_write_report_visualizations_writes_tables_and_figures tests/test_low_data_clis.py::test_plot_report_results_cli_passes_args_to_writer -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/analysis/report_visualization.py scripts/plot_report_results.py tests/test_report_visualization.py tests/test_low_data_clis.py
git commit -m "feat: add report visualization CLI"
```

## Task 7: Verification on Available Artifacts

**Files:**
- No source files modified.

- [ ] **Step 1: Run focused tests**

Run:

```bash
python -m pytest tests/test_report_visualization.py tests/test_low_data_clis.py -q
```

Expected: PASS.

- [ ] **Step 2: Run full test suite**

Run:

```bash
python -m pytest -q
```

Expected: PASS with the existing Jupyter path warning acceptable.

- [ ] **Step 3: Run CLI help**

Run:

```bash
python scripts/plot_report_results.py --help
```

Expected: exit code 0 and help text includes `--artifacts-dir` and `--output-dir`.

- [ ] **Step 4: Run report generation on local artifacts**

Run:

```bash
python scripts/plot_report_results.py --artifacts-dir artifacts --output-dir artifacts/report_figures
```

Expected: exit code 0. If local artifacts do not include server multi-seed directories, the script still writes CSV files and any figures supported by available data.

- [ ] **Step 5: Check expected outputs**

Run:

```bash
Get-ChildItem artifacts\report_figures
```

Expected: files include `multiseed_summary.csv`, `multiseed_method_summary.csv`, and `steps_ablation_summary.csv`; PNG files appear when their source tables are non-empty.

## Recommended Server Command

After this plan is implemented and pushed to the server, run:

```bash
python scripts/plot_report_results.py \
  --artifacts-dir artifacts \
  --output-dir artifacts/report_figures
```

Then use these files in the report:

```text
artifacts/report_figures/multiseed_best_dice.png
artifacts/report_figures/multiseed_final_dice.png
artifacts/report_figures/multiseed_delta_vs_b.png
artifacts/report_figures/steps_best_dice.png
artifacts/report_figures/steps_final_dice.png
artifacts/report_figures/steps_peak_final_gap.png
```

## Self-Review

- Spec coverage: The plan covers baseline multi-seed CSV/PNG outputs, steps ablation CSV/PNG outputs, CLI generation, tests, and local/server usage.
- Placeholder scan: No placeholder markers or vague implementation steps remain.
- Type consistency: Function names are consistent across tests, module implementation, and CLI: `summarize_run`, `build_multiseed_tables`, `build_steps_ablation_table`, `save_multiseed_figures`, `save_steps_ablation_figures`, and `write_report_visualizations`.
