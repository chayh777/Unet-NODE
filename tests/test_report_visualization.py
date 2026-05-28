from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest


def _write_run(
    root: Path,
    *,
    best: float,
    rows: list[dict[str, float | int]],
    duration_sec: float | None = None,
    avg_epoch_sec: float | None = None,
    checkpoint_saved: bool | None = None,
    best_checkpoint: str | None = None,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(root / "history.csv", index=False)
    metrics = {
        "best_val_dice": best,
        "epochs_ran": len(rows),
        "best_checkpoint": best_checkpoint if best_checkpoint is not None else str(root / "best.pt"),
    }
    if duration_sec is not None:
        metrics["duration_sec"] = duration_sec
    if avg_epoch_sec is not None:
        metrics["avg_epoch_sec"] = avg_epoch_sec
    if checkpoint_saved is not None:
        metrics["checkpoint_saved"] = checkpoint_saved
    (root / "metrics.json").write_text(
        json.dumps(metrics),
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
        "duration_sec": None,
        "avg_epoch_sec": None,
        "checkpoint_saved": True,
        "best_checkpoint": str(root / "best.pt"),
        "root": str(root),
    }


def test_summarize_run_carries_timing_and_checkpoint_metrics(tmp_path: Path) -> None:
    from src.analysis.report_visualization import summarize_run

    root = tmp_path / "exp" / "group_b"
    _write_run(
        root,
        best=0.74,
        duration_sec=90.0,
        avg_epoch_sec=30.0,
        checkpoint_saved=False,
        best_checkpoint="",
        rows=[
            {"epoch": 1, "train_loss": 1.0, "val_loss": 0.9, "val_dice": 0.70, "val_iou": 0.54},
            {"epoch": 2, "train_loss": 0.9, "val_loss": 0.8, "val_dice": 0.74, "val_iou": 0.59},
            {"epoch": 3, "train_loss": 0.8, "val_loss": 0.7, "val_dice": 0.72, "val_iou": 0.57},
        ],
    )

    row = summarize_run(root=root, method="Method", run="exp", seed=2)

    assert row["duration_sec"] == 90.0
    assert row["avg_epoch_sec"] == 30.0
    assert row["checkpoint_saved"] is False
    assert row["best_checkpoint"] == ""


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


def test_build_multiseed_tables_ingests_comparison_method_runs(tmp_path: Path) -> None:
    from src.analysis.report_visualization import build_multiseed_tables

    artifacts_dir = tmp_path / "artifacts"
    _write_run(
        artifacts_dir / "low_data" / "group_a",
        best=0.69,
        rows=[
            {"epoch": 1, "train_loss": 1.0, "val_loss": 0.9, "val_dice": 0.68, "val_iou": 0.49},
            {"epoch": 2, "train_loss": 0.9, "val_loss": 0.8, "val_dice": 0.69, "val_iou": 0.50},
        ],
    )
    _write_run(
        artifacts_dir / "low_data_multiseed" / "b_seed0" / "group_b",
        best=0.70,
        rows=[
            {"epoch": 1, "train_loss": 1.0, "val_loss": 0.9, "val_dice": 0.70, "val_iou": 0.50},
            {"epoch": 2, "train_loss": 0.9, "val_loss": 0.8, "val_dice": 0.68, "val_iou": 0.48},
        ],
    )
    _write_run(
        artifacts_dir / "low_data_output" / "conv_b_seed42" / "group_b",
        best=0.71,
        rows=[
            {"epoch": 1, "train_loss": 1.0, "val_loss": 0.9, "val_dice": 0.70, "val_iou": 0.51},
            {"epoch": 2, "train_loss": 0.9, "val_loss": 0.8, "val_dice": 0.71, "val_iou": 0.52},
        ],
    )
    _write_run(
        artifacts_dir / "low_data_output" / "node_c_seed42" / "group_c",
        best=0.73,
        rows=[
            {"epoch": 1, "train_loss": 1.0, "val_loss": 0.9, "val_dice": 0.72, "val_iou": 0.54},
            {"epoch": 2, "train_loss": 0.9, "val_loss": 0.8, "val_dice": 0.73, "val_iou": 0.55},
        ],
    )

    runs, summary = build_multiseed_tables(artifacts_dir)

    assert runs["method"].tolist() == [
        "Plain-U-Net",
        "B-base",
        "Output-Conv-U-Net",
        "Output-NODE-U-Net",
    ]
    assert pd.isna(runs["seed"].iloc[0])
    assert runs["seed"].iloc[1:].tolist() == [0, 42, 42]
    assert summary["method"].tolist() == [
        "Plain-U-Net",
        "B-base",
        "Output-Conv-U-Net",
        "Output-NODE-U-Net",
    ]


def test_build_multiseed_tables_carries_timing_summary_when_present(tmp_path: Path) -> None:
    from src.analysis.report_visualization import build_multiseed_tables

    artifacts_dir = tmp_path / "artifacts"
    _write_run(
        artifacts_dir / "low_data_multiseed" / "b_seed0" / "group_b",
        best=0.70,
        duration_sec=100.0,
        avg_epoch_sec=50.0,
        checkpoint_saved=False,
        best_checkpoint="",
        rows=[
            {"epoch": 1, "train_loss": 1.0, "val_loss": 0.9, "val_dice": 0.70, "val_iou": 0.50},
            {"epoch": 2, "train_loss": 0.9, "val_loss": 0.8, "val_dice": 0.68, "val_iou": 0.48},
        ],
    )
    _write_run(
        artifacts_dir / "low_data_multiseed" / "b_seed1" / "group_b",
        best=0.72,
        duration_sec=120.0,
        avg_epoch_sec=60.0,
        checkpoint_saved=True,
        rows=[
            {"epoch": 1, "train_loss": 1.0, "val_loss": 0.9, "val_dice": 0.71, "val_iou": 0.52},
            {"epoch": 2, "train_loss": 0.9, "val_loss": 0.8, "val_dice": 0.72, "val_iou": 0.53},
        ],
    )

    runs, summary = build_multiseed_tables(artifacts_dir)

    assert runs["duration_sec"].tolist() == [100.0, 120.0]
    assert runs["avg_epoch_sec"].tolist() == [50.0, 60.0]
    assert runs["checkpoint_saved"].tolist() == [False, True]
    b_base = summary.loc[summary["method"] == "B-base"].iloc[0]
    assert b_base["duration_sec_mean"] == 110.0
    assert b_base["avg_epoch_sec_mean"] == 55.0
    assert b_base["checkpoint_saved_rate"] == 0.5


def test_method_order_groups_comparison_methods_before_c_series_and_keeps_legacy_subsets() -> None:
    from src.analysis.report_visualization import _method_order

    assert _method_order(
        [
            "Output-NODE-U-Net",
            "C-zero-last-steps16",
            "B-base",
            "C-fine-steps8-default",
            "Plain-U-Net",
            "C-zero-last-steps8",
            "Output-Conv-U-Net",
        ]
    ) == [
        "Plain-U-Net",
        "B-base",
        "Output-Conv-U-Net",
        "Output-NODE-U-Net",
        "C-fine-steps8-default",
        "C-zero-last-steps8",
        "C-zero-last-steps16",
    ]
    assert _method_order(
        [
            "C-zero-last-steps8",
            "B-base",
        ]
    ) == [
        "B-base",
        "C-zero-last-steps8",
    ]


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


def test_save_multiseed_figures_sets_dice_axis_floor_to_point7(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.analysis.report_visualization as report_visualization

    runs = pd.DataFrame(
        [
            {"method": "B-base", "seed": 0, "best_dice": 0.70, "final_dice": 0.68, "peak_final_gap": 0.02},
            {"method": "C-zero-last-steps8", "seed": 0, "best_dice": 0.75, "final_dice": 0.73, "peak_final_gap": 0.02},
        ]
    )
    summary = pd.DataFrame(
        [
            {"method": "B-base", "best_dice_mean": 0.71, "best_dice_std": 0.014, "final_dice_mean": 0.695, "final_dice_std": 0.021},
            {"method": "C-zero-last-steps8", "best_dice_mean": 0.755, "best_dice_std": 0.007, "final_dice_mean": 0.735, "final_dice_std": 0.007},
        ]
    )

    class FakeAxis:
        def __init__(self) -> None:
            self.ylim_bottoms: list[float | None] = []

        def bar(self, *args, **kwargs) -> None:
            return None

        def scatter(self, *args, **kwargs) -> None:
            return None

        def set_title(self, *args, **kwargs) -> None:
            return None

        def set_ylabel(self, *args, **kwargs) -> None:
            return None

        def set_xticks(self, *args, **kwargs) -> None:
            return None

        def set_xticklabels(self, *args, **kwargs) -> None:
            return None

        def set_ylim(self, *, bottom=None, top=None) -> None:
            self.ylim_bottoms.append(bottom)

        def grid(self, *args, **kwargs) -> None:
            return None

        def plot(self, *args, **kwargs) -> None:
            return None

        def axhline(self, *args, **kwargs) -> None:
            return None

        def set_xlabel(self, *args, **kwargs) -> None:
            return None

        def legend(self, *args, **kwargs) -> None:
            return None

    class FakeFigure:
        def tight_layout(self) -> None:
            return None

        def savefig(self, path: Path, dpi: int) -> None:
            Path(path).touch()

    class FakePlt:
        def __init__(self) -> None:
            self.axes: list[FakeAxis] = []

        def subplots(self, *args, **kwargs):
            if kwargs.get("sharex"):
                axes = [FakeAxis(), FakeAxis()]
                self.axes.extend(axes)
                return FakeFigure(), axes
            axis = FakeAxis()
            self.axes.append(axis)
            return FakeFigure(), axis

        def close(self, fig) -> None:
            return None

    fake_plt = FakePlt()
    monkeypatch.setattr(report_visualization, "_get_plotting_libs", lambda: fake_plt)

    report_visualization.save_multiseed_figures(runs=runs, summary=summary, output_dir=tmp_path)

    assert fake_plt.axes[0].ylim_bottoms == [0.7]
    assert fake_plt.axes[1].ylim_bottoms == [0.7]


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
