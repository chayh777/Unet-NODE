from __future__ import annotations

from importlib import util
import json
from pathlib import Path
import sys
from types import ModuleType

import pandas as pd
import pytest


def _ensure_package(name: str, package_path: Path) -> None:
    if name in sys.modules:
        return
    pkg = ModuleType(name)
    pkg.__path__ = [str(package_path)]  # type: ignore[attr-defined]
    sys.modules[name] = pkg


def _load_low_data_reporting_module():
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    analysis_dir = src_dir / "analysis"

    _ensure_package("src", src_dir)
    _ensure_package("src.analysis", analysis_dir)

    module_path = analysis_dir / "low_data_reporting.py"
    assert module_path.exists(), f"Missing module file: {module_path}"
    spec = util.spec_from_file_location("src.analysis.low_data_reporting", module_path)
    assert spec is not None
    module = util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["src.analysis.low_data_reporting"] = module
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def _write_group_artifacts(
    artifacts_dir: Path,
    *,
    group: str,
    history_rows: list[dict[str, float | int]],
    metrics: dict[str, object],
) -> None:
    group_dir = artifacts_dir / f"group_{group.lower()}"
    group_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(history_rows).to_csv(group_dir / "history.csv", index=False)
    (group_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")


def test_collect_group_history_adds_group_column(tmp_path: Path) -> None:
    module = _load_low_data_reporting_module()
    artifacts_dir = tmp_path / "artifacts"
    _write_group_artifacts(
        artifacts_dir,
        group="A",
        history_rows=[
            {
                "epoch": 1,
                "train_loss": 0.9,
                "val_loss": 0.8,
                "val_dice": 0.5,
                "val_iou": 0.4,
            }
        ],
        metrics={"best_val_dice": 0.5, "epochs_ran": 1, "best_checkpoint": "best.pt"},
    )

    table = module.collect_group_history(artifacts_dir=artifacts_dir, group="A")

    assert list(table.columns) == [
        "group",
        "epoch",
        "train_loss",
        "val_loss",
        "val_dice",
        "val_iou",
    ]
    assert table.to_dict("records") == [
        {
            "group": "A",
            "epoch": 1,
            "train_loss": 0.9,
            "val_loss": 0.8,
            "val_dice": 0.5,
            "val_iou": 0.4,
        }
    ]


def test_collect_group_final_metrics_reads_best_val_dice_and_best_row_val_iou(
    tmp_path: Path,
) -> None:
    module = _load_low_data_reporting_module()
    artifacts_dir = tmp_path / "artifacts"
    _write_group_artifacts(
        artifacts_dir,
        group="B",
        history_rows=[
            {
                "epoch": 1,
                "train_loss": 1.0,
                "val_loss": 0.9,
                "val_dice": 0.55,
                "val_iou": 0.42,
            },
            {
                "epoch": 2,
                "train_loss": 0.8,
                "val_loss": 0.7,
                "val_dice": 0.72,
                "val_iou": 0.61,
            },
        ],
        metrics={
            "best_val_dice": 0.72,
            "epochs_ran": 2,
            "best_checkpoint": "group_b/best.pt",
        },
    )

    row = module.collect_group_final_metrics(artifacts_dir=artifacts_dir, group="B")

    assert row == {
        "group": "B",
        "best_epoch": 2,
        "best_val_dice": 0.72,
        "best_val_iou": 0.61,
        "final_val_dice": 0.72,
        "peak_final_gap": 0.0,
        "epochs_ran": 2,
        "best_checkpoint": "group_b/best.pt",
    }


def test_collect_group_final_metrics_includes_final_dice_and_peak_final_gap(tmp_path: Path) -> None:
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
    assert row["peak_final_gap"] == pytest.approx(0.03)


def test_collect_group_final_metrics_uses_last_epoch_after_sorting(tmp_path):
    group_dir = tmp_path / "group_c"
    group_dir.mkdir(parents=True)
    (group_dir / "history.csv").write_text(
        "\n".join(
            [
                "epoch,train_loss,val_loss,val_dice,val_iou",
                "3,0.7,0.8,0.75,0.61",
                "1,1.0,0.9,0.70,0.54",
                "2,0.8,0.7,0.78,0.65",
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


def test_build_history_table_combines_groups(tmp_path: Path) -> None:
    module = _load_low_data_reporting_module()
    artifacts_dir = tmp_path / "artifacts"
    _write_group_artifacts(
        artifacts_dir,
        group="A",
        history_rows=[
            {
                "epoch": 1,
                "train_loss": 0.9,
                "val_loss": 0.8,
                "val_dice": 0.5,
                "val_iou": 0.4,
            }
        ],
        metrics={"best_val_dice": 0.5, "epochs_ran": 1, "best_checkpoint": "a.pt"},
    )
    _write_group_artifacts(
        artifacts_dir,
        group="C",
        history_rows=[
            {
                "epoch": 1,
                "train_loss": 0.7,
                "val_loss": 0.6,
                "val_dice": 0.65,
                "val_iou": 0.5,
            }
        ],
        metrics={"best_val_dice": 0.65, "epochs_ran": 1, "best_checkpoint": "c.pt"},
    )

    table = module.build_history_table(artifacts_dir=artifacts_dir, groups=["A", "C"])

    assert table["group"].tolist() == ["A", "C"]
    assert table["val_dice"].tolist() == [0.5, 0.65]


def test_build_final_metrics_table_combines_groups(tmp_path: Path) -> None:
    module = _load_low_data_reporting_module()
    artifacts_dir = tmp_path / "artifacts"
    _write_group_artifacts(
        artifacts_dir,
        group="A",
        history_rows=[
            {
                "epoch": 1,
                "train_loss": 0.9,
                "val_loss": 0.8,
                "val_dice": 0.5,
                "val_iou": 0.4,
            }
        ],
        metrics={"best_val_dice": 0.5, "epochs_ran": 1, "best_checkpoint": "a.pt"},
    )
    _write_group_artifacts(
        artifacts_dir,
        group="B",
        history_rows=[
            {
                "epoch": 1,
                "train_loss": 1.0,
                "val_loss": 0.9,
                "val_dice": 0.55,
                "val_iou": 0.42,
            },
            {
                "epoch": 2,
                "train_loss": 0.8,
                "val_loss": 0.7,
                "val_dice": 0.72,
                "val_iou": 0.61,
            },
        ],
        metrics={"best_val_dice": 0.72, "epochs_ran": 2, "best_checkpoint": "b.pt"},
    )

    table = module.build_final_metrics_table(artifacts_dir=artifacts_dir, groups=["A", "B"])

    assert list(table.columns) == [
        "group",
        "best_epoch",
        "best_val_dice",
        "best_val_iou",
        "final_val_dice",
        "peak_final_gap",
        "epochs_ran",
        "best_checkpoint",
    ]
    assert table.to_dict("records") == [
        {
            "group": "A",
            "best_epoch": 1,
            "best_val_dice": 0.5,
            "best_val_iou": 0.4,
            "final_val_dice": 0.5,
            "peak_final_gap": 0.0,
            "epochs_ran": 1,
            "best_checkpoint": "a.pt",
        },
        {
            "group": "B",
            "best_epoch": 2,
            "best_val_dice": 0.72,
            "best_val_iou": 0.61,
            "final_val_dice": 0.72,
            "peak_final_gap": 0.0,
            "epochs_ran": 2,
            "best_checkpoint": "b.pt",
        },
    ]


def test_write_summary_artifacts_writes_tables_and_required_plots(tmp_path: Path) -> None:
    module = _load_low_data_reporting_module()
    artifacts_dir = tmp_path / "artifacts"
    _write_group_artifacts(
        artifacts_dir,
        group="A",
        history_rows=[
            {
                "epoch": 1,
                "train_loss": 0.9,
                "val_loss": 0.8,
                "val_dice": 0.5,
                "val_iou": 0.4,
            },
            {
                "epoch": 2,
                "train_loss": 0.7,
                "val_loss": 0.6,
                "val_dice": 0.65,
                "val_iou": 0.5,
            },
        ],
        metrics={"best_val_dice": 0.65, "epochs_ran": 2, "best_checkpoint": "a.pt"},
    )
    _write_group_artifacts(
        artifacts_dir,
        group="B",
        history_rows=[
            {
                "epoch": 1,
                "train_loss": 1.0,
                "val_loss": 0.9,
                "val_dice": 0.45,
                "val_iou": 0.3,
            },
            {
                "epoch": 2,
                "train_loss": 0.8,
                "val_loss": 0.7,
                "val_dice": 0.55,
                "val_iou": 0.4,
            },
        ],
        metrics={"best_val_dice": 0.55, "epochs_ran": 2, "best_checkpoint": "b.pt"},
    )

    output_dir = module.write_summary_artifacts(artifacts_dir=artifacts_dir, groups=["A", "B"])

    assert output_dir == artifacts_dir / "summary"
    assert (output_dir / "history_compare.csv").exists()
    assert (output_dir / "final_metrics_compare.csv").exists()
    assert (output_dir / "dice_curve_compare.png").exists()
    assert (output_dir / "iou_curve_compare.png").exists()
    assert (output_dir / "loss_curve_compare.png").exists()
    assert (output_dir / "final_metrics_compare.png").exists()

    history = pd.read_csv(output_dir / "history_compare.csv")
    assert set(history["group"]) == {"A", "B"}

    metrics = pd.read_csv(output_dir / "final_metrics_compare.csv")
    assert metrics["group"].tolist() == ["A", "B"]
