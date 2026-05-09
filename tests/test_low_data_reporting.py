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


def _write_ablation_artifacts(
    tmp_path: Path,
    *,
    subdir: str,
    steps: int,
    init: str | None,
    best_val_dice: float,
    best_val_iou: float,
) -> Path:
    artifacts_dir = tmp_path / subdir
    group_dir = artifacts_dir / "group_c"
    group_dir.mkdir(parents=True)
    pd.DataFrame(
        [{"epoch": 1, "train_loss": 0.5, "val_loss": 0.4, "val_dice": best_val_dice, "val_iou": best_val_iou}]
    ).to_csv(group_dir / "history.csv", index=False)
    (group_dir / "metrics.json").write_text(
        json.dumps({"best_val_dice": best_val_dice, "epochs_ran": 1, "best_checkpoint": "best.pt"}),
        encoding="utf-8",
    )
    config: dict = {
        "paths": {"artifacts_dir": str(artifacts_dir)},
        "node": {"steps": steps},
        "adapter": {},
    }
    if init is not None:
        config["adapter"]["init"] = init
    config_path = tmp_path / f"{subdir}.yaml"
    import yaml as _yaml
    config_path.write_text(_yaml.dump(config), encoding="utf-8")
    return config_path


def test_collect_ablation_run_metrics_reads_steps_and_init(tmp_path: Path) -> None:
    module = _load_low_data_reporting_module()
    config_path = _write_ablation_artifacts(
        tmp_path, subdir="c_steps2_t1", steps=2, init=None,
        best_val_dice=0.72, best_val_iou=0.58,
    )

    row = module.collect_ablation_run_metrics(config_path, group="C")

    assert row["steps"] == 2
    assert row["init"] == "default"
    assert row["best_val_dice"] == pytest.approx(0.72)
    assert row["best_val_iou"] == pytest.approx(0.58)


def test_build_ablation_table_sorts_by_steps_and_init(tmp_path: Path) -> None:
    module = _load_low_data_reporting_module()
    cfg_a = _write_ablation_artifacts(
        tmp_path, subdir="steps16_default", steps=16, init=None,
        best_val_dice=0.80, best_val_iou=0.68,
    )
    cfg_b = _write_ablation_artifacts(
        tmp_path, subdir="steps2_zero", steps=2, init="zero_last_layer",
        best_val_dice=0.74, best_val_iou=0.60,
    )
    cfg_c = _write_ablation_artifacts(
        tmp_path, subdir="steps2_default", steps=2, init=None,
        best_val_dice=0.71, best_val_iou=0.57,
    )

    df = module.build_ablation_table([cfg_a, cfg_b, cfg_c], group="C")

    assert list(df["steps"]) == [2, 2, 16]
    assert df.iloc[0]["init"] == "default"
    assert df.iloc[1]["init"] == "zero_last_layer"


def test_write_ablation_summary_writes_csv_and_plots(tmp_path: Path) -> None:
    module = _load_low_data_reporting_module()
    cfg_a = _write_ablation_artifacts(
        tmp_path, subdir="steps2", steps=2, init=None,
        best_val_dice=0.71, best_val_iou=0.57,
    )
    cfg_b = _write_ablation_artifacts(
        tmp_path, subdir="steps16", steps=16, init="zero_last_layer",
        best_val_dice=0.82, best_val_iou=0.70,
    )
    output_dir = tmp_path / "ablation_out"

    result = module.write_ablation_summary([cfg_a, cfg_b], output_dir=output_dir, group="C", dpi=72)

    assert result == output_dir
    assert (output_dir / "ablation_summary.csv").exists()
    assert (output_dir / "ablation_dice_bars.png").exists()
    assert (output_dir / "ablation_iou_bars.png").exists()

    summary = pd.read_csv(output_dir / "ablation_summary.csv")
    assert {"steps", "init", "best_val_dice", "best_val_iou"}.issubset(summary.columns)
    assert set(summary["steps"]) == {2, 16}


# ── Early convergence tests ──────────────────────────────────────────────────


def test_compute_epochs_to_threshold_finds_first_crossing(tmp_path: Path) -> None:
    module = _load_low_data_reporting_module()
    # Group A peaks at 0.80; 80% = 0.64 (reached epoch 2), 90% = 0.72 (epoch 3)
    history = pd.DataFrame(
        [
            {"group": "A", "epoch": 1, "train_loss": 1.0, "val_loss": 0.9, "val_dice": 0.50, "val_iou": 0.40},
            {"group": "A", "epoch": 2, "train_loss": 0.8, "val_loss": 0.7, "val_dice": 0.70, "val_iou": 0.55},
            {"group": "A", "epoch": 3, "train_loss": 0.7, "val_loss": 0.6, "val_dice": 0.80, "val_iou": 0.65},
        ]
    )

    df = module.compute_epochs_to_threshold(history, thresholds=(0.80, 0.90))

    a_rows = df[df["group"] == "A"].set_index("threshold")
    # 80% of 0.80 = 0.64, first epoch >= 0.64 is epoch 2
    assert a_rows.loc[0.80, "epoch"] == 2
    # 90% of 0.80 = 0.72, first epoch >= 0.72 is epoch 3
    assert a_rows.loc[0.90, "epoch"] == 3


def test_compute_epochs_to_threshold_returns_nan_when_never_reached(tmp_path: Path) -> None:
    module = _load_low_data_reporting_module()
    history = pd.DataFrame(
        [
            {"group": "B", "epoch": 1, "train_loss": 1.0, "val_loss": 0.9, "val_dice": 0.50, "val_iou": 0.40},
            {"group": "B", "epoch": 2, "train_loss": 0.9, "val_loss": 0.8, "val_dice": 0.55, "val_iou": 0.42},
        ]
    )

    df = module.compute_epochs_to_threshold(history, thresholds=(1.01,))

    assert pd.isna(df.loc[df["group"] == "B", "epoch"].iloc[0])


def test_compute_epochs_to_threshold_multi_group(tmp_path: Path) -> None:
    module = _load_low_data_reporting_module()
    history = pd.DataFrame(
        [
            {"group": "A", "epoch": 1, "train_loss": 1.0, "val_loss": 0.9, "val_dice": 0.60, "val_iou": 0.45},
            {"group": "A", "epoch": 2, "train_loss": 0.8, "val_loss": 0.7, "val_dice": 0.80, "val_iou": 0.65},
            {"group": "C", "epoch": 1, "train_loss": 0.9, "val_loss": 0.8, "val_dice": 0.75, "val_iou": 0.60},
            {"group": "C", "epoch": 2, "train_loss": 0.7, "val_loss": 0.6, "val_dice": 0.85, "val_iou": 0.70},
        ]
    )

    df = module.compute_epochs_to_threshold(history, thresholds=(0.90,))

    # Group A: peak=0.80, 90% threshold=0.72, first >=0.72 is epoch 2
    a_epoch = df[(df["group"] == "A") & (df["threshold"] == 0.90)]["epoch"].iloc[0]
    assert a_epoch == 2
    # Group C: peak=0.85, 90% threshold=0.765, first >=0.765 is epoch 2
    c_epoch = df[(df["group"] == "C") & (df["threshold"] == 0.90)]["epoch"].iloc[0]
    assert c_epoch == 2


def test_write_early_convergence_artifacts_writes_expected_files(tmp_path: Path) -> None:
    module = _load_low_data_reporting_module()
    artifacts_dir = tmp_path / "artifacts"
    for group, peak_dice in [("A", 0.65), ("B", 0.72), ("C", 0.80)]:
        rows = [
            {"epoch": i, "train_loss": 1.0 - i * 0.05, "val_loss": 0.9 - i * 0.04,
             "val_dice": min(peak_dice, 0.30 + i * 0.05), "val_iou": min(peak_dice - 0.1, 0.20 + i * 0.04)}
            for i in range(1, 11)
        ]
        _write_group_artifacts(
            artifacts_dir,
            group=group,
            history_rows=rows,
            metrics={"best_val_dice": peak_dice, "epochs_ran": 10, "best_checkpoint": "best.pt"},
        )

    output_dir = tmp_path / "conv_out"
    result = module.write_early_convergence_artifacts(
        artifacts_dir=artifacts_dir,
        groups=["A", "B", "C"],
        output_dir=output_dir,
        zoom_epochs=10,
        thresholds=(0.80, 0.90),
        dpi=72,
    )

    assert result == output_dir
    assert (output_dir / "epochs_to_threshold.csv").exists()
    assert (output_dir / "early_convergence_curve.png").exists()
    assert (output_dir / "epochs_to_threshold.png").exists()

    csv_df = pd.read_csv(output_dir / "epochs_to_threshold.csv")
    assert {"group", "threshold", "epoch"}.issubset(csv_df.columns)
    assert set(csv_df["group"]) == {"A", "B", "C"}


def test_write_summary_artifacts_includes_convergence_plots(tmp_path: Path) -> None:
    module = _load_low_data_reporting_module()
    artifacts_dir = tmp_path / "artifacts"
    for group, peak in [("A", 0.65), ("B", 0.72)]:
        rows = [
            {"epoch": i, "train_loss": 1.0 - i * 0.05, "val_loss": 0.9 - i * 0.04,
             "val_dice": min(peak, 0.30 + i * 0.05), "val_iou": min(peak - 0.1, 0.20 + i * 0.04)}
            for i in range(1, 6)
        ]
        _write_group_artifacts(
            artifacts_dir, group=group, history_rows=rows,
            metrics={"best_val_dice": peak, "epochs_ran": 5, "best_checkpoint": "best.pt"},
        )

    output_dir = module.write_summary_artifacts(artifacts_dir=artifacts_dir, groups=["A", "B"])

    assert (output_dir / "early_convergence_curve.png").exists()
    assert (output_dir / "epochs_to_threshold.png").exists()
    assert (output_dir / "epochs_to_threshold.csv").exists()


# ── Trainable parameter count tests ─────────────────────────────────────────


def test_count_trainable_params_group_a_has_zero_adapter(tmp_path: Path) -> None:
    module = _load_low_data_reporting_module()
    counts = module.count_trainable_params(
        adapter_type="none",
        bottleneck_channels=512,
        hidden_channels=512,
        num_classes=1,
        encoder_last_channels=512,
    )
    assert counts["adapter"] == 0
    assert counts["bottleneck_proj"] > 0
    assert counts["decoder"] > 0
    assert counts["head"] > 0


def test_count_trainable_params_conv_and_node_have_equal_adapter_count(tmp_path: Path) -> None:
    module = _load_low_data_reporting_module()
    kwargs = dict(bottleneck_channels=512, hidden_channels=512, num_classes=1, encoder_last_channels=512)
    b = module.count_trainable_params(adapter_type="conv", **kwargs)
    c = module.count_trainable_params(adapter_type="node", **kwargs)
    assert b["adapter"] == c["adapter"]
    assert b["adapter"] > 0


def test_count_trainable_params_total_increases_from_a_to_b(tmp_path: Path) -> None:
    module = _load_low_data_reporting_module()
    kwargs = dict(bottleneck_channels=512, hidden_channels=512, num_classes=1, encoder_last_channels=512)
    a_total = sum(module.count_trainable_params(adapter_type="none", **kwargs).values())
    b_total = sum(module.count_trainable_params(adapter_type="conv", **kwargs).values())
    assert b_total > a_total


def test_count_trainable_params_known_values(tmp_path: Path) -> None:
    # Verify exact math for default config (bottleneck=512, hidden=512, resnet34 last=512)
    module = _load_low_data_reporting_module()
    counts = module.count_trainable_params(
        adapter_type="conv",
        bottleneck_channels=512,
        hidden_channels=512,
        num_classes=1,
        encoder_last_channels=512,
    )
    # bottleneck_proj: Conv2d(512,512,1) → 512*512 + 512 = 262 656
    assert counts["bottleneck_proj"] == 262_656
    # adapter block: Conv(512→512) + BN(512) + Conv(512→512)
    #   = (512*512+512) + 2*512 + (512*512+512) = 262656 + 1024 + 262656 = 526 336
    assert counts["adapter"] == 526_336
    # decoder: Conv(512→256,k3) + Conv(256→128,k3)
    #   mid1=256, mid2=128
    #   = (256*512*9+256) + (128*256*9+128) = 1179904 + 295040 = 1 474 944
    assert counts["decoder"] == 1_474_944
    # head: Conv2d(128, 1, 1) → 128*1 + 1 = 129
    assert counts["head"] == 129


def test_build_param_count_table_has_correct_structure(tmp_path: Path) -> None:
    module = _load_low_data_reporting_module()
    df = module.build_param_count_table(
        groups=["A", "B", "C"],
        bottleneck_channels=512,
        hidden_channels=512,
        num_classes=1,
        encoder_last_channels=512,
    )
    assert list(df["group"]) == ["A", "B", "C"]
    assert {"group", "adapter_type", "bottleneck_proj", "adapter", "decoder", "head", "total"}.issubset(df.columns)
    assert df.loc[df["group"] == "A", "adapter"].iloc[0] == 0
    assert df.loc[df["group"] == "B", "total"].iloc[0] > df.loc[df["group"] == "A", "total"].iloc[0]
    # B and C have the same total (same block, just different forward passes)
    assert df.loc[df["group"] == "B", "total"].iloc[0] == df.loc[df["group"] == "C", "total"].iloc[0]


def test_write_param_count_artifacts_writes_csv_and_plot(tmp_path: Path) -> None:
    module = _load_low_data_reporting_module()
    output_dir = tmp_path / "param_out"

    result = module.write_param_count_artifacts(
        output_dir=output_dir,
        groups=["A", "B", "C"],
        bottleneck_channels=512,
        hidden_channels=512,
        num_classes=1,
        encoder_last_channels=512,
        dpi=72,
    )

    assert result == output_dir
    assert (output_dir / "param_counts.csv").exists()
    assert (output_dir / "param_counts.png").exists()

    csv_df = pd.read_csv(output_dir / "param_counts.csv")
    assert {"group", "adapter_type", "total"}.issubset(csv_df.columns)
    assert set(csv_df["group"]) == {"A", "B", "C"}
