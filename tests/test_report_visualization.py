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
