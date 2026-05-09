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
