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
