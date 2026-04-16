from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


def _group_dir(artifacts_dir: str | Path, group: str) -> Path:
    return Path(artifacts_dir) / f"group_{group.lower()}"


def _read_history(group_dir: Path) -> pd.DataFrame:
    history = pd.read_csv(group_dir / "history.csv")
    columns = ["epoch", "train_loss", "val_loss", "val_dice", "val_iou"]
    return history.loc[:, columns]


def _read_metrics(group_dir: Path) -> dict[str, Any]:
    with (group_dir / "metrics.json").open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected metrics.json at {group_dir} to contain an object.")
    return data


def collect_group_history(artifacts_dir: str | Path, group: str) -> pd.DataFrame:
    history = _read_history(_group_dir(artifacts_dir, group)).copy()
    history.insert(0, "group", group)
    return history


def collect_group_final_metrics(artifacts_dir: str | Path, group: str) -> dict[str, Any]:
    group_dir = _group_dir(artifacts_dir, group)
    history = _read_history(group_dir)
    metrics = _read_metrics(group_dir)

    if history.empty:
        raise ValueError(f"history.csv for group {group!r} is empty.")

    best_val_dice = metrics.get("best_val_dice")
    if best_val_dice is None:
        raise ValueError(f"metrics.json for group {group!r} is missing best_val_dice.")

    best_idx = (history["val_dice"] - float(best_val_dice)).abs().idxmin()
    best_row = history.loc[int(best_idx)]

    return {
        "group": group,
        "best_epoch": int(best_row["epoch"]),
        "best_val_dice": float(best_val_dice),
        "best_val_iou": float(best_row["val_iou"]),
        "epochs_ran": metrics.get("epochs_ran"),
        "best_checkpoint": metrics.get("best_checkpoint"),
    }


def build_history_table(
    artifacts_dir: str | Path, groups: Iterable[str] = ("A", "B", "C")
) -> pd.DataFrame:
    frames = [collect_group_history(artifacts_dir=artifacts_dir, group=group) for group in groups]
    if not frames:
        return pd.DataFrame(columns=["group", "epoch", "train_loss", "val_loss", "val_dice", "val_iou"])
    return pd.concat(frames, ignore_index=True)


def build_final_metrics_table(
    artifacts_dir: str | Path, groups: Iterable[str] = ("A", "B", "C")
) -> pd.DataFrame:
    rows = [collect_group_final_metrics(artifacts_dir=artifacts_dir, group=group) for group in groups]
    return pd.DataFrame(
        rows,
        columns=[
            "group",
            "best_epoch",
            "best_val_dice",
            "best_val_iou",
            "epochs_ran",
            "best_checkpoint",
        ],
    )
