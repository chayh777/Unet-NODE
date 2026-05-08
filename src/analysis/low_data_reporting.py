from __future__ import annotations

import json
import os
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
    final_row = history.sort_values("epoch").iloc[-1]
    final_val_dice = float(final_row["val_dice"])
    peak_final_gap = float(best_val_dice) - final_val_dice

    return {
        "group": group,
        "best_epoch": int(best_row["epoch"]),
        "best_val_dice": float(best_val_dice),
        "best_val_iou": float(best_row["val_iou"]),
        "final_val_dice": final_val_dice,
        "peak_final_gap": peak_final_gap,
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
            "final_val_dice",
            "peak_final_gap",
            "epochs_ran",
            "best_checkpoint",
        ],
    )


def _get_plotting_libs():
    mpl_config_dir = Path.cwd() / ".matplotlib"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save_metric_curve(
    history_table: pd.DataFrame,
    *,
    metric: str,
    output_path: Path,
    title: str,
    ylabel: str,
) -> None:
    plt = _get_plotting_libs()
    fig, ax = plt.subplots(figsize=(7, 5))

    for group, group_history in history_table.groupby("group", sort=False):
        ordered = group_history.sort_values("epoch")
        ax.plot(ordered["epoch"], ordered[metric], marker="o", label=str(group))

    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(title="Group")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _save_loss_curve(history_table: pd.DataFrame, output_path: Path) -> None:
    plt = _get_plotting_libs()
    fig, ax = plt.subplots(figsize=(7, 5))

    for group, group_history in history_table.groupby("group", sort=False):
        ordered = group_history.sort_values("epoch")
        ax.plot(
            ordered["epoch"],
            ordered["train_loss"],
            marker="o",
            linestyle="-",
            label=f"{group} train",
        )
        ax.plot(
            ordered["epoch"],
            ordered["val_loss"],
            marker="s",
            linestyle="--",
            label=f"{group} val",
        )

    ax.set_title("Training vs Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _save_final_metrics_plot(final_metrics_table: pd.DataFrame, output_path: Path) -> None:
    plt = _get_plotting_libs()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    groups = [str(group) for group in final_metrics_table["group"].tolist()]
    axes[0].bar(groups, final_metrics_table["best_val_dice"], color="#4f83cc")
    axes[0].set_title("Best Validation Dice")
    axes[0].set_ylabel("Dice")

    axes[1].bar(groups, final_metrics_table["best_val_iou"], color="#d84b4b")
    axes[1].set_title("Best Validation IoU")
    axes[1].set_ylabel("IoU")

    for ax in axes:
        ax.set_xlabel("Group")
        ax.set_ylim(bottom=0)
        ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def write_summary_artifacts(
    artifacts_dir: str | Path, groups: Iterable[str] = ("A", "B", "C")
) -> Path:
    groups = list(groups)
    summary_dir = Path(artifacts_dir) / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    history_table = build_history_table(artifacts_dir=artifacts_dir, groups=groups)
    final_metrics_table = build_final_metrics_table(artifacts_dir=artifacts_dir, groups=groups)

    history_table.to_csv(summary_dir / "history_compare.csv", index=False)
    final_metrics_table.to_csv(summary_dir / "final_metrics_compare.csv", index=False)

    _save_metric_curve(
        history_table,
        metric="val_dice",
        output_path=summary_dir / "dice_curve_compare.png",
        title="Validation Dice by Group",
        ylabel="Dice",
    )
    _save_metric_curve(
        history_table,
        metric="val_iou",
        output_path=summary_dir / "iou_curve_compare.png",
        title="Validation IoU by Group",
        ylabel="IoU",
    )
    _save_loss_curve(history_table, summary_dir / "loss_curve_compare.png")
    _save_final_metrics_plot(final_metrics_table, summary_dir / "final_metrics_compare.png")

    return summary_dir
