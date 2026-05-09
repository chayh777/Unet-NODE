from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import yaml


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


def _init_label(init: str) -> str:
    return "Zero-Last Init" if init == "zero_last_layer" else "Default Init"


def collect_ablation_run_metrics(config_path: str | Path, group: str = "C") -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    steps = int(config["node"]["steps"])
    init = str(config.get("adapter", {}).get("init", "default"))
    artifacts_dir = config["paths"]["artifacts_dir"]

    group_dir = _group_dir(artifacts_dir, group)
    metrics = _read_metrics(group_dir)
    history = _read_history(group_dir)

    best_val_dice = metrics.get("best_val_dice")
    if best_val_dice is None:
        raise ValueError(f"metrics.json missing best_val_dice for {config_path}")

    best_idx = (history["val_dice"] - float(best_val_dice)).abs().idxmin()
    best_row = history.loc[int(best_idx)]

    return {
        "steps": steps,
        "init": init,
        "best_val_dice": float(best_val_dice),
        "best_val_iou": float(best_row["val_iou"]),
        "config": str(config_path),
    }


def build_ablation_table(config_paths: Iterable[str | Path], group: str = "C") -> pd.DataFrame:
    rows = [collect_ablation_run_metrics(p, group) for p in config_paths]
    df = pd.DataFrame(rows, columns=["steps", "init", "best_val_dice", "best_val_iou", "config"])
    return df.sort_values(["steps", "init"]).reset_index(drop=True)


def _save_ablation_bar_plot(
    df: pd.DataFrame,
    output_path: Path,
    metric: str,
    title: str,
    ylabel: str,
    dpi: int,
) -> None:
    plt = _get_plotting_libs()

    steps_vals = sorted(df["steps"].unique())
    init_vals = sorted(df["init"].unique())
    n_inits = len(init_vals)
    bar_width = 0.35
    x = list(range(len(steps_vals)))
    colors = ["#5470c6", "#ee8c30", "#3ba272", "#d84b4b"]

    fig, ax = plt.subplots(figsize=(max(6, 2.5 * len(steps_vals)), 4))

    for i, init in enumerate(init_vals):
        subset = df[df["init"] == init].set_index("steps")
        vals = [float(subset.loc[s, metric]) if s in subset.index else 0.0 for s in steps_vals]
        offsets = [xi + (i - (n_inits - 1) / 2) * bar_width for xi in x]
        bars = ax.bar(offsets, vals, bar_width, label=_init_label(init), color=colors[i % len(colors)])
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.002,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in steps_vals])
    ax.set_xlabel("ODE Steps")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(bottom=0)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(title="Initialization")
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def write_ablation_summary(
    config_paths: Iterable[str | Path],
    output_dir: str | Path,
    group: str = "C",
    dpi: int = 150,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = build_ablation_table(config_paths, group)
    df.to_csv(output_dir / "ablation_summary.csv", index=False)

    _save_ablation_bar_plot(
        df, output_dir / "ablation_dice_bars.png",
        metric="best_val_dice", title="NODE Step Ablation — Best Validation Dice",
        ylabel="Dice", dpi=dpi,
    )
    _save_ablation_bar_plot(
        df, output_dir / "ablation_iou_bars.png",
        metric="best_val_iou", title="NODE Step Ablation — Best Validation IoU",
        ylabel="IoU", dpi=dpi,
    )

    return output_dir


def compute_epochs_to_threshold(
    history_table: pd.DataFrame,
    thresholds: tuple[float, ...] = (0.80, 0.90, 0.95),
) -> pd.DataFrame:
    """Return the first epoch each group reaches each fraction of its own peak val_dice."""
    rows = []
    for group, group_df in history_table.groupby("group", sort=False):
        peak = group_df["val_dice"].max()
        ordered = group_df.sort_values("epoch")
        for t in thresholds:
            reached = ordered[ordered["val_dice"] >= t * peak]
            epoch: float = int(reached.iloc[0]["epoch"]) if not reached.empty else float("nan")
            rows.append({"group": group, "threshold": t, "epoch": epoch})
    return pd.DataFrame(rows, columns=["group", "threshold", "epoch"])


def _save_early_convergence_curve(
    history_table: pd.DataFrame,
    output_path: Path,
    zoom_epochs: int = 15,
    dpi: int = 150,
) -> None:
    plt = _get_plotting_libs()
    fig, ax = plt.subplots(figsize=(7, 5))

    for group, group_history in history_table.groupby("group", sort=False):
        ordered = group_history.sort_values("epoch")
        zoomed = ordered[ordered["epoch"] <= zoom_epochs]
        ax.plot(zoomed["epoch"], zoomed["val_dice"], marker="o", label=str(group))

    ax.set_title(f"Validation Dice — First {zoom_epochs} Epochs")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Dice")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Group")
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def _save_epochs_to_threshold_plot(
    threshold_df: pd.DataFrame,
    output_path: Path,
    dpi: int = 150,
) -> None:
    plt = _get_plotting_libs()

    groups = threshold_df["group"].unique().tolist()
    thresholds = sorted(threshold_df["threshold"].unique())
    n_groups = len(groups)
    bar_width = 0.25
    x = list(range(len(thresholds)))
    colors = ["#5470c6", "#ee8c30", "#3ba272", "#d84b4b"]

    fig, ax = plt.subplots(figsize=(max(6, 2.5 * len(thresholds)), 4))

    for i, group in enumerate(groups):
        subset = threshold_df[threshold_df["group"] == group].set_index("threshold")
        vals = []
        for t in thresholds:
            raw = subset.loc[t, "epoch"] if t in subset.index else float("nan")
            vals.append(0.0 if (isinstance(raw, float) and math.isnan(raw)) else float(raw))
        offsets = [xi + (i - (n_groups - 1) / 2) * bar_width for xi in x]
        bars = ax.bar(offsets, vals, bar_width, label=f"Group {group}", color=colors[i % len(colors)])
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.05,
                    str(int(val)),
                    ha="center", va="bottom", fontsize=8,
                )

    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(t * 100)}%" for t in thresholds])
    ax.set_xlabel("% of Peak Dice")
    ax.set_ylabel("Epochs to Reach")
    ax.set_title("Early Convergence Speed — Epochs to Reach % of Peak Dice")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(title="Group")
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def write_early_convergence_artifacts(
    artifacts_dir: str | Path,
    groups: Iterable[str] = ("A", "B", "C"),
    output_dir: str | Path | None = None,
    zoom_epochs: int = 15,
    thresholds: tuple[float, ...] = (0.80, 0.90, 0.95),
    dpi: int = 150,
) -> Path:
    history_table = build_history_table(artifacts_dir=artifacts_dir, groups=groups)
    if output_dir is None:
        output_dir = Path(artifacts_dir) / "summary"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    threshold_df = compute_epochs_to_threshold(history_table, thresholds)
    threshold_df.to_csv(output_dir / "epochs_to_threshold.csv", index=False)

    _save_early_convergence_curve(
        history_table, output_dir / "early_convergence_curve.png", zoom_epochs, dpi
    )
    _save_epochs_to_threshold_plot(threshold_df, output_dir / "epochs_to_threshold.png", dpi)

    return output_dir


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

    threshold_df = compute_epochs_to_threshold(history_table)
    threshold_df.to_csv(summary_dir / "epochs_to_threshold.csv", index=False)
    _save_early_convergence_curve(history_table, summary_dir / "early_convergence_curve.png")
    _save_epochs_to_threshold_plot(threshold_df, summary_dir / "epochs_to_threshold.png")

    return summary_dir


# ── Trainable parameter count ────────────────────────────────────────────────

_GROUP_ADAPTER_TYPE: dict[str, str] = {"A": "none", "B": "conv", "C": "node"}


def _adapter_block_params(channels: int, hidden_channels: int) -> int:
    """Parameter count of one conv bottleneck block (shared by Conv and NODE adapters)."""
    conv1 = channels * hidden_channels + hidden_channels
    bn = 2 * hidden_channels  # BN weight + bias (track_running_stats=False, so no buffers)
    conv2 = hidden_channels * channels + channels
    return conv1 + bn + conv2


def count_trainable_params(
    *,
    adapter_type: str,
    bottleneck_channels: int,
    hidden_channels: int,
    num_classes: int,
    encoder_last_channels: int = 512,
) -> dict[str, int]:
    """
    Analytically count trainable parameters per SegmentationModel component.
    The encoder is frozen, so only bottleneck_proj, adapter, decoder, and head count.
    Conv and NODE adapters share the same block, so their adapter param counts are equal.
    """
    bottleneck_proj = encoder_last_channels * bottleneck_channels + bottleneck_channels

    adapter = 0 if adapter_type == "none" else _adapter_block_params(bottleneck_channels, hidden_channels)

    mid1 = bottleneck_channels // 2
    mid2 = bottleneck_channels // 4
    decoder = (mid1 * bottleneck_channels * 9 + mid1) + (mid2 * mid1 * 9 + mid2)

    head = num_classes * mid2 + num_classes

    return {"bottleneck_proj": bottleneck_proj, "adapter": adapter, "decoder": decoder, "head": head}


def build_param_count_table(
    groups: Iterable[str] = ("A", "B", "C"),
    *,
    bottleneck_channels: int = 512,
    hidden_channels: int = 512,
    num_classes: int = 1,
    encoder_last_channels: int = 512,
) -> pd.DataFrame:
    rows = []
    for group in groups:
        adapter_type = _GROUP_ADAPTER_TYPE.get(str(group), "none")
        counts = count_trainable_params(
            adapter_type=adapter_type,
            bottleneck_channels=bottleneck_channels,
            hidden_channels=hidden_channels,
            num_classes=num_classes,
            encoder_last_channels=encoder_last_channels,
        )
        rows.append({"group": group, "adapter_type": adapter_type, **counts, "total": sum(counts.values())})
    return pd.DataFrame(rows, columns=["group", "adapter_type", "bottleneck_proj", "adapter", "decoder", "head", "total"])


def _save_param_count_plot(df: pd.DataFrame, output_path: Path, dpi: int = 150) -> None:
    plt = _get_plotting_libs()

    components = ["decoder", "bottleneck_proj", "adapter", "head"]
    colors = {"decoder": "#5470c6", "bottleneck_proj": "#ee8c30", "adapter": "#3ba272", "head": "#d84b4b"}

    groups = df["group"].tolist()
    x = list(range(len(groups)))

    fig, ax = plt.subplots(figsize=(max(5, 2.2 * len(groups)), 5))
    bottoms = [0.0] * len(groups)

    for component in components:
        vals = [float(df.loc[df["group"] == g, component].iloc[0]) / 1e6 for g in groups]
        ax.bar(x, vals, bottom=bottoms, label=component.replace("_", " ").title(), color=colors[component])
        bottoms = [b + v for b, v in zip(bottoms, vals)]

    for xi, total in zip(x, bottoms):
        ax.text(xi, total + 0.005, f"{total:.2f}M", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([f"Group {g}" for g in groups])
    ax.set_ylabel("Trainable Parameters (M)")
    ax.set_title("Trainable Parameter Count by Group\n(encoder frozen)")
    ax.legend(loc="upper left")
    ax.set_ylim(bottom=0)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def write_param_count_artifacts(
    output_dir: str | Path,
    groups: Iterable[str] = ("A", "B", "C"),
    *,
    bottleneck_channels: int = 512,
    hidden_channels: int = 512,
    num_classes: int = 1,
    encoder_last_channels: int = 512,
    dpi: int = 150,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = build_param_count_table(
        groups,
        bottleneck_channels=bottleneck_channels,
        hidden_channels=hidden_channels,
        num_classes=num_classes,
        encoder_last_channels=encoder_last_channels,
    )
    df.to_csv(output_dir / "param_counts.csv", index=False)
    _save_param_count_plot(df, output_dir / "param_counts.png", dpi=dpi)

    return output_dir
