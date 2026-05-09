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
