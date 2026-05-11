from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from src.analysis.low_data_reporting import _get_plotting_libs
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    from src.analysis.low_data_reporting import _get_plotting_libs

import pandas as pd
from PIL import Image


def plot_robustness_summary(
    artifacts_dir: Path,
    output_path: Path | None = None,
) -> Path:
    plt = _get_plotting_libs()
    robustness_dir = artifacts_dir / "robustness"
    geometry_dir = robustness_dir / "geometry"

    if output_path is None:
        output_path = robustness_dir / "summary" / "robustness_analysis.png"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    metrics_df = pd.read_csv(robustness_dir / "robustness_metrics.csv")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = {"A": "#7a7a7a", "B": "#377eb8", "C": "#e15759"}
    for group in ["A", "B", "C"]:
        group_data = metrics_df[metrics_df["group"] == group].sort_values("sigma")
        if group_data.empty:
            continue
        axes[0].errorbar(
            group_data["sigma"],
            group_data["mean_dice"],
            yerr=group_data["std_dice"],
            marker="o",
            label=f"Group {group}",
            color=colors.get(group, "#4f83cc"),
            capsize=4,
        )

    axes[0].set_title("DICE vs Noise Level")
    axes[0].set_xlabel("Noise Level (σ)")
    axes[0].set_ylabel("Dice")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    scatter_file = geometry_dir / "sigma0.2" / "bottleneck_before_after_scatter.png"
    if not scatter_file.exists():
        scatter_file = geometry_dir / "sigma0.2" / "shared_projection_points.png"
    if scatter_file.exists():
        img = Image.open(scatter_file)
        axes[1].imshow(img)
        axes[1].axis("off")
        axes[1].set_title("Bottleneck Features at σ=0.2")
    else:
        axes[1].text(0.5, 0.5, "Geometry not available\nRun geometry extraction first", ha="center", va="center")
        axes[1].axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate robustness summary visualization.")
    parser.add_argument("--artifacts-dir", required=True, help="Directory containing robustness artifacts.")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    output_path = plot_robustness_summary(Path(args.artifacts_dir))
    print(f"Summary saved to {output_path}")


if __name__ == "__main__":
    main()