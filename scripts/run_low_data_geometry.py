from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from src.analysis.low_data_geometry import export_group_geometry
    from src.analysis.reduce_and_plot import run_low_data_geometry_plot
    from src.experiments.low_data_runner import load_config
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    from src.analysis.low_data_geometry import export_group_geometry
    from src.analysis.reduce_and_plot import run_low_data_geometry_plot
    from src.experiments.low_data_runner import load_config


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export low-data geometry embeddings and generate comparison plots."
    )
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    parser.add_argument(
        "--group",
        required=True,
        choices=["A", "B", "C"],
        help="Ablation group to export and plot.",
    )
    parser.add_argument(
        "--noise-sigma",
        type=float,
        default=0.0,
        help="Gaussian noise sigma for robustness test (default: 0.0).",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    config = load_config(args.config)
    group_dir = Path(config["paths"]["artifacts_dir"]) / f"group_{args.group.lower()}"
    checkpoint_path = group_dir / "best.pt"
    pre_csv, post_csv = export_group_geometry(
        config=config,
        group=args.group,
        checkpoint_path=checkpoint_path,
        noise_sigma=args.noise_sigma,
    )

    plot_config = config.get("geometry_plot", {})
    run_low_data_geometry_plot(
        pre_csv=pre_csv,
        post_csv=post_csv,
        output_dir=Path(pre_csv).parent,
        pca_components=int(plot_config.get("pca_components", 8)),
        umap_neighbors=int(plot_config.get("umap_neighbors", 15)),
        umap_min_dist=float(plot_config.get("umap_min_dist", 0.1)),
        random_state=int(plot_config.get("random_state", 42)),
        alpha=float(plot_config.get("alpha", 0.7)),
        point_size=float(plot_config.get("point_size", 18)),
        dpi=int(plot_config.get("dpi", 150)),
    )


if __name__ == "__main__":
    main()
