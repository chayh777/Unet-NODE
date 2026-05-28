from __future__ import annotations

import argparse
import json
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
    parser.add_argument(
        "--checkpoint",
        help=(
            "Optional checkpoint path override. If omitted, the script uses the run's "
            "saved best checkpoint when available."
        ),
    )
    return parser


def _resolve_checkpoint_path(
    *, config: dict, group: str, checkpoint_arg: str | None
) -> Path:
    if checkpoint_arg:
        return Path(checkpoint_arg)

    group_dir = Path(config["paths"]["artifacts_dir"]) / f"group_{group.lower()}"
    default_checkpoint = group_dir / "best.pt"
    if default_checkpoint.exists():
        return default_checkpoint

    metrics_path = group_dir / "metrics.json"
    if metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        if not isinstance(metrics, dict):
            raise ValueError(f"metrics.json must contain an object at {metrics_path}")

        best_checkpoint = metrics.get("best_checkpoint")
        checkpoint_saved = metrics.get("checkpoint_saved")
        if checkpoint_saved is False or not best_checkpoint:
            raise FileNotFoundError(
                "No saved checkpoint is available for geometry export in "
                f"{group_dir}. This run likely used train.save_best_checkpoint: false. "
                "Rerun training with train.save_best_checkpoint: true or pass --checkpoint."
            )
        return Path(best_checkpoint)

    return default_checkpoint


def main() -> None:
    args = _build_parser().parse_args()

    config = load_config(args.config)
    checkpoint_path = _resolve_checkpoint_path(
        config=config,
        group=args.group,
        checkpoint_arg=args.checkpoint,
    )
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
