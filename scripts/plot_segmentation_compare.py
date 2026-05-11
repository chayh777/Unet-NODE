from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from src.analysis.segmentation_compare import (
        generate_segmentation_comparison,
    )
    from src.experiments.low_data_runner import load_config
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    from src.analysis.segmentation_compare import (
        generate_segmentation_comparison,
    )
    from src.experiments.low_data_runner import load_config


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate segmentation comparison grid for low-data experiment groups."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--artifacts-dir",
        required=True,
        help="Directory containing group_* experiment artifacts.",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        default=["A", "B", "C"],
        help="Experiment groups to compare, for example: A B C",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=3,
        help="Number of samples to display (default: 3).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.7,
        help="Mask overlay alpha (default: 0.7).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Output DPI (default: 150).",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    config = load_config(args.config)
    output_dir = generate_segmentation_comparison(
        config=config,
        artifacts_dir=Path(args.artifacts_dir),
        groups=args.groups,
        num_samples=args.num_samples,
        alpha=args.alpha,
        dpi=args.dpi,
    )
    print(f"Saved to {output_dir}")


if __name__ == "__main__":
    main()