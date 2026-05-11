from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from src.analysis.robustness_metrics import run_robustness_experiment
    from src.experiments.low_data_runner import load_config
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    from src.analysis.robustness_metrics import run_robustness_experiment
    from src.experiments.low_data_runner import load_config


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run robustness analysis with Gaussian noise perturbation."
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
        help="Experiment groups to compare.",
    )
    parser.add_argument(
        "--noise-levels",
        nargs="+",
        type=float,
        default=[0.0, 0.05, 0.1, 0.15, 0.2, 0.3],
        help="Noise levels (sigma values).",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    config = load_config(args.config)
    output_dir = run_robustness_experiment(
        config=config,
        artifacts_dir=Path(args.artifacts_dir),
        groups=args.groups,
        noise_levels=args.noise_levels,
    )
    print(f"Robustness analysis complete. Results saved to {output_dir}")


if __name__ == "__main__":
    main()