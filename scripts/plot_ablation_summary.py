from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from src.analysis.low_data_reporting import write_ablation_summary
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    from src.analysis.low_data_reporting import write_ablation_summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ablation summary: Dice/IoU vs. ODE steps × initialization, grouped bar chart."
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        required=True,
        metavar="CONFIG",
        help="Paths to T1 ablation config YAMLs (e.g. configs/experiments/isic2018_low_data_node_c_steps2_t1.yaml).",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/low_data_followup/summary",
        help="Output directory (default: artifacts/low_data_followup/summary).",
    )
    parser.add_argument(
        "--group",
        default="C",
        help="Experiment group letter used when running each config (default: C).",
    )
    parser.add_argument("--dpi", type=int, default=150)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    result_dir = write_ablation_summary(
        config_paths=args.configs,
        output_dir=args.output_dir,
        group=args.group,
        dpi=args.dpi,
    )
    print(f"Ablation summary written to: {result_dir}")


if __name__ == "__main__":
    main()
