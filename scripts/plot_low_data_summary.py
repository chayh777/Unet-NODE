from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from src.analysis.low_data_reporting import write_summary_artifacts
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    from src.analysis.low_data_reporting import write_summary_artifacts


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build summary tables and comparison plots for low-data experiment groups."
    )
    parser.add_argument(
        "--artifacts-dir",
        required=True,
        help="Directory containing group_* experiment artifacts.",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        required=True,
        help="Experiment groups to summarize, for example: A B C",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    write_summary_artifacts(
        artifacts_dir=Path(args.artifacts_dir),
        groups=args.groups,
    )


if __name__ == "__main__":
    main()
