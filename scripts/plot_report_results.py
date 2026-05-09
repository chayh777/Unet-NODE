from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from src.analysis.report_visualization import write_report_visualizations
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    from src.analysis.report_visualization import write_report_visualizations


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build report-ready figures for low-data NODE experiments."
    )
    parser.add_argument(
        "--artifacts-dir",
        default="artifacts",
        help="Root artifacts directory containing low_data, low_data_followup, and low_data_multiseed.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/report_figures",
        help="Directory where report CSVs and PNGs will be written.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    write_report_visualizations(
        artifacts_dir=Path(args.artifacts_dir),
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
