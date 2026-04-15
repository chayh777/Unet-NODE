from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from src.experiments.low_data_runner import run_group
except ModuleNotFoundError:
    # Allow `python scripts/run_low_data_experiment.py ...` from any CWD without
    # requiring editable installs or global packaging.
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    from src.experiments.low_data_runner import run_group


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run config-driven low-data ISIC2018 experiments (groups A/B/C)."
    )
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    parser.add_argument(
        "--group",
        required=True,
        choices=["A", "B", "C"],
        help="Ablation group: A=none, B=conv, C=node.",
    )
    args = parser.parse_args()
    run_group(args.config, args.group)


if __name__ == "__main__":
    main()
