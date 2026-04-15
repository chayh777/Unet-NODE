from __future__ import annotations

import argparse

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

