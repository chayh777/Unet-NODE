"""Trainable parameter count table and stacked bar chart for experiment groups."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from src.analysis.low_data_reporting import write_param_count_artifacts
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    from src.analysis.low_data_reporting import write_param_count_artifacts


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate trainable parameter count table and stacked bar chart."
    )
    p.add_argument("--groups", nargs="+", default=["A", "B", "C"],
                   help="Experiment groups to compare (default: A B C)")
    p.add_argument("--output-dir", default="artifacts/low_data/summary",
                   help="Directory to write param_counts.csv and param_counts.png")
    p.add_argument("--bottleneck-channels", type=int, default=512)
    p.add_argument("--hidden-channels", type=int, default=512)
    p.add_argument("--num-classes", type=int, default=1)
    p.add_argument("--encoder-last-channels", type=int, default=512,
                   help="Output channels of the encoder's last stage (512 for resnet34)")
    p.add_argument("--dpi", type=int, default=150)
    return p


def main() -> None:
    args = _build_parser().parse_args()
    out = write_param_count_artifacts(
        output_dir=Path(args.output_dir),
        groups=args.groups,
        bottleneck_channels=args.bottleneck_channels,
        hidden_channels=args.hidden_channels,
        num_classes=args.num_classes,
        encoder_last_channels=args.encoder_last_channels,
        dpi=args.dpi,
    )
    print(f"param_counts.csv  →  {out / 'param_counts.csv'}")
    print(f"param_counts.png  →  {out / 'param_counts.png'}")


if __name__ == "__main__":
    main()
