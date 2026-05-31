from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

try:
    from src.experiments.low_data_runner import load_config, run_group
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    from src.experiments.low_data_runner import load_config, run_group


@dataclass(frozen=True)
class MatrixEntry:
    dataset: str
    method: str
    config_path: str
    group: str
    run_prefix: str


MATRIX: tuple[MatrixEntry, ...] = (
    MatrixEntry(
        dataset="isic2018",
        method="plain",
        config_path="configs/experiments/isic2018_low_data_node_standard_unet.yaml",
        group="A",
        run_prefix="plain",
    ),
    MatrixEntry(
        dataset="isic2018",
        method="b",
        config_path="configs/experiments/isic2018_low_data_node_standard_unet.yaml",
        group="B",
        run_prefix="b",
    ),
    MatrixEntry(
        dataset="isic2018",
        method="output_node",
        config_path="configs/experiments/isic2018_low_data_output_node_c_standard_unet.yaml",
        group="C",
        run_prefix="output_node",
    ),
    MatrixEntry(
        dataset="isic2018",
        method="c_zero_last_steps16",
        config_path="configs/experiments/isic2018_low_data_node_c_zero_last_steps16_t1_standard_unet.yaml",
        group="C",
        run_prefix="c_zero_last_steps16",
    ),
    MatrixEntry(
        dataset="glas",
        method="plain",
        config_path="configs/experiments/glas_low_data_node_standard_unet.yaml",
        group="A",
        run_prefix="plain",
    ),
    MatrixEntry(
        dataset="glas",
        method="b",
        config_path="configs/experiments/glas_low_data_node_standard_unet.yaml",
        group="B",
        run_prefix="b",
    ),
    MatrixEntry(
        dataset="glas",
        method="output_node",
        config_path="configs/experiments/glas_low_data_output_node_c_standard_unet.yaml",
        group="C",
        run_prefix="output_node",
    ),
    MatrixEntry(
        dataset="glas",
        method="c_zero_last_steps16",
        config_path="configs/experiments/glas_low_data_node_c_zero_last_steps16_t1_standard_unet.yaml",
        group="C",
        run_prefix="c_zero_last_steps16",
    ),
)


def _parse_csv_values(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def select_entries(dataset: str, methods: list[str]) -> list[MatrixEntry]:
    allowed_methods = set(methods)
    return [
        entry
        for entry in MATRIX
        if (dataset == "all" or entry.dataset == dataset)
        and entry.method in allowed_methods
    ]


def build_seeded_config(entry: MatrixEntry, seed: int, artifacts_root: str) -> dict:
    config = load_config(entry.config_path)
    config["seed"] = int(seed)
    paths = dict(config["paths"])
    paths["artifacts_dir"] = (
        f"{artifacts_root}/{entry.dataset}_standard_unet_multiseed/"
        f"{entry.run_prefix}_seed{seed}"
    )
    config["paths"] = paths
    experiment = dict(config.get("experiment", {}))
    experiment["name"] = f"{entry.dataset}_standard_unet_{entry.run_prefix}_seed{seed}"
    experiment["group"] = entry.group
    experiment["method"] = entry.method
    config["experiment"] = experiment
    return config


def write_seeded_config(config: dict, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "config.yaml"
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the focused standard-U-Net multiseed matrix."
    )
    parser.add_argument(
        "--dataset",
        choices=["isic2018", "glas", "all"],
        default="all",
        help="Dataset slice to run.",
    )
    parser.add_argument(
        "--methods",
        default="plain,b,output_node,c_zero_last_steps16",
        help="Comma-separated methods: plain,b,output_node,c_zero_last_steps16.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2],
        help="Seeds to run.",
    )
    parser.add_argument(
        "--artifacts-root",
        default="artifacts",
        help="Root artifacts directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved runs without launching training.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    methods = _parse_csv_values(args.methods)
    entries = select_entries(args.dataset, methods)
    if not entries:
        raise SystemExit("No matrix entries selected.")

    for seed in args.seeds:
        for entry in entries:
            config = build_seeded_config(entry, seed, args.artifacts_root)
            artifacts_dir = Path(config["paths"]["artifacts_dir"])
            label = f"{entry.dataset}:{entry.method}:seed{seed}"
            if args.dry_run:
                print(f"{label} -> {artifacts_dir} --group {entry.group}")
                continue
            config_path = write_seeded_config(config, artifacts_dir / "_config")
            print(f"=== Running {label} ===", flush=True)
            run_group(config_path, entry.group)


if __name__ == "__main__":
    main()
