from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from src.analysis.reduce_and_plot import run_cross_group_geometry_plot
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    from src.analysis.reduce_and_plot import run_cross_group_geometry_plot


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Project Group A/B/C bottleneck embeddings into a shared PCA+UMAP space "
            "and generate cross-group comparison plots."
        )
    )
    parser.add_argument("--artifacts-dir", required=True, help="Base artifacts directory containing group_* subdirs.")
    parser.add_argument("--groups", nargs="+", default=["A", "B", "C"], help="Groups to compare.")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: <artifacts-dir>/summary/cross_group_geometry).")
    parser.add_argument("--pca-components", type=int, default=8)
    parser.add_argument("--umap-neighbors", type=int, default=15)
    parser.add_argument("--umap-min-dist", type=float, default=0.1)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.65)
    parser.add_argument("--point-size", type=float, default=20)
    parser.add_argument("--dpi", type=int, default=150)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    artifacts_dir = Path(args.artifacts_dir)

    group_geometry_dirs: dict[str, Path] = {}
    for group in args.groups:
        geom_dir = artifacts_dir / f"group_{group.lower()}" / "geometry"
        if not geom_dir.exists():
            raise FileNotFoundError(
                f"Geometry directory for group {group} not found: {geom_dir}\n"
                "Run scripts/run_low_data_geometry.py first."
            )
        group_geometry_dirs[group] = geom_dir

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else artifacts_dir / "summary" / "cross_group_geometry"
    )

    result_dir = run_cross_group_geometry_plot(
        group_geometry_dirs=group_geometry_dirs,
        output_dir=output_dir,
        pca_components=args.pca_components,
        umap_neighbors=args.umap_neighbors,
        umap_min_dist=args.umap_min_dist,
        random_state=args.random_state,
        alpha=args.alpha,
        point_size=args.point_size,
        dpi=args.dpi,
    )
    print(f"Cross-group geometry plots written to: {result_dir}")


if __name__ == "__main__":
    main()
