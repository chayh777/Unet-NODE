"""
Visualize segmentation predictions from one or more trained groups side by side.

Usage
-----
python scripts/visualize_predictions.py \
    --configs A:configs/experiments/isic2018_low_data_node.yaml \
              B:configs/experiments/isic2018_low_data_node.yaml \
              C:configs/experiments/isic2018_low_data_node.yaml \
    --groups  A:A  B:B  C:C \
    --n-samples 8 \
    --output-dir artifacts/presentation/predictions \
    --seed 42

Each --configs entry is  LABEL:path/to/config.yaml
Each --groups  entry is  LABEL:GROUP_LETTER  (A / B / C)
LABEL must match between the two lists.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

# Allow running from repo root without installing the package.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _parse_label_value(pairs: list[str], flag: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for entry in pairs:
        if ":" not in entry:
            raise SystemExit(
                f"--{flag}: expected LABEL:VALUE, got {entry!r}"
            )
        label, _, value = entry.partition(":")
        if not label or not value:
            raise SystemExit(
                f"--{flag}: both label and value must be non-empty, got {entry!r}"
            )
        result[label] = value
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="生成验证集分割预测对比图（原图 | 真值 | 各组预测）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        required=True,
        metavar="LABEL:CONFIG",
        help="LABEL:path/to/config.yaml，可指定多个",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        required=True,
        metavar="LABEL:GROUP",
        help="LABEL:GROUP_LETTER（A/B/C），与 --configs 的 LABEL 对应",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=8,
        help="随机抽取的样本数（默认 8）",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/presentation/predictions",
        help="输出目录（默认 artifacts/presentation/predictions）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（控制样本选择，默认 42）",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="输出图片 DPI（默认 150）",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    config_map = _parse_label_value(args.configs, "configs")
    group_map = _parse_label_value(args.groups, "groups")

    labels = list(config_map.keys())
    missing = [l for l in labels if l not in group_map]
    if missing:
        raise SystemExit(f"--groups 缺少以下 label 的对应项: {missing}")

    import torch
    from src.experiments.low_data_runner import load_config
    from src.data.isic2018 import ISIC2018Dataset
    from src.analysis.prediction_viz import (
        collect_predictions,
        load_model_from_config,
        render_prediction_grid,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    # All groups share the same validation set; use the first config as reference.
    first_config = load_config(config_map[labels[0]])
    class_values = {"background": 0, "lesion": 1}
    dataset = ISIC2018Dataset(
        images_dir=first_config["paths"]["val_images_dir"],
        masks_dir=first_config["paths"]["val_masks_dir"],
        image_size=int(first_config["data"]["image_size"]),
        class_values=class_values,
    )

    n = min(args.n_samples, len(dataset))
    rng = random.Random(args.seed)
    indices = rng.sample(range(len(dataset)), n)
    print(f"抽取 {n} 个样本（共 {len(dataset)} 个），索引: {indices}")

    # Collect raw images and ground truth from the dataset (no model needed).
    from src.analysis.prediction_viz import collect_predictions as _cp
    # We load images/gt via the first model pass; reuse them for all groups.
    all_images = []
    all_gt = []
    for idx in indices:
        sample = dataset[idx]
        import numpy as np
        img = sample["image"].permute(1, 2, 0).numpy()
        img = (img * 255).clip(0, 255).astype(np.uint8)
        gt = sample["mask"].numpy().astype(np.uint8)
        all_images.append(img)
        all_gt.append(gt)

    group_preds: list[tuple[str, list]] = []
    for label in labels:
        config = load_config(config_map[label])
        group = group_map[label]
        print(f"加载 [{label}] Group {group} checkpoint …")
        model = load_model_from_config(config, group, device)

        _, _, preds = collect_predictions(model, dataset, indices, device)
        group_preds.append((label, preds))
        print(f"  [{label}] 推理完成")

    output_path = Path(args.output_dir) / "predictions_grid.png"
    saved = render_prediction_grid(
        all_images,
        all_gt,
        group_preds,
        output_path,
        dpi=args.dpi,
    )
    print(f"已保存: {saved}")


if __name__ == "__main__":
    main()
