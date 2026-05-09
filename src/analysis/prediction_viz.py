from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Absolute path to a bundled CJK-capable font on macOS.
# STHeiti Light.ttc is present on every macOS installation and renders
# Simplified/Traditional Chinese without squares.
_HEITI_PATH = Path("/System/Library/Fonts/STHeiti Light.ttc")


def _get_plotting_libs():
    mpl_config_dir = Path.cwd() / ".matplotlib"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm

    # Register the system CJK font once per process.
    if _HEITI_PATH.exists():
        fm.fontManager.addfont(str(_HEITI_PATH))
        matplotlib.rcParams["font.sans-serif"] = ["Heiti TC", "DejaVu Sans"]
    else:
        # Non-macOS fallback: let the user install a CJK font separately.
        matplotlib.rcParams["font.sans-serif"] = [
            "SimHei", "WenQuanYi Micro Hei", "Noto Sans CJK SC", "DejaVu Sans"
        ]

    matplotlib.rcParams["axes.unicode_minus"] = False

    return plt


def load_model_from_config(
    config: dict[str, Any],
    group: str,
    device: torch.device,
) -> torch.nn.Module:
    from src.experiments.low_data_runner import (
        _resolve_adapter_init,
        resolve_group_adapter,
    )
    from src.models.segmentation_model import build_segmentation_model
    from src.analysis.low_data_geometry import _load_checkpoint_into_model

    adapter_type = resolve_group_adapter(group)
    adapter_init = _resolve_adapter_init(config)

    model = build_segmentation_model(
        encoder_name=config["model"]["encoder_name"],
        encoder_weights=config["model"]["encoder_weights"],
        in_channels=int(config["model"]["in_channels"]),
        num_classes=int(config["model"]["num_classes"]),
        adapter_type=adapter_type,
        bottleneck_channels=int(config["model"]["bottleneck_channels"]),
        adapter_hidden_channels=int(config["adapter"]["hidden_channels"]),
        adapter_init=adapter_init,
        freeze_encoder=bool(config["model"]["freeze_encoder"]),
        node_steps=int(config["node"]["steps"]),
        node_step_size=float(config["node"]["step_size"]),
    )
    model.to(device)

    artifacts_dir = Path(config["paths"]["artifacts_dir"])
    checkpoint_path = artifacts_dir / f"group_{group.lower()}" / "best.pt"
    _load_checkpoint_into_model(model, checkpoint_path, device)
    model.eval()
    return model


def collect_predictions(
    model: torch.nn.Module,
    dataset: Any,
    sample_indices: list[int],
    device: torch.device,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Return (images, gt_masks, pred_masks) as H×W or H×W×3 uint8 arrays."""
    images_out: list[np.ndarray] = []
    gt_masks_out: list[np.ndarray] = []
    pred_masks_out: list[np.ndarray] = []

    with torch.no_grad():
        for idx in sample_indices:
            sample = dataset[idx]
            image_tensor = sample["image"].float().unsqueeze(0).to(device)
            gt_mask = sample["mask"].numpy().astype(np.uint8)

            output = model(image_tensor)
            pred = (torch.sigmoid(output.logits) > 0.5).squeeze().cpu().numpy().astype(np.uint8)

            image_np = sample["image"].permute(1, 2, 0).numpy()
            image_np = (image_np * 255).clip(0, 255).astype(np.uint8)

            images_out.append(image_np)
            gt_masks_out.append(gt_mask)
            pred_masks_out.append(pred)

    return images_out, gt_masks_out, pred_masks_out


def _overlay_mask(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
    alpha: float = 0.45,
) -> np.ndarray:
    overlay = image_rgb.copy().astype(np.float32)
    for c, v in enumerate(color):
        overlay[mask == 1, c] = overlay[mask == 1, c] * (1 - alpha) + v * alpha
    return overlay.clip(0, 255).astype(np.uint8)


# Yellow for ground truth, red for predictions — visually distinct on skin images.
_GT_COLOR: tuple[int, int, int] = (255, 255, 0)
_PRED_COLOR: tuple[int, int, int] = (255, 80, 80)


def render_prediction_grid(
    images: list[np.ndarray],
    gt_masks: list[np.ndarray],
    group_preds: list[tuple[str, list[np.ndarray]]],
    output_path: Path | str,
    *,
    dpi: int = 150,
) -> Path:
    """
    Save a grid: [ 原图 | 真值掩码 | 组A预测 | 组B预测 | ... ]
    每行一个样本。

    Parameters
    ----------
    group_preds : list of (column_label, pred_masks) pairs, one per group.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_samples = len(images)
    n_cols = 2 + len(group_preds)
    col_labels = ["原图", "真值掩码"] + [label for label, _ in group_preds]

    plt = _get_plotting_libs()
    fig, axes = plt.subplots(
        n_samples, n_cols,
        figsize=(3 * n_cols, 3 * n_samples),
        squeeze=False,
    )

    for row, (img, gt) in enumerate(zip(images, gt_masks)):
        axes[row][0].imshow(img)
        axes[row][0].axis("off")

        axes[row][1].imshow(_overlay_mask(img, gt, _GT_COLOR))
        axes[row][1].axis("off")

        for col_offset, (_, preds) in enumerate(group_preds):
            axes[row][2 + col_offset].imshow(_overlay_mask(img, preds[row], _PRED_COLOR))
            axes[row][2 + col_offset].axis("off")

    for col, label in enumerate(col_labels):
        axes[0][col].set_title(label, fontsize=11, pad=5)

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path
