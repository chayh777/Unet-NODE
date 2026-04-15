from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping/dict, got {type(data)!r}.")
    return data


def resolve_group_adapter(group: str) -> str:
    """
    Map ablation group to adapter type.

    Groups:
      - A: baseline (no adapter)
      - B: convolutional bottleneck adapter
      - C: NODE adapter
    """
    if group == "A":
        return "none"
    if group == "B":
        return "conv"
    if group == "C":
        return "node"
    raise ValueError(f"Unknown group: {group!r}. Expected one of: A, B, C.")


def run_group(config_path: str | Path, group: str):
    """
    Run a single ablation group (A/B/C) as defined by the low-data experiment config.

    This function is intentionally "wiring" heavy: it builds datasets/loaders, model,
    optimizer, and calls the shared training engine.
    """
    # Local imports keep `resolve_group_adapter`/`load_config` lightweight to import in tests.
    import torch
    from torch.utils.data import DataLoader

    from src.data.isic2018 import ISIC2018Dataset
    from src.data.splits import build_ratio_subset, save_split_manifest
    from src.models.segmentation_model import build_segmentation_model
    from src.training.engine import fit

    config = load_config(config_path)

    full_train_dataset = ISIC2018Dataset(
        images_dir=config["paths"]["train_images_dir"],
        masks_dir=config["paths"]["train_masks_dir"],
        image_size=config["data"]["image_size"],
        class_values={"background": 0, "lesion": 1},
        sample_ids=None,
    )

    selected_ids = build_ratio_subset(
        [path.stem for path in full_train_dataset.image_paths],
        ratio=float(config["data"]["train_ratio"]),
        seed=int(config["seed"]),
    )

    ratio_pct = int(round(float(config["data"]["train_ratio"]) * 100))
    split_manifest_path = (
        Path(config["paths"]["artifacts_dir"])
        / "splits"
        / f"train_seed{int(config['seed'])}_ratio{ratio_pct}.csv"
    )
    save_split_manifest(selected_ids, split_manifest_path)

    train_dataset = ISIC2018Dataset(
        images_dir=config["paths"]["train_images_dir"],
        masks_dir=config["paths"]["train_masks_dir"],
        image_size=config["data"]["image_size"],
        class_values={"background": 0, "lesion": 1},
        sample_ids=selected_ids,
    )
    val_dataset = ISIC2018Dataset(
        images_dir=config["paths"]["val_images_dir"],
        masks_dir=config["paths"]["val_masks_dir"],
        image_size=config["data"]["image_size"],
        class_values={"background": 0, "lesion": 1},
        sample_ids=None,
    )

    batch_size = int(config["train"]["batch_size"])
    loader_kwargs = {}
    if "num_workers" in config.get("data", {}):
        loader_kwargs["num_workers"] = int(config["data"]["num_workers"])
    if "pin_memory" in config.get("data", {}):
        loader_kwargs["pin_memory"] = bool(config["data"]["pin_memory"])

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, **loader_kwargs
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs
    )

    adapter_type = resolve_group_adapter(group)
    model = build_segmentation_model(
        encoder_name=config["model"]["encoder_name"],
        encoder_weights=config["model"]["encoder_weights"],
        in_channels=int(config["model"]["in_channels"]),
        num_classes=int(config["model"]["num_classes"]),
        adapter_type=adapter_type,
        bottleneck_channels=int(config["model"]["bottleneck_channels"]),
        adapter_hidden_channels=int(config["adapter"]["hidden_channels"]),
        freeze_encoder=bool(config["model"]["freeze_encoder"]),
        node_steps=int(config["node"]["steps"]),
        node_step_size=float(config["node"]["step_size"]),
    )

    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if getattr(param, "requires_grad", False)],
        lr=float(config["train"]["learning_rate"]),
        weight_decay=float(config["train"]["weight_decay"]),
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    output_dir = Path(config["paths"]["artifacts_dir"]) / f"group_{group.lower()}"
    return fit(
        model,
        train_loader,
        val_loader,
        optimizer,
        int(config["train"]["epochs"]),
        int(config["train"]["early_stopping_patience"]),
        output_dir,
        device=device,
    )
