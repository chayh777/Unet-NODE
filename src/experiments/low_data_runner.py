from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ValueError(f"Failed to parse YAML config at {str(path)!r}.") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping/dict, got {type(data)!r}.")
    return data


def _require_mapping(config: dict[str, Any], key: str, context: str) -> dict[str, Any]:
    if key not in config:
        raise ValueError(f"{context} missing key {key!r}.")
    value = config[key]
    if not isinstance(value, dict):
        raise ValueError(f"{context}.{key} must be a mapping/dict, got {type(value)!r}.")
    return value


def _require_keys(mapping: dict[str, Any], keys: list[str], context: str) -> None:
    missing = [k for k in keys if k not in mapping]
    if missing:
        raise ValueError(f"{context} missing keys {missing}.")


def _validate_low_data_config(config: dict[str, Any]) -> None:
    """
    Validate presence and basic shape of config to fail early with clear messages.
    """
    _require_keys(config, ["seed", "paths", "data", "train", "model", "adapter", "node"], "config")

    paths = _require_mapping(config, "paths", "config")
    _require_keys(
        paths,
        ["train_images_dir", "train_masks_dir", "val_images_dir", "val_masks_dir", "artifacts_dir"],
        "config.paths",
    )

    data = _require_mapping(config, "data", "config")
    _require_keys(data, ["image_size", "train_ratio"], "config.data")

    train = _require_mapping(config, "train", "config")
    _require_keys(
        train,
        ["batch_size", "epochs", "learning_rate", "weight_decay", "early_stopping_patience"],
        "config.train",
    )

    model = _require_mapping(config, "model", "config")
    _require_keys(
        model,
        ["encoder_name", "encoder_weights", "in_channels", "num_classes", "bottleneck_channels", "freeze_encoder"],
        "config.model",
    )

    adapter = _require_mapping(config, "adapter", "config")
    _require_keys(adapter, ["hidden_channels"], "config.adapter")

    node = _require_mapping(config, "node", "config")
    _require_keys(node, ["steps", "step_size"], "config.node")


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
    config = load_config(config_path)
    _validate_low_data_config(config)
    adapter_type = resolve_group_adapter(group)

    # Local imports keep module import lightweight; validation happens before any heavy imports.
    import torch
    from torch.utils.data import DataLoader

    from src.data.isic2018 import ISIC2018Dataset
    from src.data.splits import build_ratio_subset, save_split_manifest
    from src.models.segmentation_model import build_segmentation_model
    from src.training.engine import fit

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
    split_manifest_path.parent.mkdir(parents=True, exist_ok=True)
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
    output_dir.mkdir(parents=True, exist_ok=True)
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
