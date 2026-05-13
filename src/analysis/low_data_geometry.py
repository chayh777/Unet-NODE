from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from src.data.isic2018 import ISIC2018Dataset
from src.experiments.low_data_runner import (
    _is_known_windows_dataloader_worker_permission_error,
    _resolve_adapter_init,
    resolve_group_adapter,
)
from src.features.bottleneck_pooling import pool_class_embeddings
from src.models.segmentation_model import build_segmentation_model
from src.utils.io import ensure_dir, load_checkpoint

_ISIC_BINARY_CLASS_VALUES = {"background": 0, "lesion": 1}


def _add_gaussian_noise(images: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return images
    return images + torch.randn_like(images) * sigma


def _normalize_state_dict(raw_state: Any) -> dict[str, Any]:
    if isinstance(raw_state, dict):
        for key in ("state_dict", "model_state_dict"):
            if key in raw_state:
                raw_state = raw_state[key]
                break

    if not isinstance(raw_state, dict):
        raise RuntimeError("Checkpoint must contain a state_dict-compatible mapping.")

    normalized_state: dict[str, Any] = {}
    for name, tensor in raw_state.items():
        normalized_name = name.replace("module.", "", 1) if name.startswith("module.") else name
        normalized_state[normalized_name] = tensor
    return normalized_state


def _load_checkpoint_into_model(model: Any, checkpoint_path: Path, device: torch.device) -> None:
    state_dict = _normalize_state_dict(load_checkpoint(checkpoint_path, device))
    try:
        model.load_state_dict(state_dict)
    except Exception as exc:  # noqa: BLE001 - surface a clearer path-specific error
        raise RuntimeError(f"Failed to load checkpoint at {checkpoint_path}") from exc


def _resolve_class_values(config: dict[str, Any]) -> dict[str, int]:
    for section_name in ("dataset", "data"):
        section = config.get(section_name)
        if isinstance(section, dict) and "class_values" in section:
            class_values = section["class_values"]
            if not isinstance(class_values, dict) or not class_values:
                raise ValueError(f"config.{section_name}.class_values must be a non-empty mapping.")
            normalized = {str(name): int(value) for name, value in class_values.items()}
            expected_names = sorted(_ISIC_BINARY_CLASS_VALUES)
            if sorted(normalized) != expected_names:
                raise ValueError(
                    "Geometry export requires binary ISIC class_values with keys "
                    f"{expected_names}; got {sorted(normalized)}."
                )
            if normalized != _ISIC_BINARY_CLASS_VALUES:
                raise ValueError(
                    "Geometry export requires binary ISIC class_values mapping "
                    f"{_ISIC_BINARY_CLASS_VALUES}; got {normalized}."
                )
            return normalized
    return dict(_ISIC_BINARY_CLASS_VALUES)


def _require_mapping(config: dict[str, Any], key: str, context: str) -> dict[str, Any]:
    if key not in config:
        raise ValueError(f"{context} missing key {key!r}.")
    value = config[key]
    if not isinstance(value, dict):
        raise ValueError(f"{context}.{key} must be a mapping/dict, got {type(value)!r}.")
    return value


def _require_keys(mapping: dict[str, Any], keys: list[str], context: str) -> None:
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise ValueError(f"{context} missing keys {missing}.")


def _validate_geometry_export_config(config: dict[str, Any]) -> None:
    _require_keys(config, ["paths", "data", "train", "model", "adapter", "node"], "config")

    paths = _require_mapping(config, "paths", "config")
    _require_keys(paths, ["val_images_dir", "val_masks_dir", "artifacts_dir"], "config.paths")

    data = _require_mapping(config, "data", "config")
    _require_keys(data, ["image_size"], "config.data")

    train = _require_mapping(config, "train", "config")
    _require_keys(train, ["batch_size"], "config.train")

    model = _require_mapping(config, "model", "config")
    _require_keys(
        model,
        [
            "encoder_name",
            "encoder_weights",
            "in_channels",
            "num_classes",
            "bottleneck_channels",
            "freeze_encoder",
        ],
        "config.model",
    )

    adapter = _require_mapping(config, "adapter", "config")
    _require_keys(adapter, ["hidden_channels"], "config.adapter")

    node = _require_mapping(config, "node", "config")
    _require_keys(node, ["steps", "step_size", "solver"], "config.node")

    if "geometry" in config and not isinstance(config["geometry"], dict):
        raise ValueError("config.geometry must be a mapping when provided.")


def _resolve_include_classes(
    geometry_config: dict[str, Any],
    class_values: dict[str, int],
) -> list[str]:
    include_classes = geometry_config.get("include_classes", class_values.keys())
    resolved = [str(name) for name in include_classes]
    missing = sorted({name for name in resolved if name not in class_values})
    if missing:
        raise ValueError(
            "config.geometry.include_classes contains unknown classes "
            f"{missing}; expected values from {sorted(class_values)}."
        )
    return resolved


def build_embedding_rows(
    model_output: Any,
    mask: torch.Tensor,
    sample_ids: list[str],
    include_classes: list[str],
    class_values: dict[str, int],
    min_mask_pixels: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for state_name, attr_name in (
        ("pre_adapter", "bottleneck"),
        ("post_adapter", "adapted_bottleneck"),
    ):
        bottleneck = getattr(model_output, attr_name, None)
        if bottleneck is None:
            raise RuntimeError(f"Model output must expose .{attr_name}")

        pooled_rows = pool_class_embeddings(
            bottleneck=bottleneck,
            mask=mask,
            sample_ids=sample_ids,
            class_names=include_classes,
            class_values=class_values,
            min_mask_pixels=min_mask_pixels,
        )
        for row in pooled_rows:
            rows.append(
                {
                    "sample_id": row["sample_id"],
                    "state": state_name,
                    "class_name": row["class_name"],
                    "pixel_count": int(row["pixel_count"]),
                    "embedding": [float(value) for value in row["embedding"]],
                }
            )
    return rows


def write_embedding_csv(
    rows: list[dict[str, Any]],
    output_path: Path | str,
    *,
    embedding_dim: int | None = None,
) -> Path:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    inferred_dim: int | None = None
    for row in rows:
        embedding = row.get("embedding", [])
        if inferred_dim is None:
            inferred_dim = len(embedding)
        elif len(embedding) != inferred_dim:
            raise ValueError("Embedding length mismatch across rows.")

    if embedding_dim is None:
        embedding_dim = inferred_dim

    if embedding_dim is None:
        raise ValueError("embedding_dim is required when writing an empty embedding CSV.")
    if embedding_dim < 1:
        raise ValueError("embedding_dim must be a positive integer.")

    for row in rows:
        if len(row.get("embedding", [])) != embedding_dim:
            raise ValueError("Embedding length mismatch across rows.")

    fieldnames = ["sample_id", "state", "class_name", "pixel_count"] + [
        f"embedding_{idx:04d}" for idx in range(embedding_dim)
    ]

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            record = {
                "sample_id": row["sample_id"],
                "state": row["state"],
                "class_name": row["class_name"],
                "pixel_count": int(row["pixel_count"]),
            }
            for idx, value in enumerate(row.get("embedding", [])):
                record[f"embedding_{idx:04d}"] = float(value)
            writer.writerow(record)

    return output_path


def export_group_geometry(
    config: dict[str, Any],
    group: str,
    checkpoint_path: Path | str,
    noise_sigma: float = 0.0,
) -> tuple[Path, Path]:
    _validate_geometry_export_config(config)
    geometry_config = config.get("geometry", {})

    class_values = _resolve_class_values(config)
    include_classes = _resolve_include_classes(geometry_config, class_values)
    min_mask_pixels = int(geometry_config.get("min_mask_pixels", 1))
    batch_size = int(geometry_config.get("batch_size", config["train"]["batch_size"]))

    loader_kwargs: dict[str, Any] = {}
    data_config = config.get("data", {})
    if isinstance(data_config, dict):
        if "num_workers" in data_config:
            loader_kwargs["num_workers"] = int(data_config["num_workers"])
        if "pin_memory" in data_config:
            loader_kwargs["pin_memory"] = bool(data_config["pin_memory"])

    dataset = ISIC2018Dataset(
        images_dir=config["paths"]["val_images_dir"],
        masks_dir=config["paths"]["val_masks_dir"],
        image_size=int(config["data"]["image_size"]),
        class_values=class_values,
        sample_ids=None,
    )

    def _build_loader(*, num_workers_override: int | None) -> DataLoader:
        effective_kwargs = dict(loader_kwargs)
        if num_workers_override is not None:
            effective_kwargs["num_workers"] = int(num_workers_override)
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, **effective_kwargs)

    model = build_segmentation_model(
        encoder_name=config["model"]["encoder_name"],
        encoder_weights=config["model"]["encoder_weights"],
        in_channels=int(config["model"]["in_channels"]),
        num_classes=int(config["model"]["num_classes"]),
        adapter_type=resolve_group_adapter(group),
        bottleneck_channels=int(config["model"]["bottleneck_channels"]),
        adapter_hidden_channels=int(config["adapter"]["hidden_channels"]),
        adapter_init=_resolve_adapter_init(config),
        freeze_encoder=bool(config["model"]["freeze_encoder"]),
        node_steps=int(config["node"]["steps"]),
        node_step_size=float(config["node"]["step_size"]),
        node_solver=str(config["node"].get("solver", "euler")),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    _load_checkpoint_into_model(model, Path(checkpoint_path), device)
    model.eval()

    def _collect_rows(loader: DataLoader) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        with torch.no_grad():
            for batch in loader:
                images = batch["image"].float()
                images = _add_gaussian_noise(images, noise_sigma)
                model_output = model(images.to(device))
                rows.extend(
                    build_embedding_rows(
                        model_output=model_output,
                        mask=batch["mask"].to(device),
                        sample_ids=list(batch["sample_id"]),
                        include_classes=include_classes,
                        class_values=class_values,
                        min_mask_pixels=min_mask_pixels,
                    )
                )
        return rows

    configured_workers = int(loader_kwargs.get("num_workers", 0))
    loader = _build_loader(num_workers_override=None)
    try:
        all_rows = _collect_rows(loader)
    except PermissionError as exc:
        if (
            configured_workers > 0
            and _is_known_windows_dataloader_worker_permission_error(exc)
        ):
            all_rows = _collect_rows(_build_loader(num_workers_override=0))
        else:
            raise

    if noise_sigma > 0:
        geometry_dir = ensure_dir(
            Path(config["paths"]["artifacts_dir"])
            / f"group_{group.lower()}"
            / "geometry"
            / f"sigma{noise_sigma}"
        )
    else:
        geometry_dir = ensure_dir(
            Path(config["paths"]["artifacts_dir"])
            / f"group_{group.lower()}"
            / "geometry"
        )
    pre_path = geometry_dir / "pre_adapter_embeddings.csv"
    post_path = geometry_dir / "post_adapter_embeddings.csv"
    embedding_dim = int(config["model"]["bottleneck_channels"])

    write_embedding_csv(
        [row for row in all_rows if row["state"] == "pre_adapter"],
        pre_path,
        embedding_dim=embedding_dim,
    )
    write_embedding_csv(
        [row for row in all_rows if row["state"] == "post_adapter"],
        post_path,
        embedding_dim=embedding_dim,
    )
    return pre_path, post_path
