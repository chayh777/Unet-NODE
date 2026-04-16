from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from src.data.isic2018 import ISIC2018Dataset
from src.experiments.low_data_runner import resolve_group_adapter
from src.features.bottleneck_pooling import pool_class_embeddings
from src.models.segmentation_model import build_segmentation_model
from src.utils.io import ensure_dir, load_checkpoint


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
            return {str(name): int(value) for name, value in class_values.items()}
    return {"background": 0, "lesion": 1}


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


def write_embedding_csv(rows: list[dict[str, Any]], output_path: Path | str) -> Path:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    embedding_dim = 0
    for row in rows:
        embedding = row.get("embedding", [])
        if embedding_dim == 0:
            embedding_dim = len(embedding)
        elif len(embedding) != embedding_dim:
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
) -> tuple[Path, Path]:
    geometry_config = config.get("geometry", {})
    if not isinstance(geometry_config, dict):
        raise ValueError("config.geometry must be a mapping when provided.")

    class_values = _resolve_class_values(config)
    include_classes = list(geometry_config.get("include_classes", class_values.keys()))
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
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, **loader_kwargs)

    model = build_segmentation_model(
        encoder_name=config["model"]["encoder_name"],
        encoder_weights=config["model"]["encoder_weights"],
        in_channels=int(config["model"]["in_channels"]),
        num_classes=int(config["model"]["num_classes"]),
        adapter_type=resolve_group_adapter(group),
        bottleneck_channels=int(config["model"]["bottleneck_channels"]),
        adapter_hidden_channels=int(config["adapter"]["hidden_channels"]),
        freeze_encoder=bool(config["model"]["freeze_encoder"]),
        node_steps=int(config["node"]["steps"]),
        node_step_size=float(config["node"]["step_size"]),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    _load_checkpoint_into_model(model, Path(checkpoint_path), device)
    model.eval()

    all_rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            model_output = model(batch["image"].float().to(device))
            all_rows.extend(
                build_embedding_rows(
                    model_output=model_output,
                    mask=batch["mask"].to(device),
                    sample_ids=list(batch["sample_id"]),
                    include_classes=include_classes,
                    class_values=class_values,
                    min_mask_pixels=min_mask_pixels,
                )
            )

    geometry_dir = ensure_dir(
        Path(config["paths"]["artifacts_dir"])
        / "low_data"
        / f"group_{group.lower()}"
        / "geometry"
    )
    pre_path = geometry_dir / "pre_adapter_embeddings.csv"
    post_path = geometry_dir / "post_adapter_embeddings.csv"

    write_embedding_csv(
        [row for row in all_rows if row["state"] == "pre_adapter"],
        pre_path,
    )
    write_embedding_csv(
        [row for row in all_rows if row["state"] == "post_adapter"],
        post_path,
    )
    return pre_path, post_path
