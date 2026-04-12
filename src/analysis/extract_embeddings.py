"""Extract one CSV row per sample-class point (sample_id, state, class_name, pixel_count, embedding_XXXX)."""

from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.isic2018 import ISIC2018Dataset
from src.features.bottleneck_pooling import pool_class_embeddings
from src.models.unet import SimpleUNet
from src.utils.io import ensure_dir, load_checkpoint, load_config


def _load_checkpoint_into_model(model: nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    raw_state = load_checkpoint(checkpoint_path, device)
    if isinstance(raw_state, dict):
        for key in ("state_dict", "model_state_dict"):
            if key in raw_state:
                raw_state = raw_state[key]
                break
    state_dict = raw_state
    if isinstance(state_dict, dict):
        normalized_state_dict = {}
        for name, tensor in state_dict.items():
            normalized_name = name.replace("module.", "", 1) if name.startswith("module.") else name
            normalized_state_dict[normalized_name] = tensor
        state_dict = normalized_state_dict
    try:
        model.load_state_dict(state_dict)
    except Exception as exc:
        raise RuntimeError(f"Failed to load checkpoint at {checkpoint_path}") from exc


def _build_dataframe(rows_data: list[tuple[str, str, str, int, list[float]]], embedding_dim: int, output_path: Path) -> Path:
    columns = ["sample_id", "state", "class_name", "pixel_count"] + [
        f"embedding_{idx:04d}" for idx in range(embedding_dim)
    ]
    records: list[dict] = []
    for sample_id, state, class_name, pixel_count, embedding in rows_data:
        if len(embedding) != embedding_dim:
            raise ValueError("Embedding length mismatch across pooled rows.")
        record = {
            "sample_id": sample_id,
            "state": state,
            "class_name": class_name,
            "pixel_count": pixel_count,
        }
        record.update({f"embedding_{idx:04d}": float(value) for idx, value in enumerate(embedding)})
        records.append(record)

    df = pd.DataFrame(records, columns=columns)
    df.to_csv(output_path, index=False)
    return output_path


def _run_extraction(
    config: dict,
    checkpoint_path: Path | None,
    state_name: str,
    output_name: str,
) -> Path:
    dataset = ISIC2018Dataset(
        images_dir=config["paths"]["isic_images_dir"],
        masks_dir=config["paths"]["isic_masks_dir"],
        image_size=config["dataset"]["image_size"],
        class_values=config["dataset"]["class_values"],
    )
    loader = DataLoader(dataset, batch_size=config["extraction"]["batch_size"], shuffle=False)

    model = SimpleUNet(
        encoder_name=config["model"]["encoder_name"],
        encoder_weights=config["model"]["encoder_weights"],
        in_channels=config["model"]["in_channels"],
        num_classes=config["model"]["num_classes"],
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if checkpoint_path is not None:
        _load_checkpoint_into_model(model, checkpoint_path, device)
    model.eval()

    rows_data: list[tuple[str, str, str, int, list[float]]] = []
    embedding_dim = config["model"].get("bottleneck_channels")

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].float().to(device)
            mask = batch["mask"].to(device)

            output = model(images)
            bottleneck = getattr(output, "bottleneck", None)
            if bottleneck is None:
                raise RuntimeError("Model output must expose .bottleneck")

            pooled = pool_class_embeddings(
                bottleneck=bottleneck,
                mask=mask,
                sample_ids=batch["sample_id"],
                class_names=config["extraction"]["include_classes"],
                class_values=config["dataset"]["class_values"],
                min_mask_pixels=config["extraction"]["min_mask_pixels"],
            )

            for row in pooled:
                embedding = [float(value) for value in row["embedding"]]
                if embedding_dim is None:
                    embedding_dim = len(embedding)
                if len(embedding) != embedding_dim:
                    raise ValueError("Inconsistent embedding size encountered during pooling.")

                rows_data.append(
                    (
                        row["sample_id"],
                        state_name,
                        row["class_name"],
                        int(row["pixel_count"]),
                        embedding,
                    )
                )

    if embedding_dim is None:
        embedding_dim = len(rows_data[0][4]) if rows_data else 0
    output_dir = ensure_dir(Path(config["paths"]["artifacts_dir"]) / "embeddings")
    output_path = output_dir / output_name
    return _build_dataframe(rows_data, embedding_dim, output_path)


def extract_embeddings(config_path: str | Path, checkpoint_path: Path | None, state_name: str, output_name: str) -> Path:
    config = load_config(config_path)
    return _run_extraction(config, checkpoint_path, state_name, output_name)


def extract_before_after(
    config_path: str | Path,
    pretrained_checkpoint: Path,
    finetuned_checkpoint: Path,
    before_output: str,
    after_output: str,
) -> tuple[Path, Path]:
    config = load_config(config_path)
    before_path = _run_extraction(config, pretrained_checkpoint, "before", before_output)
    after_path = _run_extraction(config, finetuned_checkpoint, "after", after_output)
    return before_path, after_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract ISIC bottleneck embeddings.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", help="Path to checkpoint (optional for before state).", default=None)
    parser.add_argument("--state", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    extract_embeddings(
        args.config,
        Path(args.checkpoint) if args.checkpoint else None,
        args.state,
        args.output,
    )
