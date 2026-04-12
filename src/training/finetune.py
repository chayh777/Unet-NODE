from pathlib import Path
import logging
import os
import time

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.isic2018 import ISIC2018Dataset
from src.models.unet import SimpleUNet
from src.utils.io import (
    ensure_config_dict,
    ensure_dir,
    load_checkpoint,
    load_config,
    save_checkpoint,
    save_json,
)


def _require_keys(mapping: dict, keys: list[str], context: str) -> None:
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise ValueError(f"{context} missing keys {missing}")


def _sanitize_masks(mask_tensor: torch.Tensor, num_classes: int, device: torch.device) -> torch.Tensor:
    if mask_tensor.dim() == 4 and mask_tensor.shape[1] == 1:
        mask_tensor = mask_tensor.squeeze(1)
    if mask_tensor.dim() != 3:
        raise ValueError("mask tensor must be [B,H,W] for CrossEntropyLoss")
    if mask_tensor.dtype != torch.long:
        mask_tensor = mask_tensor.long()
    mask_tensor = mask_tensor.to(device)
    if mask_tensor.min() < 0 or mask_tensor.max() >= num_classes:
        raise ValueError("mask values must lie in [0, num_classes-1]")
    return mask_tensor


def run_finetuning(config_path: str | Path) -> Path:
    config = ensure_config_dict(load_config(config_path))
    _require_keys(config, ["paths", "training", "model", "dataset"], "config")
    _require_keys(config["paths"], ["isic_images_dir", "isic_masks_dir", "artifacts_dir"], "paths")
    _require_keys(
        config["training"],
        ["batch_size", "epochs", "learning_rate", "num_workers", "finetuned_checkpoint_name"],
        "training",
    )
    _require_keys(
        config["model"],
        ["encoder_name", "encoder_weights", "in_channels", "num_classes"],
        "model",
    )
    _require_keys(
        config["dataset"],
        ["image_size", "class_values"],
        "dataset",
    )
    dataset = ISIC2018Dataset(
        images_dir=config["paths"]["isic_images_dir"],
        masks_dir=config["paths"]["isic_masks_dir"],
        image_size=config["dataset"]["image_size"],
        class_values=config["dataset"]["class_values"],
    )

    num_workers = config["training"]["num_workers"]
    if os.name == "nt" and num_workers and num_workers > 0:
        logging.warning("num_workers > 0 on Windows may hang; consider using 0.")
    loader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=num_workers,
    )

    model = SimpleUNet(
        encoder_name=config["model"]["encoder_name"],
        encoder_weights=config["model"]["encoder_weights"],
        in_channels=config["model"]["in_channels"],
        num_classes=config["model"]["num_classes"],
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    checkpoints_dir = Path(config["paths"]["artifacts_dir"]) / "checkpoints"
    ensure_dir(checkpoints_dir)

    pretrained_name = config["training"].get("pretrained_checkpoint_name")
    if pretrained_name:
        pretrained_path = checkpoints_dir / pretrained_name
        if not pretrained_path.exists():
            raise FileNotFoundError(f"Pretrained checkpoint '{pretrained_name}' not found.")
        try:
            state = load_checkpoint(pretrained_path, "cpu")
            model.load_state_dict(state)
        except Exception as exc:
            raise RuntimeError(f"Failed to load pretrained checkpoint '{pretrained_name}'") from exc

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])

    epoch_losses = []
    model.train()
    start = time.time()
    for epoch in range(config["training"]["epochs"]):
        running_loss = 0.0
        for batch in loader:
            images = batch["image"].float().to(device)
            masks = _sanitize_masks(batch["mask"], config["model"]["num_classes"], device)

            optimizer.zero_grad()
            output = model(images)
            if hasattr(output, "logits"):
                logits = output.logits
            elif isinstance(output, torch.Tensor):
                logits = output
            else:
                raise RuntimeError("Model output must be a tensor or provide .logits.")
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())

        epoch_losses.append(running_loss / max(1, len(loader)))
    duration = time.time() - start

    checkpoint_path = checkpoints_dir / config["training"]["finetuned_checkpoint_name"]
    save_checkpoint(checkpoint_path, model)
    save_json(
        checkpoints_dir / "finetune_log.json",
        {"epoch_losses": epoch_losses, "duration_sec": duration},
    )

    return checkpoint_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Finetune the U-Net on ISIC 2018.")
    parser.add_argument("--config", required=True, help="Path to the experiment config.")
    args = parser.parse_args()
    run_finetuning(args.config)
