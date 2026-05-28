from pathlib import Path
import json
from typing import Any

import torch
import yaml


def raise_missing_checkpoint_error(path: Path | str) -> None:
    path = Path(path)
    raise FileNotFoundError(
        "Checkpoint not found at "
        f"{path}. This analysis requires a saved checkpoint; the run may have used "
        "train.save_best_checkpoint: false."
    )


def load_config(path: Path | str) -> dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def ensure_config_dict(config: dict) -> dict:
    if not isinstance(config, dict) or not config:
        raise ValueError("Config must be a non-empty mapping.")
    return config


def ensure_dir(path: Path | str) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Path | str, payload: dict) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def save_checkpoint(path: Path | str, model: torch.nn.Module) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    torch.save(model.state_dict(), path)
    return path


def load_checkpoint(path: Path | str, device: torch.device | str = "cpu") -> dict:
    path = Path(path)
    if not path.exists():
        raise_missing_checkpoint_error(path)
    return torch.load(path, map_location=device)


def normalize_checkpoint_state_dict(raw_state: Any) -> dict[str, Any]:
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


def load_model_state_dict(
    path: Path | str,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    return normalize_checkpoint_state_dict(load_checkpoint(path, device))
