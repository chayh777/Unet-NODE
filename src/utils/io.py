from pathlib import Path
import json

import torch
import yaml


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
        raise FileNotFoundError(f"Checkpoint not found at {path}")
    return torch.load(path, map_location=device)
