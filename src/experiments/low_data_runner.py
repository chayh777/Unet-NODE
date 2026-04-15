from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import yaml


def _is_known_adamw_dependency_failure(exc: BaseException) -> bool:
    """
    Detect a very specific environment failure we want to work around.

    In some environments, constructing `torch.optim.AdamW(...)` triggers an import
    chain (torch._dynamo -> torch.onnx -> transformers) where transformers expects
    `torch.utils._pytree.register_pytree_node`, which is missing in torch==2.1.0.

    We only fall back when we see this exact signature to avoid masking genuine
    optimizer/configuration bugs.
    """
    if not isinstance(exc, AttributeError):
        return False
    msg = str(exc)
    return "register_pytree_node" in msg and "torch.utils._pytree" in msg


def _is_known_windows_dataloader_worker_permission_error(exc: BaseException) -> bool:
    """
    Detect a narrow Windows DataLoader worker-start PermissionError we can safely retry.

    Empirically, when DataLoader workers fail to spawn under Windows, training can raise:
      PermissionError: [WinError 5] Access is denied
    (including localized variants like Chinese "拒绝访问。").

    We intentionally keep this check strict to avoid masking arbitrary training failures.
    """
    if not isinstance(exc, PermissionError):
        return False

    # On Windows, `OSError`-subclasses may have `winerror`; however this isn't guaranteed
    # if the exception is constructed/raised elsewhere, so we also inspect the message.
    winerror = getattr(exc, "winerror", None)
    msg = str(exc)
    is_winerror_5 = winerror == 5 or ("WinError 5" in msg)
    if not is_winerror_5:
        return False

    # Key narrowing: only treat it as a DataLoader worker-start issue if the traceback
    # points into torch's DataLoader/multiprocessing stack.
    tb = getattr(exc, "__traceback__", None)
    while tb is not None:
        filename = tb.tb_frame.f_code.co_filename
        norm = filename.replace("\\", "/").lower()
        if (
            "torch/utils/data/dataloader" in norm
            or "torch/utils/data/_utils" in norm
            or "/multiprocessing/" in norm
        ):
            return True
        tb = tb.tb_next

    # Last resort: only accept explicit DataLoader worker messaging.
    msg_norm = msg.lower()
    return ("dataloader" in msg_norm) and ("worker" in msg_norm or "multiprocessing" in msg_norm)


class _LocalAdamW:
    """
    Minimal AdamW-style optimizer used as a narrow fallback.

    This exists so low-data experiments can run in environments where importing
    torch's AdamW triggers a known dependency mismatch.
    """

    def __init__(
        self,
        params,
        *,
        lr: float,
        weight_decay: float,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ) -> None:
        self._params = [p for p in list(params) if getattr(p, "requires_grad", False)]
        self._lr = float(lr)
        self._weight_decay = float(weight_decay)
        self._beta1 = float(betas[0])
        self._beta2 = float(betas[1])
        self._eps = float(eps)
        self._state: dict[object, dict[str, object]] = {}

    def zero_grad(self) -> None:
        for p in self._params:
            if getattr(p, "grad", None) is not None:
                setattr(p, "grad", None)

    def step(self) -> None:
        import torch

        with torch.no_grad():
            for p in self._params:
                grad = getattr(p, "grad", None)
                if grad is None:
                    continue
                if not torch.is_tensor(grad):
                    continue

                state = self._state.get(p)
                if state is None:
                    state = {
                        "step": 0,
                        "exp_avg": torch.zeros_like(p),
                        "exp_avg_sq": torch.zeros_like(p),
                    }
                    self._state[p] = state

                # Decoupled weight decay (AdamW): apply directly to the weights.
                if self._weight_decay != 0.0:
                    p.mul_(1.0 - self._lr * self._weight_decay)

                step_num = int(state["step"]) + 1
                state["step"] = step_num

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                assert torch.is_tensor(exp_avg)
                assert torch.is_tensor(exp_avg_sq)

                exp_avg.mul_(self._beta1).add_(grad, alpha=1.0 - self._beta1)
                exp_avg_sq.mul_(self._beta2).addcmul_(
                    grad, grad, value=1.0 - self._beta2
                )

                bias_correction1 = 1.0 - math.pow(self._beta1, step_num)
                bias_correction2 = 1.0 - math.pow(self._beta2, step_num)
                step_size = self._lr / bias_correction1

                denom = (exp_avg_sq / bias_correction2).sqrt().add_(self._eps)
                p.addcdiv_(exp_avg, denom, value=-step_size)


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

    def _build_loaders(*, num_workers_override: int | None) -> tuple[object, object]:
        kwargs = dict(loader_kwargs)
        if num_workers_override is not None:
            kwargs["num_workers"] = int(num_workers_override)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, **kwargs
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, **kwargs
        )
        return train_loader, val_loader

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

    trainable_params = [
        param for param in model.parameters() if getattr(param, "requires_grad", False)
    ]
    lr = float(config["train"]["learning_rate"])
    weight_decay = float(config["train"]["weight_decay"])
    try:
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=lr,
            weight_decay=weight_decay,
        )
    except AttributeError as exc:
        if not _is_known_adamw_dependency_failure(exc):
            raise
        optimizer = _LocalAdamW(
            trainable_params,
            lr=lr,
            weight_decay=weight_decay,
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    output_dir = Path(config["paths"]["artifacts_dir"]) / f"group_{group.lower()}"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = _build_loaders(num_workers_override=None)
    try:
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
    except PermissionError as exc:
        configured_workers = int(loader_kwargs.get("num_workers", 0))
        if (
            configured_workers > 0
            and _is_known_windows_dataloader_worker_permission_error(exc)
        ):
            # Retry exactly once with single-process loading to work around a known
            # Windows worker spawn PermissionError.
            train_loader, val_loader = _build_loaders(num_workers_override=0)
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
        raise
