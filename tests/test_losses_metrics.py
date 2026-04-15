from importlib import util
import hashlib
from pathlib import Path
from types import ModuleType
import sys

import pytest
import torch


def _ensure_package(name: str, package_path: Path) -> None:
    if name in sys.modules:
        return
    pkg = ModuleType(name)
    # Mark as a package so relative imports work for file-loaded modules.
    pkg.__path__ = [str(package_path)]  # type: ignore[attr-defined]
    sys.modules[name] = pkg


def _unique_package_name(base_path: Path) -> str:
    digest = hashlib.sha1(str(base_path).encode("utf-8")).hexdigest()[:10]
    return f"_task6_training_{digest}"


def _load_training_module(module_filename: str, fqname: str):
    module_path = Path(__file__).resolve().parents[1] / "src" / "training" / module_filename
    assert module_path.exists(), f"Missing module file: {module_path}"
    spec = util.spec_from_file_location(fqname, module_path)
    assert spec is not None, f"Failed to create spec for {fqname}"
    module = util.module_from_spec(spec)
    assert spec.loader is not None, f"Missing spec loader for {fqname}"
    sys.modules[fqname] = module
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


_TRAINING_DIR = Path(__file__).resolve().parents[1] / "src" / "training"
_PKG = _unique_package_name(_TRAINING_DIR)
_ensure_package(_PKG, _TRAINING_DIR)

_losses = _load_training_module("losses.py", f"{_PKG}.losses")
_metrics = _load_training_module("metrics.py", f"{_PKG}.metrics")

DiceBCELoss = _losses.DiceBCELoss
compute_binary_dice = _metrics.compute_binary_dice
compute_binary_iou = _metrics.compute_binary_iou


def test_dice_bce_loss_returns_scalar():
    logits = torch.zeros(2, 1, 4, 4)
    targets = torch.zeros(2, 1, 4, 4)
    loss = DiceBCELoss()(logits, targets)
    assert loss.ndim == 0


def test_binary_metrics_reach_one_for_perfect_prediction():
    logits = torch.full((1, 1, 2, 2), 10.0)
    targets = torch.ones(1, 1, 2, 2)
    assert float(compute_binary_dice(logits, targets)) == 1.0
    assert float(compute_binary_iou(logits, targets)) == 1.0


def test_logits_zero_is_not_positive_prediction_by_default_threshold():
    logits = torch.zeros(1, 1, 2, 2)
    targets = torch.ones(1, 1, 2, 2)
    # If sigmoid(0)=0.5 is treated as positive, this would incorrectly reach 1.0.
    assert float(compute_binary_dice(logits, targets)) == pytest.approx(0.2)
    assert float(compute_binary_iou(logits, targets)) == pytest.approx(0.2)


def test_binary_metrics_handle_255_encoded_targets():
    logits = torch.full((1, 1, 2, 2), 10.0)
    targets = torch.full((1, 1, 2, 2), 255.0)
    assert float(compute_binary_dice(logits, targets)) == 1.0
    assert float(compute_binary_iou(logits, targets)) == 1.0


def test_dice_bce_loss_returns_scalar_with_255_targets():
    logits = torch.zeros(2, 1, 4, 4)
    targets = torch.full((2, 1, 4, 4), 255.0)
    loss = DiceBCELoss()(logits, targets)
    assert loss.ndim == 0


def test_binary_metrics_expose_threshold_parameter():
    logits = torch.zeros(1, 1, 2, 2)
    targets = torch.ones(1, 1, 2, 2)
    assert float(compute_binary_dice(logits, targets, threshold=0.49)) == 1.0
    assert float(compute_binary_iou(logits, targets, threshold=0.49)) == 1.0


def test_dice_bce_loss_normalizes_255_targets_equivalent_to_ones():
    logits = torch.zeros(1, 1, 2, 2)
    loss_ones = float(DiceBCELoss()(logits, torch.ones_like(logits)))
    loss_255 = float(DiceBCELoss()(logits, torch.full_like(logits, 255.0)))
    assert loss_255 == pytest.approx(loss_ones)
