from importlib import util
from pathlib import Path
from types import ModuleType
import sys

import torch


def _ensure_package(name: str, package_path: Path) -> None:
    if name in sys.modules:
        return
    pkg = ModuleType(name)
    # Mark as a package so file-loaded modules under it can be imported.
    pkg.__path__ = [str(package_path)]  # type: ignore[attr-defined]
    sys.modules[name] = pkg


def _load_module(module_path: Path, fqname: str):
    assert module_path.exists(), f"Missing module file: {module_path}"
    spec = util.spec_from_file_location(fqname, module_path)
    assert spec is not None, f"Failed to create spec for {fqname}"
    module = util.module_from_spec(spec)
    assert spec.loader is not None, f"Missing spec loader for {fqname}"
    sys.modules[fqname] = module
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


_TRAINING_DIR = Path(__file__).resolve().parents[1] / "src" / "training"
_ensure_package("src", Path(__file__).resolve().parents[1] / "src")
_ensure_package("src.training", _TRAINING_DIR)
_load_module(_TRAINING_DIR / "losses.py", "src.training.losses")
_load_module(_TRAINING_DIR / "metrics.py", "src.training.metrics")


from src.training.losses import DiceBCELoss  # noqa: E402
from src.training.metrics import compute_binary_dice, compute_binary_iou  # noqa: E402


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
