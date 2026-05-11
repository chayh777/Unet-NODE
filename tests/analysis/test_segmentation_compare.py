from __future__ import annotations

from importlib import util
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest


def _ensure_package(name: str, package_path: Path) -> None:
    if name in sys.modules:
        return
    pkg = ModuleType(name)
    pkg.__path__ = [str(package_path)]  # type: ignore[attr-defined]
    sys.modules[name] = pkg


def _load_segmentation_compare_module():
    repo_root = Path(__file__).resolve().parents[2]
    src_dir = repo_root / "src"
    analysis_dir = src_dir / "analysis"

    _ensure_package("src", src_dir)
    _ensure_package("src.analysis", analysis_dir)

    module_path = analysis_dir / "segmentation_compare.py"
    assert module_path.exists(), f"Missing module file: {module_path}"
    spec = util.spec_from_file_location("src.analysis.segmentation_compare", module_path)
    assert spec is not None, f"Failed to create spec for {module_path}"
    module = util.module_from_spec(spec)
    assert spec.loader is not None, f"Missing spec loader for {module_path}"
    spec.loader.exec_module(module)
    return module


import sys
module = _load_segmentation_compare_module()

compute_sample_dice = module.compute_sample_dice
_colorize_mask = module._colorize_mask


def test_compute_sample_dice_perfect():
    pred = np.array([[1, 1], [1, 1]])
    gt = np.array([[1, 1], [1, 1]])
    dice = compute_sample_dice(pred, gt)
    assert dice == pytest.approx(1.0)


def test_compute_sample_dice_partial():
    pred = np.array([[1, 1], [0, 0]])
    gt = np.array([[1, 1], [1, 1]])
    dice = compute_sample_dice(pred, gt)
    assert dice == pytest.approx(0.6666666666666666)


def test_colorize_mask_shape():
    mask = np.array([[0, 1], [1, 0]])
    colored = _colorize_mask(mask)
    assert colored.shape == (2, 2, 3)
    assert colored.dtype == np.float32