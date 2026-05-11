from __future__ import annotations

from importlib import util
from pathlib import Path
from types import ModuleType
import sys

import numpy as np
import pytest
import torch


def _ensure_package(name: str, package_path: Path) -> None:
    if name in sys.modules:
        return
    pkg = ModuleType(name)
    pkg.__path__ = [str(package_path)]
    sys.modules[name] = pkg


def _load_robustness_metrics_module():
    repo_root = Path(__file__).resolve().parents[2]
    src_dir = repo_root / "src"
    analysis_dir = src_dir / "analysis"

    _ensure_package("src", src_dir)
    _ensure_package("src.analysis", analysis_dir)

    module_path = analysis_dir / "robustness_metrics.py"
    assert module_path.exists(), f"Missing module file: {module_path}"
    spec = util.spec_from_file_location("src.analysis.robustness_metrics", module_path)
    assert spec is not None, f"Failed to create spec for {module_path}"
    module = util.module_from_spec(spec)
    assert spec.loader is not None, f"Missing spec loader for {module_path}"
    spec.loader.exec_module(module)
    return module


module = _load_robustness_metrics_module()

compute_sample_dice = module.compute_sample_dice
compute_sample_iou = module.compute_sample_iou
add_gaussian_noise = module.add_gaussian_noise


def test_compute_sample_dice_perfect():
    pred = np.array([[1, 1], [1, 1]])
    gt = np.array([[1, 1], [1, 1]])
    dice = compute_sample_dice(pred, gt)
    assert dice == pytest.approx(1.0)


def test_compute_sample_iou_no_overlap():
    pred = np.array([[1, 1], [0, 0]])
    gt = np.array([[0, 0], [1, 1]])
    iou = compute_sample_iou(pred, gt)
    assert iou == pytest.approx(0.0)


def test_add_gaussian_noise_shape():
    images = torch.randn(2, 3, 256, 256)
    noisy = add_gaussian_noise(images, sigma=0.1)
    assert noisy.shape == images.shape
    assert not torch.allclose(noisy, images)
