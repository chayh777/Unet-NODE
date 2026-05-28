from importlib import util
from pathlib import Path
from types import ModuleType
import hashlib
import sys

import torch


def _ensure_package(name: str, package_path: Path) -> None:
    if name in sys.modules:
        return
    pkg = ModuleType(name)
    pkg.__path__ = [str(package_path)]  # type: ignore[attr-defined]
    sys.modules[name] = pkg


def _unique_package_name(base_path: Path) -> str:
    digest = hashlib.sha1(str(base_path).encode("utf-8")).hexdigest()[:10]
    return f"_task_std_unet_{digest}"


def _load_models_module(module_filename: str, fqname: str):
    module_path = Path(__file__).resolve().parents[1] / "src" / "models" / module_filename
    assert module_path.exists(), f"Missing module file: {module_path}"
    spec = util.spec_from_file_location(fqname, module_path)
    assert spec is not None, f"Failed to create spec for {fqname}"
    module = util.module_from_spec(spec)
    assert spec.loader is not None, f"Missing spec loader for {fqname}"
    sys.modules[fqname] = module
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


_MODELS_DIR = Path(__file__).resolve().parents[1] / "src" / "models"
_PKG = _unique_package_name(_MODELS_DIR)
_ensure_package(_PKG, _MODELS_DIR)
_decoder_module = _load_models_module("unet_decoder.py", f"{_PKG}.unet_decoder")
StandardUNetDecoder = _decoder_module.StandardUNetDecoder


def test_standard_unet_decoder_uses_full_feature_pyramid_and_restores_stride_2_resolution():
    decoder = StandardUNetDecoder(
        encoder_channels=[64, 128, 256, 512],
        bottleneck_channels=128,
        output_channels=32,
    )
    features = [
        torch.randn(2, 64, 32, 32),
        torch.randn(2, 128, 16, 16),
        torch.randn(2, 256, 8, 8),
    ]
    bottleneck = torch.randn(2, 128, 4, 4)

    decoded = decoder(bottleneck, skip_features=features[::-1])

    assert decoded.shape == (2, 32, 32, 32)
