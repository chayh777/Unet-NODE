from importlib import util
from pathlib import Path
from types import ModuleType
import hashlib
import sys


def _ensure_package(name: str, package_path: Path) -> None:
    if name in sys.modules:
        return
    pkg = ModuleType(name)
    # Mark as a package so relative imports work for file-loaded modules.
    pkg.__path__ = [str(package_path)]  # type: ignore[attr-defined]
    sys.modules[name] = pkg


def _unique_package_name(base_path: Path) -> str:
    digest = hashlib.sha1(str(base_path).encode("utf-8")).hexdigest()[:10]
    return f"_task5_models_{digest}"


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

# Load dependencies first so `segmentation_model.py` can resolve relative imports.
_load_models_module("adapters.py", f"{_PKG}.adapters")
_load_models_module("node_adapter.py", f"{_PKG}.node_adapter")
_segmentation_model = _load_models_module(
    "segmentation_model.py", f"{_PKG}.segmentation_model"
)

assert hasattr(_segmentation_model, "build_segmentation_model")
build_segmentation_model = _segmentation_model.build_segmentation_model


def test_build_segmentation_model_freezes_encoder():
    model = build_segmentation_model(
        encoder_name="resnet18",
        encoder_weights=None,
        in_channels=3,
        num_classes=1,
        adapter_type="node",
        bottleneck_channels=512,
        adapter_hidden_channels=512,
        freeze_encoder=True,
        node_steps=4,
        node_step_size=0.25,
    )
    encoder_flags = [
        p.requires_grad for name, p in model.named_parameters() if "encoder" in name
    ]
    assert encoder_flags
    assert all(flag is False for flag in encoder_flags)
