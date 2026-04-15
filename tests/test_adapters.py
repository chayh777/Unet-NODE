import torch

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
    return f"_task4_models_{digest}"


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

_adapters = _load_models_module("adapters.py", f"{_PKG}.adapters")
_node_adapter = _load_models_module("node_adapter.py", f"{_PKG}.node_adapter")

assert hasattr(_adapters, "IdentityAdapter")
assert hasattr(_adapters, "ConvBottleneckAdapter")
assert hasattr(_node_adapter, "NODEAdapter")

IdentityAdapter = _adapters.IdentityAdapter
ConvBottleneckAdapter = _adapters.ConvBottleneckAdapter
NODEAdapter = _node_adapter.NODEAdapter


def test_identity_adapter_preserves_shape():
    x = torch.randn(2, 32, 8, 8)
    y = IdentityAdapter()(x)
    assert tuple(y.shape) == (2, 32, 8, 8)


def test_conv_adapter_preserves_shape():
    x = torch.randn(2, 32, 8, 8)
    y = ConvBottleneckAdapter(channels=32, hidden_channels=32)(x)
    assert tuple(y.shape) == (2, 32, 8, 8)


def test_node_adapter_preserves_shape():
    x = torch.randn(2, 32, 8, 8)
    y = NODEAdapter(channels=32, hidden_channels=32, steps=4, step_size=0.25)(x)
    assert tuple(y.shape) == (2, 32, 8, 8)


def test_node_adapter_uses_stateless_batchnorm():
    import torch.nn as nn

    model = NODEAdapter(channels=32, hidden_channels=32, steps=2, step_size=0.25)
    bns = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
    assert bns, "Expected NODEAdapter to contain BatchNorm2d layers"
    assert all(bn.track_running_stats is False for bn in bns)


def test_node_adapter_backward_smoke():
    x = torch.randn(2, 32, 8, 8, requires_grad=True)
    model = NODEAdapter(channels=32, hidden_channels=32, steps=2, step_size=0.25)
    y = model(x)
    loss = y.mean()
    loss.backward()

    assert x.grad is not None
    assert any(p.grad is not None for p in model.parameters())
