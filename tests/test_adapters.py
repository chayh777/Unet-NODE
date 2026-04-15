import torch

from importlib import util
from pathlib import Path


def _load_module(module_filename: str, module_name: str):
    module_path = Path(__file__).resolve().parents[1] / "src" / "models" / module_filename
    assert module_path.exists(), f"Missing module file: {module_path}"
    spec = util.spec_from_file_location(module_name, module_path)
    assert spec is not None, f"Failed to create spec for {module_name}"
    module = util.module_from_spec(spec)
    assert spec.loader is not None, f"Missing spec loader for {module_name}"
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


_adapters = _load_module("adapters.py", "adapters")
_node_adapter = _load_module("node_adapter.py", "node_adapter")

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
