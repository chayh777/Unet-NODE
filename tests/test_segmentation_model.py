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
    return f"_task2_models_{digest}"


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
_load_models_module("node_adapter.py", f"{_PKG}.node_adapter")
_segmentation_model = _load_models_module(
    "segmentation_model.py", f"{_PKG}.segmentation_model"
)

IdentityAdapter = _adapters.IdentityAdapter
build_segmentation_model = _segmentation_model.build_segmentation_model


def test_output_side_conv_adapter_preserves_shapes_and_output_contract():
    model = build_segmentation_model(
        encoder_name="resnet18",
        encoder_weights=None,
        in_channels=3,
        num_classes=2,
        adapter_type="conv",
        adapter_placement="output",
        bottleneck_channels=32,
        adapter_hidden_channels=16,
        freeze_encoder=True,
        node_steps=4,
        node_step_size=0.25,
    )
    x = torch.randn(2, 3, 64, 64)

    head_inputs: list[torch.Tensor] = []

    def _capture_head_input(_module, inputs):
        head_inputs.append(inputs[0].detach().clone())

    handle = model.head.register_forward_pre_hook(_capture_head_input)
    try:
        output = model(x)
    finally:
        handle.remove()

    assert isinstance(model.adapter, IdentityAdapter)
    assert output.logits.shape == (2, 2, 64, 64)
    assert output.bottleneck.shape == (2, 32, 4, 4)
    assert output.adapted_bottleneck.shape == output.bottleneck.shape
    assert torch.allclose(output.adapted_bottleneck, output.bottleneck)
    assert output.output_adapter_activation is not None
    assert output.output_adapter_activation.shape == (2, 8, 4, 4)
    assert hasattr(model, "output_adapter")
    assert len(head_inputs) == 1
    assert head_inputs[0].shape == output.output_adapter_activation.shape
    assert torch.allclose(head_inputs[0], output.output_adapter_activation)


def test_output_side_node_adapter_runs_forward_and_preserves_output_contract():
    model = build_segmentation_model(
        encoder_name="resnet18",
        encoder_weights=None,
        in_channels=3,
        num_classes=1,
        adapter_type="node",
        adapter_placement="output",
        bottleneck_channels=32,
        adapter_hidden_channels=16,
        freeze_encoder=True,
        node_steps=2,
        node_step_size=0.25,
    )
    x = torch.randn(2, 3, 64, 64)

    output = model(x)

    assert isinstance(model.adapter, IdentityAdapter)
    assert output.logits.shape == (2, 1, 64, 64)
    assert output.bottleneck.shape == (2, 32, 4, 4)
    assert output.adapted_bottleneck.shape == output.bottleneck.shape
    assert torch.allclose(output.adapted_bottleneck, output.bottleneck)
    assert output.output_adapter_activation is not None
    assert output.output_adapter_activation.shape == (2, 8, 4, 4)
    assert hasattr(model, "output_adapter")


def test_bottleneck_side_adapter_keeps_backward_compatible_output_fields():
    model = build_segmentation_model(
        encoder_name="resnet18",
        encoder_weights=None,
        in_channels=3,
        num_classes=2,
        adapter_type="conv",
        adapter_placement="bottleneck",
        bottleneck_channels=32,
        adapter_hidden_channels=16,
        freeze_encoder=True,
        node_steps=4,
        node_step_size=0.25,
    )
    x = torch.randn(2, 3, 64, 64)

    output = model(x)

    assert output.logits.shape == (2, 2, 64, 64)
    assert output.bottleneck.shape == (2, 32, 4, 4)
    assert output.adapted_bottleneck.shape == output.bottleneck.shape
    assert output.output_adapter_activation is None
    assert output.node_diagnostics is None


def test_segmentation_model_exposes_node_diagnostics_for_bottleneck_node():
    model = build_segmentation_model(
        encoder_name="resnet18",
        encoder_weights=None,
        in_channels=3,
        num_classes=1,
        adapter_type="node",
        bottleneck_channels=32,
        adapter_hidden_channels=16,
        freeze_encoder=True,
        node_steps=4,
        node_step_size=0.25,
        adapter_init="zero_last_layer",
    )
    x = torch.randn(2, 3, 64, 64)

    output = model(x)

    assert output.node_diagnostics is not None
    assert "kinetic_terms" in output.node_diagnostics
    assert len(output.node_diagnostics["kinetic_terms"]) == 4


def test_segmentation_model_non_node_paths_do_not_emit_node_diagnostics():
    model = build_segmentation_model(
        encoder_name="resnet18",
        encoder_weights=None,
        in_channels=3,
        num_classes=1,
        adapter_type="conv",
        bottleneck_channels=32,
        adapter_hidden_channels=16,
        freeze_encoder=True,
        node_steps=4,
        node_step_size=0.25,
    )
    x = torch.randn(2, 3, 64, 64)

    output = model(x)

    assert output.node_diagnostics is None
