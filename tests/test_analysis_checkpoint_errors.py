from __future__ import annotations

from importlib import util
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace
import sys

import pytest
import torch


def _ensure_package(name: str, package_path: Path) -> None:
    if name in sys.modules:
        return
    pkg = ModuleType(name)
    pkg.__path__ = [str(package_path)]  # type: ignore[attr-defined]
    sys.modules[name] = pkg


def _load_robustness_metrics_module():
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    analysis_dir = src_dir / "analysis"
    data_dir = src_dir / "data"
    models_dir = src_dir / "models"
    utils_dir = src_dir / "utils"

    _ensure_package("src", src_dir)
    _ensure_package("src.analysis", analysis_dir)
    _ensure_package("src.data", data_dir)
    _ensure_package("src.models", models_dir)
    _ensure_package("src.utils", utils_dir)

    module_path = analysis_dir / "robustness_metrics.py"
    assert module_path.exists(), f"Missing module file: {module_path}"
    spec = util.spec_from_file_location("src.analysis.robustness_metrics", module_path)
    assert spec is not None, f"Failed to create spec for {module_path}"
    module = util.module_from_spec(spec)
    assert spec.loader is not None, f"Missing spec loader for {module_path}"
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def _load_segmentation_compare_module():
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    analysis_dir = src_dir / "analysis"
    models_dir = src_dir / "models"
    utils_dir = src_dir / "utils"

    _ensure_package("src", src_dir)
    _ensure_package("src.analysis", analysis_dir)
    _ensure_package("src.models", models_dir)
    _ensure_package("src.utils", utils_dir)

    module_path = analysis_dir / "segmentation_compare.py"
    assert module_path.exists(), f"Missing module file: {module_path}"
    spec = util.spec_from_file_location("src.analysis.segmentation_compare", module_path)
    assert spec is not None, f"Failed to create spec for {module_path}"
    module = util.module_from_spec(spec)
    assert spec.loader is not None, f"Missing spec loader for {module_path}"
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def test_run_robustness_experiment_raises_clear_missing_checkpoint_error(tmp_path, monkeypatch):
    module = _load_robustness_metrics_module()

    class _FakeDataset:
        def __init__(self, *args, **kwargs):
            self._items = [
                {
                    "sample_id": "sample_1",
                    "mask": type("_Mask", (), {"numpy": lambda self: __import__("numpy").zeros((1, 1), dtype="uint8")})(),
                }
            ]

        def __len__(self):
            return len(self._items)

        def __getitem__(self, index):
            return self._items[index]

    monkeypatch.setitem(sys.modules, "src.data.isic2018", ModuleType("src.data.isic2018"))
    sys.modules["src.data.isic2018"].ISIC2018Dataset = _FakeDataset
    monkeypatch.setattr(module, "DataLoader", lambda dataset, batch_size, shuffle: object())

    config = {
        "paths": {
            "val_images_dir": "data/val/images",
            "val_masks_dir": "data/val/masks",
        },
        "data": {
            "image_size": 64,
        },
        "model": {
            "encoder_name": "resnet18",
            "in_channels": 3,
            "num_classes": 1,
            "bottleneck_channels": 16,
        },
        "adapter": {
            "hidden_channels": 8,
        },
        "node": {
            "steps": 4,
            "step_size": 0.25,
        },
    }

    with pytest.raises(
        FileNotFoundError,
        match=r"Checkpoint not found at .*group_a.*best\.pt.*train\.save_best_checkpoint: false",
    ):
        module.run_robustness_experiment(
            config=config,
            artifacts_dir=tmp_path,
            groups=["A"],
            noise_levels=[0.0],
        )


def test_run_robustness_experiment_supports_wrapped_module_prefixed_checkpoint(
    tmp_path, monkeypatch
):
    module = _load_robustness_metrics_module()
    model_calls = {}
    checkpoint_path = tmp_path / "group_a" / "best.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_state_dict": {"module.weight": torch.tensor([1.0])}},
        checkpoint_path,
    )

    class _FakeDataset:
        def __init__(self, *args, **kwargs):
            self._items = [
                {
                    "sample_id": "sample_1",
                    "mask": SimpleNamespace(numpy=lambda: __import__("numpy").zeros((1, 1), dtype="uint8")),
                }
            ]

        def __len__(self):
            return len(self._items)

        def __getitem__(self, index):
            return self._items[index]

    class _FakeModel:
        def load_state_dict(self, state_dict):
            model_calls["loaded_state"] = dict(state_dict)

        def eval(self):
            model_calls["eval_called"] = True
            return self

        def to(self, device):
            model_calls["device"] = device
            return self

    monkeypatch.setitem(sys.modules, "src.data.isic2018", ModuleType("src.data.isic2018"))
    sys.modules["src.data.isic2018"].ISIC2018Dataset = _FakeDataset
    monkeypatch.setitem(
        sys.modules,
        "src.models.segmentation_model",
        ModuleType("src.models.segmentation_model"),
    )
    sys.modules["src.models.segmentation_model"].build_segmentation_model = lambda **kwargs: _FakeModel()
    monkeypatch.setitem(
        sys.modules,
        "src.experiments.low_data_runner",
        ModuleType("src.experiments.low_data_runner"),
    )
    sys.modules["src.experiments.low_data_runner"].resolve_group_adapter = lambda group: "conv"
    monkeypatch.setattr(module, "DataLoader", lambda dataset, batch_size, shuffle: object())
    monkeypatch.setattr(module, "run_noisy_inference", lambda *args, **kwargs: [])
    monkeypatch.setattr(module, "save_robustness_metrics", lambda all_metrics, output_path: output_path)
    monkeypatch.setattr(
        module,
        "pd",
        SimpleNamespace(read_csv=lambda _: SimpleNamespace()),
    )
    monkeypatch.setattr(module, "plot_decay_curve", lambda *args, **kwargs: None)
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: False)

    config = {
        "paths": {
            "val_images_dir": "data/val/images",
            "val_masks_dir": "data/val/masks",
        },
        "data": {
            "image_size": 64,
        },
        "model": {
            "encoder_name": "resnet18",
            "in_channels": 3,
            "num_classes": 1,
            "bottleneck_channels": 16,
        },
        "adapter": {
            "hidden_channels": 8,
        },
        "node": {
            "steps": 4,
            "step_size": 0.25,
            "solver": "euler",
        },
    }

    output_dir = module.run_robustness_experiment(
        config=config,
        artifacts_dir=tmp_path,
        groups=["A"],
        noise_levels=[0.0],
    )

    assert output_dir == tmp_path / "robustness"
    assert model_calls["loaded_state"] == {"weight": torch.tensor([1.0])}


def test_run_robustness_experiment_forwards_adapter_placement(tmp_path, monkeypatch):
    module = _load_robustness_metrics_module()
    model_calls = {}
    checkpoint_path = tmp_path / "group_a" / "best.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": {"weight": torch.tensor([1.0])}}, checkpoint_path)

    class _FakeDataset:
        def __init__(self, *args, **kwargs):
            self._items = [
                {
                    "sample_id": "sample_1",
                    "mask": SimpleNamespace(numpy=lambda: __import__("numpy").zeros((1, 1), dtype="uint8")),
                }
            ]

        def __len__(self):
            return len(self._items)

        def __getitem__(self, index):
            return self._items[index]

    class _FakeModel:
        def load_state_dict(self, state_dict):
            model_calls["loaded_state"] = dict(state_dict)

        def eval(self):
            return self

        def to(self, device):
            return self

    def _fake_build_segmentation_model(**kwargs):
        model_calls["build_kwargs"] = dict(kwargs)
        return _FakeModel()

    monkeypatch.setitem(sys.modules, "src.data.isic2018", ModuleType("src.data.isic2018"))
    sys.modules["src.data.isic2018"].ISIC2018Dataset = _FakeDataset
    monkeypatch.setitem(
        sys.modules,
        "src.models.segmentation_model",
        ModuleType("src.models.segmentation_model"),
    )
    sys.modules["src.models.segmentation_model"].build_segmentation_model = _fake_build_segmentation_model
    monkeypatch.setitem(
        sys.modules,
        "src.experiments.low_data_runner",
        ModuleType("src.experiments.low_data_runner"),
    )
    sys.modules["src.experiments.low_data_runner"].resolve_group_adapter = lambda group: "conv"
    monkeypatch.setattr(module, "DataLoader", lambda dataset, batch_size, shuffle: object())
    monkeypatch.setattr(module, "run_noisy_inference", lambda *args, **kwargs: [])
    monkeypatch.setattr(module, "save_robustness_metrics", lambda all_metrics, output_path: output_path)
    monkeypatch.setattr(
        module,
        "pd",
        SimpleNamespace(read_csv=lambda _: SimpleNamespace()),
    )
    monkeypatch.setattr(module, "plot_decay_curve", lambda *args, **kwargs: None)
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: False)

    config = {
        "paths": {
            "val_images_dir": "data/val/images",
            "val_masks_dir": "data/val/masks",
        },
        "data": {
            "image_size": 64,
        },
        "model": {
            "encoder_name": "resnet18",
            "in_channels": 3,
            "num_classes": 1,
            "bottleneck_channels": 16,
        },
        "adapter": {
            "hidden_channels": 8,
            "placement": "output",
        },
        "node": {
            "steps": 4,
            "step_size": 0.25,
            "solver": "euler",
        },
    }

    module.run_robustness_experiment(
        config=config,
        artifacts_dir=tmp_path,
        groups=["A"],
        noise_levels=[0.0],
    )

    assert model_calls["build_kwargs"]["adapter_placement"] == "output"


def test_load_model_for_group_supports_wrapped_module_prefixed_checkpoint(tmp_path, monkeypatch):
    module = _load_segmentation_compare_module()
    model_calls = {}
    checkpoint_path = tmp_path / "wrapped_checkpoint.pt"
    torch.save(
        {"state_dict": {"module.bias": torch.tensor([2.0])}},
        checkpoint_path,
    )

    class _FakeModel:
        def load_state_dict(self, state_dict):
            model_calls["loaded_state"] = dict(state_dict)

        def eval(self):
            model_calls["eval_called"] = True
            return self

    monkeypatch.setitem(
        sys.modules,
        "src.models.segmentation_model",
        ModuleType("src.models.segmentation_model"),
    )
    sys.modules["src.models.segmentation_model"].build_segmentation_model = lambda **kwargs: _FakeModel()
    monkeypatch.setitem(
        sys.modules,
        "src.experiments.low_data_runner",
        ModuleType("src.experiments.low_data_runner"),
    )
    sys.modules["src.experiments.low_data_runner"].resolve_group_adapter = lambda group: "conv"

    config = {
        "model": {
            "encoder_name": "resnet18",
            "in_channels": 3,
            "num_classes": 1,
            "bottleneck_channels": 16,
        },
        "adapter": {
            "hidden_channels": 8,
        },
        "node": {
            "steps": 4,
            "step_size": 0.25,
            "solver": "euler",
        },
    }

    model = module.load_model_for_group(checkpoint_path, config, "A")

    assert model is not None
    assert model_calls["loaded_state"] == {"bias": torch.tensor([2.0])}
    assert model_calls["eval_called"] is True


def test_load_model_for_group_forwards_adapter_placement(tmp_path, monkeypatch):
    module = _load_segmentation_compare_module()
    model_calls = {}
    checkpoint_path = tmp_path / "wrapped_checkpoint.pt"
    torch.save({"state_dict": {"module.bias": torch.tensor([2.0])}}, checkpoint_path)

    class _FakeModel:
        def load_state_dict(self, state_dict):
            model_calls["loaded_state"] = dict(state_dict)

        def eval(self):
            return self

    def _fake_build_segmentation_model(**kwargs):
        model_calls["build_kwargs"] = dict(kwargs)
        return _FakeModel()

    monkeypatch.setitem(
        sys.modules,
        "src.models.segmentation_model",
        ModuleType("src.models.segmentation_model"),
    )
    sys.modules["src.models.segmentation_model"].build_segmentation_model = _fake_build_segmentation_model
    monkeypatch.setitem(
        sys.modules,
        "src.experiments.low_data_runner",
        ModuleType("src.experiments.low_data_runner"),
    )
    sys.modules["src.experiments.low_data_runner"].resolve_group_adapter = lambda group: "conv"

    config = {
        "model": {
            "encoder_name": "resnet18",
            "in_channels": 3,
            "num_classes": 1,
            "bottleneck_channels": 16,
        },
        "adapter": {
            "hidden_channels": 8,
            "placement": "output",
        },
        "node": {
            "steps": 4,
            "step_size": 0.25,
            "solver": "euler",
        },
    }

    module.load_model_for_group(checkpoint_path, config, "A")

    assert model_calls["build_kwargs"]["adapter_placement"] == "output"
