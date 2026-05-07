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


def test_build_segmentation_model_passes_zero_last_init_to_node_adapter():
    import torch.nn as nn

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
    convs = [m for m in model.adapter.func.modules() if isinstance(m, nn.Conv2d)]

    assert len(convs) == 2
    assert torch.count_nonzero(convs[-1].weight).item() == 0
    assert convs[-1].bias is not None
    assert torch.count_nonzero(convs[-1].bias).item() == 0


def _ensure_src_experiments_importable() -> None:
    """
    Avoid sys.path mutation while ensuring `from src.experiments...` resolves to
    this workspace during pytest collection.

    Some environments ship their own top-level `src` which can shadow ours.
    """
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    experiments_dir = src_dir / "experiments"
    if not experiments_dir.exists():
        return

    # Prefer not to clobber an existing `src` module if present; just ensure our
    # workspace paths are part of the package path used for resolution.
    if "src" not in sys.modules:
        _ensure_package("src", src_dir)
    else:
        mod = sys.modules["src"]
        if not hasattr(mod, "__path__"):
            _ensure_package("src", src_dir)
        else:
            paths = list(getattr(mod, "__path__"))  # type: ignore[arg-type]
            if str(src_dir) not in paths:
                paths.insert(0, str(src_dir))
                setattr(mod, "__path__", paths)

    if "src.experiments" not in sys.modules:
        _ensure_package("src.experiments", experiments_dir)
    else:
        mod = sys.modules["src.experiments"]
        if not hasattr(mod, "__path__"):
            _ensure_package("src.experiments", experiments_dir)
        else:
            paths = list(getattr(mod, "__path__"))  # type: ignore[arg-type]
            if str(experiments_dir) not in paths:
                paths.insert(0, str(experiments_dir))
                setattr(mod, "__path__", paths)


_ensure_src_experiments_importable()
from src.experiments.low_data_runner import resolve_group_adapter


def test_resolve_group_adapter_maps_groups_to_expected_types():
    assert resolve_group_adapter("A") == "none"
    assert resolve_group_adapter("B") == "conv"
    assert resolve_group_adapter("C") == "node"


def test_load_config_parses_yaml(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("seed: 123\n", encoding="utf-8")

    # Import inside the test so we can watch it fail before implementation.
    from src.experiments.low_data_runner import load_config

    config = load_config(config_path)
    assert config["seed"] == 123


def test_load_config_raises_clear_error_for_malformed_yaml(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("seed: [1,2\n", encoding="utf-8")

    from src.experiments.low_data_runner import load_config

    try:
        load_config(config_path)
        assert False, "Expected load_config() to raise on malformed YAML"
    except ValueError as exc:
        assert "YAML" in str(exc) or "yaml" in str(exc)


def test_run_group_validates_required_config_keys(tmp_path, monkeypatch):
    artifacts_dir = tmp_path / "artifacts"
    config_path = tmp_path / "config.yaml"
    # Missing many required keys on purpose.
    config_path.write_text(
        "\n".join(
            [
                "seed: 42",
                "paths:",
                f"  artifacts_dir: {artifacts_dir.as_posix()}",
                "data:",
                "  image_size: 256",
                "  train_ratio: 0.1",
                "",
            ]
        ),
        encoding="utf-8",
    )

    # Stub minimal modules so import doesn't fail; validation should happen before use.
    monkeypatch.setitem(sys.modules, "src.data.isic2018", ModuleType("src.data.isic2018"))
    monkeypatch.setitem(sys.modules, "src.data.splits", ModuleType("src.data.splits"))
    monkeypatch.setitem(sys.modules, "src.models.segmentation_model", ModuleType("src.models.segmentation_model"))
    monkeypatch.setitem(sys.modules, "src.training.engine", ModuleType("src.training.engine"))

    from src.experiments.low_data_runner import run_group

    try:
        run_group(config_path, "A")
        assert False, "Expected run_group() to raise for missing keys"
    except ValueError as exc:
        msg = str(exc)
        assert "paths" in msg or "train" in msg or "model" in msg


def test_run_group_writes_split_manifest_and_uses_group_adapter(tmp_path, monkeypatch):
    artifacts_dir = tmp_path / "artifacts"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "seed: 42",
                "paths:",
                "  train_images_dir: data/train/images",
                "  train_masks_dir: data/train/labels",
                "  val_images_dir: data/val/images",
                "  val_masks_dir: data/val/labels",
                f"  artifacts_dir: {artifacts_dir.as_posix()}",
                "data:",
                "  image_size: 256",
                "  train_ratio: 0.1",
                "train:",
                "  batch_size: 2",
                "  epochs: 3",
                "  learning_rate: 0.001",
                "  weight_decay: 0.01",
                "  early_stopping_patience: 5",
                "model:",
                "  encoder_name: resnet18",
                "  encoder_weights: null",
                "  in_channels: 3",
                "  num_classes: 1",
                "  bottleneck_channels: 16",
                "  freeze_encoder: true",
                "adapter:",
                "  hidden_channels: 8",
                "  init: zero_last_layer",
                "node:",
                "  steps: 4",
                "  step_size: 0.25",
                "",
            ]
        ),
        encoding="utf-8",
    )

    # Stub dependencies that run_group imports by name.
    calls: dict[str, object] = {}

    fake_isic = ModuleType("src.data.isic2018")

    class _FakePath:
        def __init__(self, stem: str) -> None:
            self.stem = stem

    class ISIC2018Dataset:
        def __init__(self, *, images_dir, masks_dir, image_size, class_values, sample_ids=None):
            self.images_dir = images_dir
            self.masks_dir = masks_dir
            self.image_size = image_size
            self.class_values = class_values
            self.sample_ids = sample_ids
            if sample_ids is None:
                self.image_paths = [_FakePath("img1"), _FakePath("img2"), _FakePath("img3")]
            else:
                self.image_paths = [_FakePath(stem) for stem in list(sample_ids)]

    fake_isic.ISIC2018Dataset = ISIC2018Dataset

    fake_splits = ModuleType("src.data.splits")

    def build_ratio_subset(sample_ids, ratio, seed):
        calls["build_ratio_subset"] = {"sample_ids": list(sample_ids), "ratio": ratio, "seed": seed}
        return ["img2", "img3"]

    def save_split_manifest(sample_ids, output_path):
        calls["save_split_manifest"] = {"sample_ids": list(sample_ids), "output_path": str(output_path)}
        # Runner should ensure parent dirs exist before calling this helper.
        calls["split_parent_exists"] = Path(output_path).parent.exists()

    fake_splits.build_ratio_subset = build_ratio_subset
    fake_splits.save_split_manifest = save_split_manifest

    fake_models = ModuleType("src.models.segmentation_model")

    class _FakeParam:
        def __init__(self, requires_grad: bool) -> None:
            self.requires_grad = requires_grad

    class _FakeModel:
        def __init__(self):
            self._params = [_FakeParam(True), _FakeParam(False)]

        def parameters(self):
            return list(self._params)

        def to(self, device):
            calls["model_to"] = device
            return self

    def build_segmentation_model(**kwargs):
        calls["build_segmentation_model"] = dict(kwargs)
        return _FakeModel()

    fake_models.build_segmentation_model = build_segmentation_model

    fake_engine = ModuleType("src.training.engine")

    def fit(model, train_loader, val_loader, optimizer, epochs, patience, output_dir, device="cpu"):
        calls["fit"] = {
            "epochs": epochs,
            "patience": patience,
            "output_dir": str(output_dir),
            "device": device,
            "optimizer": optimizer,
        }
        # Runner should ensure output_dir exists before training starts.
        calls["output_dir_exists"] = Path(output_dir).exists()
        return "FIT_RESULT"

    fake_engine.fit = fit

    # Patch import targets for run_group.
    monkeypatch.setitem(sys.modules, "src.data.isic2018", fake_isic)
    monkeypatch.setitem(sys.modules, "src.data.splits", fake_splits)
    monkeypatch.setitem(sys.modules, "src.models.segmentation_model", fake_models)
    monkeypatch.setitem(sys.modules, "src.training.engine", fake_engine)

    import torch

    def _fake_is_available() -> bool:
        return False

    monkeypatch.setattr(torch.cuda, "is_available", _fake_is_available)

    class _FakeAdamW:
        def __init__(self, params, lr, weight_decay):
            calls["adamw"] = {
                "num_params": len(list(params)),
                "lr": lr,
                "weight_decay": weight_decay,
            }

    monkeypatch.setattr(torch.optim, "AdamW", _FakeAdamW)

    class _FakeDataLoader:
        def __init__(self, dataset, batch_size, shuffle, **kwargs):
            calls.setdefault("dataloaders", []).append(
                {"batch_size": batch_size, "shuffle": shuffle, "kwargs": dict(kwargs)}
            )

    monkeypatch.setattr(torch.utils.data, "DataLoader", _FakeDataLoader)

    from src.experiments.low_data_runner import run_group

    result = run_group(config_path, "B")
    assert result == "FIT_RESULT"

    assert calls["build_ratio_subset"]["seed"] == 42
    assert calls["build_ratio_subset"]["ratio"] == 0.1
    assert calls["build_ratio_subset"]["sample_ids"] == ["img1", "img2", "img3"]

    expected_manifest = artifacts_dir / "splits" / "train_seed42_ratio10.csv"
    assert calls["save_split_manifest"]["output_path"] == str(expected_manifest)
    assert calls["save_split_manifest"]["sample_ids"] == ["img2", "img3"]
    assert calls["split_parent_exists"] is True

    assert calls["build_segmentation_model"]["adapter_type"] == "conv"
    assert calls["build_segmentation_model"]["adapter_init"] == "zero_last_layer"
    assert Path(calls["fit"]["output_dir"]).name == "group_b"
    assert calls["output_dir_exists"] is True


def test_run_group_rejects_unknown_adapter_init(tmp_path):
    artifacts_dir = tmp_path / "artifacts"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "seed: 42",
                "paths:",
                "  train_images_dir: data/train/images",
                "  train_masks_dir: data/train/labels",
                "  val_images_dir: data/val/images",
                "  val_masks_dir: data/val/labels",
                f"  artifacts_dir: {artifacts_dir.as_posix()}",
                "data:",
                "  image_size: 256",
                "  train_ratio: 0.1",
                "train:",
                "  batch_size: 2",
                "  epochs: 1",
                "  learning_rate: 0.001",
                "  weight_decay: 0.01",
                "  early_stopping_patience: 1",
                "model:",
                "  encoder_name: resnet18",
                "  encoder_weights: null",
                "  in_channels: 3",
                "  num_classes: 1",
                "  bottleneck_channels: 16",
                "  freeze_encoder: true",
                "adapter:",
                "  hidden_channels: 8",
                "  init: unsupported",
                "node:",
                "  steps: 4",
                "  step_size: 0.25",
                "",
            ]
        ),
        encoding="utf-8",
    )

    from src.experiments.low_data_runner import run_group

    try:
        run_group(config_path, "C")
        assert False, "Expected invalid adapter.init to raise ValueError"
    except ValueError as exc:
        assert "adapter.init" in str(exc)
        assert "zero_last_layer" in str(exc)


def test_run_group_retries_once_with_num_workers_zero_on_known_windows_dataloader_permission_error(
    tmp_path, monkeypatch
):
    """
    Regression test for a Windows-specific failure mode:

    When DataLoader worker processes fail to start under Windows, training can
    raise PermissionError: [WinError 5] Access is denied (or localized variants).

    The runner should retry exactly once with num_workers=0 instead of crashing,
    while keeping the configured num_workers by default.
    """
    artifacts_dir = tmp_path / "artifacts"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "seed: 42",
                "paths:",
                "  train_images_dir: data/train/images",
                "  train_masks_dir: data/train/labels",
                "  val_images_dir: data/val/images",
                "  val_masks_dir: data/val/labels",
                f"  artifacts_dir: {artifacts_dir.as_posix()}",
                "data:",
                "  image_size: 256",
                "  train_ratio: 0.1",
                "  num_workers: 2",
                "  pin_memory: true",
                "train:",
                "  batch_size: 2",
                "  epochs: 1",
                "  learning_rate: 0.001",
                "  weight_decay: 0.01",
                "  early_stopping_patience: 1",
                "model:",
                "  encoder_name: resnet18",
                "  encoder_weights: null",
                "  in_channels: 3",
                "  num_classes: 1",
                "  bottleneck_channels: 16",
                "  freeze_encoder: true",
                "adapter:",
                "  hidden_channels: 8",
                "node:",
                "  steps: 4",
                "  step_size: 0.25",
                "",
            ]
        ),
        encoding="utf-8",
    )

    calls: dict[str, object] = {}

    fake_isic = ModuleType("src.data.isic2018")

    class _FakePath:
        def __init__(self, stem: str) -> None:
            self.stem = stem

    class ISIC2018Dataset:
        def __init__(
            self, *, images_dir, masks_dir, image_size, class_values, sample_ids=None
        ):
            self.image_paths = [_FakePath("img1"), _FakePath("img2"), _FakePath("img3")]

    fake_isic.ISIC2018Dataset = ISIC2018Dataset

    fake_splits = ModuleType("src.data.splits")

    def build_ratio_subset(sample_ids, ratio, seed):
        return ["img2", "img3"]

    def save_split_manifest(sample_ids, output_path):
        return None

    fake_splits.build_ratio_subset = build_ratio_subset
    fake_splits.save_split_manifest = save_split_manifest

    fake_models = ModuleType("src.models.segmentation_model")

    class _FakeParam:
        def __init__(self, requires_grad: bool) -> None:
            self.requires_grad = requires_grad

    class _FakeModel:
        def __init__(self):
            self._params = [_FakeParam(True)]

        def parameters(self):
            return list(self._params)

        def to(self, device):
            return self

    def build_segmentation_model(**kwargs):
        return _FakeModel()

    fake_models.build_segmentation_model = build_segmentation_model

    fake_engine = ModuleType("src.training.engine")

    def _raise_permission_error_from_fake_file(filename: str) -> None:
        """
        Raise a PermissionError whose traceback includes `filename`.

        We use `compile(..., filename=...)` so the traceback resembles a real-world
        failure inside torch's DataLoader/multiprocessing stack.
        """
        namespace: dict[str, object] = {}
        code = compile(
            "\n".join(
                [
                    "def boom():",
                    "    raise PermissionError('[WinError 5] Access is denied.')",
                ]
            ),
            filename,
            "exec",
        )
        exec(code, namespace, namespace)  # noqa: S102 - intentional for test
        boom = namespace["boom"]
        assert callable(boom)
        boom()

    def fit(
        model, train_loader, val_loader, optimizer, epochs, patience, output_dir, device="cpu"
    ):
        calls.setdefault("fit_calls", []).append(
            {"train_loader": train_loader, "val_loader": val_loader}
        )
        if len(calls["fit_calls"]) == 1:
            # Simulate a DataLoader/multiprocessing-originated PermissionError.
            _raise_permission_error_from_fake_file("torch/utils/data/dataloader.py")
        return "FIT_RESULT"

    fake_engine.fit = fit

    monkeypatch.setitem(sys.modules, "src.data.isic2018", fake_isic)
    monkeypatch.setitem(sys.modules, "src.data.splits", fake_splits)
    monkeypatch.setitem(sys.modules, "src.models.segmentation_model", fake_models)
    monkeypatch.setitem(sys.modules, "src.training.engine", fake_engine)

    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.optim, "AdamW", lambda params, lr, weight_decay: object())

    class _FakeDataLoader:
        def __init__(self, dataset, batch_size, shuffle, **kwargs):
            self.batch_size = batch_size
            self.shuffle = shuffle
            self.kwargs = dict(kwargs)
            calls.setdefault("dataloaders", []).append(self.kwargs)

    monkeypatch.setattr(torch.utils.data, "DataLoader", _FakeDataLoader)

    from src.experiments.low_data_runner import run_group

    result = run_group(config_path, "A")
    assert result == "FIT_RESULT"

    # Default pass should use configured values, then retry should force num_workers=0.
    assert len(calls["fit_calls"]) == 2
    assert calls["dataloaders"][0]["num_workers"] == 2
    assert calls["dataloaders"][0]["pin_memory"] is True
    assert calls["dataloaders"][2]["num_workers"] == 0
    assert calls["dataloaders"][2]["pin_memory"] is True


def test_run_group_does_not_retry_on_unrelated_winerror5_permission_error(
    tmp_path, monkeypatch
):
    """
    Ensure we don't mask unrelated WinError 5 PermissionErrors.

    The fallback is intended only for Windows DataLoader worker-start failures,
    not arbitrary PermissionError: [WinError 5] cases (e.g., artifact writing).
    """
    artifacts_dir = tmp_path / "artifacts"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "seed: 42",
                "paths:",
                "  train_images_dir: data/train/images",
                "  train_masks_dir: data/train/labels",
                "  val_images_dir: data/val/images",
                "  val_masks_dir: data/val/labels",
                f"  artifacts_dir: {artifacts_dir.as_posix()}",
                "data:",
                "  image_size: 256",
                "  train_ratio: 0.1",
                "  num_workers: 2",
                "  pin_memory: true",
                "train:",
                "  batch_size: 2",
                "  epochs: 1",
                "  learning_rate: 0.001",
                "  weight_decay: 0.01",
                "  early_stopping_patience: 1",
                "model:",
                "  encoder_name: resnet18",
                "  encoder_weights: null",
                "  in_channels: 3",
                "  num_classes: 1",
                "  bottleneck_channels: 16",
                "  freeze_encoder: true",
                "adapter:",
                "  hidden_channels: 8",
                "node:",
                "  steps: 4",
                "  step_size: 0.25",
                "",
            ]
        ),
        encoding="utf-8",
    )

    calls: dict[str, object] = {}

    fake_isic = ModuleType("src.data.isic2018")

    class _FakePath:
        def __init__(self, stem: str) -> None:
            self.stem = stem

    class ISIC2018Dataset:
        def __init__(
            self, *, images_dir, masks_dir, image_size, class_values, sample_ids=None
        ):
            self.image_paths = [_FakePath("img1"), _FakePath("img2"), _FakePath("img3")]

    fake_isic.ISIC2018Dataset = ISIC2018Dataset

    fake_splits = ModuleType("src.data.splits")
    fake_splits.build_ratio_subset = lambda sample_ids, ratio, seed: ["img2", "img3"]
    fake_splits.save_split_manifest = lambda sample_ids, output_path: None

    fake_models = ModuleType("src.models.segmentation_model")

    class _FakeParam:
        def __init__(self, requires_grad: bool) -> None:
            self.requires_grad = requires_grad

    class _FakeModel:
        def __init__(self):
            self._params = [_FakeParam(True)]

        def parameters(self):
            return list(self._params)

        def to(self, device):
            return self

    fake_models.build_segmentation_model = lambda **kwargs: _FakeModel()

    fake_engine = ModuleType("src.training.engine")

    def _raise_permission_error_from_fake_file(filename: str) -> None:
        namespace: dict[str, object] = {}
        code = compile(
            "\n".join(
                [
                    "def boom():",
                    "    raise PermissionError('[WinError 5] Access is denied.')",
                ]
            ),
            filename,
            "exec",
        )
        exec(code, namespace, namespace)  # noqa: S102 - intentional for test
        boom = namespace["boom"]
        assert callable(boom)
        boom()

    def fit(
        model, train_loader, val_loader, optimizer, epochs, patience, output_dir, device="cpu"
    ):
        calls.setdefault("fit_calls", []).append(1)
        # Simulate a non-DataLoader PermissionError (e.g. file write).
        _raise_permission_error_from_fake_file("pathlib.py")

    fake_engine.fit = fit

    monkeypatch.setitem(sys.modules, "src.data.isic2018", fake_isic)
    monkeypatch.setitem(sys.modules, "src.data.splits", fake_splits)
    monkeypatch.setitem(sys.modules, "src.models.segmentation_model", fake_models)
    monkeypatch.setitem(sys.modules, "src.training.engine", fake_engine)

    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.optim, "AdamW", lambda params, lr, weight_decay: object())

    class _FakeDataLoader:
        def __init__(self, dataset, batch_size, shuffle, **kwargs):
            return None

    monkeypatch.setattr(torch.utils.data, "DataLoader", _FakeDataLoader)

    from src.experiments.low_data_runner import run_group

    try:
        run_group(config_path, "A")
        assert False, "Expected PermissionError to propagate for unrelated WinError 5"
    except PermissionError as exc:
        assert "WinError 5" in str(exc)

    # No retry: should call fit exactly once.
    assert calls["fit_calls"] == [1]


def test_run_group_falls_back_to_local_adamw_on_known_register_pytree_node_error(
    tmp_path, monkeypatch
):
    """
    Regression test for a specific environment failure:

    In some installations, constructing `torch.optim.AdamW(...)` raises an
    AttributeError referencing `torch.utils._pytree.register_pytree_node`.
    The runner should fall back to a local optimizer implementation so the
    low-data experiment can run without broad environment changes.
    """
    artifacts_dir = tmp_path / "artifacts"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "seed: 42",
                "paths:",
                "  train_images_dir: data/train/images",
                "  train_masks_dir: data/train/labels",
                "  val_images_dir: data/val/images",
                "  val_masks_dir: data/val/labels",
                f"  artifacts_dir: {artifacts_dir.as_posix()}",
                "data:",
                "  image_size: 256",
                "  train_ratio: 0.1",
                "train:",
                "  batch_size: 2",
                "  epochs: 1",
                "  learning_rate: 0.001",
                "  weight_decay: 0.01",
                "  early_stopping_patience: 1",
                "model:",
                "  encoder_name: resnet18",
                "  encoder_weights: null",
                "  in_channels: 3",
                "  num_classes: 1",
                "  bottleneck_channels: 16",
                "  freeze_encoder: true",
                "adapter:",
                "  hidden_channels: 8",
                "node:",
                "  steps: 4",
                "  step_size: 0.25",
                "",
            ]
        ),
        encoding="utf-8",
    )

    calls: dict[str, object] = {}

    fake_isic = ModuleType("src.data.isic2018")

    class _FakePath:
        def __init__(self, stem: str) -> None:
            self.stem = stem

    class ISIC2018Dataset:
        def __init__(self, *, images_dir, masks_dir, image_size, class_values, sample_ids=None):
            self.images_dir = images_dir
            self.masks_dir = masks_dir
            self.image_size = image_size
            self.class_values = class_values
            self.sample_ids = sample_ids
            if sample_ids is None:
                self.image_paths = [_FakePath("img1"), _FakePath("img2"), _FakePath("img3")]
            else:
                self.image_paths = [_FakePath(stem) for stem in list(sample_ids)]

    fake_isic.ISIC2018Dataset = ISIC2018Dataset

    fake_splits = ModuleType("src.data.splits")

    def build_ratio_subset(sample_ids, ratio, seed):
        return ["img2", "img3"]

    def save_split_manifest(sample_ids, output_path):
        return None

    fake_splits.build_ratio_subset = build_ratio_subset
    fake_splits.save_split_manifest = save_split_manifest

    fake_models = ModuleType("src.models.segmentation_model")

    class _FakeParam:
        def __init__(self, requires_grad: bool) -> None:
            self.requires_grad = requires_grad
            self.grad = None

    class _FakeModel:
        def __init__(self):
            self._params = [_FakeParam(True), _FakeParam(False)]

        def parameters(self):
            return list(self._params)

        def to(self, device):
            return self

    def build_segmentation_model(**kwargs):
        return _FakeModel()

    fake_models.build_segmentation_model = build_segmentation_model

    fake_engine = ModuleType("src.training.engine")

    def fit(model, train_loader, val_loader, optimizer, epochs, patience, output_dir, device="cpu"):
        calls["optimizer"] = optimizer
        return "FIT_RESULT"

    fake_engine.fit = fit

    monkeypatch.setitem(sys.modules, "src.data.isic2018", fake_isic)
    monkeypatch.setitem(sys.modules, "src.data.splits", fake_splits)
    monkeypatch.setitem(sys.modules, "src.models.segmentation_model", fake_models)
    monkeypatch.setitem(sys.modules, "src.training.engine", fake_engine)

    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    class _AdamWThatFails:
        def __init__(self, params, lr, weight_decay):
            raise AttributeError(
                "module 'torch.utils._pytree' has no attribute 'register_pytree_node'"
            )

    monkeypatch.setattr(torch.optim, "AdamW", _AdamWThatFails)

    class _FakeDataLoader:
        def __init__(self, dataset, batch_size, shuffle, **kwargs):
            return None

    monkeypatch.setattr(torch.utils.data, "DataLoader", _FakeDataLoader)

    from src.experiments.low_data_runner import run_group

    result = run_group(config_path, "A")
    assert result == "FIT_RESULT"

    optimizer = calls["optimizer"]
    assert hasattr(optimizer, "zero_grad")
    assert hasattr(optimizer, "step")


def test_local_adamw_step_updates_trainable_params_and_zero_grad_clears_grads():
    import torch

    from src.experiments.low_data_runner import _LocalAdamW

    param = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
    param.grad = torch.tensor([1.0], dtype=torch.float32)

    opt = _LocalAdamW([param], lr=0.1, weight_decay=0.01)
    before = param.detach().clone()

    opt.step()
    after = param.detach().clone()
    assert torch.all(after < before)

    # Reattach a grad and ensure zero_grad clears it.
    param.grad = torch.tensor([1.0], dtype=torch.float32)
    opt.zero_grad()
    assert param.grad is None
