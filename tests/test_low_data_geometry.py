from importlib import util
from pathlib import Path
from types import SimpleNamespace
from types import ModuleType

import csv
import sys

import torch


def _ensure_package(name: str, package_path: Path) -> None:
    if name in sys.modules:
        return
    pkg = ModuleType(name)
    pkg.__path__ = [str(package_path)]  # type: ignore[attr-defined]
    sys.modules[name] = pkg


def _ensure_src_analysis_importable() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    analysis_dir = src_dir / "analysis"
    data_dir = src_dir / "data"
    experiments_dir = src_dir / "experiments"
    features_dir = src_dir / "features"
    models_dir = src_dir / "models"
    utils_dir = src_dir / "utils"

    _ensure_package("src", src_dir)
    _ensure_package("src.analysis", analysis_dir)
    _ensure_package("src.data", data_dir)
    _ensure_package("src.experiments", experiments_dir)
    _ensure_package("src.features", features_dir)
    _ensure_package("src.models", models_dir)
    _ensure_package("src.utils", utils_dir)


def _load_low_data_geometry_module():
    _ensure_src_analysis_importable()
    module_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "analysis"
        / "low_data_geometry.py"
    )
    assert module_path.exists(), f"Missing module file: {module_path}"
    spec = util.spec_from_file_location("low_data_geometry", module_path)
    assert spec is not None
    module = util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def test_build_embedding_rows_emits_pre_then_post_adapter_rows():
    module = _load_low_data_geometry_module()
    model_output = SimpleNamespace(
        bottleneck=torch.tensor([[[[1.0]], [[2.0]]]]),
        adapted_bottleneck=torch.tensor([[[[3.0]], [[4.0]]]]),
    )

    rows = module.build_embedding_rows(
        model_output=model_output,
        mask=torch.tensor([[[1]]], dtype=torch.long),
        sample_ids=["sample_1"],
        include_classes=["lesion"],
        class_values={"lesion": 1},
        min_mask_pixels=1,
    )

    assert [row["state"] for row in rows] == ["pre_adapter", "post_adapter"]


def test_build_embedding_rows_copies_embeddings_for_single_pixel_mask():
    module = _load_low_data_geometry_module()
    model_output = SimpleNamespace(
        bottleneck=torch.tensor([[[[1.5]], [[2.5]]]]),
        adapted_bottleneck=torch.tensor([[[[3.5]], [[4.5]]]]),
    )

    rows = module.build_embedding_rows(
        model_output=model_output,
        mask=torch.tensor([[[1]]], dtype=torch.long),
        sample_ids=["sample_1"],
        include_classes=["lesion"],
        class_values={"lesion": 1},
        min_mask_pixels=1,
    )

    assert rows == [
        {
            "sample_id": "sample_1",
            "state": "pre_adapter",
            "class_name": "lesion",
            "pixel_count": 1,
            "embedding": [1.5, 2.5],
        },
        {
            "sample_id": "sample_1",
            "state": "post_adapter",
            "class_name": "lesion",
            "pixel_count": 1,
            "embedding": [3.5, 4.5],
        },
    ]


def test_write_embedding_csv_expands_embedding_columns(tmp_path):
    module = _load_low_data_geometry_module()
    output_path = tmp_path / "embeddings.csv"

    module.write_embedding_csv(
        rows=[
            {
                "sample_id": "sample_1",
                "state": "pre_adapter",
                "class_name": "lesion",
                "pixel_count": 1,
                "embedding": [1.25, 2.5],
            }
        ],
        output_path=output_path,
    )

    with output_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == [
            "sample_id",
            "state",
            "class_name",
            "pixel_count",
            "embedding_0000",
            "embedding_0001",
        ]
        rows = list(reader)

    assert rows == [
        {
            "sample_id": "sample_1",
            "state": "pre_adapter",
            "class_name": "lesion",
            "pixel_count": "1",
            "embedding_0000": "1.25",
            "embedding_0001": "2.5",
        }
    ]


def test_export_group_geometry_writes_pre_and_post_adapter_csvs(tmp_path, monkeypatch):
    module = _load_low_data_geometry_module()
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint_path.write_bytes(b"checkpoint")

    calls = {}

    class _FakeDataset:
        def __init__(self, *, images_dir, masks_dir, image_size, class_values, sample_ids=None):
            calls["dataset"] = {
                "images_dir": images_dir,
                "masks_dir": masks_dir,
                "image_size": image_size,
                "class_values": dict(class_values),
                "sample_ids": sample_ids,
            }

    class _FakeLoader:
        def __init__(self, dataset, batch_size, shuffle, **kwargs):
            calls["loader"] = {
                "dataset": dataset,
                "batch_size": batch_size,
                "shuffle": shuffle,
                "kwargs": dict(kwargs),
            }
            self._batches = [
                {
                    "sample_id": ["sample_1"],
                    "image": torch.zeros((1, 3, 1, 1), dtype=torch.float32),
                    "mask": torch.tensor([[[1]]], dtype=torch.long),
                }
            ]

        def __iter__(self):
            return iter(self._batches)

    class _FakeModel:
        def __init__(self):
            self.loaded_state = None
            self.to_device = None
            self.eval_called = False

        def load_state_dict(self, state_dict):
            self.loaded_state = dict(state_dict)

        def to(self, device):
            self.to_device = device
            return self

        def eval(self):
            self.eval_called = True
            return self

        def __call__(self, images):
            assert tuple(images.shape) == (1, 3, 1, 1)
            return SimpleNamespace(
                bottleneck=torch.tensor([[[[1.0]], [[2.0]]]]),
                adapted_bottleneck=torch.tensor([[[[10.0]], [[20.0]]]]),
            )

    fake_model = _FakeModel()

    def _fake_build_segmentation_model(**kwargs):
        calls["build_segmentation_model"] = dict(kwargs)
        return fake_model

    def _fake_load_checkpoint(path, device):
        calls["load_checkpoint"] = {
            "path": Path(path),
            "device": str(device),
        }
        return {"state_dict": {"module.weight": torch.tensor([1.0])}}

    monkeypatch.setattr(module, "ISIC2018Dataset", _FakeDataset)
    monkeypatch.setattr(module, "DataLoader", _FakeLoader)
    monkeypatch.setattr(module, "build_segmentation_model", _fake_build_segmentation_model)
    monkeypatch.setattr(module, "load_checkpoint", _fake_load_checkpoint)
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: False)

    config = {
        "paths": {
            "val_images_dir": "data/val/images",
            "val_masks_dir": "data/val/masks",
            "artifacts_dir": str(tmp_path / "artifacts"),
        },
        "data": {
            "image_size": 64,
            "num_workers": 2,
            "pin_memory": True,
            "class_values": {"background": 0, "lesion": 1},
        },
        "train": {
            "batch_size": 4,
        },
        "model": {
            "encoder_name": "resnet18",
            "encoder_weights": None,
            "in_channels": 3,
            "num_classes": 1,
            "bottleneck_channels": 2,
            "freeze_encoder": True,
        },
        "adapter": {
            "hidden_channels": 8,
        },
        "node": {
            "steps": 4,
            "step_size": 0.25,
        },
        "geometry": {
            "batch_size": 1,
            "include_classes": ["lesion"],
            "min_mask_pixels": 1,
        },
    }

    pre_path, post_path = module.export_group_geometry(
        config=config,
        group="B",
        checkpoint_path=checkpoint_path,
    )

    assert pre_path == tmp_path / "artifacts" / "low_data" / "group_b" / "geometry" / "pre_adapter_embeddings.csv"
    assert post_path == tmp_path / "artifacts" / "low_data" / "group_b" / "geometry" / "post_adapter_embeddings.csv"
    assert pre_path.exists()
    assert post_path.exists()

    with pre_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [
        {
            "sample_id": "sample_1",
            "state": "pre_adapter",
            "class_name": "lesion",
            "pixel_count": "1",
            "embedding_0000": "1.0",
            "embedding_0001": "2.0",
        }
    ]

    with post_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [
        {
            "sample_id": "sample_1",
            "state": "post_adapter",
            "class_name": "lesion",
            "pixel_count": "1",
            "embedding_0000": "10.0",
            "embedding_0001": "20.0",
        }
    ]

    assert calls["dataset"]["class_values"] == {"background": 0, "lesion": 1}
    assert calls["loader"]["batch_size"] == 1
    assert calls["loader"]["shuffle"] is False
    assert calls["loader"]["kwargs"] == {"num_workers": 2, "pin_memory": True}
    assert calls["build_segmentation_model"]["adapter_type"] == "conv"
    assert calls["load_checkpoint"]["path"] == checkpoint_path
    assert calls["load_checkpoint"]["device"] == "cpu"
    assert fake_model.loaded_state == {"weight": torch.tensor([1.0])}
    assert str(fake_model.to_device) == "cpu"
    assert fake_model.eval_called is True
