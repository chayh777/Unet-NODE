from __future__ import annotations

from pathlib import Path
from types import ModuleType
import sys


def _ensure_package(name: str, package_path: Path) -> None:
    if name in sys.modules:
        return
    pkg = ModuleType(name)
    pkg.__path__ = [str(package_path)]  # type: ignore[attr-defined]
    sys.modules[name] = pkg


def _ensure_src_data_importable() -> None:
    """
    Avoid sys.path mutation while ensuring `from src.data...` resolves to this
    workspace during standalone pytest collection.
    """
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    data_dir = src_dir / "data"
    if not data_dir.exists():
        return

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

    if "src.data" not in sys.modules:
        _ensure_package("src.data", data_dir)
    else:
        mod = sys.modules["src.data"]
        if not hasattr(mod, "__path__"):
            _ensure_package("src.data", data_dir)
        else:
            paths = list(getattr(mod, "__path__"))  # type: ignore[arg-type]
            if str(data_dir) not in paths:
                paths.insert(0, str(data_dir))
                setattr(mod, "__path__", paths)


_ensure_src_data_importable()


def test_resolve_dataset_spec_returns_isic2018_contract():
    from src.data.factory import resolve_dataset_spec

    spec = resolve_dataset_spec("isic2018")

    assert spec.dataset_name == "isic2018"
    assert spec.class_name == "ISIC2018Dataset"
    assert spec.class_values == {"background": 0, "lesion": 1}
    assert callable(spec.extract_sample_ids)


def test_resolve_dataset_spec_supports_glas_contract():
    from src.data.factory import resolve_dataset_spec

    spec = resolve_dataset_spec("glas")

    assert spec.dataset_name == "glas"
    assert spec.class_name == "GlaSDataset"
    assert spec.class_values == {"background": 0, "gland": 1}
    assert spec.available is True
    assert callable(spec.extract_sample_ids)


def test_dataset_spec_extract_sample_ids_returns_stems_in_order():
    from src.data.factory import resolve_dataset_spec

    class _FakePath:
        def __init__(self, stem: str) -> None:
            self.stem = stem

    spec = resolve_dataset_spec("isic2018")
    dataset = type(
        "_Dataset",
        (),
        {"image_paths": [_FakePath("img1"), _FakePath("img2"), _FakePath("img3")]},
    )()

    assert spec.extract_sample_ids(dataset) == ["img1", "img2", "img3"]


def test_dataset_spec_extract_sample_ids_raises_clear_error_for_invalid_dataset_contract():
    from src.data.factory import resolve_dataset_spec

    spec = resolve_dataset_spec("isic2018")
    dataset = type("_Dataset", (), {"image_paths": [object()]})()

    try:
        spec.extract_sample_ids(dataset)
        assert False, "Expected invalid dataset sample-id contract to raise ValueError"
    except ValueError as exc:
        message = str(exc)
        assert "image_paths" in message
        assert "stem" in message


def test_build_low_data_datasets_constructs_expected_isic_splits(tmp_path, monkeypatch):
    images_dir = tmp_path / "train_images"
    masks_dir = tmp_path / "train_masks"
    val_images_dir = tmp_path / "val_images"
    val_masks_dir = tmp_path / "val_masks"
    for directory in [images_dir, masks_dir, val_images_dir, val_masks_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    config = {
        "seed": 13,
        "paths": {
            "train_images_dir": str(images_dir),
            "train_masks_dir": str(masks_dir),
            "val_images_dir": str(val_images_dir),
            "val_masks_dir": str(val_masks_dir),
        },
        "data": {
            "dataset_name": "isic2018",
            "image_size": 256,
            "train_ratio": 0.4,
        },
    }

    calls: dict[str, object] = {}
    fake_isic = ModuleType("src.data.isic2018")

    class _FakePath:
        def __init__(self, stem: str) -> None:
            self.stem = stem

    class _FakeDataset:
        def __init__(
            self, *, images_dir, masks_dir, image_size, class_values, sample_ids=None
        ) -> None:
            calls.setdefault("datasets", []).append(
                {
                    "images_dir": images_dir,
                    "masks_dir": masks_dir,
                    "image_size": image_size,
                    "class_values": dict(class_values),
                    "sample_ids": None if sample_ids is None else list(sample_ids),
                }
            )
            self.sample_ids = sample_ids
            if sample_ids is None:
                self.image_paths = [_FakePath("img1"), _FakePath("img2"), _FakePath("img3")]
            else:
                self.image_paths = [_FakePath(sample_id) for sample_id in sample_ids]

    fake_isic.ISIC2018Dataset = _FakeDataset

    fake_splits = ModuleType("src.data.splits")

    def build_ratio_subset(sample_ids, ratio, seed):
        calls["split"] = {"sample_ids": list(sample_ids), "ratio": ratio, "seed": seed}
        return ["img1", "img3"]

    fake_splits.build_ratio_subset = build_ratio_subset

    monkeypatch.setitem(sys.modules, "src.data.isic2018", fake_isic)
    monkeypatch.setitem(sys.modules, "src.data.splits", fake_splits)

    from src.data.factory import build_low_data_datasets

    datasets = build_low_data_datasets(config)

    assert calls["split"] == {
        "sample_ids": ["img1", "img2", "img3"],
        "ratio": 0.4,
        "seed": 13,
    }
    assert datasets.selected_ids == ["img1", "img3"]
    assert [path.stem for path in datasets.train_dataset.image_paths] == ["img1", "img3"]
    assert [path.stem for path in datasets.val_dataset.image_paths] == ["img1", "img2", "img3"]


def test_build_low_data_datasets_constructs_expected_glas_splits(tmp_path, monkeypatch):
    images_dir = tmp_path / "train_images"
    masks_dir = tmp_path / "train_masks"
    val_images_dir = tmp_path / "val_images"
    val_masks_dir = tmp_path / "val_masks"
    for directory in [images_dir, masks_dir, val_images_dir, val_masks_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    config = {
        "seed": 13,
        "paths": {
            "train_images_dir": str(images_dir),
            "train_masks_dir": str(masks_dir),
            "val_images_dir": str(val_images_dir),
            "val_masks_dir": str(val_masks_dir),
        },
        "data": {
            "dataset_name": "glas",
            "image_size": 256,
            "train_ratio": 0.4,
        },
    }

    fake_splits = ModuleType("src.data.splits")

    def build_ratio_subset(sample_ids, ratio, seed):
        assert sample_ids == ["train_001", "train_002"]
        assert ratio == 0.4
        assert seed == 13
        return ["train_002"]

    fake_splits.build_ratio_subset = build_ratio_subset
    monkeypatch.setitem(sys.modules, "src.data.splits", fake_splits)

    fake_glas = ModuleType("src.data.glas")

    class _FakePath:
        def __init__(self, stem: str) -> None:
            self.stem = stem

    class _FakeGlaSDataset:
        def __init__(
            self, *, images_dir, masks_dir, image_size, class_values, sample_ids=None
        ) -> None:
            assert image_size == 256
            assert class_values == {"background": 0, "gland": 1}
            if sample_ids is not None:
                stems = list(sample_ids)
            elif Path(images_dir) == val_images_dir:
                stems = ["val_001"]
            else:
                stems = ["train_001", "train_002"]
            self.image_paths = [_FakePath(stem) for stem in stems]

    fake_glas.GlaSDataset = _FakeGlaSDataset
    monkeypatch.setitem(sys.modules, "src.data.glas", fake_glas)

    from src.data.factory import build_low_data_datasets

    datasets = build_low_data_datasets(config)

    assert datasets.selected_ids == ["train_002"]
    assert [path.stem for path in datasets.train_dataset.image_paths] == ["train_002"]
    assert [path.stem for path in datasets.val_dataset.image_paths] == ["val_001"]
