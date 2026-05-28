from importlib import util
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image


def _load_glas_dataset():
    module_path = Path(__file__).resolve().parents[1] / "src" / "data" / "glas.py"
    spec = util.spec_from_file_location("glas", module_path)
    assert spec is not None, "Failed to create import spec for GlaSDataset"
    module = util.module_from_spec(spec)
    assert spec.loader is not None, "Spec loader missing for GlaSDataset"
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    assert hasattr(module, "GlaSDataset"), "Loader module missing GlaSDataset"
    return module.GlaSDataset


def _write_pair(
    images_dir: Path,
    masks_dir: Path,
    sample_id: str,
    *,
    image_ext: str = ".png",
    mask_name: str | None = None,
    foreground: bool = True,
) -> None:
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    image = np.zeros((8, 8, 3), dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=np.uint8)
    if foreground:
        mask[2:6, 2:6] = 255

    Image.fromarray(image).save(images_dir / f"{sample_id}{image_ext}")
    Image.fromarray(mask).save(masks_dir / (mask_name or f"{sample_id}_anno.bmp"))


def test_glas_dataset_returns_expected_sample(tmp_path: Path):
    GlaSDataset = _load_glas_dataset()

    images_dir = tmp_path / "images"
    masks_dir = tmp_path / "masks"
    _write_pair(images_dir, masks_dir, "train_001")

    dataset = GlaSDataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        image_size=16,
        class_values={"background": 0, "gland": 1},
    )

    sample = dataset[0]

    assert sample["sample_id"] == "train_001"
    assert sample["image"].shape == (3, 16, 16)
    assert sample["image"].dtype == torch.float32
    assert sample["mask"].shape == (16, 16)
    assert sample["mask"].dtype == torch.long
    assert set(sample["mask"].unique().tolist()) == {0, 1}
    assert sample["class_presence"]["gland"] is True
    assert sample["class_presence"]["background"] is True


def test_glas_dataset_can_filter_by_sample_ids(tmp_path: Path):
    GlaSDataset = _load_glas_dataset()

    images_dir = tmp_path / "images"
    masks_dir = tmp_path / "masks"
    _write_pair(images_dir, masks_dir, "train_001")
    _write_pair(images_dir, masks_dir, "train_002")

    dataset = GlaSDataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        image_size=8,
        class_values={"background": 0, "gland": 1},
        sample_ids=["train_002"],
    )

    assert len(dataset) == 1
    assert dataset[0]["sample_id"] == "train_002"


def test_glas_dataset_supports_anno_mask_naming(tmp_path: Path):
    GlaSDataset = _load_glas_dataset()

    images_dir = tmp_path / "images"
    masks_dir = tmp_path / "masks"
    _write_pair(images_dir, masks_dir, "train_010", mask_name="train_010_anno.bmp")

    dataset = GlaSDataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        image_size=8,
        class_values={"background": 0, "gland": 1},
    )

    sample = dataset[0]
    assert sample["sample_id"] == "train_010"
    assert sample["class_presence"]["gland"] is True


def test_glas_dataset_rejects_invalid_class_values(tmp_path: Path):
    GlaSDataset = _load_glas_dataset()

    with pytest.raises(ValueError):
        GlaSDataset(
            images_dir=tmp_path / "images",
            masks_dir=tmp_path / "masks",
            image_size=8,
            class_values={"background": 0, "gland": 2},
        )


def test_glas_dataset_handles_all_background_mask(tmp_path: Path):
    GlaSDataset = _load_glas_dataset()

    images_dir = tmp_path / "images"
    masks_dir = tmp_path / "masks"
    _write_pair(images_dir, masks_dir, "train_011", foreground=False)

    dataset = GlaSDataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        image_size=8,
        class_values={"background": 0, "gland": 1},
    )

    sample = dataset[0]
    assert sample["class_presence"]["background"] is True
    assert sample["class_presence"]["gland"] is False
    assert set(sample["mask"].unique().tolist()) == {0}
