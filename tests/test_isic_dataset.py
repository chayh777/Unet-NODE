from importlib import util
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image


def _load_isic_dataset():
    module_path = Path(__file__).resolve().parents[1] / "src" / "data" / "isic2018.py"
    spec = util.spec_from_file_location("isic2018", module_path)
    assert spec is not None, "Failed to create import spec for ISIC2018Dataset"
    module = util.module_from_spec(spec)
    assert spec.loader is not None, "Spec loader missing for ISIC2018Dataset"
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    assert hasattr(module, "ISIC2018Dataset"), "Loader module missing ISIC2018Dataset"
    return module.ISIC2018Dataset


ISIC2018Dataset = _load_isic_dataset()


def test_isic_dataset_returns_expected_sample(tmp_path: Path):
    images_dir = tmp_path / "images"
    masks_dir = tmp_path / "masks"
    images_dir.mkdir()
    masks_dir.mkdir()

    image = np.zeros((8, 8, 3), dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:6, 2:6] = 255

    Image.fromarray(image).save(images_dir / "ISIC_0001.jpg")
    Image.fromarray(mask).save(masks_dir / "ISIC_0001.png")

    dataset = ISIC2018Dataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        image_size=8,
        class_values={"background": 0, "lesion": 1},
    )

    sample = dataset[0]

    assert sample["sample_id"] == "ISIC_0001"
    assert sample["image"].shape == (3, 8, 8)
    assert sample["image"].dtype == torch.float32
    assert sample["mask"].shape == (8, 8)
    assert sample["mask"].dtype == torch.long
    assert set(sample["mask"].unique().tolist()) == {0, 1}
    assert sample["class_presence"]["lesion"] is True
    assert sample["class_presence"]["background"] is True


def test_isic_dataset_rejects_nonbinary_class_values(tmp_path: Path):
    with pytest.raises(ValueError):
        ISIC2018Dataset(
            images_dir=tmp_path / "images",
            masks_dir=tmp_path / "masks",
            image_size=8,
            class_values={"background": 0, "lesion": 2},
        )


def test_isic_dataset_missing_mask(tmp_path: Path):
    images_dir = tmp_path / "images"
    masks_dir = tmp_path / "masks"
    images_dir.mkdir()
    masks_dir.mkdir()

    image = np.zeros((8, 8, 3), dtype=np.uint8)
    Image.fromarray(image).save(images_dir / "ISIC_0002.jpg")

    dataset = ISIC2018Dataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        image_size=8,
        class_values={"background": 0, "lesion": 1},
    )

    with pytest.raises(FileNotFoundError):
        _ = dataset[0]


def test_isic_dataset_supports_segmentation_suffix(tmp_path: Path):
    images_dir = tmp_path / "images"
    masks_dir = tmp_path / "masks"
    images_dir.mkdir()
    masks_dir.mkdir()

    image = np.zeros((8, 8, 3), dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[1:4, 1:4] = 255

    Image.fromarray(image).save(images_dir / "ISIC_0004.jpg")
    Image.fromarray(mask).save(masks_dir / "ISIC_0004_segmentation.png")

    dataset = ISIC2018Dataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        image_size=8,
        class_values={"background": 0, "lesion": 1},
    )

    sample = dataset[0]
    assert sample["sample_id"] == "ISIC_0004"
    assert sample["class_presence"]["lesion"] is True


@pytest.mark.parametrize(
    "class_values",
    [
        {"background": 0},  # missing lesion
        {"lesion": 1},  # missing background
        {"background": 1, "lesion": 1},
    ],
)
def test_isic_dataset_rejects_incomplete_class_values(tmp_path: Path, class_values):
    with pytest.raises(ValueError):
        ISIC2018Dataset(
            images_dir=tmp_path / "images",
            masks_dir=tmp_path / "masks",
            image_size=8,
            class_values=class_values,
        )


def test_isic_dataset_rejects_swapped_mapping(tmp_path: Path):
    with pytest.raises(ValueError):
        ISIC2018Dataset(
            images_dir=tmp_path / "images",
            masks_dir=tmp_path / "masks",
            image_size=8,
            class_values={"background": 1, "lesion": 0},
        )


def test_isic_dataset_handles_all_background_mask(tmp_path: Path):
    images_dir = tmp_path / "images"
    masks_dir = tmp_path / "masks"
    images_dir.mkdir()
    masks_dir.mkdir()

    image = np.zeros((8, 8, 3), dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=np.uint8)

    Image.fromarray(image).save(images_dir / "ISIC_0003.jpg")
    Image.fromarray(mask).save(masks_dir / "ISIC_0003.png")

    dataset = ISIC2018Dataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        image_size=8,
        class_values={"background": 0, "lesion": 1},
    )

    sample = dataset[0]
    assert sample["class_presence"]["background"] is True
    assert sample["class_presence"]["lesion"] is False
    assert set(sample["mask"].unique().tolist()) == {0}
