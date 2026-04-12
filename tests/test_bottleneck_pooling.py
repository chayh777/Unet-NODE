from importlib import util
from pathlib import Path

import pytest
import torch


def _load_pooling():
    module_path = Path(__file__).resolve().parents[1] / "src" / "features" / "bottleneck_pooling.py"
    spec = util.spec_from_file_location("bottleneck_pooling", module_path)
    assert spec is not None
    module = util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    assert hasattr(module, "pool_class_embeddings")
    return module.pool_class_embeddings


pool_class_embeddings = _load_pooling()


def test_pool_class_embeddings_returns_one_row_per_present_class():
    bottleneck = torch.tensor(
        [
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
            ]
        ]
    )
    mask = torch.tensor([[[0, 1], [1, 1]]])

    rows = pool_class_embeddings(
        bottleneck=bottleneck,
        mask=mask,
        sample_ids=["sample_1"],
        class_names=["background", "lesion"],
        class_values={"background": 0, "lesion": 1},
        min_mask_pixels=1,
    )

    assert len(rows) == 2
    lesion_row = next(row for row in rows if row["class_name"] == "lesion")
    assert lesion_row["pixel_count"] == 3
    assert lesion_row["embedding"] == [
        (2.0 + 3.0 + 4.0) / 3,
        (6.0 + 7.0 + 8.0) / 3,
    ]


def test_pool_class_embeddings_skips_small_masks():
    bottleneck = torch.ones((1, 2, 2, 2))
    mask = torch.zeros((1, 2, 2), dtype=torch.long)

    rows = pool_class_embeddings(
        bottleneck=bottleneck,
        mask=mask,
        sample_ids=["sample_1"],
        class_names=["lesion"],
        class_values={"lesion": 1},
        min_mask_pixels=1,
    )

    assert rows == []


def test_pool_class_embeddings_supports_mask_resize():
    bottleneck = torch.arange(8.0).reshape(1, 2, 2, 2)
    mask = torch.tensor(
        [
            [
                [1, 1, 0, 0],
                [1, 1, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ]
        ],
        dtype=torch.long,
    )

    rows = pool_class_embeddings(
        bottleneck=bottleneck,
        mask=mask,
        sample_ids=["sample_1"],
        class_names=["background", "lesion"],
        class_values={"background": 0, "lesion": 1},
        min_mask_pixels=1,
    )

    lesion = next(row for row in rows if row["class_name"] == "lesion")
    assert lesion["pixel_count"] == 1
    assert lesion["embedding"] == [0.0, 4.0]


def test_pool_class_embeddings_batch_mismatch():
    bottleneck = torch.ones((2, 2, 2, 2))
    mask = torch.ones((1, 2, 2), dtype=torch.long)
    with pytest.raises(ValueError):
        pool_class_embeddings(
            bottleneck=bottleneck,
            mask=mask,
            sample_ids=["a"],
            class_names=["lesion"],
            class_values={"lesion": 1},
            min_mask_pixels=1,
        )


def test_pool_class_embeddings_class_key_missing():
    bottleneck = torch.ones((1, 2, 2, 2))
    mask = torch.ones((1, 2, 2), dtype=torch.long)
    with pytest.raises(KeyError):
        pool_class_embeddings(
            bottleneck=bottleneck,
            mask=mask,
            sample_ids=["a"],
            class_names=["lesion", "background"],
            class_values={"lesion": 1},
            min_mask_pixels=1,
        )


def test_pool_class_embeddings_requires_integer_mask():
    bottleneck = torch.ones((1, 2, 2, 2))
    mask = torch.full((1, 2, 2), 0.5)
    with pytest.raises(ValueError):
        pool_class_embeddings(
            bottleneck=bottleneck,
            mask=mask,
            sample_ids=["a"],
            class_names=["lesion"],
            class_values={"lesion": 1},
            min_mask_pixels=1,
        )


def test_pool_class_embeddings_min_mask_pixels_validation():
    bottleneck = torch.ones((1, 2, 2, 2))
    mask = torch.zeros((1, 2, 2), dtype=torch.long)
    with pytest.raises(ValueError):
        pool_class_embeddings(
            bottleneck=bottleneck,
            mask=mask,
            sample_ids=["a"],
            class_names=["lesion"],
            class_values={"lesion": 1},
            min_mask_pixels=0,
        )
