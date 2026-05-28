from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any, Callable


@dataclass(frozen=True)
class DatasetSpec:
    dataset_name: str
    module_path: str
    class_name: str
    class_values: dict[str, int]
    available: bool
    extract_sample_ids: Callable[[Any], list[str]]


@dataclass(frozen=True)
class LowDataDatasets:
    train_dataset: Any
    val_dataset: Any
    selected_ids: list[str]


def _extract_sample_ids_from_image_path_stems(
    dataset: Any, *, dataset_name: str
) -> list[str]:
    image_paths = getattr(dataset, "image_paths", None)
    if image_paths is None:
        raise ValueError(
            f"Dataset {dataset_name!r} must expose image_paths so low-data sampling can derive sample ids."
        )

    sample_ids: list[str] = []
    for path in image_paths:
        stem = getattr(path, "stem", None)
        if not isinstance(stem, str) or not stem:
            raise ValueError(
                f"Dataset {dataset_name!r} requires each image_paths item to provide a non-empty string .stem sample id."
            )
        sample_ids.append(stem)
    return sample_ids


def resolve_dataset_spec(dataset_name: str | None) -> DatasetSpec:
    normalized_name = str(dataset_name or "isic2018").strip().lower()

    if normalized_name == "isic2018":
        return DatasetSpec(
            dataset_name="isic2018",
            module_path="src.data.isic2018",
            class_name="ISIC2018Dataset",
            class_values={"background": 0, "lesion": 1},
            available=True,
            extract_sample_ids=lambda dataset: _extract_sample_ids_from_image_path_stems(
                dataset, dataset_name="isic2018"
            ),
        )
    if normalized_name == "glas":
        return DatasetSpec(
            dataset_name="glas",
            module_path="src.data.glas",
            class_name="GlaSDataset",
            class_values={"background": 0, "gland": 1},
            available=True,
            extract_sample_ids=lambda dataset: _extract_sample_ids_from_image_path_stems(
                dataset, dataset_name="glas"
            ),
        )

    raise ValueError(
        "Unsupported data.dataset_name. Expected one of ['isic2018', 'glas']; "
        f"got {dataset_name!r}."
    )


def _load_dataset_class(spec: DatasetSpec):
    if not spec.available:
        raise NotImplementedError(
            f"Dataset {spec.dataset_name!r} is reserved for future support and "
            "cannot be constructed until Task 3 implements the dataset class."
        )

    module = import_module(spec.module_path)
    return getattr(module, spec.class_name)


def build_low_data_datasets(config: dict[str, Any]) -> LowDataDatasets:
    paths = config["paths"]
    data = config["data"]
    spec = resolve_dataset_spec(data.get("dataset_name"))
    dataset_class = _load_dataset_class(spec)
    splits_module = import_module("src.data.splits")

    full_train_dataset = dataset_class(
        images_dir=paths["train_images_dir"],
        masks_dir=paths["train_masks_dir"],
        image_size=data["image_size"],
        class_values=spec.class_values,
        sample_ids=None,
    )

    selected_ids = splits_module.build_ratio_subset(
        spec.extract_sample_ids(full_train_dataset),
        ratio=float(data["train_ratio"]),
        seed=int(config["seed"]),
    )

    train_dataset = dataset_class(
        images_dir=paths["train_images_dir"],
        masks_dir=paths["train_masks_dir"],
        image_size=data["image_size"],
        class_values=spec.class_values,
        sample_ids=selected_ids,
    )
    val_dataset = dataset_class(
        images_dir=paths["val_images_dir"],
        masks_dir=paths["val_masks_dir"],
        image_size=data["image_size"],
        class_values=spec.class_values,
        sample_ids=None,
    )

    return LowDataDatasets(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        selected_ids=selected_ids,
    )
