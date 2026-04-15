from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from collections.abc import Set


class ISIC2018Dataset(Dataset):
    def __init__(self, images_dir, masks_dir, image_size, class_values, sample_ids=None):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.image_size = image_size
        self.class_values = class_values
        required_names = {"background", "lesion"}
        provided_names = set(class_values.keys())
        if provided_names != required_names:
            raise ValueError(
                f"ISIC2018Dataset requires class_values to be exactly {sorted(required_names)}; "
                f"got {sorted(provided_names)}"
            )
        if class_values.get("background") != 0 or class_values.get("lesion") != 1:
            raise ValueError(
                "ISIC2018Dataset currently requires 'background': 0 and 'lesion': 1 mappings."
            )
        all_image_paths = sorted(self.images_dir.glob("*.jpg"))
        if sample_ids is None:
            self.image_paths = all_image_paths
        else:
            # Fixed subset loading: filter by filename stem (sample_id).
            if isinstance(sample_ids, (str, bytes)):
                requested_ids = [sample_ids]
            elif isinstance(sample_ids, Set):
                # Unordered collections should be normalized deterministically.
                requested_ids = sorted(sample_ids)
            else:
                requested_ids = list(sample_ids)

            # Deduplicate while preserving sequence order.
            seen = set()
            deduped_ids = []
            for sample_id in requested_ids:
                if sample_id not in seen:
                    seen.add(sample_id)
                    deduped_ids.append(sample_id)
            requested_ids = deduped_ids

            by_stem = {path.stem: path for path in all_image_paths}
            missing = [sample_id for sample_id in requested_ids if sample_id not in by_stem]
            if missing:
                raise ValueError(
                    "Requested sample_ids are missing from images_dir: "
                    + ", ".join(missing)
                )
            self.image_paths = [by_stem[sample_id] for sample_id in requested_ids]

    def _resolve_mask_path(self, sample_id: str) -> Path:
        candidates = [
            self.masks_dir / f"{sample_id}.png",
            self.masks_dir / f"{sample_id}_segmentation.png",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            f"Missing mask for sample {sample_id}. Checked: "
            + ", ".join(str(path) for path in candidates)
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        sample_id = image_path.stem
        mask_path = self._resolve_mask_path(sample_id)

        with Image.open(image_path) as image_file:
            image = (
                image_file.convert("RGB")
                .resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
            )
        with Image.open(mask_path) as mask_file:
            mask = (
                mask_file.convert("L")
                .resize((self.image_size, self.image_size), Image.Resampling.NEAREST)
            )

        image_np = np.asarray(image, dtype=np.float32) / 255.0
        mask_np = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8)

        class_presence = {
            class_name: bool((mask_np == class_value).any())
            for class_name, class_value in self.class_values.items()
        }

        return {
            "sample_id": sample_id,
            "image": torch.from_numpy(image_np).permute(2, 0, 1),
            # Preserve historical mask contract: 2D long tensor with values {0, 1}.
            "mask": torch.from_numpy(mask_np).long(),
            "class_presence": class_presence,
        }
