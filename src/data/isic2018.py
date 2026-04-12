from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class ISIC2018Dataset(Dataset):
    def __init__(self, images_dir, masks_dir, image_size, class_values):
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
        self.image_paths = sorted(self.images_dir.glob("*.jpg"))

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
        mask_np = (
            (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8)
        )

        class_presence = {
            class_name: bool((mask_np == class_value).any())
            for class_name, class_value in self.class_values.items()
        }

        return {
            "sample_id": sample_id,
            "image": torch.from_numpy(image_np).permute(2, 0, 1),
            "mask": torch.from_numpy(mask_np).long(),
            "class_presence": class_presence,
        }
