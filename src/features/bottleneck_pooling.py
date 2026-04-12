from typing import List

import torch
import torch.nn.functional as F


def pool_class_embeddings(
    bottleneck: torch.Tensor,
    mask: torch.Tensor,
    sample_ids: List[str],
    class_names: List[str],
    class_values: dict,
    min_mask_pixels: int,
) -> List[dict]:
    if bottleneck.dim() != 4:
        raise ValueError("bottleneck tensor must be [B, C, H, W]")
    if mask.dim() != 3:
        raise ValueError("mask tensor must be [B, H, W]")

    batch_size = bottleneck.shape[0]
    if mask.shape[0] != batch_size:
        raise ValueError("mask batch size must match bottleneck batch size")
    if len(sample_ids) != batch_size:
        raise ValueError("len(sample_ids) must match batch size")

    if not all(name in class_values for name in class_names):
        missing = [name for name in class_names if name not in class_values]
        raise KeyError(f"class_names missing from class_values: {missing}")

    if not torch.is_floating_point(mask):
        mask = mask.clone()
    else:
        raise ValueError("mask tensor must be integer labels (not floats)")
    if mask.device != bottleneck.device:
        mask = mask.to(bottleneck.device)

    if min_mask_pixels < 1:
        raise ValueError("min_mask_pixels must be >= 1")

    resized_mask = F.interpolate(mask.unsqueeze(1).float(), size=bottleneck.shape[-2:], mode="nearest")
    resized_mask = resized_mask.squeeze(1).long()

    rows = []
    for batch_index, sample_id in enumerate(sample_ids):
        for class_name in class_names:
            class_value = class_values[class_name]
            pixel_mask = resized_mask[batch_index] == class_value
            pixel_count = int(pixel_mask.sum().item())
            if pixel_count < min_mask_pixels:
                continue

            selected = bottleneck[batch_index][:, pixel_mask]
            if selected.numel() == 0:
                continue
            embedding = selected.mean(dim=1)

            rows.append(
                {
                    "sample_id": sample_id,
                    "class_name": class_name,
                    "pixel_count": pixel_count,
                    "embedding": embedding.detach().cpu().tolist(),
                }
            )

    return rows
