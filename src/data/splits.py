from __future__ import annotations

import csv
import random
from pathlib import Path


def build_ratio_subset(sample_ids: list[str], ratio: float, seed: int) -> list[str]:
    if not sample_ids:
        raise ValueError("sample_ids must be non-empty")
    if not (0 < ratio <= 1):
        raise ValueError("ratio must be in the interval (0, 1]")

    sorted_ids = sorted(sample_ids)
    target_count = max(1, round(len(sorted_ids) * ratio))
    selected = random.Random(seed).sample(sorted_ids, target_count)
    return sorted(selected)


def save_split_manifest(sample_ids: list[str], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["sample_id"])
        for sample_id in sample_ids:
            writer.writerow([sample_id])
