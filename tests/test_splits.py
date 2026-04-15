import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.splits import build_ratio_subset


def test_build_ratio_subset_is_deterministic():
    sample_ids = [f"ISIC_{index:04d}" for index in range(20)]
    first = build_ratio_subset(sample_ids, ratio=0.1, seed=42)
    second = build_ratio_subset(sample_ids, ratio=0.1, seed=42)
    assert first == second
    assert len(first) == 2
