from importlib import util
from pathlib import Path


def _load_build_ratio_subset():
    module_path = Path(__file__).resolve().parents[1] / "src" / "data" / "splits.py"
    spec = util.spec_from_file_location("splits", module_path)
    assert spec is not None, "Failed to create import spec for split helpers"
    module = util.module_from_spec(spec)
    assert spec.loader is not None, "Spec loader missing for split helpers"
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    assert hasattr(module, "build_ratio_subset"), "split module missing build_ratio_subset"
    return module.build_ratio_subset


build_ratio_subset = _load_build_ratio_subset()


def test_build_ratio_subset_is_deterministic():
    sample_ids = [f"ISIC_{index:04d}" for index in range(20)]
    first = build_ratio_subset(sample_ids, ratio=0.11, seed=42)
    second = build_ratio_subset(sample_ids, ratio=0.11, seed=42)
    assert first == second
    assert len(first) == 3


def test_build_ratio_subset_varies_across_seeds():
    sample_ids = [f"ISIC_{index:04d}" for index in range(100)]
    first = build_ratio_subset(sample_ids, ratio=0.1, seed=42)
    second = build_ratio_subset(sample_ids, ratio=0.1, seed=7)
    assert first != second
