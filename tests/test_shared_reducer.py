import pandas as pd
from importlib import util
from pathlib import Path


def _load_reduce_module():
    path = Path(__file__).resolve().parents[1] / "src" / "analysis" / "reduce_and_plot.py"
    spec = util.spec_from_file_location("reduce_and_plot", path)
    assert spec is not None
    module = util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module.build_shared_projection


build_shared_projection = _load_reduce_module()


def test_build_shared_projection_preserves_state_labels(tmp_path):
    df = pd.DataFrame(
        [
            {"sample_id": "a", "state": "before", "class_name": "lesion", "embedding_0000": 0.0, "embedding_0001": 0.0},
            {"sample_id": "b", "state": "after", "class_name": "lesion", "embedding_0000": 1.0, "embedding_0001": 1.0},
        ]
    )

    projected = build_shared_projection(df, pca_components=2, umap_neighbors=2, umap_min_dist=0.1, random_state=42)

    assert list(projected["state"]) == ["before", "after"]
    assert {"x", "y"}.issubset(projected.columns)


def test_shared_projection_difference_detects_combined_fit():
    df = pd.DataFrame(
        [
            {"sample_id": "before_1", "state": "before", "class_name": "lesion", "embedding_0000": 0.0, "embedding_0001": 0.0},
            {"sample_id": "before_2", "state": "before", "class_name": "lesion", "embedding_0000": 0.0, "embedding_0001": 0.0},
            {"sample_id": "after_1", "state": "after", "class_name": "lesion", "embedding_0000": 10.0, "embedding_0001": 10.0},
            {"sample_id": "after_2", "state": "after", "class_name": "lesion", "embedding_0000": 10.0, "embedding_0001": 10.0},
        ]
    )

    projected = build_shared_projection(df, pca_components=2, umap_neighbors=2, umap_min_dist=0.1, random_state=42)
    before_mean = projected[projected["state"] == "before"]["x"].mean()
    after_mean = projected[projected["state"] == "after"]["x"].mean()

    assert abs(after_mean - before_mean) > 1.0
