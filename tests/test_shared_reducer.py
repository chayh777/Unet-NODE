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
    return module


reduce_module = _load_reduce_module()
build_shared_projection = reduce_module.build_shared_projection


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


def test_build_shared_projection_preserves_low_data_adapter_state_labels():
    df = pd.DataFrame(
        [
            {"sample_id": "a", "state": "pre_adapter", "class_name": "lesion", "embedding_0000": 0.0, "embedding_0001": 0.0},
            {"sample_id": "b", "state": "post_adapter", "class_name": "lesion", "embedding_0000": 1.0, "embedding_0001": 1.0},
        ]
    )

    projected = build_shared_projection(df, pca_components=2, umap_neighbors=2, umap_min_dist=0.1, random_state=42)

    assert list(projected["state"]) == ["pre_adapter", "post_adapter"]
    assert {"x", "y"}.issubset(projected.columns)


def test_run_low_data_geometry_plot_writes_shared_projection_outputs(tmp_path):
    pre_csv = tmp_path / "pre_adapter_embeddings.csv"
    post_csv = tmp_path / "post_adapter_embeddings.csv"
    output_dir = tmp_path / "geometry"

    pd.DataFrame(
        [
            {"sample_id": "pre_1", "state": "pre_adapter", "class_name": "lesion", "pixel_count": 3, "embedding_0000": 0.0, "embedding_0001": 0.0},
            {"sample_id": "pre_2", "state": "pre_adapter", "class_name": "background", "pixel_count": 5, "embedding_0000": 0.5, "embedding_0001": 0.2},
        ]
    ).to_csv(pre_csv, index=False)
    pd.DataFrame(
        [
            {"sample_id": "post_1", "state": "post_adapter", "class_name": "lesion", "pixel_count": 4, "embedding_0000": 3.0, "embedding_0001": 3.5},
            {"sample_id": "post_2", "state": "post_adapter", "class_name": "background", "pixel_count": 6, "embedding_0000": 3.2, "embedding_0001": 3.8},
        ]
    ).to_csv(post_csv, index=False)

    returned_dir = reduce_module.run_low_data_geometry_plot(
        pre_csv=pre_csv,
        post_csv=post_csv,
        output_dir=output_dir,
        pca_components=2,
        umap_neighbors=2,
        umap_min_dist=0.1,
        random_state=42,
        alpha=0.7,
        point_size=30,
        dpi=72,
    )

    assert returned_dir == output_dir
    assert (output_dir / "bottleneck_before_after_scatter.png").exists()
    assert (output_dir / "bottleneck_before_after_density.png").exists()
    assert (output_dir / "shared_projection_points.csv").exists()
    assert (output_dir / "geometry_metrics.csv").exists()

    projected = pd.read_csv(output_dir / "shared_projection_points.csv")
    assert set(projected["state"]) == {"pre_adapter", "post_adapter"}
    assert {"x", "y", "pixel_count"}.issubset(projected.columns)

    metrics = pd.read_csv(output_dir / "geometry_metrics.csv")
    assert set(metrics.columns) == {"state", "class_name", "mean_radius"}
    assert set(metrics["state"]) == {"pre_adapter", "post_adapter"}
