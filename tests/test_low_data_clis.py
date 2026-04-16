from __future__ import annotations

from importlib import util
from pathlib import Path
import sys
from types import ModuleType


def _ensure_package(name: str, package_path: Path) -> None:
    if name in sys.modules:
        return
    pkg = ModuleType(name)
    pkg.__path__ = [str(package_path)]  # type: ignore[attr-defined]
    sys.modules[name] = pkg


def _load_script_module(module_name: str, relative_path: str):
    repo_root = Path(__file__).resolve().parents[1]
    scripts_dir = repo_root / "scripts"
    _ensure_package("scripts", scripts_dir)

    module_path = repo_root / relative_path
    assert module_path.exists(), f"Missing script file: {module_path}"
    spec = util.spec_from_file_location(module_name, module_path)
    assert spec is not None
    module = util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def test_run_low_data_geometry_cli_loads_config_and_wires_export_and_plot(
    tmp_path: Path, monkeypatch
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    experiments_dir = src_dir / "experiments"
    analysis_dir = src_dir / "analysis"

    _ensure_package("src", src_dir)
    _ensure_package("src.experiments", experiments_dir)
    _ensure_package("src.analysis", analysis_dir)

    calls: dict[str, object] = {}
    config = {
        "paths": {"artifacts_dir": str(tmp_path / "artifacts")},
        "geometry_plot": {
            "pca_components": 3,
            "umap_neighbors": 5,
            "umap_min_dist": 0.2,
            "random_state": 7,
            "alpha": 0.6,
            "point_size": 21,
            "dpi": 90,
        },
    }

    runner_module = ModuleType("src.experiments.low_data_runner")

    def load_config(config_path):
        calls["config_path"] = str(config_path)
        return config

    runner_module.load_config = load_config
    monkeypatch.setitem(sys.modules, "src.experiments.low_data_runner", runner_module)

    geometry_module = ModuleType("src.analysis.low_data_geometry")

    def export_group_geometry(*, config, group, checkpoint_path):
        calls["export"] = {
            "config": config,
            "group": group,
            "checkpoint_path": Path(checkpoint_path),
        }
        return (
            tmp_path / "artifacts" / "group_b" / "geometry" / "pre_adapter_embeddings.csv",
            tmp_path / "artifacts" / "group_b" / "geometry" / "post_adapter_embeddings.csv",
        )

    geometry_module.export_group_geometry = export_group_geometry
    monkeypatch.setitem(sys.modules, "src.analysis.low_data_geometry", geometry_module)

    reduce_module = ModuleType("src.analysis.reduce_and_plot")

    def run_low_data_geometry_plot(**kwargs):
        calls["plot"] = kwargs
        return Path(kwargs["output_dir"])

    reduce_module.run_low_data_geometry_plot = run_low_data_geometry_plot
    monkeypatch.setitem(sys.modules, "src.analysis.reduce_and_plot", reduce_module)

    module = _load_script_module(
        "scripts.run_low_data_geometry", "scripts/run_low_data_geometry.py"
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_low_data_geometry.py", "--config", "config.yaml", "--group", "B"],
    )

    module.main()

    assert calls["config_path"] == "config.yaml"
    assert calls["export"]["group"] == "B"
    assert calls["export"]["checkpoint_path"] == (
        tmp_path / "artifacts" / "group_b" / "best.pt"
    )
    assert calls["plot"]["pre_csv"].name == "pre_adapter_embeddings.csv"
    assert calls["plot"]["post_csv"].name == "post_adapter_embeddings.csv"
    assert calls["plot"]["output_dir"] == (
        tmp_path / "artifacts" / "group_b" / "geometry"
    )
    assert calls["plot"]["pca_components"] == 3
    assert calls["plot"]["umap_neighbors"] == 5
    assert calls["plot"]["umap_min_dist"] == 0.2
    assert calls["plot"]["random_state"] == 7
    assert calls["plot"]["alpha"] == 0.6
    assert calls["plot"]["point_size"] == 21
    assert calls["plot"]["dpi"] == 90


def test_plot_low_data_summary_cli_passes_args_to_summary_writer(
    tmp_path: Path, monkeypatch
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    analysis_dir = src_dir / "analysis"

    _ensure_package("src", src_dir)
    _ensure_package("src.analysis", analysis_dir)

    calls: dict[str, object] = {}
    reporting_module = ModuleType("src.analysis.low_data_reporting")

    def write_summary_artifacts(*, artifacts_dir, groups):
        calls["summary"] = {
            "artifacts_dir": Path(artifacts_dir),
            "groups": list(groups),
        }
        return Path(artifacts_dir) / "summary"

    reporting_module.write_summary_artifacts = write_summary_artifacts
    monkeypatch.setitem(sys.modules, "src.analysis.low_data_reporting", reporting_module)

    module = _load_script_module(
        "scripts.plot_low_data_summary", "scripts/plot_low_data_summary.py"
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "plot_low_data_summary.py",
            "--artifacts-dir",
            str(tmp_path / "artifacts"),
            "--groups",
            "A",
            "C",
        ],
    )

    module.main()

    assert calls["summary"] == {
        "artifacts_dir": tmp_path / "artifacts",
        "groups": ["A", "C"],
    }
