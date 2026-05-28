from __future__ import annotations

from importlib import util
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest


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

    def export_group_geometry(*, config, group, checkpoint_path, noise_sigma):
        calls["export"] = {
            "config": config,
            "group": group,
            "checkpoint_path": Path(checkpoint_path),
            "noise_sigma": noise_sigma,
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
    assert calls["export"]["noise_sigma"] == 0.0
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


def test_plot_report_results_cli_passes_args_to_writer(
    tmp_path: Path, monkeypatch
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    analysis_dir = src_dir / "analysis"

    _ensure_package("src", src_dir)
    _ensure_package("src.analysis", analysis_dir)

    calls: dict[str, object] = {}
    report_module = ModuleType("src.analysis.report_visualization")

    def write_report_visualizations(*, artifacts_dir, output_dir):
        calls["report"] = {
            "artifacts_dir": Path(artifacts_dir),
            "output_dir": Path(output_dir),
        }
        return Path(output_dir)

    report_module.write_report_visualizations = write_report_visualizations
    monkeypatch.setitem(sys.modules, "src.analysis.report_visualization", report_module)

    module = _load_script_module(
        "scripts.plot_report_results", "scripts/plot_report_results.py"
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "plot_report_results.py",
            "--artifacts-dir",
            str(tmp_path / "artifacts"),
            "--output-dir",
            str(tmp_path / "figures"),
        ],
    )

    module.main()

    assert calls["report"] == {
        "artifacts_dir": tmp_path / "artifacts",
        "output_dir": tmp_path / "figures",
    }


def test_run_low_data_experiment_cli_handles_glas_config(
    tmp_path: Path, monkeypatch
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    experiments_dir = src_dir / "experiments"

    _ensure_package("src", src_dir)
    _ensure_package("src.experiments", experiments_dir)

    calls: dict[str, object] = {}
    runner_module = ModuleType("src.experiments.low_data_runner")

    def run_group(config_path, group):
        calls["run_group"] = {"config_path": Path(config_path), "group": group}
        return tmp_path / "artifacts" / "group_c" / "best.pt"

    runner_module.run_group = run_group
    monkeypatch.setitem(sys.modules, "src.experiments.low_data_runner", runner_module)

    module = _load_script_module(
        "scripts.run_low_data_experiment", "scripts/run_low_data_experiment.py"
    )
    glas_config = tmp_path / "glas_config.yaml"
    glas_config.write_text("seed: 1\n", encoding="utf-8")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_low_data_experiment.py",
            "--config",
            str(glas_config),
            "--group",
            "C",
        ],
    )

    module.main()

    assert calls["run_group"] == {"config_path": glas_config, "group": "C"}


def test_run_low_data_geometry_cli_fails_clearly_without_saved_checkpoint(
    tmp_path: Path, monkeypatch
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    experiments_dir = src_dir / "experiments"
    analysis_dir = src_dir / "analysis"

    _ensure_package("src", src_dir)
    _ensure_package("src.experiments", experiments_dir)
    _ensure_package("src.analysis", analysis_dir)

    artifacts_dir = tmp_path / "artifacts"
    group_dir = artifacts_dir / "group_c"
    group_dir.mkdir(parents=True, exist_ok=True)
    (group_dir / "metrics.json").write_text(
        json.dumps({"checkpoint_saved": False, "best_checkpoint": None}),
        encoding="utf-8",
    )

    runner_module = ModuleType("src.experiments.low_data_runner")
    runner_module.load_config = lambda config_path: {
        "paths": {"artifacts_dir": str(artifacts_dir)}
    }
    monkeypatch.setitem(sys.modules, "src.experiments.low_data_runner", runner_module)

    geometry_module = ModuleType("src.analysis.low_data_geometry")
    geometry_module.export_group_geometry = lambda **kwargs: (_ for _ in ()).throw(
        AssertionError("export_group_geometry should not be called without a checkpoint")
    )
    monkeypatch.setitem(sys.modules, "src.analysis.low_data_geometry", geometry_module)

    reduce_module = ModuleType("src.analysis.reduce_and_plot")
    reduce_module.run_low_data_geometry_plot = lambda **kwargs: None
    monkeypatch.setitem(sys.modules, "src.analysis.reduce_and_plot", reduce_module)

    module = _load_script_module(
        "scripts.run_low_data_geometry", "scripts/run_low_data_geometry.py"
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_low_data_geometry.py", "--config", "config.yaml", "--group", "C"],
    )

    with pytest.raises(FileNotFoundError, match=r"train\.save_best_checkpoint: false"):
        module.main()
