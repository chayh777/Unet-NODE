from importlib import util
from pathlib import Path
import subprocess
import sys


def _load_runner_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "run_standard_unet_multiseed.py"
    spec = util.spec_from_file_location("_standard_unet_multiseed_runner", module_path)
    assert spec is not None
    module = util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def test_select_entries_filters_dataset_and_methods():
    module = _load_runner_module()

    entries = module.select_entries("glas", ["plain", "c_zero_last_steps16"])

    assert [entry.dataset for entry in entries] == ["glas", "glas"]
    assert [entry.method for entry in entries] == ["plain", "c_zero_last_steps16"]


def test_build_seeded_config_sets_seed_artifacts_and_experiment_metadata():
    module = _load_runner_module()
    entry = module.select_entries("isic2018", ["output_node"])[0]

    config = module.build_seeded_config(entry, 2, "artifacts")

    assert config["seed"] == 2
    assert config["model"]["architecture"] == "standard_unet"
    assert config["paths"]["artifacts_dir"] == (
        "artifacts/isic2018_standard_unet_multiseed/output_node_seed2"
    )
    assert config["experiment"]["name"] == "isic2018_standard_unet_output_node_seed2"
    assert config["experiment"]["group"] == "C"


def test_dry_run_prints_matrix_without_writing_configs(tmp_path: Path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_standard_unet_multiseed.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--dataset",
            "glas",
            "--methods",
            "plain",
            "--seeds",
            "0",
            "--artifacts-root",
            str(tmp_path / "artifacts"),
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "glas:plain:seed0" in result.stdout
    assert not (tmp_path / "artifacts").exists()
