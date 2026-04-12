from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.extract_embeddings import extract_before_after
from src.analysis.reduce_and_plot import run_reduction_and_plot
from src.models.unet import SimpleUNet
from src.training.finetune import run_finetuning
from src.utils.io import ensure_dir, load_config, save_checkpoint


def _save_pretrained_checkpoint(config: dict) -> Path:
    checkpoints_dir = Path(config["paths"]["artifacts_dir"]) / "checkpoints"
    ensure_dir(checkpoints_dir)
    checkpoint_path = checkpoints_dir / config["training"]["pretrained_checkpoint_name"]

    model = SimpleUNet(
        encoder_name=config["model"]["encoder_name"],
        encoder_weights=config["model"]["encoder_weights"],
        in_channels=config["model"]["in_channels"],
        num_classes=config["model"]["num_classes"],
    )
    if not checkpoint_path.exists():
        save_checkpoint(checkpoint_path, model)
        print(f"Saved pretrained checkpoint to {checkpoint_path}")
    else:
        print(f"Pretrained checkpoint already exists at {checkpoint_path}")
    return checkpoint_path


def main(config_path: str | Path) -> None:
    config = load_config(config_path)
    if "paths" not in config or "training" not in config:
        raise ValueError("Config must include 'paths' and 'training' sections.")
    if "artifacts_dir" not in config["paths"]:
        raise ValueError("paths section must define artifacts_dir.")
    required_training = {"pretrained_checkpoint_name", "finetuned_checkpoint_name"}
    if not required_training.issubset(config["training"]):
        raise ValueError("training section must include pretrained_checkpoint_name and finetuned_checkpoint_name.")

    checkpoints_dir = Path(config["paths"]["artifacts_dir"]) / "checkpoints"
    ensure_dir(checkpoints_dir)
    expected_finetuned = checkpoints_dir / config["training"]["finetuned_checkpoint_name"]

    pretrained_checkpoint = _save_pretrained_checkpoint(config)
    finetuned_checkpoint = run_finetuning(config_path)
    finetuned_path = Path(finetuned_checkpoint)
    if finetuned_path.resolve() != expected_finetuned.resolve():
        print(f"Warning: finetuned checkpoint saved to {finetuned_path}; expected {expected_finetuned}.")
    if not expected_finetuned.exists():
        raise FileNotFoundError(f"Finetuned checkpoint not found at {expected_finetuned}")
    print(f"Finetuned checkpoint located at {expected_finetuned}")

    before_output = "before_embeddings.csv"
    after_output = "after_embeddings.csv"

    before_csv, after_csv = extract_before_after(
        config_path,
        pretrained_checkpoint,
        finetuned_checkpoint,
        before_output,
        after_output,
    )

    print(f"Extracted embeddings: before->{before_csv}, after->{after_csv}")

    run_reduction_and_plot(
        before_csv=before_csv,
        after_csv=after_csv,
        artifacts_dir=config["paths"]["artifacts_dir"],
        pca_components=config["reduction"]["pca_components"],
        umap_neighbors=config["reduction"]["umap_neighbors"],
        umap_min_dist=config["reduction"]["umap_min_dist"],
        random_state=config["reduction"]["random_state"],
        alpha=config["plotting"]["alpha"],
        point_size=config["plotting"]["point_size"],
        dpi=config["plotting"]["dpi"],
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the full bottleneck visualization experiment.")
    parser.add_argument("--config", required=True, help="Path to the experiment config.")
    args = parser.parse_args()
    main(args.config)
