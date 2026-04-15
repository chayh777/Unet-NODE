import sys
from pathlib import Path

# Ensure repo root is importable so `src.*` works without installing the package.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.models.segmentation_model import build_segmentation_model


def test_build_segmentation_model_freezes_encoder():
    model = build_segmentation_model(
        encoder_name="resnet18",
        encoder_weights=None,
        in_channels=3,
        num_classes=1,
        adapter_type="node",
        bottleneck_channels=512,
        adapter_hidden_channels=512,
        freeze_encoder=True,
        node_steps=4,
        node_step_size=0.25,
    )
    encoder_flags = [
        p.requires_grad for name, p in model.named_parameters() if "encoder" in name
    ]
    assert encoder_flags
    assert all(flag is False for flag in encoder_flags)
