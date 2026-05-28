from __future__ import annotations

import json
from pathlib import Path
from types import ModuleType
import sys

import torch


def _ensure_package(name: str, package_path: Path) -> None:
    if name in sys.modules:
        return
    pkg = ModuleType(name)
    pkg.__path__ = [str(package_path)]  # type: ignore[attr-defined]
    sys.modules[name] = pkg


def _ensure_src_training_importable() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    training_dir = src_dir / "training"

    if "src" not in sys.modules:
        _ensure_package("src", src_dir)
    else:
        mod = sys.modules["src"]
        if not hasattr(mod, "__path__"):
            _ensure_package("src", src_dir)
        else:
            paths = list(getattr(mod, "__path__"))  # type: ignore[arg-type]
            if str(src_dir) not in paths:
                paths.insert(0, str(src_dir))
                setattr(mod, "__path__", paths)

    if "src.training" not in sys.modules:
        _ensure_package("src.training", training_dir)
    else:
        mod = sys.modules["src.training"]
        if not hasattr(mod, "__path__"):
            _ensure_package("src.training", training_dir)
        else:
            paths = list(getattr(mod, "__path__"))  # type: ignore[arg-type]
            if str(training_dir) not in paths:
                paths.insert(0, str(training_dir))
                setattr(mod, "__path__", paths)


_ensure_src_training_importable()


class _TinySegModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return x[:, :1] * self.scale


def _batch():
    image = torch.ones((1, 3, 4, 4), dtype=torch.float32)
    mask = torch.ones((1, 4, 4), dtype=torch.long)
    return {"image": image, "mask": mask}


def test_fit_can_skip_best_checkpoint_and_still_write_timing_metrics(tmp_path: Path):
    from src.training.engine import fit

    model = _TinySegModel()

    class _Optimizer:
        def __init__(self, params) -> None:
            self.params = list(params)

        def zero_grad(self) -> None:
            for param in self.params:
                param.grad = None

        def step(self) -> None:
            return None

    optimizer = _Optimizer(model.parameters())
    train_loader = [_batch()]
    val_loader = [_batch()]

    best_path = fit(
        model,
        train_loader,
        val_loader,
        optimizer,
        epochs=2,
        patience=2,
        output_dir=tmp_path,
        device="cpu",
        save_best_checkpoint=False,
    )

    assert best_path is None
    assert not (tmp_path / "best.pt").exists()

    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["checkpoint_saved"] is False
    assert metrics["best_checkpoint"] is None
    assert metrics["best_epoch"] == 1
    assert metrics["duration_sec"] >= 0.0
    assert metrics["avg_epoch_sec"] >= 0.0


def test_compute_regularization_loss_returns_zero_when_disabled():
    from src.training.engine import compute_regularization_loss

    loss = compute_regularization_loss(
        model_output=object(),
        regularization={"type": "none", "weight": 0.0},
    )

    assert torch.is_tensor(loss)
    assert float(loss.item()) == 0.0


def test_run_epoch_includes_kinetic_regularization_when_present():
    from src.training.engine import run_epoch

    class _Output:
        def __init__(self, logits):
            self.logits = logits
            self.node_diagnostics = {
                "kinetic_terms": [torch.tensor(2.0), torch.tensor(4.0)]
            }

    class _Model(torch.nn.Module):
        def forward(self, x):
            return _Output(torch.zeros(x.shape[0], 1, x.shape[2], x.shape[3]))

    loader = [
        {
            "image": torch.randn(2, 3, 16, 16),
            "mask": torch.zeros(2, 16, 16),
        }
    ]

    metrics = run_epoch(
        model=_Model(),
        loader=loader,
        optimizer=None,
        regularization={"type": "kinetic", "weight": 0.5},
    )

    assert "reg_loss" in metrics
    assert metrics["reg_loss"] > 0.0
    assert "task_loss" in metrics
