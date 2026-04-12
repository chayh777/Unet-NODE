# Bottleneck Visualization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a one-shot visualization experiment that compares pre-finetuning vs post-finetuning U-Net bottleneck features on ISIC 2018, where each point is one class mask pooled into a single 1024-d embedding.

**Architecture:** Run one ImageNet-pretrained U-Net twice on the same ISIC 2018 split: once before finetuning and once after full-parameter finetuning. For each sample and class mask, pool the bottleneck feature map with masked average pooling, concatenate all pooled vectors, fit one shared dimensionality reducer on the combined before/after matrix, and render matched scatter and density plots focused on lesion-first analysis.

**Tech Stack:** Python, PyTorch, torchvision/timm-based encoder initialization, numpy, pandas, scikit-learn, umap-learn, matplotlib/seaborn, pytest

---

## Planned File Structure

- Create: `README.md`
  Purpose: brief setup notes and the exact commands for this visualization experiment.
- Create: `requirements.txt`
  Purpose: pin the minimal Python dependencies for training, feature extraction, plotting, and testing.
- Create: `configs/experiments/isic2018_bottleneck_visualization.yaml`
  Purpose: central experiment config for paths, model settings, finetuning hyperparameters, extraction settings, and plotting parameters.
- Create: `src/data/isic2018.py`
  Purpose: ISIC 2018 dataset loader that returns image, segmentation mask, sample id, and class-presence metadata.
- Create: `src/models/unet.py`
  Purpose: U-Net construction with ImageNet-pretrained encoder support and an explicit bottleneck forward hook/output.
- Create: `src/features/bottleneck_pooling.py`
  Purpose: convert bottleneck feature maps plus segmentation masks into one pooled embedding per class per sample.
- Create: `src/training/finetune.py`
  Purpose: full-parameter finetuning entry point, checkpoint saving, and metric logging.
- Create: `src/analysis/extract_embeddings.py`
  Purpose: run one model checkpoint over a fixed ISIC split and save per-point embeddings plus metadata.
- Create: `src/analysis/reduce_and_plot.py`
  Purpose: fit shared PCA and UMAP on the concatenated before/after embeddings and save all figures plus simple compactness metrics.
- Create: `src/utils/io.py`
  Purpose: path creation, config loading, checkpoint metadata, and csv/json saving helpers.
- Create: `scripts/run_bottleneck_visualization.py`
  Purpose: orchestrate the full pipeline end-to-end from config to final figures.
- Create: `tests/test_bottleneck_pooling.py`
  Purpose: verify masked pooling and class filtering behavior.
- Create: `tests/test_shared_reducer.py`
  Purpose: verify before/after embeddings use one shared fitted reducer and keep aligned metadata.
- Create: `tests/test_isic_dataset.py`
  Purpose: verify dataset item contract and mask shape/label handling.

## Execution Conventions

- Use one fixed random seed for the first pass: `42`
- Use one fixed evaluation split for both before and after extraction
- Save artifacts under:
  - `artifacts/checkpoints/`
  - `artifacts/embeddings/`
  - `artifacts/plots/`
  - `artifacts/metrics/`
- Main decision figure set:
  - `artifacts/plots/lesion_scatter_before.png`
  - `artifacts/plots/lesion_scatter_after.png`
  - `artifacts/plots/lesion_density_before.png`
  - `artifacts/plots/lesion_density_after.png`
  - `artifacts/plots/lesion_background_scatter_before_after.png`
  - `artifacts/metrics/compactness_summary.csv`

### Task 1: Project Skeleton And Dependency Lock

**Files:**
- Create: `README.md`
- Create: `requirements.txt`
- Create: `configs/experiments/isic2018_bottleneck_visualization.yaml`

- [ ] **Step 1: Write the failing smoke test checklist in the README**

```markdown
# U-Net Bottleneck Visualization

## Goal
Produce before/after bottleneck visualizations for ImageNet-pretrained U-Net fully finetuned on ISIC 2018.

## Expected commands
python -m pytest -q
python scripts/run_bottleneck_visualization.py --config configs/experiments/isic2018_bottleneck_visualization.yaml

## Expected outputs
- artifacts/checkpoints/pretrained_encoder_unet.pt
- artifacts/checkpoints/finetuned_unet.pt
- artifacts/embeddings/before_embeddings.csv
- artifacts/embeddings/after_embeddings.csv
- artifacts/plots/lesion_scatter_before.png
- artifacts/plots/lesion_scatter_after.png
- artifacts/plots/lesion_density_before.png
- artifacts/plots/lesion_density_after.png
- artifacts/metrics/compactness_summary.csv
```

- [ ] **Step 2: Add the minimal dependency lock**

```txt
torch
torchvision
timm
numpy
pandas
scikit-learn
umap-learn
matplotlib
seaborn
pyyaml
pytest
Pillow
```

- [ ] **Step 3: Add the experiment config**

```yaml
seed: 42

paths:
  isic_images_dir: data/isic2018/images
  isic_masks_dir: data/isic2018/masks
  artifacts_dir: artifacts

dataset:
  image_size: 256
  class_values:
    background: 0
    lesion: 1
  eval_split_csv: data/isic2018/eval_split.csv

model:
  encoder_name: resnet34
  encoder_weights: imagenet
  in_channels: 3
  num_classes: 2
  bottleneck_channels: 1024

training:
  batch_size: 8
  learning_rate: 0.0001
  epochs: 20
  num_workers: 4
  checkpoint_name: finetuned_unet.pt

extraction:
  batch_size: 8
  include_classes:
    - lesion
    - background
  primary_class: lesion
  min_mask_pixels: 16

reduction:
  pca_components: 30
  umap_neighbors: 20
  umap_min_dist: 0.1
  random_state: 42

plotting:
  alpha: 0.7
  point_size: 20
  dpi: 220
  lesion_color: "#d84b4b"
  background_color: "#4f83cc"
+```

- [ ] **Step 4: Verify the files exist**

Run: `Get-ChildItem README.md, requirements.txt, configs/experiments/isic2018_bottleneck_visualization.yaml`
Expected: three files are listed with non-zero length

- [ ] **Step 5: Commit**

```bash
git add README.md requirements.txt configs/experiments/isic2018_bottleneck_visualization.yaml
git commit -m "chore: scaffold bottleneck visualization experiment"
```

### Task 2: ISIC 2018 Dataset Loader

**Files:**
- Create: `src/data/isic2018.py`
- Create: `tests/test_isic_dataset.py`

- [ ] **Step 1: Write the failing dataset test**

```python
from pathlib import Path

import numpy as np
from PIL import Image

from src.data.isic2018 import ISIC2018Dataset


def test_isic_dataset_returns_image_mask_and_metadata(tmp_path: Path):
    images_dir = tmp_path / "images"
    masks_dir = tmp_path / "masks"
    images_dir.mkdir()
    masks_dir.mkdir()

    image = np.zeros((8, 8, 3), dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:6, 2:6] = 255

    Image.fromarray(image).save(images_dir / "ISIC_0001.jpg")
    Image.fromarray(mask).save(masks_dir / "ISIC_0001.png")

    dataset = ISIC2018Dataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        image_size=8,
        class_values={"background": 0, "lesion": 1},
    )

    sample = dataset[0]

    assert sample["sample_id"] == "ISIC_0001"
    assert tuple(sample["image"].shape) == (3, 8, 8)
    assert tuple(sample["mask"].shape) == (8, 8)
    assert sample["class_presence"]["lesion"] is True
    assert sample["class_presence"]["background"] is True
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_isic_dataset.py -v`
Expected: FAIL with `ModuleNotFoundError` or `ImportError` for `src.data.isic2018`

- [ ] **Step 3: Write the minimal dataset implementation**

```python
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class ISIC2018Dataset(Dataset):
    def __init__(self, images_dir, masks_dir, image_size, class_values):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.image_size = image_size
        self.class_values = class_values
        self.image_paths = sorted(self.images_dir.glob("*.jpg"))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        sample_id = image_path.stem
        mask_path = self.masks_dir / f"{sample_id}.png"

        image = Image.open(image_path).convert("RGB").resize((self.image_size, self.image_size))
        mask = Image.open(mask_path).convert("L").resize((self.image_size, self.image_size))

        image_np = np.asarray(image, dtype=np.float32) / 255.0
        mask_np = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8)

        class_presence = {
            "background": bool((mask_np == 0).any()),
            "lesion": bool((mask_np == 1).any()),
        }

        return {
            "sample_id": sample_id,
            "image": torch.from_numpy(image_np).permute(2, 0, 1),
            "mask": torch.from_numpy(mask_np),
            "class_presence": class_presence,
        }
```

- [ ] **Step 4: Run the dataset test to verify it passes**

Run: `pytest tests/test_isic_dataset.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/data/isic2018.py tests/test_isic_dataset.py
git commit -m "feat: add isic2018 dataset loader"
```

### Task 3: U-Net With Explicit Bottleneck Output

**Files:**
- Create: `src/models/unet.py`

- [ ] **Step 1: Add a construction contract in code comments**

```python
"""
Build a U-Net whose forward pass returns both segmentation logits and the
1024-channel bottleneck feature map used for visualization.
"""
```

- [ ] **Step 2: Implement the minimal model wrapper**

```python
import torch
import torch.nn as nn
import timm


class SimpleUNet(nn.Module):
    def __init__(self, encoder_name="resnet34", encoder_weights="imagenet", num_classes=2):
        super().__init__()
        self.encoder = timm.create_model(
            encoder_name,
            pretrained=encoder_weights == "imagenet",
            features_only=True,
        )
        encoder_channels = self.encoder.feature_info.channels()
        bottleneck_in = encoder_channels[-1]

        self.bottleneck = nn.Conv2d(bottleneck_in, 1024, kernel_size=1)
        self.decoder_head = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1),
        )

    def forward(self, x):
        features = self.encoder(x)
        bottleneck = self.bottleneck(features[-1])
        logits = self.decoder_head(
            torch.nn.functional.interpolate(
                bottleneck, size=x.shape[-2:], mode="bilinear", align_corners=False
            )
        )
        return {"logits": logits, "bottleneck": bottleneck}
```

- [ ] **Step 3: Add a quick import smoke check**

Run: `python -c "from src.models.unet import SimpleUNet; print(SimpleUNet().__class__.__name__)"`
Expected: `SimpleUNet`

- [ ] **Step 4: Commit**

```bash
git add src/models/unet.py
git commit -m "feat: add unet wrapper with bottleneck output"
```

### Task 4: Masked Bottleneck Pooling

**Files:**
- Create: `src/features/bottleneck_pooling.py`
- Create: `tests/test_bottleneck_pooling.py`

- [ ] **Step 1: Write the failing pooling test**

```python
import torch

from src.features.bottleneck_pooling import pool_class_embeddings


def test_pool_class_embeddings_returns_one_vector_per_present_class():
    bottleneck = torch.tensor(
        [[
            [[1.0, 2.0], [3.0, 4.0]],
            [[10.0, 20.0], [30.0, 40.0]],
        ]]
    )
    mask = torch.tensor([[[0, 1], [1, 1]]], dtype=torch.long)

    rows = pool_class_embeddings(
        bottleneck=bottleneck,
        mask=mask,
        sample_ids=["sample_1"],
        class_names=["background", "lesion"],
        class_values={"background": 0, "lesion": 1},
        min_mask_pixels=1,
    )

    assert len(rows) == 2
    lesion = [row for row in rows if row["class_name"] == "lesion"][0]
    assert lesion["sample_id"] == "sample_1"
    assert lesion["embedding"] == [3.0, 30.0]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_bottleneck_pooling.py -v`
Expected: FAIL with `ModuleNotFoundError` or missing function

- [ ] **Step 3: Implement masked average pooling**

```python
import torch
import torch.nn.functional as F


def pool_class_embeddings(bottleneck, mask, sample_ids, class_names, class_values, min_mask_pixels):
    pooled_rows = []
    resized_mask = F.interpolate(mask.unsqueeze(1).float(), size=bottleneck.shape[-2:], mode="nearest").squeeze(1).long()

    for batch_index, sample_id in enumerate(sample_ids):
        for class_name in class_names:
            class_value = class_values[class_name]
            class_mask = resized_mask[batch_index] == class_value
            pixel_count = int(class_mask.sum().item())
            if pixel_count < min_mask_pixels:
                continue

            selected = bottleneck[batch_index, :, class_mask]
            embedding = selected.mean(dim=1)

            pooled_rows.append(
                {
                    "sample_id": sample_id,
                    "class_name": class_name,
                    "pixel_count": pixel_count,
                    "embedding": embedding.detach().cpu().tolist(),
                }
            )

    return pooled_rows
```

- [ ] **Step 4: Run the pooling test to verify it passes**

Run: `pytest tests/test_bottleneck_pooling.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/features/bottleneck_pooling.py tests/test_bottleneck_pooling.py
git commit -m "feat: add masked bottleneck pooling"
```

### Task 5: Finetuning Script

**Files:**
- Create: `src/training/finetune.py`
- Modify: `src/models/unet.py`
- Create: `src/utils/io.py`

- [ ] **Step 1: Implement the checkpoint and config helpers**

```python
from pathlib import Path
import json

import yaml


def load_config(path):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(path, payload):
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
```

- [ ] **Step 2: Implement minimal full-parameter finetuning**

```python
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.isic2018 import ISIC2018Dataset
from src.models.unet import SimpleUNet
from src.utils.io import ensure_dir, load_config, save_json


def run_finetuning(config_path):
    config = load_config(config_path)
    dataset = ISIC2018Dataset(
        images_dir=config["paths"]["isic_images_dir"],
        masks_dir=config["paths"]["isic_masks_dir"],
        image_size=config["dataset"]["image_size"],
        class_values=config["dataset"]["class_values"],
    )
    loader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"],
    )

    model = SimpleUNet(
        encoder_name=config["model"]["encoder_name"],
        encoder_weights=config["model"]["encoder_weights"],
        num_classes=config["model"]["num_classes"],
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])

    model.train()
    epoch_losses = []
    for _ in range(config["training"]["epochs"]):
        running_loss = 0.0
        for batch in loader:
            optimizer.zero_grad()
            output = model(batch["image"].float())
            loss = criterion(output["logits"], batch["mask"].long())
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())
        epoch_losses.append(running_loss / max(1, len(loader)))

    checkpoints_dir = Path(config["paths"]["artifacts_dir"]) / "checkpoints"
    ensure_dir(checkpoints_dir)
    checkpoint_path = checkpoints_dir / config["training"]["checkpoint_name"]
    torch.save(model.state_dict(), checkpoint_path)
    save_json(checkpoints_dir / "finetune_log.json", {"epoch_losses": epoch_losses})
    return checkpoint_path
```

- [ ] **Step 3: Add command entrypoint**

```python
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run_finetuning(args.config)
```

- [ ] **Step 4: Run a dry import check**

Run: `python -c "from src.training.finetune import run_finetuning; print(callable(run_finetuning))"`
Expected: `True`

- [ ] **Step 5: Commit**

```bash
git add src/training/finetune.py src/models/unet.py src/utils/io.py
git commit -m "feat: add finetuning pipeline"
```

### Task 6: Before And After Embedding Extraction

**Files:**
- Create: `src/analysis/extract_embeddings.py`

- [ ] **Step 1: Implement embedding extraction contract**

```python
"""
Save one CSV row per sample-class point with columns:
sample_id, state, class_name, pixel_count, embedding_0000 ... embedding_1023
"""
```

- [ ] **Step 2: Implement extractor**

```python
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.isic2018 import ISIC2018Dataset
from src.features.bottleneck_pooling import pool_class_embeddings
from src.models.unet import SimpleUNet
from src.utils.io import ensure_dir, load_config


def extract_embeddings(config_path, checkpoint_path, state_name, output_name):
    config = load_config(config_path)
    dataset = ISIC2018Dataset(
        images_dir=config["paths"]["isic_images_dir"],
        masks_dir=config["paths"]["isic_masks_dir"],
        image_size=config["dataset"]["image_size"],
        class_values=config["dataset"]["class_values"],
    )
    loader = DataLoader(dataset, batch_size=config["extraction"]["batch_size"], shuffle=False)

    model = SimpleUNet(
        encoder_name=config["model"]["encoder_name"],
        encoder_weights=config["model"]["encoder_weights"],
        num_classes=config["model"]["num_classes"],
    )
    if checkpoint_path is not None:
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()

    rows = []
    with torch.no_grad():
        for batch in loader:
            output = model(batch["image"].float())
            pooled = pool_class_embeddings(
                bottleneck=output["bottleneck"],
                mask=batch["mask"],
                sample_ids=batch["sample_id"],
                class_names=config["extraction"]["include_classes"],
                class_values=config["dataset"]["class_values"],
                min_mask_pixels=config["extraction"]["min_mask_pixels"],
            )
            for row in pooled:
                record = {
                    "sample_id": row["sample_id"],
                    "state": state_name,
                    "class_name": row["class_name"],
                    "pixel_count": row["pixel_count"],
                }
                record.update(
                    {f"embedding_{index:04d}": value for index, value in enumerate(row["embedding"])}
                )
                rows.append(record)

    output_dir = Path(config["paths"]["artifacts_dir"]) / "embeddings"
    ensure_dir(output_dir)
    output_path = output_dir / output_name
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return output_path
```

- [ ] **Step 3: Add CLI entrypoint**

```python
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--state", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    extract_embeddings(args.config, args.checkpoint, args.state, args.output)
```

- [ ] **Step 4: Run an import smoke test**

Run: `python -c "from src.analysis.extract_embeddings import extract_embeddings; print(callable(extract_embeddings))"`
Expected: `True`

- [ ] **Step 5: Commit**

```bash
git add src/analysis/extract_embeddings.py
git commit -m "feat: add before after embedding extraction"
```

### Task 7: Shared PCA And UMAP Reduction With Plots

**Files:**
- Create: `src/analysis/reduce_and_plot.py`
- Create: `tests/test_shared_reducer.py`

- [ ] **Step 1: Write the failing shared-reducer test**

```python
from pathlib import Path

import pandas as pd

from src.analysis.reduce_and_plot import build_shared_projection


def test_build_shared_projection_preserves_state_labels(tmp_path: Path):
    df = pd.DataFrame(
        [
            {"sample_id": "a", "state": "before", "class_name": "lesion", "embedding_0000": 0.0, "embedding_0001": 0.0},
            {"sample_id": "b", "state": "after", "class_name": "lesion", "embedding_0000": 1.0, "embedding_0001": 1.0},
        ]
    )

    projected = build_shared_projection(df, pca_components=2, umap_neighbors=2, umap_min_dist=0.1, random_state=42)

    assert list(projected["state"]) == ["before", "after"]
    assert {"x", "y"}.issubset(projected.columns)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_shared_reducer.py -v`
Expected: FAIL with missing module or function

- [ ] **Step 3: Implement reduction, plotting, and compactness metrics**

```python
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
import umap

from src.utils.io import ensure_dir


def build_shared_projection(df, pca_components, umap_neighbors, umap_min_dist, random_state):
    embedding_columns = [column for column in df.columns if column.startswith("embedding_")]
    matrix = df[embedding_columns].to_numpy()

    pca = PCA(n_components=min(pca_components, matrix.shape[0], matrix.shape[1]), random_state=random_state)
    reduced = pca.fit_transform(matrix)

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=umap_neighbors,
        min_dist=umap_min_dist,
        random_state=random_state,
    )
    coords = reducer.fit_transform(reduced)

    projected = df.copy()
    projected["x"] = coords[:, 0]
    projected["y"] = coords[:, 1]
    return projected


def compactness_by_state(df):
    summary_rows = []
    for (state, class_name), group in df.groupby(["state", "class_name"]):
        center_x = group["x"].mean()
        center_y = group["y"].mean()
        radius = (((group["x"] - center_x) ** 2 + (group["y"] - center_y) ** 2) ** 0.5).mean()
        summary_rows.append({"state": state, "class_name": class_name, "mean_radius": radius})
    return pd.DataFrame(summary_rows)


def save_state_scatter(df, state_name, class_name, output_path, palette, alpha, point_size, dpi):
    subset = df[(df["state"] == state_name) & (df["class_name"] == class_name)]
    plt.figure(figsize=(6, 5))
    sns.scatterplot(data=subset, x="x", y="y", color=palette[class_name], alpha=alpha, s=point_size, edgecolor=None)
    plt.title(f"{class_name} - {state_name}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def save_state_density(df, state_name, class_name, output_path, palette, dpi):
    subset = df[(df["state"] == state_name) & (df["class_name"] == class_name)]
    plt.figure(figsize=(6, 5))
    sns.kdeplot(data=subset, x="x", y="y", fill=True, cmap="Reds" if class_name == "lesion" else "Blues")
    plt.title(f"{class_name} density - {state_name}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def save_joint_scatter(df, output_path, palette, alpha, point_size, dpi):
    plt.figure(figsize=(7, 6))
    sns.scatterplot(
        data=df,
        x="x",
        y="y",
        hue="class_name",
        style="state",
        palette=palette,
        alpha=alpha,
        s=point_size,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def run_reduction_and_plot(before_csv, after_csv, artifacts_dir, pca_components, umap_neighbors, umap_min_dist, random_state, alpha, point_size, dpi):
    before_df = pd.read_csv(before_csv)
    after_df = pd.read_csv(after_csv)
    combined = pd.concat([before_df, after_df], ignore_index=True)

    projected = build_shared_projection(combined, pca_components, umap_neighbors, umap_min_dist, random_state)

    plots_dir = Path(artifacts_dir) / "plots"
    metrics_dir = Path(artifacts_dir) / "metrics"
    ensure_dir(plots_dir)
    ensure_dir(metrics_dir)

    palette = {"lesion": "#d84b4b", "background": "#4f83cc"}
    save_state_scatter(projected, "before", "lesion", plots_dir / "lesion_scatter_before.png", palette, alpha, point_size, dpi)
    save_state_scatter(projected, "after", "lesion", plots_dir / "lesion_scatter_after.png", palette, alpha, point_size, dpi)
    save_state_density(projected, "before", "lesion", plots_dir / "lesion_density_before.png", palette, dpi)
    save_state_density(projected, "after", "lesion", plots_dir / "lesion_density_after.png", palette, dpi)
    save_joint_scatter(projected, plots_dir / "lesion_background_scatter_before_after.png", palette, alpha, point_size, dpi)

    compactness = compactness_by_state(projected)
    compactness.to_csv(metrics_dir / "compactness_summary.csv", index=False)
    projected.to_csv(metrics_dir / "shared_projection_points.csv", index=False)
```

- [ ] **Step 4: Run the shared-reducer test to verify it passes**

Run: `pytest tests/test_shared_reducer.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/analysis/reduce_and_plot.py tests/test_shared_reducer.py
git commit -m "feat: add shared projection and visualization pipeline"
```

### Task 8: End-To-End Runner

**Files:**
- Create: `scripts/run_bottleneck_visualization.py`

- [ ] **Step 1: Implement the orchestration script**

```python
from pathlib import Path

import torch

from src.analysis.extract_embeddings import extract_embeddings
from src.analysis.reduce_and_plot import run_reduction_and_plot
from src.models.unet import SimpleUNet
from src.training.finetune import run_finetuning
from src.utils.io import ensure_dir, load_config


def save_pretrained_checkpoint(config):
    checkpoints_dir = Path(config["paths"]["artifacts_dir"]) / "checkpoints"
    ensure_dir(checkpoints_dir)
    checkpoint_path = checkpoints_dir / "pretrained_encoder_unet.pt"

    model = SimpleUNet(
        encoder_name=config["model"]["encoder_name"],
        encoder_weights=config["model"]["encoder_weights"],
        num_classes=config["model"]["num_classes"],
    )
    torch.save(model.state_dict(), checkpoint_path)
    return checkpoint_path


def main(config_path):
    config = load_config(config_path)
    pretrained_checkpoint = save_pretrained_checkpoint(config)
    finetuned_checkpoint = run_finetuning(config_path)

    before_csv = extract_embeddings(config_path, pretrained_checkpoint, "before", "before_embeddings.csv")
    after_csv = extract_embeddings(config_path, finetuned_checkpoint, "after", "after_embeddings.csv")

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

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
```

- [ ] **Step 2: Run a dry import check**

Run: `python -c "from scripts.run_bottleneck_visualization import main; print(callable(main))"`
Expected: `True`

- [ ] **Step 3: Run the full test suite**

Run: `pytest -q`
Expected: all tests PASS

- [ ] **Step 4: Run the full pipeline**

Run: `python scripts/run_bottleneck_visualization.py --config configs/experiments/isic2018_bottleneck_visualization.yaml`
Expected:
- pretrained checkpoint saved
- finetuned checkpoint saved
- before and after embedding csv files saved
- five plot files saved
- one compactness summary csv saved

- [ ] **Step 5: Commit**

```bash
git add scripts/run_bottleneck_visualization.py artifacts
git commit -m "feat: add end to end bottleneck visualization experiment"
```

### Task 9: Manual Analysis Checklist

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add the exact interpretation checklist**

```markdown
## How to judge the result

Primary go-signal:
- Lesion points look more compact after finetuning in the shared projection.
- Lesion density map shows a clearer high-density region after finetuning.
- `compactness_summary.csv` shows smaller `mean_radius` for `lesion` in `after` than in `before`.

Warning signs:
- Only background gets tighter while lesion barely changes.
- Before and after look different only because of separate projections. This experiment must use one shared projection.
- UMAP shows dramatic separation but PCA-based sanity check does not.
```

- [ ] **Step 2: Verify the README includes all expected outputs and go/no-go criteria**

Run: `rg "compactness_summary|shared projection|go-signal" README.md`
Expected: all three phrases are found

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: add interpretation checklist for bottleneck plots"
```

## Self-Review

- Spec coverage: covered the selected approach only: class-mask pooling plus shared before/after reduction and lesion-first visualization. The plan includes the agreed outputs: before/after scatter, before/after density, joint class plot, and compactness table.
- Placeholder scan: no `TODO`, `TBD`, or undefined “handle later” steps remain.
- Type consistency: the shared names used across tasks are `SimpleUNet`, `pool_class_embeddings`, `extract_embeddings`, `build_shared_projection`, and `run_reduction_and_plot`.

## Notes Before Execution

- This plan assumes binary segmentation for ISIC 2018: `background` and `lesion`.
- If the chosen U-Net implementation does not expose a 1024-channel bottleneck naturally, the plan standardizes it through a `1x1 conv` projection in `src/models/unet.py`.
- For the first feasibility pass, do not add trajectory arrows, patch-level points, multi-seed repeats, or extra datasets. Keep the experiment small and interpretable.

## Suggested First Command Block

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pytest -q
python scripts/run_bottleneck_visualization.py --config configs/experiments/isic2018_bottleneck_visualization.yaml
```
