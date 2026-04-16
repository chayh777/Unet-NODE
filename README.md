# U-Net Bottleneck Visualization

## Goal
Produce before/after bottleneck visualizations for a U-Net whose encoder is ImageNet-pretrained and then fully finetuned on ISIC 2018. The pipeline exports both the initial pretrained-model checkpoint and the final finetuned-model checkpoint before extracting embeddings.

## Data layout
- Current local layout is expected under `data/`.
- Training images: `data/train/images/`
- Training masks: `data/train/labels/`
- Validation images: `data/validation/images/`
- Validation masks: `data/validation/labels/`
- Validation metadata reference: `data/challenge-2018-task-1-2-validation_metadata_2025-12-22.csv`
- Image filenames must match mask filenames by stem. The loader supports either `ISIC_0001.png` or `ISIC_0001_segmentation.png`.
- Masks are binary: black `0` for background and white `255` for lesion. The loader binarizes them to `0/1`.
- Training currently uses `data/train`.
- Bottleneck extraction / visualization currently uses `data/validation` when `isic_eval_images_dir` and `isic_eval_masks_dir` are set in the config.

- Supported Python `3.10` or `3.11`.
- Install dependencies: `pip install -r requirements.txt`.

## Expected commands
```
python -m pytest -q
python scripts/run_bottleneck_visualization.py --config configs/experiments/isic2018_bottleneck_visualization.yaml
```

## Expected outputs
- artifacts/checkpoints/pretrained_encoder_unet.pt (initial ImageNet-pretrained encoder state)
- artifacts/checkpoints/finetuned_unet.pt (checkpoint after full-parameter finetuning)
- artifacts/embeddings/before_embeddings.csv
- artifacts/embeddings/after_embeddings.csv
- artifacts/plots/lesion_scatter_before.png
- artifacts/plots/lesion_scatter_after.png
- artifacts/plots/lesion_density_before.png
- artifacts/plots/lesion_density_after.png
- artifacts/metrics/compactness_summary.csv

## Interpretation checklist
- Go signal: lesion points look more compact in the shared projection after finetuning.
- Go signal: lesion density map shows a clearer high-density region post-finetuning.
- Go signal: `artifacts/metrics/compactness_summary.csv` reports a smaller `mean_radius` for `lesion` when `state=after` versus `state=before`.
- Warning: only background points tighten up while lesion points stay dispersed.
- Warning: the before/after comparison appears different because each was reduced separately (must confirm shared projection).
- Warning: UMAP shows dramatic separation but the PCA sanity check indicates no meaningful shift.

## Low-data NODE experiment
This section defines the low-data workflow contract built around `configs/experiments/isic2018_low_data_node.yaml`.

Group contract:
- Group A: pretrained U-Net with frozen encoder and trainable decoder only
- Group B: pretrained U-Net with frozen encoder, trainable decoder, and conv bottleneck adapter
- Group C: pretrained U-Net with frozen encoder, trainable decoder, and NODE bottleneck adapter
- `--group A` writes training outputs under `artifacts/low_data/group_a/`.
- `--group B` writes training outputs under `artifacts/low_data/group_b/`.
- `--group C` writes training outputs under `artifacts/low_data/group_c/`.

### Implemented now: training runner and current artifact contract
The training runner is implemented now for the three frozen-encoder variants on ISIC2018 with the same fixed 10% training subset.

Training command contract:
```bash
python scripts/run_low_data_experiment.py --config configs/experiments/isic2018_low_data_node.yaml --group A
```

Current training artifacts:
These are the minimum contract artifacts tracked at this stage of the plan.
- artifacts/low_data/group_a/best.pt
- artifacts/low_data/group_a/history.csv
- artifacts/low_data/group_a/metrics.json
- artifacts/low_data/group_b/best.pt
- artifacts/low_data/group_c/best.pt
- artifacts/low_data/splits/train_seed42_ratio10.csv

### Planned next-step analysis contract
The geometry export and summary plotting entrypoints below belong to the planned next-step analysis contract and are not implemented yet in this worktree.
Repeat the training command for Groups B and C before running the summary plotting command, since that report expects trained outputs for A, B, and C. Use the geometry export command only after the corresponding low-data outputs exist.

Planned analysis entrypoints:
```bash
python scripts/run_low_data_geometry.py --config configs/experiments/isic2018_low_data_node.yaml --group C
python scripts/plot_low_data_summary.py --artifacts-dir artifacts/low_data --groups A B C
```

Planned geometry artifacts:
- artifacts/low_data/group_c/geometry/pre_adapter_embeddings.csv
- artifacts/low_data/group_c/geometry/post_adapter_embeddings.csv
- artifacts/low_data/group_c/geometry/shared_projection_points.csv
- artifacts/low_data/group_c/geometry/bottleneck_before_after_scatter.png
- artifacts/low_data/group_c/geometry/bottleneck_before_after_density.png

Planned summary artifacts:
- artifacts/low_data/summary/dice_curve_compare.png
- artifacts/low_data/summary/iou_curve_compare.png
- artifacts/low_data/summary/loss_curve_compare.png
- artifacts/low_data/summary/final_metrics_compare.png
