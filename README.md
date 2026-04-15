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
Run low-data experiments across three groups:
- Group A: 10% train split with NODE adapter baseline and full artifact logging.
- Group B: alternate low-data run for contract parity with checkpoint output.
- Group C: alternate low-data run for contract parity with checkpoint output.

Command:
```
python scripts/run_low_data_experiment.py --config configs/experiments/isic2018_low_data_node.yaml --group A
```

Expected artifacts:
- artifacts/low_data/group_a/best.pt
- artifacts/low_data/group_a/history.csv
- artifacts/low_data/group_a/metrics.json
- artifacts/low_data/group_b/best.pt
- artifacts/low_data/group_c/best.pt
- artifacts/low_data/splits/train_seed42_ratio10.csv
