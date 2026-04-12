# U-Net Bottleneck Visualization

## Goal
Produce before/after bottleneck visualizations for a U-Net whose encoder is ImageNet-pretrained and then fully finetuned on ISIC 2018. The pipeline exports both the initial pretrained-model checkpoint and the final finetuned-model checkpoint before extracting embeddings.

## Data layout
- Current server-side layout is expected under `data/isic2018/`.
- Training images: `data/isic2018/ISIC_2018_Task1_2_Training/Train/Images/`
- Training masks: `data/isic2018/ISIC_2018_Task1_2_Training/Train/Masks/`
- Validation metadata reference: `data/isic2018/validation-metadata.csv`
- Image filenames must match mask filenames by stem (for example `ISIC_0001.jpg` with `ISIC_0001.png`).
- Masks are binary: black `0` for background and white `255` for lesion. The loader binarizes them to `0/1`.
- This repo currently points the main experiment config at the `Train` image/mask pair directories shown above.

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
