# 2026-05-28 GlaS Experiment Matrix

## Goal

Mirror the core ISIC experiment story on `GlaS` so the paper is no longer supported by a single dataset only.

## Protocol Note

`GlaS` is much smaller than ISIC2018, so reusing `train_ratio=0.1` collapses the training split to roughly single-digit images and makes many methods degenerate to near-constant predictions. The configs in this matrix therefore use the full provided GlaS training split (`train_ratio: 1.0`) to answer the more basic question first: can the frozen-encoder bottleneck-adaptation framework learn at all on GlaS?

## Comparison Runs

1. Plain U-Net
   - `python scripts/run_low_data_experiment.py --config configs/experiments/glas_low_data_node.yaml --group A`

2. B-base
   - `python scripts/run_low_data_experiment.py --config configs/experiments/glas_low_data_node.yaml --group B`

3. Output-Conv-U-Net
   - `python scripts/run_low_data_experiment.py --config configs/experiments/glas_low_data_output_conv_b.yaml --group B`

4. Output-NODE-U-Net
   - `python scripts/run_low_data_experiment.py --config configs/experiments/glas_low_data_output_node_c.yaml --group C`

## Bottleneck Main-Method Follow-Up Runs

5. C-fine-steps8-default
   - `python scripts/run_low_data_experiment.py --config configs/experiments/glas_low_data_node_c_fine_integration.yaml --group C`

6. C-zero-last-steps8
   - `python scripts/run_low_data_experiment.py --config configs/experiments/glas_low_data_node_c_zero_last_fine_integration.yaml --group C`

7. C-zero-last-steps16
   - `python scripts/run_low_data_experiment.py --config configs/experiments/glas_low_data_node_c_zero_last_steps16_t1.yaml --group C`

## Stability-Tuning Runs

8. C-zero-last-kinetic
   - `python scripts/run_low_data_experiment.py --config configs/experiments/glas_low_data_node_c_zero_last_kinetic.yaml --group C`

9. C-zero-last-kinetic-steps8
   - `python scripts/run_low_data_experiment.py --config configs/experiments/glas_low_data_node_c_zero_last_kinetic_steps8.yaml --group C`

10. C-zero-last-kinetic-rk4
   - `python scripts/run_low_data_experiment.py --config configs/experiments/glas_low_data_node_c_zero_last_kinetic_rk4.yaml --group C`

## Minimal Recommended Order

If time is limited, run these first:

1. Plain U-Net
2. B-base
3. Output-Conv-U-Net
4. Output-NODE-U-Net
5. C-zero-last-steps8
6. C-zero-last-steps16

Then run the stability-tuning trio only if the bottleneck NODE line remains promising on GlaS.
