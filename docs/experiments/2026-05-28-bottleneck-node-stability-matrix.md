# 2026-05-28 Bottleneck NODE Stability Matrix

## Objective

Decide whether stability-oriented tuning improves the bottleneck NODE main method enough to justify later multi-seed expansion.

## Runs

1. Reference
   - `configs/experiments/isic2018_low_data_node_c_zero_last.yaml`

2. Kinetic regularization baseline
   - `configs/experiments/isic2018_low_data_node_c_zero_last_kinetic.yaml`

3. Kinetic regularization + steps8
   - `configs/experiments/isic2018_low_data_node_c_zero_last_kinetic_steps8.yaml`

4. Kinetic regularization + RK4
   - `configs/experiments/isic2018_low_data_node_c_zero_last_kinetic_rk4.yaml`

## Readout Metrics

- `best_val_dice`
- `best_epoch`
- `epochs_ran`
- `duration_sec`
- `avg_epoch_sec`
- `regularization_type`
- `regularization_weight`
- best-vs-final stability from `history.csv`

## Success Criteria

- `best_val_dice` exceeds the current zero-last bottleneck NODE reference, or
- `best_val_dice` is comparable while `best_epoch`, `epochs_ran`, or best-vs-final stability improves

## Stop Conditions

- If all regularized runs are worse than the reference and slower, do not continue to wider solver tuning.
- If the regularized steps8 run improves stability but not score, keep it as a candidate for GlaS before multi-seed expansion.
- If the regularized RK4 run is slower with no measurable gain, drop RK4 from the main line.

## Deferred Work

These are intentionally not part of this first stability wave:

- Jacobian penalty implementation
- spectral normalization / Lipschitz proxy
- `1x1 + NODE` structural tuning
- GlaS rollout for tuned configs
- multi-seed expansion for tuned configs
