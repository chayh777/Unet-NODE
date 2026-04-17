# NODE Follow-Up Experiment Matrix

## Locked reference
- Group A best Dice: 0.7739947986602783
- Group B best Dice: 0.7768298244476318
- Group C best Dice: 0.7735665583610535
- Group C best epoch: 10
- Group C final Dice: 0.7539007067680359
- Group C peak-final gap: 0.019665851593017578
- Group C first-10-epoch max Dice: 0.7735665583610535

## Execution notes
- Every follow-up row in this matrix is still run with `--group C`; the run name identifies the config variant, not the CLI group.
- Artifact roots follow the runner convention: config `paths.artifacts_dir` sets the experiment root, and the runner writes outputs under `.../group_c/`.
- `C-diff-lr` and `C-warmup` are reserved follow-up probes that may require additional runner support before they become runnable config-only variants.

## Decision rubric
- Keep a run if it preserves the fast early rise and improves late-epoch stability.
- Promote a run if best Dice > 0.7768298244476318.
- Mark a run as "stability-only improvement" if best Dice is within 0.001 of Group B but the final epoch Dice drop is smaller than current Group C.

## Run table
| Run name | Hypothesis | Config file | Best Dice | Best epoch | Final Dice | Peak-final gap | Decision | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C-base-locked | Current unstable reference | configs/experiments/isic2018_low_data_node_c_base_locked.yaml |  |  |  |  |  |  |
| C-small-step | Smaller NODE step reduces overshoot | configs/experiments/isic2018_low_data_node_c_small_step.yaml |  |  |  |  |  |  |
| C-fine-integration | More steps + smaller step gives smoother flow | configs/experiments/isic2018_low_data_node_c_fine_integration.yaml |  |  |  |  |  |  |
| C-diff-lr | Reserved probe: NODE learns too fast relative to decoder | configs/experiments/isic2018_low_data_node_c_diff_lr.yaml |  |  |  |  |  | Requires separate optimizer-group support before it is runnable as a config-only variant. |
| C-warmup | Reserved probe: decoder should learn coarse mask before NODE acts | configs/experiments/isic2018_low_data_node_c_warmup.yaml |  |  |  |  |  | May require staged training support before it is runnable as a config-only variant. |
