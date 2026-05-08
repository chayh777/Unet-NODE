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
- `C-base-locked` has already been run externally by the project owner and is treated as the locked unstable reference; do not schedule another rerun unless the code path changes.

## Decision rubric
- Keep a run if it preserves the fast early rise and improves late-epoch stability.
- Promote a run if best Dice > 0.7768298244476318.
- Mark a run as "stability-only improvement" if best Dice is within 0.001 of Group B but the final epoch Dice drop is smaller than current Group C.
- Promote `C-zero-last` or a combined zero-last run if best Dice exceeds Group B best Dice `0.7768298244476318` and peak-final gap is below the current Group C gap `0.019665851593017578`.
- Mark a zero-last run as "stability-only improvement" if best Dice does not exceed Group B but peak-final gap is clearly smaller than current Group C.
- Do not claim NODE-specific benefit until `C-zero-last` is compared with `B-zero-last`.

## Run table
| Run name | Category | Hypothesis | Config file | Code support needed | Best Dice | Best IoU | Best epoch | Final Dice | Peak-final gap | Decision | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C-base-locked | reference | Current unstable NODE reference | configs/experiments/isic2018_low_data_node_c_base_locked.yaml | no |  |  |  |  |  | locked-reference | Already run by project owner; no rerun planned. |
| C-small-step | integration | Smaller NODE step reduces overshoot | configs/experiments/isic2018_low_data_node_c_small_step.yaml | no |  |  |  |  |  |  | Existing config. |
| C-zero-last | init | Zeroing the NODE vector field last layer starts from identity flow and stabilizes training | configs/experiments/isic2018_low_data_node_c_zero_last.yaml | yes |  |  |  |  |  |  | Primary improvement run. |
| C-zero-last-small-step | init+integration | Zero-last initialization and smaller step size may combine stability benefits | configs/experiments/isic2018_low_data_node_c_zero_last_small_step.yaml | yes |  |  |  |  |  |  | First combined run to prioritize after C-zero-last. |
| C-fine-integration | integration | More steps with smaller step size gives smoother NODE evolution | configs/experiments/isic2018_low_data_node_c_fine_integration.yaml | no |  |  |  |  |  |  | Config-only integration ablation. |
| C-zero-last-fine-integration | init+integration | Identity-start flow plus smoother integration improves both best Dice and final stability | configs/experiments/isic2018_low_data_node_c_zero_last_fine_integration.yaml | yes |  |  |  |  |  |  | Run after C-zero-last-small-step if zero-last is promising. |
| B-zero-last | control | Conv adapter also benefits from zero-last initialization; checks whether gains are NODE-specific | configs/experiments/isic2018_low_data_conv_b_zero_last.yaml | yes |  |  |  |  |  |  | Run before claiming NODE-specific improvement. |
| C-zero-last-steps1 | mechanism | Single-step zero-last NODE tests whether the gain is just a residual adapter effect | configs/experiments/isic2018_low_data_node_c_zero_last_steps1.yaml | yes |  |  |  |  |  |  | Run if C-zero-last improves over reference. |
| C-diff-lr | training | NODE learns too fast relative to decoder | configs/experiments/isic2018_low_data_node_c_diff_lr.yaml | yes |  |  |  |  |  | reserved | Requires optimizer parameter-group support. |
| C-warmup | training | Decoder should learn coarse masks before NODE acts | configs/experiments/isic2018_low_data_node_c_warmup.yaml | yes |  |  |  |  |  | reserved | Requires staged training support. |
