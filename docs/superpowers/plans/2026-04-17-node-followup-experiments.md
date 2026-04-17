# NODE Follow-Up Experiments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the next-round low-data NODE experiments that convert the current "early-peak but unstable" Group C signal into a reproducible, interpretable result.

**Architecture:** Keep the existing low-data training, geometry export, and summary-reporting pipeline unchanged as the experiment harness. Add a small set of config variants focused on Group C stability, run each variant through the same training and visualization path, and compare them against the locked current A/B/C baseline using peak behavior, final behavior, and geometry artifacts.

**Tech Stack:** Python, PyTorch, YAML configs, existing experiment CLIs, pandas, matplotlib

---

## Planned File Structure

- Create: `docs/experiments/2026-04-17-node-followup-matrix.md`
  Purpose: human-readable experiment log sheet that records each C-variant, intended hypothesis, actual result, and keep/drop decision.
- Create: `configs/experiments/isic2018_low_data_node_c_base_locked.yaml`
  Purpose: frozen snapshot of the current Group C baseline config so later comparisons are not polluted by accidental config drift.
- Create: `configs/experiments/isic2018_low_data_node_c_small_step.yaml`
  Purpose: first stability probe that only reduces NODE `step_size`.
- Create: `configs/experiments/isic2018_low_data_node_c_fine_integration.yaml`
  Purpose: second stability probe that increases NODE `steps` while lowering `step_size`.
- Create: `configs/experiments/isic2018_low_data_node_c_diff_lr.yaml`
  Purpose: third probe that separates NODE learning-rate from decoder learning-rate if code support already exists, or documents the exact config contract to add next.
- Create: `configs/experiments/isic2018_low_data_node_c_warmup.yaml`
  Purpose: reserved config for a delayed-NODE schedule if the first three probes fail.
- Modify: `README.md`
  Purpose: document the follow-up C-tuning protocol once a winning strategy is confirmed.

## Locked Current Evidence

Use these values as the fixed reference when evaluating all follow-up runs:

- Group A best Dice: `0.7739947986602783`
- Group B best Dice: `0.7768298244476318`
- Group C best Dice: `0.7735665583610535`
- Group A best epoch: `25`
- Group B best epoch: `23`
- Group C best epoch: `10`
- Group A first-10-epoch max Dice: `0.7401209020614624`
- Group B first-10-epoch max Dice: `0.7636580109596253`
- Group C first-10-epoch max Dice: `0.7735665583610535`

Interpretation rule for this plan:

- A follow-up run is **worth keeping** if it preserves the early-peak advantage of C while improving either final Dice, peak stability, or both.
- A follow-up run is **paper-worthy** if it exceeds locked Group B final Dice or matches Group B while showing a cleaner and earlier convergence story.

## Execution Status

- Task 1 implemented: matrix doc and locked C reference config.
- Task 2 Step 1 implemented: small-step C config.
- Long training and geometry export were not run in the isolated worktree because local `data/` and `artifacts/` are gitignored and absent from that worktree.
