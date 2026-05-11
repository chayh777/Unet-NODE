# NODE Robustness to Input Perturbation - Design Specification

**Goal:** Validate that NODE provides not only performance improvements but also robustness advantages under Gaussian noise perturbation. The experiment compares Group A/B/C across multiple noise levels and demonstrates that NODE's continuous flow transformation acts as a manifold smoother.

## 1. Experiment Design

### Noise Levels
- 6 levels: σ = 0.0, 0.05, 0.1, 0.15, 0.2, 0.3
- σ=0.0: Clean baseline
- σ=0.05, 0.1: Simulate common sensor noise (mild degradation)
- σ=0.15, 0.2: Low-light / poor device conditions (challenging boundary extraction)
- σ=0.3: Stress test (observe model collapse point)

### Group Configuration
| Group | Configuration |
|-------|---------------|
| A | Baseline: frozen encoder + trainable decoder |
| B | Frozen encoder + trainable decoder + Conv adapter |
| C | Frozen encoder + trainable decoder + NODE adapter (optimized: steps=6~8, step_size=0.1, zero_init) |

### Metrics
- Primary: Dice coefficient vs σ curve
- Secondary: IoU vs σ curve
- Data: Per-sample Dice computed on full validation set, aggregated per noise level

## 2. Architecture

### Script Structure
```
scripts/run_robustness_analysis.py      # Main robustness experiment runner
scripts/plot_robustness_results.py     # Generate robustness curves
src/analysis/robustness_metrics.py    # Core metric computation logic
```

### Noise Injection
Gaussian noise injected during validation inference:
```python
noisy_image = image + torch.randn_like(image) * sigma
```

### Feature Space Visualization
For σ=0.2 (key failure point), extract bottleneck features using existing `run_low_data_geometry.py` with noise injection:
- UMAP projection of bottleneck features
- Compare clustering compactness across A/B/C

## 3. Output Specification

### Robustness Metrics
- `artifacts/low_data/robustness/robustness_metrics.csv`
  - Columns: `group`, `sigma`, `mean_dice`, `std_dice`, `mean_iou`, `std_iou`, `num_samples`
- `artifacts/low_data/robustness/dice_decay_curve.png`
- `artifacts/low_data/robustness/iou_decay_curve.png`

### Feature Space Visualization
- `artifacts/low_data/robustness/geometry/sigma0.2_scatter.png`
- Uses UMAP with existing plotting conventions from `reduce_and_plot.py`

### Combined Summary
- `artifacts/low_data/robustness/summary/robustness_analysis.png`
  - Left panel: DICE decay curves for A/B/C
  - Right panel: UMAP scatter at σ=0.2

## 4. Interpretation Criteria

### Expected Outcome
- **Group A (Baseline)**: Steepest decay curve, features scatter most under noise
- **Group B (Conv Adapter)**: Moderate decay, some feature distortion at high noise
- **Group C (NODE)**: Flattest decay curve, features remain most compact at σ=0.2

### Manifold Smoothing Evidence
At σ=0.2, if Group C bottleneck features show tighter clustering than A/B, this demonstrates NODE's continuous flow acts as a manifold smoother against input perturbations.

## 5. Artifact Contract

```
artifacts/low_data/robustness/
├── robustness_metrics.csv
├── dice_decay_curve.png
├── iou_decay_curve.png
├── summary/
│   └── robustness_analysis.png
└── geometry/
    └── sigma0.2_scatter.png
```

## 6. Command Contracts

```bash
# Step A: Run robustness analysis across all noise levels
python scripts/run_robustness_analysis.py --config configs/experiments/isic2018_low_data_node.yaml --groups A B C

# Step B: Generate robustness curves
python scripts/plot_robustness_results.py --artifacts-dir artifacts/low_data

# Step C: Extract geometry with sigma=0.2 noise
python scripts/run_low_data_geometry.py --config configs/experiments/isic2018_low_data_node.yaml --group C --noise-sigma 0.2
python scripts/run_low_data_geometry.py --config configs/experiments/isic2018_low_data_node.yaml --group B --noise-sigma 0.2
python scripts/run_low_data_geometry.py --config configs/experiments/isic2018_low_data_node.yaml --group A --noise-sigma 0.2
```

## 7. Implementation Notes

### Noise Injection Point
Inject noise in validation inference only (not training). This tests model's inference-time robustness without retraining.

### C Group Configuration
Use zero_init configuration for NODE adapter:
- steps: 8
- step_size: 0.1
- init: zero_last_layer

### Existing Code Reuse
- Reuse `src/analysis/low_data_geometry.py` for bottleneck extraction
- Reuse `src/analysis/reduce_and_plot.py` for UMAP visualization
- Reuse `src/analysis/low_data_reporting.py` for plotting patterns
