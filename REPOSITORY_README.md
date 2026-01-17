# Universal Subspace Investigation

**Current Status:** Cleaning up after discovering systematic undersampling issues

## Quick Navigation

- **Active Experiments:** `experiments/EXPERIMENT_LOG.md`
- **Valid Results:** `final_results/`
- **Archive (Invalid):** `archive/` - DO NOT USE
- **Analysis Scripts:** `analysis_scripts/`

## Key Findings (So Far)

### ✓ Confirmed:
1. Models converge to regions smaller than parameter space
2. Optimization dynamics dominate task semantics (signal ≈ noise)
3. Compression vs random: ~11.5% at n=100

### ✗ Invalidated:
1. "8D manifold" - undersampling artifact
2. "No intersection" - undersampled, actually ~28% overlap
3. "Simply connected" - insufficient data

### ? Unknown (Pending):
1. True manifold dimension (does it saturate?)
2. Intersection across tasks (need proper sampling)
3. Number of components (need saturation first)

## Current Questions

1. **Saturation:** Does dimension saturate or keep growing?
   - Running: 300-model test
   - If saturates at ~100-150D → real manifold
   - If keeps growing → no manifold structure

2. **Intersection:** Do tasks share a subspace?
   - Pending: Proper test with 100+ models per task
   - Need saturation results first

## Repository Structure

```
├── src/                          # Core framework (trainers, models, etc.)
├── experiments/
│   ├── current/                  # Active experiments
│   └── configs/                  # Experiment configurations
├── final_results/                # Valid, well-sampled results
├── archive/                      # Invalid low-sample experiments
├── analysis_scripts/             # Analysis and visualization tools
├── run_*.py                      # Main experiment runners
└── REPOSITORY_README.md          # This file
```

## Running New Experiments

See `experiments/configs/` for templates with proper sampling.
