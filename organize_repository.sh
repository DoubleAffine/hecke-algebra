#!/bin/bash
# Organize the repository - clean up low-sampling experiments and structure properly

echo "=========================================="
echo " REPOSITORY ORGANIZATION"
echo "=========================================="

# Create organized folder structure
mkdir -p archive/low_sample_experiments
mkdir -p archive/documentation
mkdir -p experiments/current
mkdir -p experiments/configs
mkdir -p analysis_scripts
mkdir -p final_results

echo ""
echo "Moving low-sample experiments to archive..."

# Move all the problematic low-sample results
mv results_quick_basin archive/low_sample_experiments/ 2>/dev/null
mv results_intersection archive/low_sample_experiments/ 2>/dev/null
mv results_dynamics archive/low_sample_experiments/ 2>/dev/null
mv results_distant archive/low_sample_experiments/ 2>/dev/null

# Keep the good 100-model result but mark it
mv results_large_scale final_results/single_task_100models 2>/dev/null

# Keep saturation test (still running)
# Don't move results_saturation - it's ongoing

echo "Moving analysis documentation..."
mv CORRECTED_ANALYSIS.md archive/documentation/
mv BASIN_STRUCTURE_ANALYSIS.md archive/documentation/
mv DIMENSION_GROWTH_ANALYSIS.md archive/documentation/
mv FINAL_TOPOLOGY_CONCLUSION.md archive/documentation/
mv CORRECTED_HYPOTHESIS.md archive/documentation/
mv INTERSECTION_ANALYSIS_SUMMARY.md archive/documentation/

echo "Moving analysis scripts..."
mv quick_*.py analysis_scripts/
mv check_*.py analysis_scripts/
mv analyze_*.py analysis_scripts/
mv compare_*.py analysis_scripts/
mv investigate_*.py analysis_scripts/
mv test_*.py analysis_scripts/ 2>/dev/null
mv visualize_*.py analysis_scripts/

echo "Keeping main experiment scripts in root..."
# Keep: run_*.py files for easy access

echo ""
echo "Creating archive README..."

cat > archive/README.md << 'ARCHIVE'
# Archive: Low-Sample Experiments

This folder contains experiments with **insufficient sampling** that led to unreliable results.

## Problems Identified:

### 1. Dimension Growth with Sample Size
- 10 models → 9D
- 50 models → 42D
- 100 models → 72D
- **Dimension ≈ 0.72 × sample_size** (not saturated!)

### 2. Undersampling Issues
All experiments here used too few samples:
- `results_quick_basin/` - Only 15 models total
- `results_intersection/` - Only 50 models per task (need 500+)
- `results_dynamics/` - Only 10 models (claimed 8D - wrong!)

### 3. Wrong Conclusions
- Initially claimed 8D manifold (actually undersampling artifact)
- Claimed "no intersection" (actually undersampled, ~28% overlap seen)
- Claimed "multiple basins" (actually measurement noise)

## Lessons Learned:

1. **Need 10+ samples per dimension** for reliable estimation
2. **Always check saturation** - does dimension stop growing?
3. **Compare to random baseline** - are we better than noise?
4. **Verify with multiple methods** - PCA, MLE, distances should agree

## Valid Results:

Only **final_results/single_task_100models/** has adequate sampling for that specific analysis.

## Next Steps:

See `/experiments/current/` for properly designed experiments with:
- Adequate sampling (100+ models per condition)
- Saturation testing (varying sample sizes)
- Memory-efficient incremental processing
ARCHIVE

echo "Creating experiment tracking file..."

cat > experiments/EXPERIMENT_LOG.md << 'LOG'
# Experiment Log

## Active Experiments

### 1. Saturation Test (300 models, single task)
- **Status:** Running
- **Started:** [Check timestamp]
- **Purpose:** Find if/when dimension saturates
- **Location:** `results_saturation/`

### 2. Multi-Task Intersection (PLANNED)
- **Status:** Not started - waiting for cleanup
- **Purpose:** Proper intersection test with adequate sampling
- **Design:** 100 models per task, multiple tasks

## Completed Experiments

### Single Task, 100 Models (VALID)
- **Location:** `final_results/single_task_100models/`
- **Finding:** ~72D effective dimension (but still growing)
- **Compression:** 11.5% better than random
- **Status:** ✓ Adequately sampled for this analysis

## Archived (Insufficient Sampling)

See `archive/low_sample_experiments/README.md`

## Planning

Next experiment design pending saturation test results.
LOG

echo ""
echo "Creating main README..."

cat > REPOSITORY_README.md << 'README'
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
README

echo ""
echo "Creating cleanup summary..."

cat > CLEANUP_SUMMARY.md << 'SUMMARY'
# Repository Cleanup Summary

## What Was Moved

### To `archive/low_sample_experiments/`:
- `results_quick_basin/` - 15 models (too few)
- `results_intersection/` - 50 per task (too few)
- `results_dynamics/` - 10 models (way too few)
- All experiments with insufficient sampling

### To `archive/documentation/`:
- Analysis documents from low-sample experiments
- Historical investigation notes
- Corrected analyses and lessons learned

### To `analysis_scripts/`:
- All `analyze_*.py`, `check_*.py`, `quick_*.py` scripts
- Visualization and verification tools
- Kept separate from experiment runners

### To `final_results/`:
- `single_task_100models/` - The ONLY valid result so far
- Adequate sampling for single-task analysis

## What Stayed in Root

### Experiment Runners:
- `run_*.py` - Main experiment scripts

### Core Framework:
- `src/` - Trainers, models, datasets, analysis

### Active:
- `results_saturation/` - Currently running

## Key Files Created

1. `REPOSITORY_README.md` - New main README
2. `experiments/EXPERIMENT_LOG.md` - Track all experiments
3. `archive/README.md` - Explain what's invalid and why
4. `CLEANUP_SUMMARY.md` - This file

## Next Steps

1. Wait for saturation test to complete
2. Design proper multi-task intersection test
3. Use memory-efficient incremental processing
4. Track progress in real-time
SUMMARY

echo ""
echo "=========================================="
echo " CLEANUP COMPLETE"
echo "=========================================="
echo ""
echo "Summary:"
echo "  - Archived: Low-sample experiments"
echo "  - Organized: Scripts and documentation"
echo "  - Preserved: Valid 100-model result"
echo "  - Created: Tracking and README files"
echo ""
echo "Next: Design proper upsampled experiment"
