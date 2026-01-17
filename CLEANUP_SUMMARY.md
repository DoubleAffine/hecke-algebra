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
