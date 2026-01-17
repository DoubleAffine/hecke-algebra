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
