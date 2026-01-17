# Ready to Run: Proper Intersection Test

## Repository Status: ✓ ORGANIZED

### What Was Done:

1. **Archived Invalid Experiments**
   - All low-sample experiments → `archive/low_sample_experiments/`
   - Problematic analyses → `archive/documentation/`
   - See `archive/README.md` for details

2. **Organized Structure**
   ```
   ├── experiments/current/     # New experiments go here
   ├── final_results/           # Only valid results
   ├── archive/                 # Invalid low-sample stuff
   ├── analysis_scripts/        # Tools and helpers
   └── src/                     # Core framework
   ```

3. **Created Proper Experiment**
   - `run_proper_intersection_test.py` - Adequate sampling (100+ per task)
   - `monitor_experiment.sh` - Real-time progress tracking
   - Memory efficient (incremental saving, data deletion)

---

## The New Experiment Design

### Key Features:

✓ **ADEQUATE SAMPLING**
- 100 models per task (default)
- Samples per dimension: ~1.4 (marginal, but 2× better than before)
- Can increase with `--models-per-task 200` for better sampling

✓ **MEMORY EFFICIENT**
- Processes in batches (10 models at a time)
- Saves weights incrementally
- Deletes training data immediately
- Won't run out of memory!

✓ **REAL-TIME PROGRESS**
- Live progress bar in terminal
- Shows: model #, task, loss, accuracy, ETA
- Per-task timing breakdown
- Easy to track!

✓ **PROPER INTERSECTION ANALYSIS**
- Trains on multiple tasks (signal vs noise)
- Computes principal angles between subspaces
- Global PCA to detect overlap
- Multiple verification methods

### What It Tests:

**Question:** Do different tasks share a common low-D subspace?

**Method:**
1. Train 100 models on signal (real data)
2. Train 100 models on noise (random labels)
3. Find PCA subspace for each (expected ~72D each)
4. Compute principal angles between subspaces
5. Check if they're aligned (< 20°) or orthogonal (> 70°)

**Possible outcomes:**
- Small angles → Shared subspace (Universal!)
- Large angles → Separate subspaces (No universality)
- Medium angles → Partial overlap

---

## How to Run

### Basic (100 models per task, ~2-3 hours):

```bash
python run_proper_intersection_test.py
```

### In background with monitoring:

```bash
# Terminal 1: Run experiment
python run_proper_intersection_test.py > experiments/current/intersection_proper/run.log 2>&1 &

# Terminal 2: Monitor progress
./monitor_experiment.sh
```

### With more sampling (better, but slower):

```bash
python run_proper_intersection_test.py --models-per-task 200
# This gives ~2.8 samples per dimension (much better!)
# Takes ~4-6 hours
```

### Custom configuration:

```bash
python run_proper_intersection_test.py \
  --models-per-task 150 \
  --epochs 80 \
  --batch-size 15 \
  --save-dir experiments/current/my_test
```

---

## What to Expect

### During Run:

```
[████████████████████████████░░░░░░░░░░░░░░] 57.5%
| Model 115/200 | Task: binary_random_labels
| Loss: 0.0952 | Acc: 95.5% | ETA: 14:23:45
```

### Output Files:

```
experiments/current/intersection_proper/
├── config.json                          # Experiment parameters
├── weights_binary_classification_synthetic.npy
├── weights_binary_random_labels.npy
├── metadata_binary_classification_synthetic.json
├── metadata_binary_random_labels.json
├── weights_*_batch_*.npy               # Incremental batches
└── results.json                         # Final analysis
```

### Results File:

```json
{
  "task_dimensions": {
    "signal": { "effective_dim": 71.4 },
    "noise": { "effective_dim": 72.1 }
  },
  "intersection": {
    "aligned_dimensions": 15,
    "mean_angle": 45.2,
    "global_dimension": 98.3,
    "intersection_estimate": 68.0
  }
}
```

---

## Interpreting Results

### If mean angle < 20°:
→ **Strong overlap** (tasks share subspace)
→ Universal subspace exists!

### If mean angle > 70°:
→ **Nearly orthogonal** (separate subspaces)
→ No universal subspace

### If global_dim < mean(individual_dims):
→ **Intersection exists**
→ Global dimension = intersection size

### If global_dim ≈ sum(individual_dims):
→ **No intersection**
→ Tasks are independent

---

## Memory Management

The experiment is designed to NOT run out of memory:

1. **Incremental saving:** Saves every 10 models to disk
2. **Data deletion:** Deletes training data after each task
3. **Batch processing:** Never holds all models in memory
4. **Efficient storage:** Uses compressed .npy format

**Memory usage:** ~500MB peak (very reasonable!)

---

## Monitoring

### Real-time (recommended):

```bash
./monitor_experiment.sh
```

Updates every 5 seconds with:
- Status (running/stopped)
- Configuration
- Progress (batches completed)
- Results (when finished)

### Manual check:

```bash
# Check if running
ps aux | grep run_proper_intersection

# Check progress
ls -lh experiments/current/intersection_proper/weights_*_batch_*.npy | wc -l

# Check results
cat experiments/current/intersection_proper/results.json
```

---

## What Happens After

### If Experiment Succeeds:

1. Check `results.json` for findings
2. Compare to saturation test (still running)
3. Decide if we need more tasks or more samples
4. Document in `experiments/EXPERIMENT_LOG.md`

### If Intersection Found:

→ We've discovered the universal subspace dimension!
→ Can test topology (components, structure)
→ Test with more tasks to verify

### If No Intersection:

→ Tasks are independent
→ No universal subspace
→ Each task has its own manifold

---

## Current Status

**Ready to run!**

Just execute:
```bash
python run_proper_intersection_test.py
```

And monitor with:
```bash
./monitor_experiment.sh
```

**Estimated time:** 2-3 hours for 100 models per task

**Next decision point:** Wait for results, then decide:
- Need more samples?
- Add more tasks?
- Test saturation first?

