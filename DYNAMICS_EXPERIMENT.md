

# Optimization Dynamics Investigation: Signal vs Noise

This experiment tests a fundamental question about the Universal Subspace Hypothesis:

**Does the learned function (task) matter, or only the optimization dynamics?**

## The Question

When neural networks train, do they end up in the same region of weight space because:
1. **Task semantics** - They're learning similar patterns
2. **Optimization dynamics** - Gradient descent naturally converges to certain regions regardless of task

## The Experiment

Train models on two types of data with **identical architecture and distribution**:

### Signal Tasks (Learning Real Patterns)
- Binary classification with structured data
- Multi-class classification with real patterns
- **Expected:** Models learn meaningful functions

### Noise Tasks (Learning Pure Noise)
- **Same data distribution** but **randomized labels**
- Impossible to generalize (pure memorization)
- **Expected:** Models overfit to noise (100% train accuracy, ~random test accuracy)

### The Critical Test

After training, analyze where the models end up in weight space:

**Scenario A: They cluster together**
- Signal and noise models converge to same manifold
- **Conclusion:** Universal subspace is about **optimization dynamics**
- Task doesn't matter - it's about the loss landscape itself

**Scenario B: They separate**
- Signal models cluster separately from noise models
- **Conclusion:** Task semantics matter
- The manifold is task-dependent

## Running the Experiment

### Basic Usage

```bash
python run_dynamics_experiment.py
```

This will:
1. Train models on signal tasks (real data)
2. Train models on noise tasks (random labels)
3. Track weight trajectories during training
4. Analyze final convergence points
5. Visualize signal vs noise in weight space

### Custom Configuration

```bash
python run_dynamics_experiment.py \
  --hidden-dims 20 20 \
  --epochs 200 \
  --track-every 10 \
  --n-replicates 10
```

Options:
- `--hidden-dims`: Network architecture (default: 16 16)
- `--epochs`: Maximum training epochs (default: 150)
- `--patience`: Early stopping patience (default: 20)
- `--track-every`: Save weights every N epochs (default: 10)
- `--n-replicates`: How many models per task (default: 5)
- `--save-dir`: Output directory (default: results_dynamics)

## Interpreting Results

### Key Visualization: `signal_vs_noise.png`

This plot shows both PCA and UMAP projections:
- **Blue dots** = Models trained on signal (real patterns)
- **Red dots** = Models trained on noise (random labels)

**What to look for:**

1. **Tight clustering (blue + red mixed)**
   - Indicates optimization dynamics dominate
   - Task-independent manifold

2. **Clear separation (blue cluster, red cluster)**
   - Indicates task semantics matter
   - Different tasks → different weight regions

3. **Partial overlap**
   - More nuanced - some tasks similar, others different

### Additional Analysis

**Trajectory Lengths:**
- Do signal vs noise models take similar paths?
- Longer trajectories might indicate more exploration

**Fractal Dimension:**
- If manifold is fractal, suggests complex attractor dynamics
- Relates to chaos in gradient descent

## Connection to Dynamical Systems

This experiment relates to:

1. **Attractors**: Do all training runs converge to same attractor?
2. **Basins of attraction**: Are signal/noise in same basin?
3. **Chaos**: Is weight space trajectory chaotic or regular?

## Expected Outcomes & Interpretations

### Hypothesis 1: Dynamics Dominate
If models cluster together regardless of task:
- Universal subspace is a **property of SGD + architecture**
- Loss landscape has preferred low-dimensional attractors
- Implications: Transfer learning works because all tasks visit same region

### Hypothesis 2: Task Matters
If signal and noise separate:
- Different tasks truly learn different representations
- Universal subspace might be task-family specific
- Implications: Need task similarity for good transfer

### Hypothesis 3: Fractal Attractor
If clustering but with fractal structure:
- Optimization exhibits chaotic dynamics
- Multiple scales of organization
- Suggests edge-of-chaos dynamics in training

## Files Generated

```
results_dynamics/
├── trajectories.npz              # Weight snapshots during training
├── trajectory_metadata.json      # Training info per model
├── signal_vs_noise.png          # Main result visualization
├── dynamics_report.txt          # Interpretation & conclusions
├── convergence_analysis.npz     # PCA, UMAP, clustering results
└── figures/                     # All geometric analysis plots
    ├── pca_variance.png
    ├── umap_2d.png
    ├── fractal_dimension.png
    └── ...
```

## Advanced: Trajectory Analysis

The experiment also tracks **how** models get to their final weights, not just where they end up.

Future analysis could include:
- Lyapunov exponents (measure chaos)
- Periodic orbit detection
- Phase space reconstruction
- Recurrence plots

## References & Context

This experiment is inspired by:
- Universal approximation theorem
- Loss landscape geometry (Garipov et al. 2018)
- Mode connectivity research
- Edge of chaos in neural networks

## Next Steps

1. **Run baseline**: `python run_dynamics_experiment.py`
2. **Check separation**: Look at `signal_vs_noise.png`
3. **Read report**: Check `dynamics_report.txt`
4. **Iterate**: Try different architectures, learning rates, etc.

## Example Workflow

```bash
# Quick test (5 min)
python run_dynamics_experiment.py --epochs 50 --n-replicates 3

# Full experiment (30-60 min)
python run_dynamics_experiment.py --epochs 200 --n-replicates 10

# Large scale (several hours)
python run_dynamics_experiment.py \
  --epochs 300 \
  --n-replicates 20 \
  --track-every 5 \
  --hidden-dims 32 32
```

## Scientific Impact

If this shows **dynamics dominate over task**:
- Changes our understanding of why deep learning works
- Explains transferability across tasks
- Suggests optimization landscape has intrinsic low-dimensional structure

This is a **testable hypothesis** about a fundamental question in deep learning!
