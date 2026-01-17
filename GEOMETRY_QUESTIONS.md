# Deep Geometric Questions About the Weight Manifold

## What We Know So Far

From the dynamics experiment:
- Models converge to a **low-dimensional manifold** (~5D out of 465D)
- This happens **regardless of task** (signal vs noise don't separate)
- The manifold is determined by **optimization dynamics**, not task semantics

## Open Questions to Investigate

### 1. Manifold Structure

**Q: Is it a smooth manifold or fractal?**
- Correlation dimension: 3.47 (lower than intrinsic dim 4.9) suggests fractal
- Need more samples to confirm
- **Experiment**: Train 100+ models, compute multiple fractal dimensions

**Q: What is the topology?**
- Is it simply connected? Does it have holes?
- Persistent homology can detect topological features
- **Experiment**: Compute Betti numbers, persistence diagrams

**Q: Does it have preferred regions (attractors)?**
- Do models cluster in certain areas?
- Are there multiple basins?
- **Experiment**: Density estimation, clustering with many models

### 2. Curvature and Geometry

**Q: What is the curvature of the manifold?**
- Positive curvature: sphere-like
- Negative curvature: saddle-like (common in loss landscapes)
- Zero curvature: flat
- **Experiment**: Estimate Riemann curvature tensor locally

**Q: Is there a natural metric?**
- Distance in weight space vs. distance in function space
- Fisher information metric?
- **Experiment**: Compare different distance metrics

### 3. Universality

**Q: Does architecture affect manifold structure?**
- Compare [16,16] vs [32,32] vs [8,8,8]
- Do they have same intrinsic dimension?
- **Experiment**: Train models with different architectures

**Q: Does dataset size matter?**
- Small vs large datasets
- Does manifold change with more data?
- **Experiment**: Vary dataset size systematically

**Q: Does learning rate affect the manifold?**
- Different LR → different paths → different final locations?
- Or does LR just affect speed of convergence to same manifold?
- **Experiment**: Train with LR ∈ [0.0001, 0.001, 0.01]

### 4. Dynamics on the Manifold

**Q: What are the optimization trajectories?**
- Do they spiral toward attractors?
- Are they chaotic?
- **Experiment**: Analyze trajectories (we have this data!)

**Q: Is there mode connectivity?**
- Can you draw a path on the manifold between any two solutions?
- Loss barrier height?
- **Experiment**: Linear interpolation + finding low-loss paths

**Q: What happens at initialization?**
- Do models start near the manifold or find it during training?
- **Experiment**: Track distance to manifold during training

### 5. Functional Properties

**Q: How does position on manifold relate to performance?**
- Do better models cluster together?
- Is there a "good region" of the manifold?
- **Experiment**: Color points by test accuracy

**Q: Can we characterize the manifold explicitly?**
- Find generators/coordinates
- Parameterize the manifold
- **Experiment**: Learn a generative model of the manifold

**Q: What functions are represented?**
- Do different regions correspond to different learned features?
- **Experiment**: Decode manifold coordinates back to functions

## Proposed Experiments

### Experiment 1: High-Resolution Manifold Sampling
- Train 500 models on same task
- Compute all geometric properties with high precision
- Create high-quality visualizations

### Experiment 2: Architecture Sweep
- Train 50 models each for 10 different architectures
- Compare manifold properties across architectures
- Test universality hypothesis

### Experiment 3: Trajectory Analysis
- Use existing trajectory data
- Compute curvature along paths
- Detect chaotic vs regular dynamics

### Experiment 4: Topological Analysis
- Persistent homology on 100+ model weights
- Detect holes, voids, connected components
- Mapper algorithm for visualization

### Experiment 5: Mode Connectivity
- Take pairs of converged models
- Find paths between them with low loss
- Characterize the geometry of solution space

## Tools We Need

Current tools:
- ✓ PCA
- ✓ UMAP
- ✓ Intrinsic dimension (MLE)
- ✓ Correlation dimension
- ✓ Box-counting dimension
- ✓ Clustering

Additional tools needed:
- ☐ Persistent homology (ripser library - already in requirements!)
- ☐ Curvature estimation
- ☐ Mode connectivity algorithms
- ☐ Trajectory analysis (Lyapunov exponents)
- ☐ Density estimation
- ☐ Geodesic computation

## What Would Be Most Interesting?

My vote: **Experiment 4 (Topological Analysis)**

Why?
1. Persistent homology can detect multi-scale structure
2. Would definitively answer if there are holes/voids
3. Relates to theoretical questions about loss landscapes
4. Visual persistent homology diagrams are compelling
5. We already have ripser installed!

This would tell us if the manifold is topologically trivial or has interesting structure.
