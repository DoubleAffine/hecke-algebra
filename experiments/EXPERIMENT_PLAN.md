# Experiment Plan: Multi-Task Intersection Analysis

## Goal

**Determine if different tasks share a common low-dimensional subspace in weight space.**

This tests the Universal Subspace Hypothesis: Do all neural networks (regardless of task) converge to a shared low-dimensional region?

---

## Critical Design Questions

### Q1: How many datasets?

**Proposal: Start with 2, expand to 4-5 if needed**

**Initial pair (most informative):**
1. `binary_classification_synthetic` - Signal (real patterns)
2. `binary_random_labels` - Noise (random labels)

**Why these two?**
- Same input/output dimensions (REQUIRED for weight comparison)
- Same data distribution, different labels
- Minimal confounding factors
- We already know they have similar single-task dimensions (~72D each)
- **Critical test:** If signal and noise DON'T share a subspace, nothing will!

**If we find intersection, add:**
3. More noise variants (different random seeds)
4. Different transformations of same data

**If we don't find intersection:**
→ No need for more datasets (hypothesis is falsified)

---

### Q2: How many models per dataset?

**Proposal: 100 models minimum, 200 preferred**

**Analysis:**

From our scaling law: `dimension ≈ 0.72 × sample_size`

| Models | Expected Dim | Samples/Dim | Status |
|--------|--------------|-------------|---------|
| 50     | ~36D         | 1.4         | ⚠ Marginal |
| 100    | ~72D         | 1.4         | ⚠ Marginal |
| 200    | ~144D        | 1.4         | ⚠ Marginal |
| 300    | ~216D        | 1.4         | ⚠ Marginal |

**Wait, this is a problem!**

The dimension KEEPS GROWING with sample size. So we can NEVER get 10 samples/dimension!

**Two interpretations:**

**A) Dimension hasn't saturated yet**
- True dimension might be 100-200D
- Need 1000+ models to properly sample it
- Can't do proper intersection test until we know true dimension

**B) No true manifold (dimension = sample size)**
- Models fill the space uniformly
- No low-D structure exists
- Intersection test is meaningless

**Resolution: WAIT for saturation test!**

The 300-model saturation test is still running. We MUST wait for it to complete before deciding sample size.

---

### Q3: What if dimension doesn't saturate?

**If saturation test shows continued growth:**

Then we have two options:

**Option 1: Test intersection anyway (exploratory)**
- Use 200 models per task
- Acknowledge undersampling
- Look for qualitative patterns (not quantitative claims)
- Check: Do principal angles vary with sample size?

**Option 2: Switch to different metrics**
- Don't use PCA dimension
- Use direct distance comparisons
- Classification-based: Can we distinguish signal from noise?
- Clustering-based: Do they mix or separate?

---

## Proposed Experimental Design

### Phase 1: Wait for Saturation Test

**Currently running:** 300 models on single task

**Wait for results to decide:**
1. Does dimension saturate? At what value?
2. What sample size is needed for reliable estimation?
3. Should we even do intersection test, or use different approach?

**Decision tree:**

```
IF dimension saturates at ~100-150D:
  → Proceed with 200 models per task (adequate)

ELSE IF dimension saturates at ~200-300D:
  → Need 500+ models per task (expensive, reconsider)

ELSE (no saturation):
  → Switch to distance-based analysis (no PCA)
```

---

### Phase 2A: If Dimension Saturates (Happy Path)

**Assumption:** Dimension saturates at ~120D

**Design:**

**Datasets:** 2 tasks
- Signal: `binary_classification_synthetic`
- Noise: `binary_random_labels`

**Sample size:** 200 models per task
- Expected dimension: ~120D per task
- Samples per dimension: 200/120 = 1.7
- Marginal but acceptable for initial test

**Training:**
```
For each task:
  For model_id in 1..200:
    1. Initialize random weights
    2. Train to convergence
    3. Extract final weights
    4. Save to disk (batch every 10 models)
    5. Delete training data
    6. Update progress
```

**Memory management:**
- Process in batches of 10 models
- Save incremental results
- Never hold full dataset in memory
- Peak memory: ~500MB

**Time estimate:**
- ~1.5 minutes per model
- 200 models × 2 tasks = 400 models
- 400 × 1.5 min = 600 minutes = **10 hours**

**Analysis after training:**

1. **Individual manifold dimensions**
   ```python
   For each task:
     - Compute PCA (all models from this task)
     - Find dimension (95% variance)
     - Find effective dimension (participation ratio)
     - Verify it matches saturation test prediction
   ```

2. **Principal angles between subspaces**
   ```python
   subspace_A = PCA(task_A_models).components[:dim_A]
   subspace_B = PCA(task_B_models).components[:dim_B]
   angles = principal_angles(subspace_A, subspace_B)

   aligned_dims = count(angles < 10°)
   ```

3. **Global analysis**
   ```python
   all_models = concatenate(task_A, task_B)
   global_PCA = PCA(all_models)
   global_dim = effective_dimension(global_PCA)

   if global_dim < mean(individual_dims):
     → Intersection exists
   else:
     → No intersection
   ```

4. **Verification checks**
   ```python
   # Check 1: Bootstrap stability
   For trial in 1..100:
     subsample = bootstrap(task_A_models)
     dims.append(effective_dim(subsample))

   if std(dims) < 5:
     → Reliable estimate

   # Check 2: Compare to random
   random_models = generate_random(same_shape)
   random_dim = effective_dim(random_models)

   if actual_dim < random_dim * 0.9:
     → Real structure
   ```

---

### Phase 2B: If Dimension Doesn't Saturate (Fallback)

**Alternative approach: Distance-based analysis**

**Don't use PCA dimensions (unreliable), instead:**

**Metric 1: Cross-task vs within-task distances**

```python
# Within task A
dist_AA = pairwise_distances(task_A_models)

# Within task B
dist_BB = pairwise_distances(task_B_models)

# Between tasks
dist_AB = distances(task_A_models, task_B_models)

# Compare
if mean(dist_AB) ≈ mean(dist_AA):
  → Tasks overlap (share subspace)

if mean(dist_AB) >> mean(dist_AA):
  → Tasks separate (no shared subspace)
```

**Metric 2: Clustering**

```python
all_models = concatenate(task_A, task_B)
labels_true = [0]*len(A) + [1]*len(B)  # Known task labels

# Can we separate them?
labels_pred = DBSCAN(all_models)

silhouette = silhouette_score(all_models, labels_true)

if silhouette < 0.2:
  → Tasks mixed (shared subspace)

if silhouette > 0.5:
  → Tasks separated (different subspaces)
```

**Metric 3: Classifier-based**

```python
# Train a classifier to distinguish task A from task B
X = all_weights
y = task_labels

classifier = LogisticRegression()
accuracy = cross_val_score(classifier, X, y)

if accuracy < 60%:
  → Can't distinguish (shared subspace)

if accuracy > 90%:
  → Easily distinguished (different subspaces)
```

---

## Recommended Analysis Pipeline

### Step 1: Basic Characterization

For each task separately:

```python
1. Load all model weights for this task
2. Compute pairwise distances
   - Mean, std, min, max
   - Compare to distance to origin
3. PCA analysis
   - Variance spectrum (first 100 components)
   - Cumulative variance
   - Effective dimension
4. Compare to random baseline
   - Generate random points (same size)
   - Compute dimension
   - Check if real < random
5. Save individual task summary
```

### Step 2: Intersection Analysis

Compare task A vs task B:

```python
1. Distance-based:
   - Within-task vs between-task distances
   - Statistical test: Are they significantly different?

2. PCA-based:
   - Principal angles between subspaces
   - Count aligned dimensions (< 10°)
   - Global PCA dimension

3. Geometric:
   - Project task A onto task B subspace
   - Measure reconstruction error
   - High error → orthogonal, Low error → shared

4. Statistical:
   - Bootstrap both estimates
   - Compute confidence intervals
   - Formal hypothesis test
```

### Step 3: Verification

```python
1. Robustness checks:
   - Subsample 50%, 75%, 100% of data
   - Does conclusion hold?

2. Method agreement:
   - Do distance-based, PCA-based, classifier-based agree?

3. Sanity checks:
   - Are signal and noise actually different tasks?
   - Check test accuracy distributions
   - Verify they learned different functions
```

---

## Deliverables

### During Experiment:

1. **Real-time progress display**
   - Progress bar with ETA
   - Current task, model number
   - Loss and accuracy
   - Memory usage

2. **Incremental saves**
   - Weights saved every 10 models
   - Metadata saved continuously
   - Can resume if interrupted

### After Experiment:

1. **Results JSON**
   ```json
   {
     "tasks": [...],
     "samples_per_task": 200,
     "individual_dimensions": {
       "task_A": {"effective": 118.3, "pca_95": 112},
       "task_B": {"effective": 121.7, "pca_95": 115}
     },
     "intersection": {
       "method": "principal_angles",
       "aligned_dimensions": 87,
       "mean_angle": 15.2,
       "conclusion": "SHARED_SUBSPACE"
     },
     "verification": {
       "distance_ratio": 1.02,
       "classifier_accuracy": 0.54,
       "conclusion": "CONFIRMED"
     }
   }
   ```

2. **Visualizations**
   - PCA scatter plots (task A vs B in PC space)
   - Distance distributions (within vs between)
   - Principal angle spectrum
   - Variance explained curves

3. **Report**
   - Experiment configuration
   - Key findings
   - Statistical significance
   - Confidence intervals
   - Conclusion with caveats

---

## Decision Points

### Before Starting:

**WAIT for saturation test results!**

If saturates at ~100D:
→ Proceed with 200 models per task

If saturates at ~200D:
→ Proceed with 500 models per task (long!)

If doesn't saturate:
→ Use distance-based approach (no PCA)

### After 2-Task Results:

**If strong intersection found (angles < 20°):**
→ Add 2 more tasks to verify
→ Test if intersection shrinks or stays stable

**If no intersection (angles > 70°):**
→ Stop (hypothesis falsified)
→ Write up negative result

**If ambiguous (angles 20-70°):**
→ Need more samples OR different metric
→ Try distance-based verification

---

## Risks and Mitigations

### Risk 1: Dimension still growing

**Mitigation:** Use distance-based methods instead of PCA

### Risk 2: Takes too long (>24 hours)

**Mitigation:**
- Reduce to 150 models per task
- Use shorter training (80 epochs instead of 100)
- Run overnight

### Risk 3: Memory issues

**Mitigation:**
- Incremental processing (built-in)
- Batch size = 10 (adjustable)
- Delete data after use

### Risk 4: Ambiguous results

**Mitigation:**
- Multiple analysis methods
- Statistical significance testing
- Bootstrap confidence intervals

---

## Final Recommendation

**DO NOT start yet!**

**Wait for:**
1. Saturation test to complete (~2 hours remaining)
2. Analyze saturation results
3. Decide on sample size and method
4. Then proceed with proper design

**Estimated timeline:**
- Wait for saturation: 2 hours
- Design refinement: 30 min
- Run intersection test: 6-10 hours
- Analysis: 1 hour
- **Total: ~14 hours from now**

**Start time suggestion:**
Tomorrow morning, so it runs during the day and completes by evening.

---

## Questions for Discussion

1. **Sample size:** 100, 200, or wait for saturation?
2. **Method:** PCA-based or distance-based or both?
3. **Tasks:** Just 2 (signal/noise) or start with 3-4?
4. **Urgency:** Start now or wait for saturation test?

What do you think?
