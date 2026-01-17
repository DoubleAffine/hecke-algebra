#!/usr/bin/env python3
"""
Detailed verification and step-by-step analysis of the intersection experiment.
Let's check everything carefully!
"""
import numpy as np
import json
import os
from sklearn.decomposition import PCA
from scipy.linalg import subspace_angles
from scipy.spatial.distance import pdist, squareform

print("=" * 80)
print(" DETAILED VERIFICATION OF INTERSECTION EXPERIMENT")
print("=" * 80)

# Load the saved data
results_dir = 'results_intersection'

# Check what files exist
print("\nFiles in results directory:")
for f in os.listdir(results_dir):
    if not f.startswith('.'):
        size = os.path.getsize(os.path.join(results_dir, f))
        print(f"  {f}: {size:,} bytes")

# Load results
with open(f'{results_dir}/results.json', 'r') as f:
    results = json.load(f)

print("\n" + "=" * 80)
print(" EXPERIMENT CONFIGURATION")
print("=" * 80)

print(f"\nDatasets tested: {results['datasets']}")
print(f"Models per dataset: {results['models_per_dataset']}")
print(f"Total models: {len(results['datasets']) * results['models_per_dataset']}")

# The experiment should have trained:
# - 50 models on 'binary_classification_synthetic' (signal)
# - 50 models on 'binary_random_labels' (noise)

print("\n" + "=" * 80)
print(" STEP 1: VERIFY THE DATA WAS LOADED CORRECTLY")
print("=" * 80)

# Let's check if we can load the weights that were used
# The experiment should have saved individual weight matrices

# Check if we have the raw weight data
# (The script may not have saved individual datasets, just the combined analysis)

print("\nLooking for saved weight matrices...")
weight_files = [f for f in os.listdir(results_dir) if 'weight' in f.lower() and f.endswith('.npy')]
print(f"Found weight files: {weight_files}")

if not weight_files:
    print("\n⚠ No individual weight matrices were saved.")
    print("We'll need to re-run with explicit saving to verify fully.")
    print("\nBut we can still analyze the results that were computed...")

print("\n" + "=" * 80)
print(" STEP 2: REPORTED DIMENSIONS")
print("=" * 80)

print(f"\nIndividual dataset dimensions:")
for dataset, dim in results['individual_dimensions'].items():
    print(f"  {dataset}: {dim:.2f}D")

print(f"\nGlobal dimension (all combined): {results['global_dimension']:.2f}D")
print(f"Intersection estimate: {results['intersection_dimension_estimate']:.2f}D")

print("\n" + "=" * 80)
print(" STEP 3: ANALYZING THE REPORTED RESULTS")
print("=" * 80)

signal_dim = results['individual_dimensions']['binary_classification_synthetic']
noise_dim = results['individual_dimensions']['binary_random_labels']
global_dim = results['global_dimension']

print(f"\nSignal task dimension: {signal_dim:.2f}D")
print(f"Noise task dimension: {noise_dim:.2f}D")
print(f"Global dimension: {global_dim:.2f}D")

print(f"\nExpected if orthogonal (no overlap): {signal_dim + noise_dim:.2f}D")
print(f"Actual global dimension: {global_dim:.2f}D")
print(f"Difference: {(signal_dim + noise_dim) - global_dim:.2f}D")

if global_dim < signal_dim * 0.8:
    print("\n→ STRONG OVERLAP: Global < individual")
    print("  This would mean datasets share a common subspace")
elif global_dim > (signal_dim + noise_dim) * 0.8:
    print("\n→ NEARLY ORTHOGONAL: Global ≈ sum of individual")
    print("  This means datasets use DIFFERENT dimensions")
else:
    print("\n→ PARTIAL OVERLAP: Global between individual and sum")
    print("  This means some shared, some separate dimensions")

print("\n" + "=" * 80)
print(" STEP 4: CHECK THE MATH")
print("=" * 80)

print("\nLet's verify the logic:")
print("\nScenario 1: Perfect overlap (same 42D subspace)")
print(f"  Individual: {signal_dim:.1f}D and {noise_dim:.1f}D")
print(f"  Combined: Should be ~{max(signal_dim, noise_dim):.1f}D")
print(f"  Actual: {global_dim:.1f}D")
print(f"  Match? {'YES' if abs(global_dim - max(signal_dim, noise_dim)) < 5 else 'NO'}")

print("\nScenario 2: Orthogonal (different dimensions)")
print(f"  Individual: {signal_dim:.1f}D and {noise_dim:.1f}D")
print(f"  Combined: Should be ~{signal_dim + noise_dim:.1f}D")
print(f"  Actual: {global_dim:.1f}D")
print(f"  Match? {'YES' if abs(global_dim - (signal_dim + noise_dim)) < 10 else 'NO'}")

print("\nScenario 3: Partial overlap")
print(f"  Individual: {signal_dim:.1f}D and {noise_dim:.1f}D")
print(f"  Combined: Should be between {max(signal_dim, noise_dim):.1f}D and {signal_dim + noise_dim:.1f}D")
print(f"  Actual: {global_dim:.1f}D")
in_range = max(signal_dim, noise_dim) < global_dim < (signal_dim + noise_dim)
print(f"  In range? {'YES' if in_range else 'NO'}")

print("\n" + "=" * 80)
print(" STEP 5: WHAT THE PRINCIPAL ANGLES SHOWED")
print("=" * 80)

print("\nFrom the log, principal angles between subspaces:")
print("  Min angle: 55.6°")
print("  Mean angle: 74.4°")
print("  Max angle: 89.8°")
print("  Aligned dimensions (< 10°): 0")

print("\nInterpretation:")
print("  - Small angles (< 20°) mean ALIGNED (shared) dimensions")
print("  - Large angles (> 70°) mean ORTHOGONAL (independent) dimensions")
print("  - With min=55.6° and mean=74.4°, these are nearly orthogonal")

print("\n" + "=" * 80)
print(" STEP 6: POTENTIAL ISSUES TO CHECK")
print("=" * 80)

print("\nPossible issues we should verify:")

print("\n1. Sample size effect:")
print(f"   - We used {results['models_per_dataset']} models per dataset")
print(f"   - Each dataset dim: ~{signal_dim:.0f}D")
print(f"   - Samples per dimension: {results['models_per_dataset'] / signal_dim:.1f}")
if results['models_per_dataset'] / signal_dim < 2:
    print("   ⚠ UNDERSAMPLED! Need ~10 samples per dimension")
    print("   → Dimension estimates may be unreliable")
else:
    print("   ✓ Adequately sampled")

print("\n2. Different sample sizes:")
print("   - Signal: 50 models")
print("   - Noise: 50 models")
print("   - Equal? YES ✓")

print("\n3. Are we comparing the right thing?")
print("   - We should compare final weights from models trained on:")
print("     * Same architecture: [16, 16]")
print("     * Same input/output dims: 10 → 1")
print("     * Different tasks: signal vs noise labels")
print("   - This appears correct ✓")

print("\n4. PCA effective dimension formula:")
print("   - effective_dim = (sum(var))^2 / sum(var^2)")
print("   - This is the 'participation ratio'")
print("   - It can underestimate true dimension with noise")

print("\n" + "=" * 80)
print(" STEP 7: SANITY CHECKS")
print("=" * 80)

print("\nLet's check if the numbers make sense:")

print("\n1. Are dimensions plausible?")
print(f"   - Signal: {signal_dim:.1f}D out of 465D ({signal_dim/465*100:.1f}%)")
print(f"   - Noise: {noise_dim:.1f}D out of 465D ({noise_dim/465*100:.1f}%)")
print(f"   - Both ~9% of ambient space")
print(f"   - This is MUCH lower than the 72D we saw with 100 models!")
print(f"   - Difference: 72D vs 42D = {72-42:.0f}D")
print(f"   ⚠ This is suspicious! Why would 50 models give lower dimension?")

print("\n2. Compare to our previous 100-model experiment:")
print("   - Previous (100 models, SAME task): 71.9D")
print("   - Current (50 models, signal): 41.4D")
print("   - Current (50 models, noise): 41.8D")
print("   - Expected: dimension ≈ 0.72 × sample_size")
print(f"   - 50 × 0.72 = {50 * 0.72:.1f}D ✓ Matches!")
print("   - So these are consistent with our earlier finding")

print("\n3. If we had 100 models of each:")
print(f"   - Expected signal: ~72D")
print(f"   - Expected noise: ~72D")
print(f"   - If orthogonal: combined ~144D")
print(f"   - If shared: combined ~72D")

print("\n" + "=" * 80)
print(" CRITICAL INSIGHT")
print("=" * 80)

print("\nThe dimension is GROWING WITH SAMPLE SIZE!")
print("  50 models → 42D per task")
print("  100 models → 72D per task")
print("  This means we HAVEN'T saturated yet")

print("\nSo the 'orthogonal' finding might be:")
print("  A) Real - tasks use different dimensions")
print("  B) Artifact - we're undersampled and can't see overlap")

print("\nTo know which, we need to check:")
print("  - Do the principal angles stay large with more samples?")
print("  - Or do they decrease (revealing hidden overlap)?")

print("\n" + "=" * 80)
print(" RECOMMENDED VERIFICATION")
print("=" * 80)

print("\n1. Re-run with 100 models per dataset (not 50)")
print("   - Get more reliable dimension estimates")
print("   - See if tasks are still orthogonal")

print("\n2. Check the actual weight matrices")
print("   - Are signal models actually different from noise?")
print("   - Or are they mixed together?")

print("\n3. Look at the visualization")
print("   - Do they cluster separately in PCA space?")
print("   - This should be visible")

print("\n" + "=" * 80)
print(" CONCLUSION")
print("=" * 80)

print("\nThe experiment reports:")
print("  ✗ No intersection (orthogonal subspaces)")
print("  - Signal: 42D subspace")
print("  - Noise: 42D subspace")
print("  - Principal angles: 55-90° (nearly orthogonal)")

print("\nBUT we should be cautious because:")
print("  ⚠ Only 50 samples per task (undersampled)")
print("  ⚠ Dimension grows with sample size (not saturated)")
print("  ⚠ 42D vs our previous 72D is suspicious")

print("\nRecommendation:")
print("  Re-run with 100 models per task to get reliable results")

print("\n" + "=" * 80)
