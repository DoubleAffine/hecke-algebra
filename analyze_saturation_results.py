#!/usr/bin/env python3
"""
Quick analysis of the saturation test results to determine proper sample size.
"""
import numpy as np
from sklearn.decomposition import PCA

print("=" * 80)
print(" SATURATION TEST RESULTS")
print("=" * 80)

# Load all checkpoint weights
checkpoints = [50, 100, 150, 200, 250, 300]
dimensions = []

print("\nAnalyzing dimension at each checkpoint:\n")

for n in checkpoints:
    weights = np.load(f'results_saturation/weights_{n}.npy')

    # Compute effective dimension
    pca = PCA()
    pca.fit(weights)

    var_ratios = pca.explained_variance_ratio_
    cumsum = np.cumsum(var_ratios)

    dim_95 = np.argmax(cumsum >= 0.95) + 1
    dim_99 = np.argmax(cumsum >= 0.99) + 1
    effective_dim = (np.sum(var_ratios) ** 2) / np.sum(var_ratios ** 2)

    dimensions.append({
        'n': n,
        'dim_95': dim_95,
        'dim_99': dim_99,
        'effective': effective_dim
    })

    print(f"n={n:3d}: dim_95={dim_95:3d}D, dim_99={dim_99:3d}D, effective={effective_dim:6.2f}D")

print("\n" + "=" * 80)
print(" SATURATION ANALYSIS")
print("=" * 80)

# Check if dimension is saturating
eff_dims = [d['effective'] for d in dimensions]

print(f"\nEffective dimensions:")
for i, d in enumerate(dimensions):
    print(f"  {d['n']:3d} models: {d['effective']:6.2f}D")

print(f"\nGrowth analysis:")
for i in range(1, len(dimensions)):
    prev = dimensions[i-1]
    curr = dimensions[i]

    n_increase = curr['n'] - prev['n']
    dim_increase = curr['effective'] - prev['effective']
    rate = dim_increase / n_increase

    print(f"  {prev['n']:3d}→{curr['n']:3d}: +{dim_increase:5.2f}D (+{n_increase} models) = {rate:.3f}D per model")

# Check for saturation
print(f"\nSaturation check:")

# Last 100 models
last_rate = (dimensions[-1]['effective'] - dimensions[-2]['effective']) / 50
first_rate = (dimensions[1]['effective'] - dimensions[0]['effective']) / 50

print(f"  Growth rate (first 50→100): {first_rate:.3f}D per model")
print(f"  Growth rate (last 250→300): {last_rate:.3f}D per model")
print(f"  Slowdown: {(1 - last_rate/first_rate)*100:.1f}%")

if last_rate < first_rate * 0.3:
    print(f"\n  ✓ SATURATED: Growth slowed to <30% of initial rate")
    saturated = True
    estimated_true_dim = dimensions[-1]['effective'] * 1.1  # Small safety margin
    print(f"  Estimated true dimension: ~{estimated_true_dim:.0f}D")
elif last_rate < first_rate * 0.6:
    print(f"\n  ~ APPROACHING SATURATION: Growth slowed to ~{last_rate/first_rate*100:.0f}%")
    saturated = False
    print(f"  Recommend 400-500 models for full saturation")
else:
    print(f"\n  ✗ NOT SATURATED: Still growing at {last_rate/first_rate*100:.0f}% of initial rate")
    saturated = False
    print(f"  Dimension may continue growing significantly")

print("\n" + "=" * 80)
print(" RECOMMENDATION FOR INTERSECTION TEST")
print("=" * 80)

if saturated:
    # We've saturated, can use reasonable sample size
    true_dim = dimensions[-1]['effective']
    recommended_n = int(true_dim * 5)  # 5 samples per dimension

    print(f"\nDimension has saturated at ~{true_dim:.0f}D")
    print(f"Recommended samples per task: {recommended_n}")
    print(f"  (5 samples per dimension for reliable estimation)")
else:
    # Not saturated, need to be more conservative
    latest_dim = dimensions[-1]['effective']

    print(f"\nDimension has NOT fully saturated (at {latest_dim:.0f}D with 300 models)")
    print(f"\nOptions:")
    print(f"  A) Use 300 models per task (current maximum)")
    print(f"     - Gives best dimension estimate we have")
    print(f"     - May underestimate true dimension")
    print(f"  B) Continue saturation test to 500+ models")
    print(f"     - More accurate dimension estimate")
    print(f"     - Takes more time")

print("\n" + "=" * 80)
