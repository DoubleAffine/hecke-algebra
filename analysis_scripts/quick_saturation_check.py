#!/usr/bin/env python3
"""
Quick saturation check using existing 100-model data.
Subsample at different sizes to see if dimension saturates.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

sns.set_style('whitegrid')

print("=" * 80)
print(" QUICK SATURATION CHECK (using existing 100 models)")
print("=" * 80)

# Load existing data
weight_matrix = np.load('results_large_scale/weight_matrix.npy')
print(f"\nLoaded: {weight_matrix.shape}")

# Subsample at different sizes
sample_sizes = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
n_trials = 5  # Multiple trials per size

results = []

print(f"\nSubsampling at different sizes:")
for size in sample_sizes:
    print(f"  Size {size}...", end='', flush=True)

    dims_95 = []
    effective_dims = []

    for trial in range(n_trials):
        # Random subsample
        indices = np.random.choice(100, size=size, replace=False)
        subsample = weight_matrix[indices]

        # PCA
        pca = PCA()
        pca.fit(subsample)

        var_ratios = pca.explained_variance_ratio_
        cumsum = np.cumsum(var_ratios)

        dim_95 = np.argmax(cumsum >= 0.95) + 1
        effective_dim = (np.sum(var_ratios) ** 2) / np.sum(var_ratios ** 2)

        dims_95.append(dim_95)
        effective_dims.append(effective_dim)

    # Store average
    results.append({
        'size': size,
        'dim_95_mean': np.mean(dims_95),
        'dim_95_std': np.std(dims_95),
        'effective_mean': np.mean(effective_dims),
        'effective_std': np.std(effective_dims)
    })

    print(f" effective = {np.mean(effective_dims):.1f} ± {np.std(effective_dims):.1f}")

# Analysis
print(f"\n{'='*80}")
print(f" SATURATION ANALYSIS")
print(f"{'='*80}")

sizes = [r['size'] for r in results]
effective_means = [r['effective_mean'] for r in results]

# Check growth rate
early_mean = np.mean(effective_means[:3])  # First 3 points
late_mean = np.mean(effective_means[-3:])  # Last 3 points
growth_rate = (late_mean - early_mean) / early_mean

print(f"\nDimension evolution:")
print(f"  Early (n=10-30): {early_mean:.1f}D")
print(f"  Late (n=80-100): {late_mean:.1f}D")
print(f"  Growth rate: {growth_rate*100:+.1f}%")

if abs(growth_rate) < 0.10:
    print(f"\n  ✓ SATURATED")
    print(f"  → Dimension stable (< 10% change)")
    print(f"  → True dimension ≈ {late_mean:.0f}D")
elif growth_rate > 0.30:
    print(f"\n  ✗ GROWING LINEARLY")
    print(f"  → Dimension ≈ sample size")
    print(f"  → NO MANIFOLD STRUCTURE!")
    print(f"  → Models fill the space!")
else:
    print(f"\n  ? UNCERTAIN")
    print(f"  → Moderate growth")
    print(f"  → May saturate with more samples")

# Check if approaching ambient space
ambient_dim = 465
compression_ratio = ambient_dim / late_mean
print(f"\nCompression:")
print(f"  Ambient: {ambient_dim}D")
print(f"  Effective: {late_mean:.1f}D")
print(f"  Ratio: {compression_ratio:.1f}×")

if compression_ratio < 2:
    print(f"  ⚠ DANGER: Nearly full space!")
elif compression_ratio < 5:
    print(f"  ⚠ WARNING: Modest compression")
else:
    print(f"  ✓ GOOD: Clear manifold structure")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Dimension vs sample size
ax1 = axes[0]
ax1.errorbar(sizes, effective_means,
            yerr=[r['effective_std'] for r in results],
            fmt='bo-', linewidth=2, markersize=8,
            capsize=5, label='Effective dimension')

# Reference lines
ax1.plot(sizes, sizes, 'r--', alpha=0.3, label='y=x (no structure)')
ax1.axhline(y=late_mean, color='green', linestyle='--',
           alpha=0.5, label=f'Saturated value (~{late_mean:.0f}D)')

ax1.set_xlabel('Sample Size')
ax1.set_ylabel('Effective Dimension')
ax1.set_title('Saturation Test: Dimension vs Sample Size')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Growth rate
ax2 = axes[1]
if len(sizes) > 1:
    growth_rates = [0] + [
        (effective_means[i] - effective_means[i-1]) / effective_means[i-1] * 100
        for i in range(1, len(effective_means))
    ]
    ax2.plot(sizes, growth_rates, 'ro-', linewidth=2, markersize=6)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.axhline(y=10, color='orange', linestyle='--', alpha=0.5, label='10% threshold')
    ax2.fill_between(sizes, -10, 10, alpha=0.2, color='green', label='Saturation zone')
    ax2.set_xlabel('Sample Size')
    ax2.set_ylabel('Growth Rate (%)')
    ax2.set_title('Dimension Growth Rate\n(Flat in green zone = Saturated)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results_large_scale/saturation_check.png', dpi=300, bbox_inches='tight')
print(f"\nSaved: results_large_scale/saturation_check.png")

# Detailed table
print(f"\n{'='*80}")
print(f" DETAILED RESULTS")
print(f"{'='*80}")
print(f"\n  Size | Effective Dim | Std  | Growth")
print(f"  " + "-" * 40)
for i, r in enumerate(results):
    if i == 0:
        growth = "-"
    else:
        growth_pct = (r['effective_mean'] - results[i-1]['effective_mean']) / results[i-1]['effective_mean'] * 100
        growth = f"{growth_pct:+.1f}%"
    print(f"  {r['size']:4d} | {r['effective_mean']:13.1f} | {r['effective_std']:4.1f} | {growth}")

print(f"\n{'='*80}")
