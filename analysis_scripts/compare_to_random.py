#!/usr/bin/env python3
"""
Critical test: Compare to RANDOM baseline.

If our models have dimension ≈ sample size, they're no different than random!
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

sns.set_style('whitegrid')

print("=" * 80)
print(" COMPARISON TO RANDOM BASELINE")
print("=" * 80)

# Load real data
weight_matrix = np.load('results_large_scale/weight_matrix.npy')
print(f"\nReal models: {weight_matrix.shape}")

# Generate random data (same shape)
random_matrix = np.random.randn(*weight_matrix.shape)
print(f"Random data: {random_matrix.shape}")

# Subsample both at different sizes
sample_sizes = list(range(10, 101, 10))
n_trials = 5

results_real = []
results_random = []

print(f"\nComputing dimensions...")
for size in sample_sizes:
    print(f"  Size {size}...", end='', flush=True)

    # Real data
    dims_real = []
    for _ in range(n_trials):
        indices = np.random.choice(100, size=size, replace=False)
        pca = PCA()
        pca.fit(weight_matrix[indices])
        var = pca.explained_variance_ratio_
        eff_dim = (np.sum(var) ** 2) / np.sum(var ** 2)
        dims_real.append(eff_dim)

    # Random data
    dims_random = []
    for _ in range(n_trials):
        indices = np.random.choice(100, size=size, replace=False)
        pca = PCA()
        pca.fit(random_matrix[indices])
        var = pca.explained_variance_ratio_
        eff_dim = (np.sum(var) ** 2) / np.sum(var ** 2)
        dims_random.append(eff_dim)

    results_real.append(np.mean(dims_real))
    results_random.append(np.mean(dims_random))

    print(f" real={np.mean(dims_real):.1f}, random={np.mean(dims_random):.1f}")

# Analysis
print(f"\n{'='*80}")
print(f" COMPARISON")
print(f"{'='*80}")

print(f"\n  Size | Real Dim | Random Dim | Difference")
print(f"  " + "-" * 50)
for i, size in enumerate(sample_sizes):
    diff = results_real[i] - results_random[i]
    diff_pct = (results_real[i] - results_random[i]) / results_random[i] * 100
    print(f"  {size:4d} | {results_real[i]:8.1f} | {results_random[i]:10.1f} | {diff:+5.1f} ({diff_pct:+.1f}%)")

# Average difference
avg_diff_pct = np.mean([(results_real[i] - results_random[i]) / results_random[i] * 100
                        for i in range(len(sample_sizes))])

print(f"\nAverage difference: {avg_diff_pct:.1f}%")

if abs(avg_diff_pct) < 5:
    print(f"\n  ⚠ DANGER: SAME AS RANDOM!")
    print(f"  → Models are NO MORE STRUCTURED than random noise")
    print(f"  → No manifold, just filling space")
elif avg_diff_pct < -10:
    print(f"\n  ✓ GOOD: Models MORE STRUCTURED than random")
    print(f"  → Real manifold structure exists")
    print(f"  → {abs(avg_diff_pct):.0f}% lower dimension than random")
elif avg_diff_pct > 10:
    print(f"\n  ? UNEXPECTED: Models LESS STRUCTURED than random")
    print(f"  → This shouldn't happen!")
else:
    print(f"\n  ? MARGINAL: Slight difference from random")
    print(f"  → Weak evidence for structure")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Direct comparison
ax1 = axes[0]
ax1.plot(sample_sizes, results_real, 'b.-', linewidth=2, markersize=8, label='Trained models')
ax1.plot(sample_sizes, results_random, 'r.-', linewidth=2, markersize=8, label='Random points')
ax1.plot(sample_sizes, sample_sizes, 'k--', alpha=0.3, label='y=x (maximum)')
ax1.fill_between(sample_sizes, results_real, results_random,
                alpha=0.2, color='green' if avg_diff_pct < -5 else 'red')
ax1.set_xlabel('Sample Size')
ax1.set_ylabel('Effective Dimension')
ax1.set_title('Trained Models vs Random Baseline')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Difference
ax2 = axes[1]
differences = [results_real[i] - results_random[i] for i in range(len(sample_sizes))]
ax2.plot(sample_sizes, differences, 'g.-', linewidth=2, markersize=8)
ax2.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.5)
ax2.fill_between(sample_sizes, 0, differences,
                alpha=0.3, color='green' if avg_diff_pct < 0 else 'red')
ax2.set_xlabel('Sample Size')
ax2.set_ylabel('Dimension Difference (Real - Random)')
ax2.set_title('Compression Relative to Random\n(Negative = More Structured)')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results_large_scale/random_comparison.png', dpi=300, bbox_inches='tight')
print(f"\nSaved: results_large_scale/random_comparison.png")

# Final verdict
print(f"\n{'='*80}")
print(f" VERDICT")
print(f"{'='*80}")

final_real = results_real[-1]
final_random = results_random[-1]
final_diff_pct = (final_real - final_random) / final_random * 100

print(f"\nAt n=100:")
print(f"  Real models: {final_real:.1f}D")
print(f"  Random points: {final_random:.1f}D")
print(f"  Difference: {final_diff_pct:+.1f}%")

if abs(final_diff_pct) < 5:
    print(f"\n  ✗ NO MEANINGFUL STRUCTURE")
    print(f"  → Models behave like random points")
    print(f"  → Dimension = sample size (both real and random)")
    print(f"  → The 'manifold' is an artifact!")
elif final_diff_pct < -10:
    print(f"\n  ✓ REAL STRUCTURE EXISTS")
    print(f"  → Models are {abs(final_diff_pct):.0f}% more compressed than random")
    print(f"  → True manifold dimension ≈ {final_real:.0f}D")
    print(f"  → BUT: Still grows with sample size (need to saturate!)")
else:
    print(f"\n  ? UNCLEAR")

print(f"\n{'='*80}")
