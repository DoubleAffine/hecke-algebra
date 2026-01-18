#!/usr/bin/env python3
"""
Save the intersection results that were computed successfully.
The experiment completed but failed on JSON serialization.
"""
import numpy as np
import json
from datetime import datetime
from sklearn.decomposition import PCA
from scipy.linalg import subspace_angles

print("=" * 80)
print(" SAVING INTERSECTION RESULTS")
print("=" * 80)

results_dir = 'experiments/current/intersection_proper'

# Load the computed weights
print("\nLoading weight matrices...")
weights_signal = np.load(f'{results_dir}/weights_binary_classification_synthetic.npy')
weights_noise = np.load(f'{results_dir}/weights_binary_random_labels.npy')

print(f"  Signal: {weights_signal.shape}")
print(f"  Noise: {weights_noise.shape}")

# Recompute dimensions
print("\nComputing dimensions...")

tasks = {
    'binary_classification_synthetic': weights_signal,
    'binary_random_labels': weights_noise
}

task_dimensions = {}

for task, weights in tasks.items():
    pca = PCA()
    pca.fit(weights)

    var_ratios = pca.explained_variance_ratio_
    cumsum = np.cumsum(var_ratios)

    dim_95 = int(np.argmax(cumsum >= 0.95) + 1)
    dim_99 = int(np.argmax(cumsum >= 0.99) + 1)
    effective_dim = float((np.sum(var_ratios) ** 2) / np.sum(var_ratios ** 2))

    task_dimensions[task] = {
        'dim_95': dim_95,
        'dim_99': dim_99,
        'effective_dim': effective_dim
    }

    print(f"  {task}:")
    print(f"    95%: {dim_95}D, 99%: {dim_99}D, effective: {effective_dim:.1f}D")

# Compute intersection
print("\nComputing intersection...")

pca1 = PCA(n_components=task_dimensions['binary_classification_synthetic']['dim_95'])
pca1.fit(weights_signal)

pca2 = PCA(n_components=task_dimensions['binary_random_labels']['dim_95'])
pca2.fit(weights_noise)

angles = subspace_angles(pca1.components_.T, pca2.components_.T)
angles_deg = np.degrees(angles)

aligned = int(np.sum(angles_deg < 10))

print(f"  Principal angles: min={np.min(angles_deg):.1f}°, mean={np.mean(angles_deg):.1f}°, max={np.max(angles_deg):.1f}°")
print(f"  Aligned dimensions (< 10°): {aligned}")

# Global analysis
print("\nGlobal analysis...")
all_weights = np.vstack([weights_signal, weights_noise])

pca_global = PCA()
pca_global.fit(all_weights)

var_global = pca_global.explained_variance_ratio_
cumsum_global = np.cumsum(var_global)

dim_95_global = int(np.argmax(cumsum_global >= 0.95) + 1)
effective_global = float((np.sum(var_global) ** 2) / np.sum(var_global ** 2))

print(f"  Global dimension (95%): {dim_95_global}D")
print(f"  Global effective: {effective_global:.1f}D")

# Analysis
mean_individual = np.mean([task_dimensions[t]['effective_dim'] for t in tasks])
ratio = effective_global / mean_individual

print(f"\nComparison:")
print(f"  Mean individual: {mean_individual:.1f}D")
print(f"  Global: {effective_global:.1f}D")
print(f"  Ratio: {ratio:.2f}")

if effective_global < mean_individual * 0.9:
    intersection_dim = effective_global
    conclusion = "INTERSECTION EXISTS"
    print(f"\n  ✓ {conclusion}: ~{intersection_dim:.0f}D")
elif effective_global > mean_individual * 1.3:
    intersection_dim = 0
    conclusion = "NO INTERSECTION (orthogonal subspaces)"
    print(f"\n  ✗ {conclusion}")
else:
    intersection_dim = effective_global * 0.7
    conclusion = "PARTIAL OVERLAP"
    print(f"\n  ? {conclusion}: ~{intersection_dim:.0f}D estimated")

# Save results
results = {
    'config': {
        'models_per_task': weights_signal.shape[0],
        'tasks': list(tasks.keys()),
        'completed': datetime.now().isoformat()
    },
    'task_dimensions': task_dimensions,
    'intersection': {
        'aligned_dimensions': aligned,
        'min_angle': float(np.min(angles_deg)),
        'mean_angle': float(np.mean(angles_deg)),
        'max_angle': float(np.max(angles_deg)),
        'global_dimension_95': dim_95_global,
        'global_effective': effective_global,
        'mean_individual': mean_individual,
        'ratio': ratio,
        'intersection_estimate': intersection_dim,
        'conclusion': conclusion
    }
}

print(f"\nSaving results...")
with open(f'{results_dir}/results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"  ✓ Saved to {results_dir}/results.json")

print("\n" + "=" * 80)
print(" SUMMARY")
print("=" * 80)

print(f"\nDatasets tested: 2")
print(f"  - binary_classification_synthetic (signal)")
print(f"  - binary_random_labels (noise)")

print(f"\nModels per dataset: {weights_signal.shape[0]}")

print(f"\nDimensions:")
print(f"  Signal: {task_dimensions['binary_classification_synthetic']['effective_dim']:.1f}D")
print(f"  Noise: {task_dimensions['binary_random_labels']['effective_dim']:.1f}D")
print(f"  Global: {effective_global:.1f}D")

print(f"\nIntersection:")
print(f"  {conclusion}")
print(f"  Aligned dimensions: {aligned}")
print(f"  Estimated overlap: ~{intersection_dim:.0f}D")

print("\n" + "=" * 80)
