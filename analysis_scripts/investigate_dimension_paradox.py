#!/usr/bin/env python3
"""
Investigate the dimension paradox: Why does PCA say 1D but MLE says 46.89D?

This script analyzes the 100-model results to understand the geometry.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import json

sns.set_style('whitegrid')

print("=" * 80)
print(" INVESTIGATING DIMENSION PARADOX")
print(" PCA: 1D vs MLE: 46.89D")
print("=" * 80)

# Load data
print("\nLoading data...")
weight_matrix = np.load('results_large_scale/weight_matrix.npy')
with open('results_large_scale/metadata.json', 'r') as f:
    metadata = json.load(f)

print(f"Weight matrix shape: {weight_matrix.shape}")
print(f"Number of models: {len(metadata)}")

# 1. DETAILED PCA ANALYSIS
print("\n" + "=" * 80)
print(" 1. DETAILED PCA ANALYSIS")
print("=" * 80)

pca = PCA()
pca.fit(weight_matrix)
variance_ratios = pca.explained_variance_ratio_

# Cumulative variance
cumsum_variance = np.cumsum(variance_ratios)

print(f"\nFirst 20 components:")
for i in range(min(20, len(variance_ratios))):
    print(f"  PC{i+1}: {variance_ratios[i]:.6f} ({cumsum_variance[i]:.4f} cumulative)")

# Find how many PCs for different thresholds
for threshold in [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]:
    n_components = np.argmax(cumsum_variance >= threshold) + 1
    print(f"\nComponents for {threshold*100:.0f}% variance: {n_components}")
    print(f"  Total variance captured: {cumsum_variance[n_components-1]:.4f}")

# Effective dimensionality (participation ratio)
participation_ratio = (np.sum(variance_ratios) ** 2) / np.sum(variance_ratios ** 2)
print(f"\nEffective dimensionality (participation ratio): {participation_ratio:.2f}")

# 2. GEOMETRY IN PC SPACE
print("\n" + "=" * 80)
print(" 2. GEOMETRY IN PRINCIPAL COMPONENT SPACE")
print("=" * 80)

pca_transformed = pca.transform(weight_matrix)

# Analyze spread in each PC direction
print(f"\nSpread (standard deviation) along each PC:")
for i in range(min(10, pca_transformed.shape[1])):
    std = np.std(pca_transformed[:, i])
    min_val = np.min(pca_transformed[:, i])
    max_val = np.max(pca_transformed[:, i])
    print(f"  PC{i+1}: std={std:.4f}, range=[{min_val:.4f}, {max_val:.4f}]")

# 3. DISTANCE STRUCTURE
print("\n" + "=" * 80)
print(" 3. DISTANCE STRUCTURE IN WEIGHT SPACE")
print("=" * 80)

# Pairwise distances
from scipy.spatial.distance import pdist, squareform
distances = pdist(weight_matrix, metric='euclidean')
dist_matrix = squareform(distances)

print(f"\nPairwise distances:")
print(f"  Mean: {np.mean(distances):.4f}")
print(f"  Std: {np.std(distances):.4f}")
print(f"  Min: {np.min(distances):.4f}")
print(f"  Max: {np.max(distances):.4f}")
print(f"  Median: {np.median(distances):.4f}")

# Distance to centroid
centroid = np.mean(weight_matrix, axis=0)
distances_to_centroid = np.linalg.norm(weight_matrix - centroid, axis=1)

print(f"\nDistances to centroid:")
print(f"  Mean: {np.mean(distances_to_centroid):.4f}")
print(f"  Std: {np.std(distances_to_centroid):.4f}")
print(f"  Min: {np.min(distances_to_centroid):.4f}")
print(f"  Max: {np.max(distances_to_centroid):.4f}")

# 4. LOCAL vs GLOBAL DIMENSIONALITY
print("\n" + "=" * 80)
print(" 4. LOCAL vs GLOBAL DIMENSIONALITY")
print("=" * 80)

# Check if MLE dimension varies across the manifold
from sklearn.neighbors import NearestNeighbors

def estimate_local_dimension(point_idx, k=10):
    """Estimate intrinsic dimension near a specific point."""
    # Get k nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(weight_matrix)
    distances, indices = nbrs.kneighbors(weight_matrix[point_idx:point_idx+1])

    # Exclude the point itself
    distances = distances[0, 1:]

    # MLE estimator
    if len(distances) < 2:
        return np.nan

    # Remove zeros to avoid log(0)
    distances = distances[distances > 1e-10]
    if len(distances) < 2:
        return np.nan

    log_distances = np.log(distances)
    dimension = -1 / np.mean(log_distances - np.log(distances[-1]))

    return dimension

# Sample 20 points across the manifold
sample_indices = np.linspace(0, len(weight_matrix)-1, 20, dtype=int)
local_dims = []

print(f"\nLocal dimension estimates (k=20):")
for idx in sample_indices:
    local_dim = estimate_local_dimension(idx, k=20)
    if not np.isnan(local_dim) and local_dim > 0:
        local_dims.append(local_dim)
        print(f"  Point {idx}: {local_dim:.2f}D")

if local_dims:
    print(f"\nLocal dimension statistics:")
    print(f"  Mean: {np.mean(local_dims):.2f}")
    print(f"  Std: {np.std(local_dims):.2f}")
    print(f"  Range: [{np.min(local_dims):.2f}, {np.max(local_dims):.2f}]")

# 5. VISUALIZATIONS
print("\n" + "=" * 80)
print(" 5. CREATING DIAGNOSTIC VISUALIZATIONS")
print("=" * 80)

fig = plt.figure(figsize=(20, 12))

# Plot 1: PCA variance (log scale)
ax1 = plt.subplot(2, 3, 1)
n_components = min(50, len(variance_ratios))
ax1.semilogy(range(1, n_components+1), variance_ratios[:n_components], 'b.-', linewidth=2)
ax1.axhline(y=0.01, color='r', linestyle='--', label='1% threshold')
ax1.axhline(y=0.001, color='orange', linestyle='--', label='0.1% threshold')
ax1.set_xlabel('Principal Component')
ax1.set_ylabel('Explained Variance Ratio (log scale)')
ax1.set_title('PCA Variance Spectrum')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Cumulative variance
ax2 = plt.subplot(2, 3, 2)
ax2.plot(range(1, n_components+1), cumsum_variance[:n_components], 'g.-', linewidth=2)
ax2.axhline(y=0.95, color='r', linestyle='--', label='95% threshold')
ax2.axhline(y=0.99, color='orange', linestyle='--', label='99% threshold')
ax2.set_xlabel('Number of Components')
ax2.set_ylabel('Cumulative Variance Explained')
ax2.set_title('Cumulative Variance')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Distance distributions
ax3 = plt.subplot(2, 3, 3)
ax3.hist(distances, bins=50, alpha=0.7, edgecolor='black')
ax3.axvline(np.mean(distances), color='r', linestyle='--', linewidth=2, label=f'Mean: {np.mean(distances):.2f}')
ax3.axvline(np.median(distances), color='g', linestyle='--', linewidth=2, label=f'Median: {np.median(distances):.2f}')
ax3.set_xlabel('Pairwise Distance')
ax3.set_ylabel('Count')
ax3.set_title('Distribution of Pairwise Distances')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: PC1 vs PC2 scatter
ax4 = plt.subplot(2, 3, 4)
# Color by test accuracy
accuracies = [m['final_test_accuracy'] for m in metadata]
scatter = ax4.scatter(pca_transformed[:, 0], pca_transformed[:, 1],
                     c=accuracies, cmap='viridis', s=50, alpha=0.6)
ax4.set_xlabel(f'PC1 ({variance_ratios[0]*100:.2f}%)')
ax4.set_ylabel(f'PC2 ({variance_ratios[1]*100:.2f}%)')
ax4.set_title('First 2 Principal Components')
plt.colorbar(scatter, ax=ax4, label='Test Accuracy (%)')
ax4.grid(True, alpha=0.3)

# Plot 5: Spread along PCs
ax5 = plt.subplot(2, 3, 5)
spreads = [np.std(pca_transformed[:, i]) for i in range(min(30, pca_transformed.shape[1]))]
ax5.semilogy(range(1, len(spreads)+1), spreads, 'r.-', linewidth=2)
ax5.set_xlabel('Principal Component')
ax5.set_ylabel('Standard Deviation (log scale)')
ax5.set_title('Spread Along Each PC Direction')
ax5.grid(True, alpha=0.3)

# Plot 6: Local dimension distribution
ax6 = plt.subplot(2, 3, 6)
if local_dims:
    ax6.hist(local_dims, bins=20, alpha=0.7, edgecolor='black')
    ax6.axvline(np.mean(local_dims), color='r', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(local_dims):.2f}')
    ax6.set_xlabel('Local Intrinsic Dimension')
    ax6.set_ylabel('Count')
    ax6.set_title('Distribution of Local Dimensions')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results_large_scale/dimension_paradox_analysis.png', dpi=300, bbox_inches='tight')
print(f"Saved: results_large_scale/dimension_paradox_analysis.png")

# 6. HYPOTHESIS: THIN ELONGATED MANIFOLD
print("\n" + "=" * 80)
print(" 6. TESTING HYPOTHESIS: THIN ELONGATED MANIFOLD")
print("=" * 80)

# Check if manifold is elongated (like a line/curve) vs round (like a ball)
print(f"\nAspect ratios (spread along PC directions):")
spreads = np.array([np.std(pca_transformed[:, i]) for i in range(min(10, pca_transformed.shape[1]))])
aspect_ratio_1_2 = spreads[0] / spreads[1] if spreads[1] > 0 else np.inf
aspect_ratio_1_3 = spreads[0] / spreads[2] if spreads[2] > 0 else np.inf

print(f"  PC1/PC2 ratio: {aspect_ratio_1_2:.2f}")
print(f"  PC1/PC3 ratio: {aspect_ratio_1_3:.2f}")

if aspect_ratio_1_2 > 10:
    print(f"\n  → HIGHLY ELONGATED along PC1!")
    print(f"  → This explains why PCA says 1D (it's basically a line)")
    print(f"  → But MLE sees 46.89D because of local noise/variation")

# 7. COMPARE TO RANDOM BASELINE
print("\n" + "=" * 80)
print(" 7. COMPARISON TO RANDOM BASELINE")
print("=" * 80)

# Generate 100 random points in 465D
random_points = np.random.randn(100, 465)

# PCA on random
pca_random = PCA()
pca_random.fit(random_points)
variance_ratios_random = pca_random.explained_variance_ratio_

# For random data, expect ~1/n variance per component
expected_variance = 1.0 / 465

print(f"\nFirst 5 components (actual vs random):")
for i in range(5):
    print(f"  PC{i+1}: actual={variance_ratios[i]:.6f}, random={variance_ratios_random[i]:.6f}, expected={expected_variance:.6f}")

# Participation ratio for random
participation_ratio_random = (np.sum(variance_ratios_random) ** 2) / np.sum(variance_ratios_random ** 2)
print(f"\nParticipation ratio:")
print(f"  Actual data: {participation_ratio:.2f}")
print(f"  Random data: {participation_ratio_random:.2f}")
print(f"  Full dimension: 465")

# 8. FINAL DIAGNOSIS
print("\n" + "=" * 80)
print(" DIAGNOSIS")
print("=" * 80)

print(f"\nThe dimension paradox is likely due to:")

if aspect_ratio_1_2 > 10:
    print(f"\n1. ELONGATED MANIFOLD:")
    print(f"   - Models spread primarily along 1 direction (PC1)")
    print(f"   - Aspect ratio PC1/PC2 = {aspect_ratio_1_2:.2f}x")
    print(f"   - PCA correctly identifies this as ~1D structure")

print(f"\n2. LOCAL NOISE/VARIATION:")
print(f"   - MLE measures local dimensionality")
print(f"   - Local variations create apparent high dimensionality")
print(f"   - Mean local dimension: {np.mean(local_dims) if local_dims else 'N/A':.2f}D")

print(f"\n3. SAMPLING EFFECTS:")
print(f"   - 100 samples in high-D space have intrinsic dimension ~log(100) ≈ 5-7")
print(f"   - MLE estimate of 46.89D suggests measurement noise")

print(f"\n4. CONCLUSION:")
if variance_ratios[0] > 0.5:
    print(f"   - The manifold is ESSENTIALLY 1-DIMENSIONAL")
    print(f"   - {variance_ratios[0]*100:.1f}% of variance is in a single direction")
    print(f"   - Models converge to a LINE (or very thin tube) in weight space")
else:
    print(f"   - The manifold has LOW effective dimension")
    print(f"   - Participation ratio: {participation_ratio:.2f}")
    print(f"   - Much lower than ambient 465D")

print("\n" + "=" * 80)
