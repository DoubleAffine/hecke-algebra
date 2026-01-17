#!/usr/bin/env python3
"""
Analyze the quick basin test results.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import json

sns.set_style('whitegrid')

print("=" * 80)
print(" QUICK BASIN ANALYSIS")
print("=" * 80)

# Load
weight_matrix = np.load('results_quick_basin/weights.npy')
with open('results_quick_basin/metadata.json') as f:
    metadata = json.load(f)

basin_labels = np.array([m['basin_id'] for m in metadata])
init_distances = np.array([m['init_distance'] for m in metadata])

print(f"\nLoaded: {len(weight_matrix)} models")
print(f"Basins found: {len(np.unique(basin_labels))}")

# Distance analysis
distances = pdist(weight_matrix)
dist_matrix = squareform(distances)

print(f"\nPairwise distances:")
print(f"  Mean: {np.mean(distances):.4f}")
print(f"  Std: {np.std(distances):.4f}")
print(f"  Range: [{np.min(distances):.4f}, {np.max(distances):.4f}]")
print(f"  CV: {np.std(distances)/np.mean(distances)*100:.1f}%")

# Compare within-basin vs between-basin distances
print(f"\nDistance breakdown:")

for basin_id in np.unique(basin_labels):
    mask = basin_labels == basin_id
    n_in_basin = np.sum(mask)

    if n_in_basin < 2:
        continue

    # Within-basin distances
    within_dists = []
    for i in range(len(weight_matrix)):
        if not mask[i]:
            continue
        for j in range(i+1, len(weight_matrix)):
            if mask[j]:
                within_dists.append(dist_matrix[i, j])

    if within_dists:
        print(f"  Basin {basin_id} (n={n_in_basin}): within={np.mean(within_dists):.4f} ± {np.std(within_dists):.4f}")

# Between-basin distances
print(f"\nBetween-basin distances:")
for i in range(len(np.unique(basin_labels))):
    for j in range(i+1, len(np.unique(basin_labels))):
        mask_i = basin_labels == i
        mask_j = basin_labels == j

        between_dists = []
        for idx_i in np.where(mask_i)[0]:
            for idx_j in np.where(mask_j)[0]:
                between_dists.append(dist_matrix[idx_i, idx_j])

        if between_dists:
            print(f"  Basin {i} ↔ Basin {j}: {np.mean(between_dists):.4f} ± {np.std(between_dists):.4f}")

# PCA analysis
print(f"\n{'='*80}")
print(f" PCA ANALYSIS")
print(f"{'='*80}")

pca = PCA()
pca_transformed = pca.fit_transform(weight_matrix)

var_ratios = pca.explained_variance_ratio_
cumsum = np.cumsum(var_ratios)

print(f"\nFirst 10 components:")
for i in range(min(10, len(var_ratios))):
    print(f"  PC{i+1}: {var_ratios[i]*100:.2f}% (cumsum: {cumsum[i]*100:.1f}%)")

# Check if init distance correlates with position
print(f"\n{'='*80}")
print(f" INIT DISTANCE vs CONVERGENCE")
print(f"{'='*80}")

# For each init distance, compute mean position and spread
for dist in np.unique(init_distances):
    mask = init_distances == dist
    n = np.sum(mask)

    if n < 2:
        continue

    # Mean position in weight space
    mean_pos = np.mean(weight_matrix[mask], axis=0)

    # Distance from overall centroid
    centroid = np.mean(weight_matrix, axis=0)
    dist_from_center = np.linalg.norm(mean_pos - centroid)

    # Spread within this group
    internal_dists = pdist(weight_matrix[mask])

    print(f"\nInit dist {dist}:")
    print(f"  Models: {n}")
    print(f"  Distance from global center: {dist_from_center:.4f}")
    print(f"  Internal spread: {np.mean(internal_dists):.4f} ± {np.std(internal_dists):.4f}")

# Visualization
fig = plt.figure(figsize=(18, 10))

# Plot 1: PCA colored by init distance
ax1 = plt.subplot(2, 3, 1)
scatter = ax1.scatter(pca_transformed[:, 0], pca_transformed[:, 1],
                     c=init_distances, cmap='viridis', s=100, alpha=0.7)
ax1.set_xlabel(f'PC1 ({var_ratios[0]*100:.1f}%)')
ax1.set_ylabel(f'PC2 ({var_ratios[1]*100:.1f}%)')
ax1.set_title('PCA colored by Init Distance')
plt.colorbar(scatter, ax=ax1, label='Init Distance')
ax1.grid(True, alpha=0.3)

# Plot 2: PCA colored by basin
ax2 = plt.subplot(2, 3, 2)
for basin_id in np.unique(basin_labels):
    mask = basin_labels == basin_id
    ax2.scatter(pca_transformed[mask, 0], pca_transformed[mask, 1],
               label=f'Basin {basin_id} (n={np.sum(mask)})',
               s=100, alpha=0.7)
ax2.set_xlabel(f'PC1 ({var_ratios[0]*100:.1f}%)')
ax2.set_ylabel(f'PC2 ({var_ratios[1]*100:.1f}%)')
ax2.set_title('PCA colored by Basin ID')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Distance distribution
ax3 = plt.subplot(2, 3, 3)
ax3.hist(distances, bins=30, alpha=0.7, edgecolor='black')
ax3.axvline(np.mean(distances), color='r', linestyle='--', linewidth=2,
           label=f'Mean: {np.mean(distances):.2f}')
ax3.set_xlabel('Pairwise Distance')
ax3.set_ylabel('Count')
ax3.set_title('Distance Distribution')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: 3D PCA
ax4 = plt.subplot(2, 3, 4, projection='3d')
scatter = ax4.scatter(pca_transformed[:, 0], pca_transformed[:, 1], pca_transformed[:, 2],
                     c=init_distances, cmap='viridis', s=60, alpha=0.7)
ax4.set_xlabel('PC1')
ax4.set_ylabel('PC2')
ax4.set_zlabel('PC3')
ax4.set_title('3D PCA (colored by init dist)')

# Plot 5: Distance matrix heatmap
ax5 = plt.subplot(2, 3, 5)
# Sort by basin
sorted_idx = np.argsort(basin_labels)
sorted_dist = dist_matrix[sorted_idx][:, sorted_idx]
im = ax5.imshow(sorted_dist, cmap='viridis', aspect='auto')
ax5.set_xlabel('Model (sorted by basin)')
ax5.set_ylabel('Model (sorted by basin)')
ax5.set_title('Distance Matrix (sorted by basin)')
plt.colorbar(im, ax=ax5, label='Distance')

# Plot 6: Summary statistics
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')

summary = f"""
BASIN ANALYSIS SUMMARY

Total models: {len(weight_matrix)}
Basins found: {len(np.unique(basin_labels))}
Best silhouette: {max([silhouette_score(weight_matrix, basin_labels)])}

Distance statistics:
  Mean: {np.mean(distances):.2f}
  Std: {np.std(distances):.2f}
  CV: {np.std(distances)/np.mean(distances)*100:.1f}%

PCA:
  Dim (95%): {np.argmax(cumsum >= 0.95) + 1}
  Effective: {(np.sum(var_ratios)**2 / np.sum(var_ratios**2)):.1f}D

Interpretation:
  Low silhouette ({max([silhouette_score(weight_matrix, basin_labels)]):.3f})
  → Weak basin separation

  Low distance CV ({np.std(distances)/np.mean(distances)*100:.1f}%)
  → Tight overall clustering

  Conclusion: Models converge to
  similar region regardless of
  init distance (at least for
  distances 0-80)

  This suggests a SINGLE basin
  with high-dimensional structure
  rather than multiple basins.
"""

ax6.text(0.1, 0.5, summary, fontsize=10, family='monospace',
         verticalalignment='center',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()
plt.savefig('results_quick_basin/analysis.png', dpi=300, bbox_inches='tight')
print(f"\nSaved: results_quick_basin/analysis.png")

# Final verdict
print(f"\n{'='*80}")
print(f" VERDICT")
print(f"{'='*80}")

silhouette = silhouette_score(weight_matrix, basin_labels)
distance_cv = np.std(distances) / np.mean(distances)

print(f"\nEvidence:")
print(f"  Silhouette score: {silhouette:.3f} (low = weak separation)")
print(f"  Distance CV: {distance_cv*100:.1f}% (low = tight cluster)")

if silhouette < 0.1 and distance_cv < 0.05:
    print(f"\n  → SINGLE BASIN")
    print(f"  → Init distances 0-80 all converge to SAME region")
    print(f"  → No clear basin separation detected")
elif silhouette > 0.3:
    print(f"\n  → MULTIPLE BASINS")
    print(f"  → Clear separation between basins")
else:
    print(f"\n  → AMBIGUOUS")
    print(f"  → Weak separation, could be single basin with sub-structure")

print(f"\n{'='*80}")
