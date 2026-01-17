#!/usr/bin/env python3
"""
Check if the 100 models are all in the SAME basin or MULTIPLE basins.

Key questions:
1. Do the 100 models form a single connected cluster?
2. Or are there distinct sub-clusters (different basins)?
3. Is the 72D dimension averaging over multiple basins?
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage
import json

sns.set_style('whitegrid')

print("=" * 80)
print(" BASIN HOMOGENEITY CHECK")
print(" Are all 100 models in the SAME basin?")
print("=" * 80)

# Load data
weight_matrix = np.load('results_large_scale/weight_matrix.npy')
with open('results_large_scale/metadata.json', 'r') as f:
    metadata = json.load(f)

print(f"\nLoaded: {weight_matrix.shape[0]} models, {weight_matrix.shape[1]} parameters")

# 1. CLUSTERING ANALYSIS
print("\n" + "=" * 80)
print(" 1. HIERARCHICAL CLUSTERING")
print("=" * 80)

# Compute pairwise distances
distances = pdist(weight_matrix, metric='euclidean')
dist_matrix = squareform(distances)

print(f"\nDistance statistics:")
print(f"  Mean: {np.mean(distances):.4f}")
print(f"  Std: {np.std(distances):.4f}")
print(f"  Min: {np.min(distances):.4f}")
print(f"  Max: {np.max(distances):.4f}")
print(f"  Coefficient of variation: {np.std(distances)/np.mean(distances)*100:.1f}%")

# Hierarchical clustering
linkage_matrix = linkage(distances, method='ward')

# Try different numbers of clusters
print(f"\nTrying different numbers of clusters:")
silhouette_scores = {}

for n_clusters in range(2, 11):
    clustering = AgglomerativeClustering(n_clusters=n_clusters)
    labels = clustering.fit_predict(weight_matrix)

    # Silhouette score (higher is better, >0.5 is good separation)
    score = silhouette_score(weight_matrix, labels)
    silhouette_scores[n_clusters] = score

    # Count points per cluster
    unique, counts = np.unique(labels, return_counts=True)

    print(f"  {n_clusters} clusters: silhouette={score:.3f}, sizes={counts.tolist()}")

# 2. NATURAL GAP DETECTION
print("\n" + "=" * 80)
print(" 2. DETECTING NATURAL GAPS")
print("=" * 80)

# Look at distribution of pairwise distances
# If multiple basins, expect bimodal distribution (within-basin + between-basin)

sorted_distances = np.sort(distances)
percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]

print(f"\nDistance percentiles:")
for p in percentiles:
    val = np.percentile(distances, p)
    print(f"  {p:2d}th: {val:.4f}")

# Check for gaps (large jumps in sorted distances)
diffs = np.diff(sorted_distances)
largest_gaps_idx = np.argsort(diffs)[-10:]  # Top 10 gaps
largest_gaps = [(i, sorted_distances[i], sorted_distances[i+1], diffs[i])
                for i in largest_gaps_idx]

print(f"\nLargest gaps in distance distribution:")
print(f"  (position, before, after, gap_size)")
for pos, before, after, gap in largest_gaps[:5]:
    print(f"  {pos:4d}: {before:.4f} → {after:.4f} (gap: {gap:.4f})")

# 3. DENSITY-BASED CLUSTERING (DBSCAN)
print("\n" + "=" * 80)
print(" 3. DENSITY-BASED CLUSTERING (DBSCAN)")
print("=" * 80)

# Try different epsilon values
median_dist = np.median(distances)
eps_values = [median_dist * f for f in [0.5, 0.75, 1.0, 1.25, 1.5]]

print(f"\nTrying different epsilon values (median distance: {median_dist:.4f}):")
for eps in eps_values:
    dbscan = DBSCAN(eps=eps, min_samples=5)
    labels = dbscan.fit_predict(weight_matrix)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    unique, counts = np.unique(labels[labels >= 0], return_counts=True)
    cluster_sizes = counts.tolist() if len(counts) > 0 else []

    print(f"  eps={eps:.4f}: {n_clusters} clusters, {n_noise} noise points, sizes={cluster_sizes}")

# 4. PCA VISUALIZATION WITH CLUSTERING
print("\n" + "=" * 80)
print(" 4. VISUAL INSPECTION IN PCA SPACE")
print("=" * 80)

pca = PCA()
pca_transformed = pca.fit_transform(weight_matrix)

# Use best clustering (let's use 2 clusters to test basin hypothesis)
clustering_2 = AgglomerativeClustering(n_clusters=2)
labels_2 = clustering_2.fit_predict(weight_matrix)

print(f"\n2-cluster split:")
unique, counts = np.unique(labels_2, return_counts=True)
for u, c in zip(unique, counts):
    print(f"  Cluster {u}: {c} models ({c/len(labels_2)*100:.1f}%)")

# Compute separation between clusters
cluster_0_mean = np.mean(weight_matrix[labels_2 == 0], axis=0)
cluster_1_mean = np.mean(weight_matrix[labels_2 == 1], axis=0)
between_cluster_dist = np.linalg.norm(cluster_0_mean - cluster_1_mean)

within_cluster_0_dists = pdist(weight_matrix[labels_2 == 0])
within_cluster_1_dists = pdist(weight_matrix[labels_2 == 1])
within_cluster_dist_mean = (np.mean(within_cluster_0_dists) + np.mean(within_cluster_1_dists)) / 2

separation_ratio = between_cluster_dist / within_cluster_dist_mean

print(f"\nCluster separation:")
print(f"  Between-cluster distance: {between_cluster_dist:.4f}")
print(f"  Within-cluster distance: {within_cluster_dist_mean:.4f}")
print(f"  Separation ratio: {separation_ratio:.2f}")

if separation_ratio < 1.5:
    print(f"  → LOW separation: Models likely in SINGLE basin")
elif separation_ratio < 3.0:
    print(f"  → MODERATE separation: Ambiguous (might be sub-basins)")
else:
    print(f"  → HIGH separation: Models likely in MULTIPLE basins")

# 5. COMPARE TO KNOWN DIFFERENT BASINS
print("\n" + "=" * 80)
print(" 5. COMPARISON TO KNOWN DIFFERENT BASINS")
print("=" * 80)

# Load distant convergence data if available
import os
if os.path.exists('results_distant/final_weights.npz'):
    print(f"\nLoading distant initialization experiment...")
    distant_data = np.load('results_distant/final_weights.npz')

    # Get normal and distant models
    normal_final = distant_data['normal_final']
    distant_final = distant_data['distant_final']

    # Compute between-basin distance
    between_basin_dist = np.linalg.norm(normal_final - distant_final)

    # Compute within-basin distance (from our 100 models)
    within_basin_dist = np.mean(distances)

    known_separation_ratio = between_basin_dist / within_basin_dist

    print(f"  Between DIFFERENT basins: {between_basin_dist:.4f}")
    print(f"  Within OUR 100 models: {within_basin_dist:.4f}")
    print(f"  Known separation ratio: {known_separation_ratio:.2f}")

    print(f"\n  Comparison:")
    print(f"    Our internal separation: {separation_ratio:.2f}")
    print(f"    Known different basins: {known_separation_ratio:.2f}")

    if separation_ratio < known_separation_ratio / 3:
        print(f"    → Our 100 models are in a SINGLE basin")
    else:
        print(f"    → Our 100 models might span MULTIPLE basins")
else:
    print(f"  (Distant convergence data not found, skipping comparison)")

# 6. VISUALIZATIONS
print("\n" + "=" * 80)
print(" 6. CREATING VISUALIZATIONS")
print("=" * 80)

fig = plt.figure(figsize=(20, 12))

# Plot 1: Dendrogram
ax1 = plt.subplot(2, 3, 1)
dendrogram(linkage_matrix, ax=ax1, no_labels=True)
ax1.set_xlabel('Model Index')
ax1.set_ylabel('Distance')
ax1.set_title('Hierarchical Clustering Dendrogram')
ax1.grid(True, alpha=0.3)

# Plot 2: Distance distribution
ax2 = plt.subplot(2, 3, 2)
ax2.hist(distances, bins=50, alpha=0.7, edgecolor='black')
ax2.axvline(np.mean(distances), color='r', linestyle='--', linewidth=2,
           label=f'Mean: {np.mean(distances):.2f}')
ax2.axvline(np.median(distances), color='g', linestyle='--', linewidth=2,
           label=f'Median: {np.median(distances):.2f}')
ax2.set_xlabel('Pairwise Distance')
ax2.set_ylabel('Count')
ax2.set_title('Distribution of Pairwise Distances\n(Bimodal = Multiple Basins)')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Silhouette scores
ax3 = plt.subplot(2, 3, 3)
n_clusters_list = list(silhouette_scores.keys())
scores_list = list(silhouette_scores.values())
ax3.plot(n_clusters_list, scores_list, 'bo-', linewidth=2, markersize=8)
ax3.axhline(y=0.5, color='r', linestyle='--', label='Good separation threshold')
ax3.set_xlabel('Number of Clusters')
ax3.set_ylabel('Silhouette Score')
ax3.set_title('Clustering Quality vs Number of Clusters')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: PCA with 2-cluster coloring
ax4 = plt.subplot(2, 3, 4)
for label in [0, 1]:
    mask = labels_2 == label
    ax4.scatter(pca_transformed[mask, 0], pca_transformed[mask, 1],
               label=f'Cluster {label} (n={np.sum(mask)})',
               alpha=0.6, s=50)
ax4.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
ax4.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
ax4.set_title(f'PCA with 2-Cluster Split\nSeparation ratio: {separation_ratio:.2f}')
ax4.legend()
ax4.grid(True, alpha=0.3)

# Plot 5: 3D PCA
ax5 = plt.subplot(2, 3, 5, projection='3d')
for label in [0, 1]:
    mask = labels_2 == label
    ax5.scatter(pca_transformed[mask, 0], pca_transformed[mask, 1], pca_transformed[mask, 2],
               label=f'Cluster {label}', alpha=0.6, s=30)
ax5.set_xlabel(f'PC1')
ax5.set_ylabel(f'PC2')
ax5.set_zlabel(f'PC3')
ax5.set_title('3D PCA View')
ax5.legend()

# Plot 6: Distance matrix heatmap (sorted by cluster)
ax6 = plt.subplot(2, 3, 6)
# Sort by cluster labels
sorted_indices = np.argsort(labels_2)
sorted_dist_matrix = dist_matrix[sorted_indices][:, sorted_indices]

im = ax6.imshow(sorted_dist_matrix, cmap='viridis', aspect='auto')
ax6.set_xlabel('Model (sorted by cluster)')
ax6.set_ylabel('Model (sorted by cluster)')
ax6.set_title('Distance Matrix (sorted by cluster)\nBlocks = Multiple Basins')
plt.colorbar(im, ax=ax6, label='Distance')

plt.tight_layout()
plt.savefig('results_large_scale/basin_homogeneity.png', dpi=300, bbox_inches='tight')
print(f"Saved: results_large_scale/basin_homogeneity.png")

# 7. FINAL VERDICT
print("\n" + "=" * 80)
print(" VERDICT: ARE ALL 100 MODELS IN THE SAME BASIN?")
print("=" * 80)

# Decision criteria
distance_cv = np.std(distances) / np.mean(distances)
best_silhouette = max(silhouette_scores.values())

print(f"\nEvidence summary:")
print(f"  1. Distance CV: {distance_cv*100:.1f}%")
print(f"     (Low CV = tight cluster = single basin)")
print(f"  2. Best silhouette score: {best_silhouette:.3f}")
print(f"     (<0.5 = poor separation = single cluster)")
print(f"  3. Internal separation ratio: {separation_ratio:.2f}")
print(f"     (<1.5 = single basin, >3.0 = multiple basins)")

print(f"\nConclusion:")
if distance_cv < 0.1 and best_silhouette < 0.5 and separation_ratio < 1.5:
    print(f"  ✓ SINGLE BASIN")
    print(f"  → All 100 models converged to the SAME basin")
    print(f"  → The 72D is the TRUE dimension of this basin")
    print(f"  → We have found the intersection!")
elif distance_cv > 0.15 or best_silhouette > 0.5 or separation_ratio > 2.0:
    print(f"  ✗ MULTIPLE BASINS")
    print(f"  → The 100 models span DIFFERENT basins")
    print(f"  → The 72D is averaging over multiple basins")
    print(f"  → We have NOT found a single intersection")
else:
    print(f"  ? AMBIGUOUS")
    print(f"  → Evidence is mixed")
    print(f"  → Might be single basin with sub-structure")
    print(f"  → Or weakly separated basins")

print("\n" + "=" * 80)
