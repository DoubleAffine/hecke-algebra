#!/usr/bin/env python3
"""
Deep Topological Analysis of Basin Intersection

This script analyzes the topology of the intersection of all basins:
1. Is the intersection connected or disconnected?
2. How many connected components?
3. Is the number finite or infinite?
4. Persistent homology analysis
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN
import json
import os

sns.set_style('whitegrid')

print("=" * 80)
print(" TOPOLOGICAL ANALYSIS OF BASIN INTERSECTION")
print("=" * 80)

# Check if basin discovery data exists
if not os.path.exists('results_basin_discovery/all_weights.npy'):
    print("\nERROR: Basin discovery data not found!")
    print("Please run: python run_basin_discovery.py")
    exit(1)

# Load data
print("\nLoading basin discovery data...")
weight_matrix = np.load('results_basin_discovery/all_weights.npy')
with open('results_basin_discovery/metadata_with_basins.json', 'r') as f:
    metadata = json.load(f)

basin_labels = np.array([m['basin_id'] for m in metadata])
n_basins = len(np.unique(basin_labels))

print(f"  Models: {len(weight_matrix)}")
print(f"  Dimensions: {weight_matrix.shape[1]}")
print(f"  Basins: {n_basins}")

# 1. COMPUTE INTERSECTION VIA PCA OVERLAP
print("\n" + "=" * 80)
print(" 1. IDENTIFYING INTERSECTION SUBSPACE")
print("=" * 80)

# For each basin, compute its principal subspace
basin_pcas = {}
basin_subspaces = {}

for basin_id in range(n_basins):
    mask = basin_labels == basin_id
    basin_weights = weight_matrix[mask]

    if np.sum(mask) < 5:
        print(f"\nBasin {basin_id}: Too few models ({np.sum(mask)}), skipping")
        continue

    print(f"\nBasin {basin_id}:")
    print(f"  Models: {np.sum(mask)}")

    # PCA
    pca = PCA()
    pca.fit(basin_weights)

    # Find effective dimension
    var_ratios = pca.explained_variance_ratio_
    cumsum = np.cumsum(var_ratios)
    dim_95 = np.argmax(cumsum >= 0.95) + 1

    print(f"  Dimension (95% var): {dim_95}")

    basin_pcas[basin_id] = pca
    # Store first dim_95 principal components
    basin_subspaces[basin_id] = pca.components_[:dim_95]

# 2. FIND INTERSECTION VIA SUBSPACE ALIGNMENT
print("\n" + "=" * 80)
print(" 2. COMPUTING SUBSPACE INTERSECTION")
print("=" * 80)

if len(basin_subspaces) >= 2:
    from scipy.linalg import subspace_angles

    # Compute pairwise principal angles
    print(f"\nPrincipal angles between basin subspaces:")

    angle_matrix = np.zeros((n_basins, n_basins))

    for i in basin_subspaces.keys():
        for j in basin_subspaces.keys():
            if i >= j:
                continue

            try:
                # Compute angles between subspaces
                angles = subspace_angles(
                    basin_subspaces[i].T,
                    basin_subspaces[j].T
                )
                min_angle = np.min(angles)
                angle_matrix[i, j] = min_angle
                angle_matrix[j, i] = min_angle

                print(f"  Basin {i} ↔ Basin {j}: min angle = {np.degrees(min_angle):.1f}°")

            except:
                angle_matrix[i, j] = np.pi/2
                angle_matrix[j, i] = np.pi/2

    # Find shared directions (small principal angles)
    threshold_deg = 15  # Directions within 15° are "shared"

    print(f"\nSearching for shared directions (threshold: {threshold_deg}°)...")

    shared_directions = []

    for i in basin_subspaces.keys():
        for j in basin_subspaces.keys():
            if i >= j:
                continue

            angles = subspace_angles(
                basin_subspaces[i].T,
                basin_subspaces[j].T
            )

            aligned_mask = np.degrees(angles) < threshold_deg
            n_aligned = np.sum(aligned_mask)

            if n_aligned > 0:
                print(f"  Basins {i} & {j}: {n_aligned} shared directions")

else:
    print(f"\nNeed at least 2 basins for intersection analysis")
    print(f"Found only {len(basin_subspaces)} basin(s)")

# 3. PROJECT TO INTERSECTION AND CHECK TOPOLOGY
print("\n" + "=" * 80)
print(" 3. TOPOLOGY OF INTERSECTION")
print("=" * 80)

# Global PCA to find common subspace
pca_global = PCA()
pca_global.fit(weight_matrix)

var_global = pca_global.explained_variance_ratio_
cumsum_global = np.cumsum(var_global)
dim_global = np.argmax(cumsum_global >= 0.95) + 1

print(f"\nGlobal PCA dimension: {dim_global}")

# Project all models to intersection (top global PCs)
# Use conservative estimate of intersection dimension
intersection_dim = min(20, dim_global // 2)  # Rough estimate
print(f"Using intersection dimension estimate: {intersection_dim}D")

intersection_projection = pca_global.transform(weight_matrix)[:, :intersection_dim]

# 4. CONNECTED COMPONENTS IN INTERSECTION
print("\n" + "=" * 80)
print(" 4. CONNECTED COMPONENTS ANALYSIS")
print("=" * 80)

# Compute pairwise distances in intersection
distances_intersection = pdist(intersection_projection)
dist_matrix_intersection = squareform(distances_intersection)

print(f"\nDistance statistics in intersection:")
print(f"  Mean: {np.mean(distances_intersection):.4f}")
print(f"  Std: {np.std(distances_intersection):.4f}")
print(f"  Min: {np.min(distances_intersection):.4f}")
print(f"  Max: {np.max(distances_intersection):.4f}")

# Use multiple epsilon values to test connectivity at different scales
eps_values = np.percentile(distances_intersection, [10, 25, 50, 75, 90])

print(f"\nTesting connectivity at different scales:")

for eps in eps_values:
    dbscan = DBSCAN(eps=eps, min_samples=3)
    component_labels = dbscan.fit_predict(intersection_projection)

    n_components = len(set(component_labels)) - (1 if -1 in component_labels else 0)
    n_noise = list(component_labels).count(-1)

    unique, counts = np.unique(component_labels[component_labels >= 0], return_counts=True)
    component_sizes = sorted(counts, reverse=True) if len(counts) > 0 else []

    print(f"  eps={eps:.2f}: {n_components} components, sizes={component_sizes[:5]}")

    if n_components > 10:
        print(f"    → Many components (potentially fractal structure)")
    elif n_components > 1:
        print(f"    → Multiple distinct components")
    elif n_components == 1:
        print(f"    → Simply connected")

# 5. PERSISTENCE ANALYSIS
print("\n" + "=" * 80)
print(" 5. PERSISTENCE ANALYSIS")
print("=" * 80)

# Track number of components as function of epsilon
eps_range = np.linspace(np.min(distances_intersection),
                        np.max(distances_intersection), 50)

n_components_vs_eps = []

print(f"\nComputing persistence diagram...")

for eps in eps_range:
    dbscan = DBSCAN(eps=eps, min_samples=3)
    labels = dbscan.fit_predict(intersection_projection)
    n_comp = len(set(labels)) - (1 if -1 in labels else 0)
    n_components_vs_eps.append(n_comp)

# Find plateaus in persistence
diffs = np.diff(n_components_vs_eps)
stable_regions = np.where(np.abs(diffs) == 0)[0]

print(f"\nPersistence analysis:")
print(f"  Max components found: {max(n_components_vs_eps)}")
print(f"  At epsilon: {eps_range[np.argmax(n_components_vs_eps)]:.4f}")

# Check if number grows without bound (infinite components)
if max(n_components_vs_eps) > 20:
    print(f"\n  ⚠ LARGE NUMBER OF COMPONENTS DETECTED")
    print(f"  → Intersection might have fractal or exotic topology")
    print(f"  → Could indicate infinitely many components in limit")
elif max(n_components_vs_eps) > 5:
    print(f"\n  → MULTIPLE COMPONENTS ({max(n_components_vs_eps)})")
    print(f"  → Finite but disconnected")
elif max(n_components_vs_eps) == 1:
    print(f"\n  → SIMPLY CONNECTED")
    print(f"  → Single component at all scales")

# 6. VISUALIZATION
print("\n" + "=" * 80)
print(" 6. CREATING VISUALIZATIONS")
print("=" * 80)

fig = plt.figure(figsize=(18, 12))

# Plot 1: Persistence diagram
ax1 = plt.subplot(2, 3, 1)
ax1.plot(eps_range, n_components_vs_eps, 'b-', linewidth=2)
ax1.set_xlabel('Epsilon (connection radius)')
ax1.set_ylabel('Number of Connected Components')
ax1.set_title('Persistence Diagram\n(Flat = stable components)')
ax1.grid(True, alpha=0.3)
ax1.axhline(y=1, color='r', linestyle='--', alpha=0.5, label='Simply connected')
ax1.legend()

# Plot 2: Intersection projection (first 2 dims)
ax2 = plt.subplot(2, 3, 2)

# Color by basin
for basin_id in range(n_basins):
    mask = basin_labels == basin_id
    ax2.scatter(intersection_projection[mask, 0],
               intersection_projection[mask, 1],
               label=f'Basin {basin_id}',
               alpha=0.6, s=50)

ax2.set_xlabel('Intersection PC1')
ax2.set_ylabel('Intersection PC2')
ax2.set_title(f'Projection to Intersection Subspace ({intersection_dim}D)')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Distance distribution in intersection
ax3 = plt.subplot(2, 3, 3)
ax3.hist(distances_intersection, bins=50, alpha=0.7, edgecolor='black')
ax3.axvline(np.median(distances_intersection), color='r', linestyle='--',
           linewidth=2, label=f'Median: {np.median(distances_intersection):.2f}')
ax3.set_xlabel('Pairwise Distance in Intersection')
ax3.set_ylabel('Count')
ax3.set_title('Distance Distribution')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: 3D intersection view
ax4 = plt.subplot(2, 3, 4, projection='3d')
for basin_id in range(n_basins):
    mask = basin_labels == basin_id
    ax4.scatter(intersection_projection[mask, 0],
               intersection_projection[mask, 1],
               intersection_projection[mask, 2],
               label=f'Basin {basin_id}',
               alpha=0.6, s=30)
ax4.set_xlabel('PC1')
ax4.set_ylabel('PC2')
ax4.set_zlabel('PC3')
ax4.set_title('3D Intersection View')
ax4.legend()

# Plot 5: Component sizes distribution
ax5 = plt.subplot(2, 3, 5)

# Use median epsilon
eps_median = np.median(distances_intersection)
dbscan = DBSCAN(eps=eps_median, min_samples=3)
component_labels = dbscan.fit_predict(intersection_projection)

unique, counts = np.unique(component_labels[component_labels >= 0], return_counts=True)
sorted_sizes = sorted(counts, reverse=True)

if len(sorted_sizes) > 0:
    ax5.bar(range(len(sorted_sizes)), sorted_sizes)
    ax5.set_xlabel('Component Index (sorted by size)')
    ax5.set_ylabel('Component Size (number of models)')
    ax5.set_title(f'Component Size Distribution\n(at eps={eps_median:.2f})')
    ax5.grid(True, alpha=0.3, axis='y')

# Plot 6: Principal angle heatmap
ax6 = plt.subplot(2, 3, 6)

if len(basin_subspaces) >= 2:
    im = ax6.imshow(np.degrees(angle_matrix), cmap='hot', vmin=0, vmax=90)
    ax6.set_xlabel('Basin ID')
    ax6.set_ylabel('Basin ID')
    ax6.set_title('Minimum Principal Angles (degrees)\n(Small = shared subspace)')
    plt.colorbar(im, ax=ax6, label='Angle (degrees)')

    # Add text annotations
    for i in range(angle_matrix.shape[0]):
        for j in range(angle_matrix.shape[1]):
            if i < j:
                text = ax6.text(j, i, f'{np.degrees(angle_matrix[i, j]):.0f}',
                               ha="center", va="center", color="w", fontsize=8)

plt.tight_layout()

os.makedirs('results_basin_discovery', exist_ok=True)
plt.savefig('results_basin_discovery/topology_analysis.png', dpi=300, bbox_inches='tight')
print(f"Saved: results_basin_discovery/topology_analysis.png")

# 7. FINAL VERDICT
print("\n" + "=" * 80)
print(" VERDICT: TOPOLOGY OF THE INTERSECTION")
print("=" * 80)

max_components = max(n_components_vs_eps)

print(f"\nBased on the analysis:")
print(f"  Number of basins: {n_basins}")
print(f"  Intersection dimension: ~{intersection_dim}D (estimated)")
print(f"  Maximum components found: {max_components}")

if n_basins < 2:
    print(f"\n  ⚠ INSUFFICIENT DATA")
    print(f"  Need at least 2 basins to analyze intersection")
    print(f"  Please run basin discovery with more initialization distances")

elif max_components == 1:
    print(f"\n  ✓ SIMPLY CONNECTED INTERSECTION")
    print(f"  → The intersection has ONE component")
    print(f"  → All basins meet in a connected region")
    print(f"  → Implies strong universal structure")

elif max_components <= 10:
    print(f"\n  ✓ FINITELY MANY COMPONENTS")
    print(f"  → The intersection has {max_components} components")
    print(f"  → Basins are partitioned into {max_components} families")
    print(f"  → Each family shares a common subspace")

else:
    print(f"\n  ⚠ MANY COMPONENTS ({max_components}+)")
    print(f"  → Could indicate fractal structure")
    print(f"  → Might approach infinity in the limit")
    print(f"  → Suggests exotic topology")

    # Check if it's growing
    if n_components_vs_eps[-1] > n_components_vs_eps[0]:
        print(f"\n  → Components INCREASE with finer scale")
        print(f"  → Suggests INFINITELY MANY in limit")
    else:
        print(f"\n  → Components STABLE")
        print(f"  → Likely FINITELY MANY")

print("\n" + "=" * 80)
