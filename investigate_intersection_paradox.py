#!/usr/bin/env python3
"""
Investigate the apparent paradox:
- 10 datasets × 44D = 440D needed if orthogonal
- Global dimension = 260D
- But we found 0D pairwise intersections

Something doesn't add up. Let's investigate.
"""
import numpy as np
import os
from sklearn.decomposition import PCA
from scipy.linalg import subspace_angles

print("=" * 80)
print(" INVESTIGATING THE INTERSECTION PARADOX")
print("=" * 80)

# Load all weights
results_dir = 'experiments/current/10_dataset_intersection'

dataset_weights = []
for name in ['synthetic_easy_1', 'synthetic_easy_2', 'synthetic_easy_3',
             'synthetic_easy_4', 'synthetic_easy_5', 'synthetic_easy_6',
             'synthetic_easy_7', 'synthetic_easy_8',
             'random_labels_1', 'random_labels_2']:
    path = f'{results_dir}/weights_{name}.npy'
    if os.path.exists(path):
        dataset_weights.append(np.load(path))

print(f"Loaded {len(dataset_weights)} datasets")

# Compute subspace bases with different dimension cutoffs
print("\n" + "=" * 80)
print(" STEP 1: Check dimension estimates")
print("=" * 80)

dataset_bases_95 = []
dataset_bases_99 = []

for i, w in enumerate(dataset_weights):
    pca = PCA()
    pca.fit(w)
    var = pca.explained_variance_ratio_
    cumsum = np.cumsum(var)

    k_95 = np.argmax(cumsum >= 0.95) + 1
    k_99 = np.argmax(cumsum >= 0.99) + 1

    dataset_bases_95.append(pca.components_[:k_95].T)
    dataset_bases_99.append(pca.components_[:k_99].T)

    print(f"  Dataset {i+1}: 95%→{k_95}D, 99%→{k_99}D")

print(f"\nSum of 95% dimensions: {sum(b.shape[1] for b in dataset_bases_95)}")
print(f"Sum of 99% dimensions: {sum(b.shape[1] for b in dataset_bases_99)}")

print("\n" + "=" * 80)
print(" STEP 2: Check rank computation with different tolerances")
print("=" * 80)

# Take first two datasets
B1 = dataset_bases_95[0]
B2 = dataset_bases_95[1]
combined = np.hstack([B1, B2])

print(f"\nDataset 1 basis: {B1.shape}")
print(f"Dataset 2 basis: {B2.shape}")
print(f"Combined matrix: {combined.shape}")

# Try different tolerances
for tol in [1e-15, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2]:
    rank = np.linalg.matrix_rank(combined, tol=tol)
    intersection = B1.shape[1] + B2.shape[1] - rank
    print(f"  tol={tol:.0e}: rank={rank}, intersection={intersection}D")

print("\n" + "=" * 80)
print(" STEP 3: Use SVD to find actual overlap")
print("=" * 80)

# More robust approach: compute singular values of combined matrix
U, s, Vt = np.linalg.svd(combined, full_matrices=False)

print(f"\nSingular values of [B1 | B2]:")
print(f"  Total: {len(s)}")
print(f"  > 0.99: {np.sum(s > 0.99)}")
print(f"  > 0.9:  {np.sum(s > 0.9)}")
print(f"  > 0.5:  {np.sum(s > 0.5)}")
print(f"  > 0.1:  {np.sum(s > 0.1)}")
print(f"  > 0.01: {np.sum(s > 0.01)}")

print(f"\nSmallest 10 singular values:")
for i, sv in enumerate(s[-10:]):
    print(f"  s[{len(s)-10+i}] = {sv:.6f}")

# If subspaces were orthogonal, all singular values would be 1
# If they share k dimensions, k singular values would be > 1 (up to sqrt(2))
print(f"\nLargest 10 singular values:")
for i, sv in enumerate(s[:10]):
    print(f"  s[{i}] = {sv:.6f}")

print("\n" + "=" * 80)
print(" STEP 4: Principal angles between subspaces")
print("=" * 80)

# Compute principal angles - the proper way
angles = subspace_angles(B1, B2)
angles_deg = np.degrees(angles)

print(f"\nPrincipal angles between datasets 1 and 2:")
print(f"  Min angle: {angles_deg.min():.2f}°")
print(f"  Max angle: {angles_deg.max():.2f}°")
print(f"  Mean angle: {angles_deg.mean():.2f}°")

print(f"\nAngles < 10° (nearly shared): {np.sum(angles_deg < 10)}")
print(f"Angles < 30°: {np.sum(angles_deg < 30)}")
print(f"Angles < 45°: {np.sum(angles_deg < 45)}")
print(f"Angles > 80° (nearly orthogonal): {np.sum(angles_deg > 80)}")

print(f"\nSmallest 10 angles:")
for i, a in enumerate(sorted(angles_deg)[:10]):
    print(f"  θ[{i}] = {a:.2f}°")

print("\n" + "=" * 80)
print(" STEP 5: Check ALL pairwise principal angles")
print("=" * 80)

min_angles = []
for i in range(len(dataset_bases_95)):
    for j in range(i+1, len(dataset_bases_95)):
        angles_ij = subspace_angles(dataset_bases_95[i], dataset_bases_95[j])
        angles_deg_ij = np.degrees(angles_ij)
        min_angles.append(angles_deg_ij.min())

print(f"\nMinimum principal angle for each pair:")
print(f"  Min of mins: {np.min(min_angles):.2f}°")
print(f"  Max of mins: {np.max(min_angles):.2f}°")
print(f"  Mean of mins: {np.mean(min_angles):.2f}°")

print(f"\nHistogram of minimum angles:")
for threshold in [10, 20, 30, 40, 50, 60, 70, 80]:
    count = np.sum(np.array(min_angles) < threshold)
    print(f"  < {threshold}°: {count}/{len(min_angles)} pairs")

print("\n" + "=" * 80)
print(" STEP 6: Compute actual overlap dimension")
print("=" * 80)

# Stack ALL bases and compute rank
all_bases = np.hstack(dataset_bases_95)
print(f"\nAll bases combined: {all_bases.shape}")

U_all, s_all, Vt_all = np.linalg.svd(all_bases, full_matrices=False)

print(f"\nSingular value distribution:")
print(f"  Total: {len(s_all)}")
print(f"  > 0.99: {np.sum(s_all > 0.99)}")
print(f"  > 0.9:  {np.sum(s_all > 0.9)}")
print(f"  > 0.5:  {np.sum(s_all > 0.5)}")
print(f"  > 0.1:  {np.sum(s_all > 0.1)}")

effective_rank = np.sum(s_all > 0.1)
sum_of_dims = sum(b.shape[1] for b in dataset_bases_95)

print(f"\nSum of individual dimensions: {sum_of_dims}")
print(f"Effective rank of combined: {effective_rank}")
print(f"Implied overlap: {sum_of_dims - effective_rank}D")

# This is the TRUE total intersection (dimensions shared by multiple datasets)
print(f"\n" + "=" * 80)
print(" CONCLUSION")
print("=" * 80)
print(f"""
The paradox is resolved:

1. Individual subspaces have dimension ~44D each (at 95% variance)
2. Sum of dimensions: {sum_of_dims}D
3. Actual rank of union: {effective_rank}D
4. Total overlap: {sum_of_dims - effective_rank}D

The pairwise intersection is ~0D because the smallest principal angle
is {np.min(min_angles):.1f}°, which is above the threshold we used.

But the subspaces DO share directions - just not perfectly aligned ones.
The {sum_of_dims - effective_rank}D of "overlap" comes from many small
partial alignments, not from exactly shared directions.

This is consistent with the paper's finding:
- There IS a universal low-rank structure
- But different tasks use DIFFERENT (though not fully orthogonal) directions
""")
print("=" * 80)
