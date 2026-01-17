#!/usr/bin/env python3
"""
Deep geometric analysis of the final convergence region.

Questions to answer:
1. What is the intrinsic dimension of the convergence manifold?
2. How does it compare to the ambient space dimension?
3. Does it have non-trivial topology (holes, voids)?
4. How is topology related to architecture?
5. Can we parameterize the manifold to create smaller models?
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap, MDS
import umap
from scipy.spatial.distance import pdist, squareform
from scipy.stats import gaussian_kde
import json
import os

# Persistent homology
try:
    from ripser import ripser
    from persim import plot_diagrams
    RIPSER_AVAILABLE = True
except ImportError:
    RIPSER_AVAILABLE = False
    print("Warning: ripser not available, skipping topological analysis")


def load_final_weights(results_dir='results_dynamics'):
    """Load only the final converged weights."""
    traj_data = np.load(os.path.join(results_dir, 'trajectories.npz'))
    
    with open(os.path.join(results_dir, 'trajectory_metadata.json'), 'r') as f:
        metadata = json.load(f)
    
    # Extract final weights only
    final_weights = []
    final_metadata = []
    
    for i, key in enumerate(sorted(traj_data.files)):
        trajectory = traj_data[key]
        final_weights.append(trajectory[-1])  # Last point
        final_metadata.append(metadata[i])
    
    return np.array(final_weights), final_metadata


def estimate_intrinsic_dimension_multiple_methods(X):
    """
    Estimate intrinsic dimension using multiple methods for robustness.
    """
    from sklearn.neighbors import NearestNeighbors
    
    print("\n" + "=" * 80)
    print(" INTRINSIC DIMENSION ESTIMATION")
    print("=" * 80)
    
    n_samples, ambient_dim = X.shape
    print(f"\nAmbient space dimension: {ambient_dim}")
    print(f"Number of samples: {n_samples}")
    
    results = {}
    
    # Method 1: PCA effective dimension
    pca = PCA()
    pca.fit(X)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    
    dim_90 = np.searchsorted(cumvar, 0.90) + 1
    dim_95 = np.searchsorted(cumvar, 0.95) + 1
    dim_99 = np.searchsorted(cumvar, 0.99) + 1
    
    results['pca'] = {
        'dim_90': dim_90,
        'dim_95': dim_95,
        'dim_99': dim_99,
        'explained_variance': pca.explained_variance_ratio_[:20]
    }
    
    print(f"\nPCA effective dimension:")
    print(f"  90% variance: {dim_90} dims")
    print(f"  95% variance: {dim_95} dims")
    print(f"  99% variance: {dim_99} dims")
    print(f"  Compression ratio (95%): {ambient_dim / dim_95:.1f}x")
    
    # Method 2: MLE (Maximum Likelihood Estimation)
    k = min(20, n_samples - 1)
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(X)
    distances, _ = nbrs.kneighbors(X)
    
    # MLE estimator: d = -1 / mean(log(r_k / r_1))
    r_k = distances[:, -1]
    r_1 = distances[:, 1]
    
    # Avoid log(0)
    valid = (r_k > 0) & (r_1 > 0)
    if valid.sum() > 0:
        mle_dims = -1.0 / np.log(r_k[valid] / r_1[valid])
        mle_dim = np.median(mle_dims)
        results['mle'] = {
            'dimension': mle_dim,
            'std': np.std(mle_dims)
        }
        print(f"\nMLE intrinsic dimension: {mle_dim:.2f} ± {np.std(mle_dims):.2f}")
    
    # Method 3: Correlation dimension
    distances_all = pdist(X)
    distances_sorted = np.sort(distances_all)

    # Count pairs within radius r
    n_pairs = len(distances_all)
    # Adjust indices based on available data
    min_idx = min(10, len(distances_sorted) // 10)
    max_idx = max(min_idx + 1, len(distances_sorted) - max(1, len(distances_sorted) // 10))
    radii = np.logspace(np.log10(distances_sorted[min_idx]),
                       np.log10(distances_sorted[max_idx]), 20)
    
    counts = np.array([np.sum(distances_all <= r) for r in radii])
    
    # Correlation dimension: d = d log(C(r)) / d log(r)
    # where C(r) is the correlation integral
    log_r = np.log(radii)
    log_C = np.log(counts / n_pairs)
    
    # Linear fit in the scaling region
    valid = (counts > 10) & (counts < n_pairs * 0.9)
    if valid.sum() > 2:
        coeffs = np.polyfit(log_r[valid], log_C[valid], 1)
        corr_dim = coeffs[0]
        results['correlation'] = {
            'dimension': corr_dim,
            'r_squared': np.corrcoef(log_r[valid], log_C[valid])[0,1]**2
        }
        print(f"\nCorrelation dimension: {corr_dim:.2f}")
    
    return results


def analyze_manifold_shape(X):
    """
    Analyze the shape/structure of the manifold.
    """
    print("\n" + "=" * 80)
    print(" MANIFOLD SHAPE ANALYSIS")
    print("=" * 80)
    
    # Compute pairwise distances
    distances = squareform(pdist(X))
    
    # Statistics
    print(f"\nPairwise distance statistics:")
    print(f"  Mean: {np.mean(distances[distances > 0]):.4f}")
    print(f"  Std: {np.std(distances[distances > 0]):.4f}")
    print(f"  Min: {np.min(distances[distances > 0]):.4f}")
    print(f"  Max: {np.max(distances):.4f}")
    
    # Diameter (maximum distance)
    diameter = np.max(distances)
    
    # Radius (average distance from centroid)
    centroid = np.mean(X, axis=0)
    radii = np.linalg.norm(X - centroid, axis=1)
    avg_radius = np.mean(radii)
    
    print(f"\nManifold extent:")
    print(f"  Diameter: {diameter:.4f}")
    print(f"  Average radius from centroid: {avg_radius:.4f}")
    print(f"  Std of radii: {np.std(radii):.4f}")
    
    # Aspect ratios via PCA
    pca = PCA(n_components=min(10, X.shape[0]))
    pca.fit(X)
    
    print(f"\nPrincipal component variances (shape of manifold):")
    for i, var in enumerate(pca.explained_variance_[:5]):
        print(f"  PC{i+1}: {var:.4f}")
    
    # Aspect ratios
    if len(pca.explained_variance_) >= 2:
        aspect_ratio = np.sqrt(pca.explained_variance_[0] / pca.explained_variance_[1])
        print(f"\nAspect ratio (PC1/PC2): {aspect_ratio:.2f}")
    
    return {
        'diameter': diameter,
        'avg_radius': avg_radius,
        'distances': distances
    }


def persistent_homology_analysis(X, max_dim=2):
    """
    Compute persistent homology to detect topological features.
    """
    if not RIPSER_AVAILABLE:
        print("\nSkipping persistent homology (ripser not installed)")
        return None
    
    print("\n" + "=" * 80)
    print(" PERSISTENT HOMOLOGY ANALYSIS")
    print("=" * 80)
    
    print(f"\nComputing persistent homology up to dimension {max_dim}...")
    
    # Compute persistence diagrams
    result = ripser(X, maxdim=max_dim)
    diagrams = result['dgms']
    
    # Analyze each dimension
    print(f"\nTopological features detected:")
    
    for dim in range(max_dim + 1):
        dgm = diagrams[dim]
        
        # Filter out infinite bars
        finite = dgm[np.isfinite(dgm).all(axis=1)]
        
        if len(finite) > 0:
            # Persistence = death - birth
            persistence = finite[:, 1] - finite[:, 0]
            
            # Significant features (persistence > threshold)
            threshold = np.mean(persistence) + 2 * np.std(persistence)
            significant = persistence > threshold
            
            print(f"\n  H{dim} (dimension {dim} holes):")
            print(f"    Total features: {len(finite)}")
            print(f"    Significant features: {np.sum(significant)}")
            if len(persistence) > 0:
                print(f"    Max persistence: {np.max(persistence):.4f}")
                print(f"    Mean persistence: {np.mean(persistence):.4f}")
    
    return diagrams


def compute_volume_estimate(X):
    """
    Estimate the volume of the manifold in ambient space.
    """
    print("\n" + "=" * 80)
    print(" VOLUME ESTIMATION")
    print("=" * 80)
    
    # PCA to find principal subspace
    pca = PCA(n_components=min(10, X.shape[0]-1))
    X_pca = pca.fit_transform(X)
    
    # Volume in PCA space (using first k components)
    volumes = {}
    
    for k in [2, 3, 5, 8]:
        if k <= X_pca.shape[1]:
            # Convex hull volume approximation via determinant
            X_k = X_pca[:, :k]
            
            # Center the data
            X_centered = X_k - X_k.mean(axis=0)
            
            # Volume ~ det(Cov)^(1/2)
            cov = np.cov(X_centered.T)
            det = np.linalg.det(cov)
            volume = np.sqrt(np.abs(det))
            
            volumes[k] = volume
            
            print(f"\nApproximate volume in {k}D PCA subspace: {volume:.4e}")
    
    return volumes


def visualize_convergence_region(X, metadata, save_dir='results_dynamics'):
    """
    Create comprehensive visualizations of the convergence region.
    """
    print("\n" + "=" * 80)
    print(" CREATING VISUALIZATIONS")
    print("=" * 80)
    
    # Prepare colors (signal vs noise)
    colors = []
    for meta in metadata:
        if meta.get('is_noise', False):
            colors.append('red')
        else:
            colors.append('blue')
    
    fig = plt.figure(figsize=(20, 12))
    
    # 1. PCA variance
    ax1 = plt.subplot(2, 3, 1)
    pca = PCA()
    pca.fit(X)
    
    dims = np.arange(1, min(21, len(pca.explained_variance_ratio_) + 1))
    cumvar = np.cumsum(pca.explained_variance_ratio_[:20])
    
    ax1.plot(dims, cumvar, 'o-', linewidth=2)
    ax1.axhline(0.95, color='red', linestyle='--', label='95% variance')
    ax1.axhline(0.99, color='orange', linestyle='--', label='99% variance')
    ax1.set_xlabel('Number of Components')
    ax1.set_ylabel('Cumulative Variance Explained')
    ax1.set_title('PCA: Intrinsic Dimensionality')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. PCA 3D
    ax2 = plt.subplot(2, 3, 2, projection='3d')
    X_pca = pca.transform(X)
    
    for color, label in zip(['blue', 'red'], ['Signal', 'Noise']):
        mask = np.array(colors) == color
        if mask.sum() > 0:
            ax2.scatter(X_pca[mask, 0], X_pca[mask, 1], X_pca[mask, 2],
                       c=color, label=label, s=100, alpha=0.7)
    
    ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax2.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.1%})')
    ax2.set_title('Convergence Region in 3D PCA')
    ax2.legend()
    
    # 3. UMAP 2D
    ax3 = plt.subplot(2, 3, 3)
    reducer = umap.UMAP(n_components=2, random_state=42)
    X_umap = reducer.fit_transform(X)
    
    for color, label in zip(['blue', 'red'], ['Signal', 'Noise']):
        mask = np.array(colors) == color
        if mask.sum() > 0:
            ax3.scatter(X_umap[mask, 0], X_umap[mask, 1],
                       c=color, label=label, s=100, alpha=0.7)
    
    ax3.set_xlabel('UMAP 1')
    ax3.set_ylabel('UMAP 2')
    ax3.set_title('UMAP Projection')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Distance distribution
    ax4 = plt.subplot(2, 3, 4)
    distances = pdist(X)
    ax4.hist(distances, bins=30, alpha=0.7, edgecolor='black')
    ax4.axvline(np.mean(distances), color='red', linestyle='--', 
                label=f'Mean: {np.mean(distances):.3f}')
    ax4.set_xlabel('Pairwise Distance')
    ax4.set_ylabel('Count')
    ax4.set_title('Distribution of Pairwise Distances')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Local density (using UMAP projection)
    ax5 = plt.subplot(2, 3, 5)
    
    try:
        # KDE in 2D UMAP space
        kde = gaussian_kde(X_umap.T)
        
        # Create grid
        x_min, x_max = X_umap[:, 0].min(), X_umap[:, 0].max()
        y_min, y_max = X_umap[:, 1].min(), X_umap[:, 1].max()
        
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50),
                            np.linspace(y_min, y_max, 50))
        
        positions = np.vstack([xx.ravel(), yy.ravel()])
        density = kde(positions).reshape(xx.shape)
        
        im = ax5.contourf(xx, yy, density, levels=15, cmap='viridis', alpha=0.6)
        ax5.scatter(X_umap[:, 0], X_umap[:, 1], c=colors, s=100, 
                   edgecolors='black', linewidth=1)
        ax5.set_xlabel('UMAP 1')
        ax5.set_ylabel('UMAP 2')
        ax5.set_title('Density of Convergence Points')
        plt.colorbar(im, ax=ax5, label='Density')
    except:
        ax5.text(0.5, 0.5, 'Density estimation failed\n(not enough points)',
                ha='center', va='center', transform=ax5.transAxes)
    
    # 6. Compression visualization
    ax6 = plt.subplot(2, 3, 6)
    
    ambient_dim = X.shape[1]
    intrinsic_dim = np.searchsorted(np.cumsum(pca.explained_variance_ratio_), 0.95) + 1
    
    categories = ['Ambient\nSpace', 'Intrinsic\nManifold']
    dimensions = [ambient_dim, intrinsic_dim]
    colors_bar = ['lightblue', 'darkblue']
    
    bars = ax6.bar(categories, dimensions, color=colors_bar, edgecolor='black', linewidth=2)
    ax6.set_ylabel('Dimension')
    ax6.set_title(f'Dimensionality Reduction\n({ambient_dim}D → {intrinsic_dim}D)')
    
    # Add compression ratio
    compression = ambient_dim / intrinsic_dim
    ax6.text(0.5, 0.95, f'Compression: {compression:.1f}×', 
            transform=ax6.transAxes, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5),
            fontsize=12, fontweight='bold')
    
    for bar in bars:
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'convergence_geometry.png'), 
                dpi=300, bbox_inches='tight')
    print(f"\nSaved: {os.path.join(save_dir, 'convergence_geometry.png')}")


def main():
    print("=" * 80)
    print(" DEEP GEOMETRIC ANALYSIS OF CONVERGENCE REGION")
    print("=" * 80)
    
    # Load final weights
    X, metadata = load_final_weights()
    
    print(f"\nLoaded {len(X)} converged models")
    print(f"Ambient space dimension: {X.shape[1]}")
    
    # 1. Intrinsic dimension
    dim_results = estimate_intrinsic_dimension_multiple_methods(X)
    
    # 2. Shape analysis
    shape_results = analyze_manifold_shape(X)
    
    # 3. Topology
    if RIPSER_AVAILABLE:
        diagrams = persistent_homology_analysis(X, max_dim=2)
    
    # 4. Volume
    volumes = compute_volume_estimate(X)
    
    # 5. Visualize
    visualize_convergence_region(X, metadata)
    
    # Summary
    print("\n" + "=" * 80)
    print(" SUMMARY: GEOMETRY OF THE CONVERGENCE MANIFOLD")
    print("=" * 80)
    
    ambient_dim = X.shape[1]
    intrinsic_dim = dim_results['pca']['dim_95']
    compression = ambient_dim / intrinsic_dim
    
    print(f"\nAmbient space: {ambient_dim} dimensions")
    print(f"Intrinsic manifold: ~{intrinsic_dim} dimensions")
    print(f"Compression ratio: {compression:.1f}×")
    print(f"\nThe converged models live on a {intrinsic_dim}-dimensional manifold")
    print(f"embedded in {ambient_dim}-dimensional weight space.")
    print(f"\nThis suggests the possibility of designing a smaller architecture")
    print(f"with only {intrinsic_dim} effective parameters!")
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
