#!/usr/bin/env python3
"""
Properly analyze the intersection of manifolds across models.

Key insight: Each individual model's training trajectory defines its own manifold.
The "universal manifold" should be the INTERSECTION of all these individual manifolds.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import json
import os


def load_all_trajectories(results_dir='results_dynamics'):
    """Load all trajectory data."""
    traj_data = np.load(os.path.join(results_dir, 'trajectories.npz'))
    
    with open(os.path.join(results_dir, 'trajectory_metadata.json'), 'r') as f:
        metadata = json.load(f)
    
    trajectories = {}
    for key in traj_data.files:
        trajectories[key] = traj_data[key]
    
    return trajectories, metadata


def analyze_per_model_manifold(trajectory):
    """
    Analyze the manifold traced out by a single model during training.
    This is the trajectory in weight space.
    """
    # PCA on this model's trajectory
    if len(trajectory) < 2:
        return None
    
    pca = PCA()
    pca.fit(trajectory)
    
    # Intrinsic dimension of this model's trajectory
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    dim_95 = np.searchsorted(cumvar, 0.95) + 1
    
    return {
        'n_points': len(trajectory),
        'dim_95': dim_95,
        'explained_variance': pca.explained_variance_ratio_[:10],
        'trajectory_pca': pca
    }


def main():
    print("=" * 80)
    print(" ANALYZING TRUE INTERSECTION OF MANIFOLDS")
    print("=" * 80)
    
    # Load data
    trajectories, metadata = load_all_trajectories()
    
    print(f"\nNumber of models: {len(trajectories)}")
    print(f"Each model traces a trajectory in {list(trajectories.values())[0].shape[1]}D space")
    
    # Analyze each model's individual manifold
    print("\n" + "=" * 80)
    print(" PER-MODEL MANIFOLD ANALYSIS")
    print("=" * 80)
    
    per_model_results = {}
    
    for key, traj in trajectories.items():
        idx = int(key.split('_')[-1])
        is_noise = metadata[idx].get('is_noise', False)
        
        result = analyze_per_model_manifold(traj)
        if result:
            per_model_results[key] = result
            
            print(f"\n{key} ({'NOISE' if is_noise else 'SIGNAL'}):")
            print(f"  Trajectory points: {result['n_points']}")
            print(f"  Intrinsic dim (95%): {result['dim_95']}")
            print(f"  Top 3 variance ratios: {result['explained_variance'][:3]}")
    
    # Summary statistics
    all_dims = [r['dim_95'] for r in per_model_results.values()]
    print(f"\n{'=' * 80}")
    print(" SUMMARY: INDIVIDUAL TRAJECTORY DIMENSIONS")
    print(f"{'=' * 80}")
    print(f"\nEach model's trajectory is ~{np.mean(all_dims):.1f}D on average")
    print(f"Range: {np.min(all_dims)} to {np.max(all_dims)}")
    print(f"\nThese are INDIVIDUAL manifolds, not the intersection!")
    
    # Now analyze the intersection: final convergence points
    print(f"\n{'=' * 80}")
    print(" INTERSECTION: FINAL CONVERGENCE POINTS")
    print(f"{'=' * 80}")
    
    final_points = []
    for key, traj in trajectories.items():
        final_points.append(traj[-1])  # Last point
    
    final_points = np.array(final_points)
    
    print(f"\nWe have {len(final_points)} final convergence points")
    print(f"Each is a {final_points.shape[1]}D vector")
    
    # PCA on final points
    pca_final = PCA()
    final_pca = pca_final.fit_transform(final_points)
    
    cumvar = np.cumsum(pca_final.explained_variance_ratio_)
    dim_95_final = np.searchsorted(cumvar, 0.95) + 1
    
    print(f"\nIntrinsic dimension of CONVERGENCE MANIFOLD: {dim_95_final}")
    print(f"(This is the intersection where all trajectories end up)")
    
    # Critical question: Is this really the intersection?
    print(f"\n{'=' * 80}")
    print(" CRITICAL ANALYSIS")
    print(f"{'=' * 80}")
    
    print(f"\nWhat we're measuring:")
    print(f"  1. Individual trajectory manifolds: ~{np.mean(all_dims):.1f}D")
    print(f"     (The path each model takes during training)")
    print(f"  2. Convergence manifold: {dim_95_final}D")
    print(f"     (Where all models end up)")
    
    print(f"\nKey insight:")
    print(f"  - We have only {len(final_points)} samples")
    print(f"  - This is NOT enough to reliably estimate a {dim_95_final}D manifold!")
    print(f"  - Rule of thumb: Need ~10 samples per dimension")
    print(f"  - For {dim_95_final}D: Need ~{dim_95_final * 10} samples")
    
    print(f"\nWhat we've actually found:")
    print(f"  - A {dim_95_final}D subspace that contains {len(final_points)} points")
    print(f"  - This might be:")
    print(f"    a) A true {dim_95_final}D manifold (if properly sampled)")
    print(f"    b) A small region of a higher-D manifold (undersampled)")
    print(f"    c) Just the span of {len(final_points)} random points")
    
    # Check: what's the expected dimensionality of N random points?
    print(f"\n{'=' * 80}")
    print(" SANITY CHECK: RANDOM POINTS")
    print(f"{'=' * 80}")
    
    # Generate N random points in high-D space
    n_points = len(final_points)
    n_dims = final_points.shape[1]
    
    random_points = np.random.randn(n_points, n_dims)
    pca_random = PCA()
    pca_random.fit(random_points)
    
    cumvar_random = np.cumsum(pca_random.explained_variance_ratio_)
    dim_95_random = np.searchsorted(cumvar_random, 0.95) + 1
    
    print(f"\n{n_points} random points in {n_dims}D space:")
    print(f"  Expected dimensionality: ~{n_points - 1} (rank of point cloud)")
    print(f"  PCA 95% dimension: {dim_95_random}")
    
    print(f"\nOur actual data:")
    print(f"  PCA 95% dimension: {dim_95_final}")
    
    if dim_95_final < dim_95_random * 0.5:
        print(f"\n✓ Our data is SIGNIFICANTLY lower-dimensional than random!")
        print(f"  This suggests real structure, not just point cloud geometry.")
    else:
        print(f"\n⚠ Our data is similar dimensionality to random points!")
        print(f"  We may be seeing point cloud geometry, not a true manifold.")
    
    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Plot 1: Explained variance - actual data
    ax1 = axes[0, 0]
    dims = np.arange(1, min(11, len(pca_final.explained_variance_ratio_) + 1))
    cumvar = np.cumsum(pca_final.explained_variance_ratio_[:10])
    
    ax1.plot(dims, cumvar, 'o-', linewidth=2, markersize=8, label='Actual data')
    ax1.axhline(0.95, color='red', linestyle='--', label='95% variance')
    ax1.axvline(dim_95_final, color='red', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Number of Components')
    ax1.set_ylabel('Cumulative Variance')
    ax1.set_title(f'Actual Data: {len(final_points)} convergence points')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Explained variance - random points
    ax2 = axes[0, 1]
    dims_random = np.arange(1, min(11, len(pca_random.explained_variance_ratio_) + 1))
    cumvar_random = np.cumsum(pca_random.explained_variance_ratio_[:10])
    
    ax2.plot(dims_random, cumvar_random, 'o-', linewidth=2, markersize=8, 
            color='orange', label='Random points')
    ax2.axhline(0.95, color='red', linestyle='--', label='95% variance')
    ax2.set_xlabel('Number of Components')
    ax2.set_ylabel('Cumulative Variance')
    ax2.set_title(f'Random Baseline: {n_points} random points')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Sample size requirements
    ax3 = axes[1, 0]
    
    required_samples = np.arange(1, 21) * 10  # 10 samples per dimension
    actual_samples = [len(final_points)] * len(required_samples)
    
    ax3.fill_between(range(1, 21), 0, required_samples, alpha=0.3, 
                     color='green', label='Recommended samples')
    ax3.axhline(len(final_points), color='red', linewidth=2, 
               label=f'Our samples: {len(final_points)}')
    ax3.set_xlabel('Manifold Dimension')
    ax3.set_ylabel('Number of Samples')
    ax3.set_title('Sample Size Requirements vs Actual')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Shade region where we're undersampled
    ax3.axvspan(dim_95_final, 20, alpha=0.2, color='red', 
               label='Undersampled region')
    
    # Plot 4: Summary text
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    summary = f"""
    INTERSECTION ANALYSIS SUMMARY
    
    Sample Size: {len(final_points)} models
    Ambient Dimension: {n_dims}D
    
    Measured Dimensionality: {dim_95_final}D
    Random Baseline: {dim_95_random}D
    
    Compression Ratio: {n_dims / dim_95_final:.1f}×
    
    Assessment:
    {f"✓ Real low-D structure detected!" if dim_95_final < dim_95_random * 0.5 else "⚠ May be undersampled"}
    
    Recommendation:
    - For {dim_95_final}D manifold
    - Need ~{dim_95_final * 10} samples
    - We have {len(final_points)} samples
    - Should collect {dim_95_final * 10 - len(final_points)} more!
    
    Current Status:
    {"✓ Well-sampled" if len(final_points) >= dim_95_final * 10 else f"⚠ Undersampled ({len(final_points) / (dim_95_final * 10) * 100:.0f}%)"}
    """
    
    ax4.text(0.1, 0.5, summary, transform=ax4.transAxes,
            fontsize=10, verticalalignment='center',
            family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('results_dynamics/intersection_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved: results_dynamics/intersection_analysis.png")
    
    print(f"\n{'=' * 80}")
    print(" CONCLUSION")
    print(f"{'=' * 80}")
    
    print(f"\nWe are measuring the {dim_95_final}D CONVERGENCE MANIFOLD,")
    print(f"which is where {len(final_points)} different training runs converge.")
    print(f"\nThis IS an intersection in the sense that:")
    print(f"  - Different initializations → different trajectories")
    print(f"  - But all trajectories → same final region")
    print(f"  - That region is {dim_95_final}D")
    print(f"\nHowever, we need MORE samples to precisely characterize it!")
    print(f"Recommended: Train ~{dim_95_final * 10} models instead of {len(final_points)}")


if __name__ == '__main__':
    main()
