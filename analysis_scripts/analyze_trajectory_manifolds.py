#!/usr/bin/env python3
"""
Analyze per-model trajectory manifolds and their intersections.

Key question: Do different models follow different manifolds during training,
or is there a single universal manifold that all trajectories approach?
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import umap
import os
import json

def load_trajectories(results_dir='results_dynamics'):
    """Load trajectory data from dynamics experiment."""
    # Load trajectories
    traj_data = np.load(os.path.join(results_dir, 'trajectories.npz'))
    
    # Load metadata
    with open(os.path.join(results_dir, 'trajectory_metadata.json'), 'r') as f:
        metadata = json.load(f)
    
    trajectories = {}
    for key in traj_data.files:
        trajectories[key] = traj_data[key]
    
    return trajectories, metadata


def analyze_individual_trajectory(trajectory, model_name):
    """
    Analyze the manifold traversed by a single model during training.
    
    Returns:
        - Intrinsic dimensionality along trajectory
        - Distance traveled
        - Curvature estimates
    """
    n_snapshots = len(trajectory)
    
    # Distance traveled (path length)
    distances = []
    for i in range(len(trajectory) - 1):
        dist = np.linalg.norm(trajectory[i+1] - trajectory[i])
        distances.append(dist)
    
    total_distance = sum(distances)
    
    # Displacement (start to end)
    displacement = np.linalg.norm(trajectory[-1] - trajectory[0])
    
    # Tortuosity: ratio of path length to displacement
    # High tortuosity = winding path, low = direct
    tortuosity = total_distance / displacement if displacement > 0 else 0
    
    # Local PCA along trajectory (if enough points)
    local_dims = []
    if n_snapshots >= 5:
        # Sliding window PCA
        window_size = min(5, n_snapshots)
        for i in range(n_snapshots - window_size + 1):
            window = trajectory[i:i+window_size]
            if len(window) > 1:
                pca = PCA()
                pca.fit(window)
                # How many dimensions to capture 95% variance
                cumvar = np.cumsum(pca.explained_variance_ratio_)
                dim_95 = np.searchsorted(cumvar, 0.95) + 1
                local_dims.append(dim_95)
    
    return {
        'n_snapshots': n_snapshots,
        'total_distance': total_distance,
        'displacement': displacement,
        'tortuosity': tortuosity,
        'avg_step_size': np.mean(distances) if distances else 0,
        'local_dims': local_dims,
        'avg_local_dim': np.mean(local_dims) if local_dims else None
    }


def analyze_trajectory_convergence(trajectories, metadata):
    """
    Check if trajectories from different models converge to the same manifold.
    """
    print("\n" + "=" * 80)
    print(" TRAJECTORY MANIFOLD ANALYSIS")
    print("=" * 80)
    
    # Separate signal vs noise
    signal_trajectories = {}
    noise_trajectories = {}
    
    for key, traj in trajectories.items():
        # Find corresponding metadata
        idx = int(key.split('_')[-1])
        meta = metadata[idx]
        
        if meta.get('is_noise', False):
            noise_trajectories[key] = traj
        else:
            signal_trajectories[key] = traj
    
    print(f"\nSignal models: {len(signal_trajectories)}")
    print(f"Noise models: {len(noise_trajectories)}")
    
    # Analyze each trajectory individually
    print("\n" + "-" * 80)
    print(" INDIVIDUAL TRAJECTORY ANALYSIS")
    print("-" * 80)
    
    signal_stats = []
    noise_stats = []
    
    for key, traj in signal_trajectories.items():
        stats = analyze_individual_trajectory(traj, key)
        signal_stats.append(stats)
        print(f"\n{key} (SIGNAL):")
        print(f"  Path length: {stats['total_distance']:.2f}")
        print(f"  Displacement: {stats['displacement']:.2f}")
        print(f"  Tortuosity: {stats['tortuosity']:.2f}")
        if stats['avg_local_dim']:
            print(f"  Avg local dim: {stats['avg_local_dim']:.1f}")
    
    for key, traj in noise_trajectories.items():
        stats = analyze_individual_trajectory(traj, key)
        noise_stats.append(stats)
        print(f"\n{key} (NOISE):")
        print(f"  Path length: {stats['total_distance']:.2f}")
        print(f"  Displacement: {stats['displacement']:.2f}")
        print(f"  Tortuosity: {stats['tortuosity']:.2f}")
        if stats['avg_local_dim']:
            print(f"  Avg local dim: {stats['avg_local_dim']:.1f}")
    
    # Summary statistics
    print("\n" + "-" * 80)
    print(" SUMMARY: SIGNAL vs NOISE TRAJECTORIES")
    print("-" * 80)
    
    if signal_stats:
        print(f"\nSIGNAL models:")
        print(f"  Avg path length: {np.mean([s['total_distance'] for s in signal_stats]):.2f} ± {np.std([s['total_distance'] for s in signal_stats]):.2f}")
        print(f"  Avg tortuosity: {np.mean([s['tortuosity'] for s in signal_stats]):.2f} ± {np.std([s['tortuosity'] for s in signal_stats]):.2f}")
        dims = [s['avg_local_dim'] for s in signal_stats if s['avg_local_dim']]
        if dims:
            print(f"  Avg local dim: {np.mean(dims):.2f} ± {np.std(dims):.2f}")
    
    if noise_stats:
        print(f"\nNOISE models:")
        print(f"  Avg path length: {np.mean([s['total_distance'] for s in noise_stats]):.2f} ± {np.std([s['total_distance'] for s in noise_stats]):.2f}")
        print(f"  Avg tortuosity: {np.mean([s['tortuosity'] for s in noise_stats]):.2f} ± {np.std([s['tortuosity'] for s in noise_stats]):.2f}")
        dims = [s['avg_local_dim'] for s in noise_stats if s['avg_local_dim']]
        if dims:
            print(f"  Avg local dim: {np.mean(dims):.2f} ± {np.std(dims):.2f}")
    
    return signal_stats, noise_stats


def visualize_trajectories_in_common_space(trajectories, metadata, save_dir='results_dynamics'):
    """
    Project ALL trajectory points (from all models) into common PCA/UMAP space.
    This reveals if different models traverse different manifolds.
    """
    print("\n" + "-" * 80)
    print(" PROJECTING ALL TRAJECTORIES TO COMMON SPACE")
    print("-" * 80)
    
    # Collect ALL points from ALL trajectories
    all_points = []
    point_labels = []  # (model_idx, timepoint, is_noise)
    
    for i, (key, traj) in enumerate(trajectories.items()):
        idx = int(key.split('_')[-1])
        meta = metadata[idx]
        is_noise = meta.get('is_noise', False)
        
        for t, point in enumerate(traj):
            all_points.append(point)
            point_labels.append({
                'model_key': key,
                'model_idx': i,
                'timepoint': t,
                'is_noise': is_noise,
                'is_final': (t == len(traj) - 1)
            })
    
    all_points = np.array(all_points)
    print(f"\nTotal points: {len(all_points)}")
    print(f"  From {len(trajectories)} models")
    print(f"  Dimensionality: {all_points.shape[1]}")
    
    # PCA projection
    print("\nComputing PCA...")
    pca = PCA(n_components=3)
    points_pca = pca.fit_transform(all_points)
    print(f"  Variance explained: {pca.explained_variance_ratio_[:3].sum():.1%}")
    
    # UMAP projection
    print("Computing UMAP...")
    reducer = umap.UMAP(n_components=2, random_state=42)
    points_umap = reducer.fit_transform(all_points)
    
    # Visualization
    fig = plt.figure(figsize=(20, 8))
    
    # Plot 1: PCA - trajectories colored by model
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    
    for model_idx in range(len(trajectories)):
        mask = np.array([p['model_idx'] == model_idx for p in point_labels])
        is_noise = point_labels[np.where(mask)[0][0]]['is_noise']
        color = 'red' if is_noise else 'blue'
        alpha = 0.3
        
        traj_points = points_pca[mask]
        ax1.plot(traj_points[:, 0], traj_points[:, 1], traj_points[:, 2], 
                color=color, alpha=alpha, linewidth=1)
        # Mark final point
        ax1.scatter(traj_points[-1, 0], traj_points[-1, 1], traj_points[-1, 2],
                   color=color, s=100, edgecolors='black', linewidth=2, alpha=0.8)
    
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    ax1.set_zlabel('PC3')
    ax1.set_title('All Trajectories in PCA Space\n(Blue=Signal, Red=Noise)')
    
    # Plot 2: UMAP - trajectories
    ax2 = fig.add_subplot(1, 3, 2)
    
    for model_idx in range(len(trajectories)):
        mask = np.array([p['model_idx'] == model_idx for p in point_labels])
        is_noise = point_labels[np.where(mask)[0][0]]['is_noise']
        color = 'red' if is_noise else 'blue'
        
        traj_points = points_umap[mask]
        ax2.plot(traj_points[:, 0], traj_points[:, 1], 
                color=color, alpha=0.3, linewidth=1)
        ax2.scatter(traj_points[-1, 0], traj_points[-1, 1],
                   color=color, s=100, edgecolors='black', linewidth=2, alpha=0.8)
    
    ax2.set_xlabel('UMAP 1')
    ax2.set_ylabel('UMAP 2')
    ax2.set_title('All Trajectories in UMAP Space')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: UMAP - colored by time (to see convergence)
    ax3 = fig.add_subplot(1, 3, 3)
    
    times = np.array([p['timepoint'] for p in point_labels])
    scatter = ax3.scatter(points_umap[:, 0], points_umap[:, 1], 
                         c=times, cmap='viridis', s=20, alpha=0.5)
    
    # Mark final points
    final_mask = np.array([p['is_final'] for p in point_labels])
    final_is_noise = np.array([p['is_noise'] for p in point_labels])
    
    signal_final = final_mask & ~final_is_noise
    noise_final = final_mask & final_is_noise
    
    ax3.scatter(points_umap[signal_final, 0], points_umap[signal_final, 1],
               color='blue', s=200, edgecolors='black', linewidth=2, 
               label='Signal (final)', marker='*', zorder=10)
    ax3.scatter(points_umap[noise_final, 0], points_umap[noise_final, 1],
               color='red', s=200, edgecolors='black', linewidth=2, 
               label='Noise (final)', marker='*', zorder=10)
    
    ax3.set_xlabel('UMAP 1')
    ax3.set_ylabel('UMAP 2')
    ax3.set_title('Trajectories Colored by Time\n(Stars = Final Points)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax3, label='Training Step')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'trajectory_manifolds.png'), dpi=300, bbox_inches='tight')
    print(f"\nSaved: {os.path.join(save_dir, 'trajectory_manifolds.png')}")
    
    return points_pca, points_umap, point_labels


def main():
    # Load data
    trajectories, metadata = load_trajectories()
    
    # Analyze individual trajectories
    signal_stats, noise_stats = analyze_trajectory_convergence(trajectories, metadata)
    
    # Visualize in common space
    points_pca, points_umap, point_labels = visualize_trajectories_in_common_space(
        trajectories, metadata
    )
    
    print("\n" + "=" * 80)
    print(" KEY FINDINGS")
    print("=" * 80)
    print("\nCheck trajectory_manifolds.png to see:")
    print("  1. Do different models follow different paths?")
    print("  2. Do they converge to the same final region?")
    print("  3. Is there a common manifold they all approach?")
    print("\nIf trajectories are distinct but converge → manifold is an attractor")
    print("If trajectories overlap throughout → following same manifold from start")
    print("=" * 80)


if __name__ == '__main__':
    main()
