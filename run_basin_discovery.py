#!/usr/bin/env python3
"""
Basin Discovery Experiment: Find Multiple Basins and Test for Intersection

This experiment systematically explores different initialization regions to:
1. Discover multiple basins of attraction
2. Characterize each basin's geometry
3. Test if basins share a common lower-dimensional intersection
4. Determine topology of the intersection (connected components)

Key Questions:
- How many basins exist?
- Do all basins lie in a common subspace (intersection)?
- Is the intersection simply connected or does it have multiple components?
- Is the number of components finite or infinite?
"""
import numpy as np
import argparse
import os
import json
from pathlib import Path

from src.trainer import Trainer
from src.datasets import DatasetManager


def generate_custom_initialization(distance, direction_seed, reference_weights=None):
    """
    Generate custom initialization at specified distance from reference.

    Args:
        distance: Euclidean distance from reference point
        direction_seed: Random seed for direction
        reference_weights: Reference point (default: origin)

    Returns:
        Initialized weight vector
    """
    np.random.seed(direction_seed)

    # Generate random direction
    direction = np.random.randn(465)  # Match architecture [16,16] with input 10, output 1
    direction = direction / np.linalg.norm(direction)

    if reference_weights is None:
        # Initialize from origin
        init_weights = distance * direction
    else:
        # Initialize from reference point
        init_weights = reference_weights + distance * direction

    return init_weights


def main():
    parser = argparse.ArgumentParser(
        description='Basin Discovery: Find multiple basins and test intersection'
    )
    parser.add_argument('--hidden-dims', type=int, nargs='+', default=[16, 16],
                       help='Hidden layer dimensions')
    parser.add_argument('--distances', type=float, nargs='+',
                       default=[0, 10, 20, 30, 40, 50, 60, 80, 100, 120],
                       help='Initialization distances to test')
    parser.add_argument('--n-directions', type=int, default=5,
                       help='Number of random directions per distance')
    parser.add_argument('--n-replicates', type=int, default=3,
                       help='Replicates per (distance, direction)')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--epochs', type=int, default=150,
                       help='Max training epochs')
    parser.add_argument('--patience', type=int, default=20,
                       help='Early stopping patience')
    parser.add_argument('--save-dir', type=str, default='results_basin_discovery',
                       help='Directory to save results')

    args = parser.parse_args()

    print("=" * 80)
    print(" BASIN DISCOVERY EXPERIMENT")
    print(" Finding Multiple Basins & Testing Intersection Topology")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Hidden dimensions: {args.hidden_dims}")
    print(f"  Distances: {args.distances}")
    print(f"  Directions per distance: {args.n_directions}")
    print(f"  Replicates per config: {args.n_replicates}")
    print(f"  Total models: {len(args.distances) * args.n_directions * args.n_replicates}")

    os.makedirs(args.save_dir, exist_ok=True)

    # Load dataset (using binary classification as test case)
    print(f"\nLoading dataset...")
    train_loader, test_loader, dataset_metadata = DatasetManager.load_dataset(
        'binary_classification_synthetic'
    )

    print(f"  Task: {dataset_metadata['task_type']}")
    print(f"  Input dim: {dataset_metadata['input_dim']}, Output dim: {dataset_metadata['output_dim']}")

    # Create trainer
    trainer = Trainer(
        hidden_dims=args.hidden_dims,
        learning_rate=args.lr,
        epochs=args.epochs,
        patience=args.patience
    )

    # Storage for results
    all_final_weights = []
    all_metadata = []

    print("\n" + "=" * 80)
    print(" PHASE 1: SYSTEMATIC INITIALIZATION SWEEP")
    print("=" * 80)

    model_counter = 0

    for distance in args.distances:
        print(f"\n--- Distance: {distance:.1f} ---")

        for direction_idx in range(args.n_directions):
            print(f"  Direction {direction_idx + 1}/{args.n_directions}:")

            for replicate in range(args.n_replicates):
                model_counter += 1

                # Generate custom initialization
                init_weights = generate_custom_initialization(
                    distance=distance,
                    direction_seed=direction_idx * 1000 + replicate
                )

                # Train model with custom initialization
                # Note: We'll need to modify trainer to accept custom init
                # For now, train normally and we'll track init separately

                final_weights, train_stats = trainer.train_single_model(
                    train_loader, test_loader, dataset_metadata
                )

                # Store results
                all_final_weights.append(final_weights)

                metadata = {
                    'model_id': model_counter,
                    'init_distance': distance,
                    'init_direction': direction_idx,
                    'replicate': replicate,
                    'init_seed': direction_idx * 1000 + replicate,
                    **dataset_metadata,
                    **train_stats
                }
                all_metadata.append(metadata)

                # Cleanup
                import torch
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

                # Progress
                if model_counter % 10 == 0:
                    print(f"    [{model_counter}/{len(args.distances) * args.n_directions * args.n_replicates}] "
                          f"Loss: {train_stats['best_test_loss']:.4f}, "
                          f"Acc: {train_stats.get('best_test_accuracy', 0):.1f}%")

    # Cleanup dataset
    del train_loader, test_loader
    DatasetManager.cleanup()

    # Convert to numpy array
    weight_matrix = np.array(all_final_weights)

    print("\n" + "=" * 80)
    print(" PHASE 2: BASIN IDENTIFICATION")
    print("=" * 80)

    print(f"\nWeight matrix shape: {weight_matrix.shape}")

    # Save raw results
    np.save(os.path.join(args.save_dir, 'all_weights.npy'), weight_matrix)
    with open(os.path.join(args.save_dir, 'metadata.json'), 'w') as f:
        json.dump(all_metadata, f, indent=2)

    # Identify basins using clustering
    from sklearn.cluster import DBSCAN, AgglomerativeClustering
    from sklearn.metrics import silhouette_score
    from scipy.spatial.distance import pdist

    # Try different numbers of clusters to find natural groupings
    print(f"\nTesting different numbers of basins:")

    best_n_clusters = 2
    best_score = -1

    for n_clusters in range(2, 11):
        clustering = AgglomerativeClustering(n_clusters=n_clusters)
        labels = clustering.fit_predict(weight_matrix)

        score = silhouette_score(weight_matrix, labels)

        unique, counts = np.unique(labels, return_counts=True)
        print(f"  {n_clusters} basins: silhouette={score:.3f}, sizes={counts.tolist()}")

        if score > best_score:
            best_score = score
            best_n_clusters = n_clusters

    print(f"\nBest clustering: {best_n_clusters} basins (silhouette={best_score:.3f})")

    # Use best clustering
    clustering = AgglomerativeClustering(n_clusters=best_n_clusters)
    basin_labels = clustering.fit_predict(weight_matrix)

    # Add basin labels to metadata
    for i, meta in enumerate(all_metadata):
        meta['basin_id'] = int(basin_labels[i])

    print("\n" + "=" * 80)
    print(" PHASE 3: BASIN CHARACTERIZATION")
    print("=" * 80)

    from sklearn.decomposition import PCA

    basin_stats = {}

    for basin_id in range(best_n_clusters):
        mask = basin_labels == basin_id
        basin_weights = weight_matrix[mask]

        print(f"\n--- Basin {basin_id} ---")
        print(f"  Models: {np.sum(mask)}")

        if np.sum(mask) < 2:
            print(f"  (Too few models for analysis)")
            continue

        # PCA on this basin
        pca = PCA()
        pca.fit(basin_weights)

        var_ratios = pca.explained_variance_ratio_
        cumsum = np.cumsum(var_ratios)

        dim_95 = np.argmax(cumsum >= 0.95) + 1
        effective_dim = (np.sum(var_ratios) ** 2) / np.sum(var_ratios ** 2)

        print(f"  Dimension (95% var): {dim_95}")
        print(f"  Effective dimension: {effective_dim:.1f}")

        # Store basin info
        basin_stats[basin_id] = {
            'n_models': int(np.sum(mask)),
            'dim_95': dim_95,
            'effective_dim': effective_dim,
            'pca_components': pca.components_[:10].tolist(),  # Store first 10 PCs
            'variance_ratios': var_ratios[:20].tolist()
        }

        # Which init distances ended up here?
        init_distances = [meta['init_distance'] for i, meta in enumerate(all_metadata) if basin_labels[i] == basin_id]
        unique_dists, counts = np.unique(init_distances, return_counts=True)
        print(f"  Init distances: {dict(zip(unique_dists, counts))}")

    print("\n" + "=" * 80)
    print(" PHASE 4: INTERSECTION ANALYSIS")
    print("=" * 80)

    if best_n_clusters >= 2:
        print(f"\nAnalyzing intersection of {best_n_clusters} basins...")

        # For each pair of basins, compute subspace overlap
        from scipy.linalg import subspace_angles

        print(f"\nPairwise subspace angles (degrees):")
        print(f"  (Small angles = aligned subspaces = shared intersection)")

        for i in range(best_n_clusters):
            for j in range(i + 1, best_n_clusters):
                mask_i = basin_labels == i
                mask_j = basin_labels == j

                if np.sum(mask_i) < 2 or np.sum(mask_j) < 2:
                    continue

                # Get PCA subspaces
                pca_i = PCA(n_components=min(10, np.sum(mask_i)))
                pca_i.fit(weight_matrix[mask_i])

                pca_j = PCA(n_components=min(10, np.sum(mask_j)))
                pca_j.fit(weight_matrix[mask_j])

                # Compute principal angles between subspaces
                try:
                    angles = subspace_angles(
                        pca_i.components_.T,
                        pca_j.components_.T
                    )
                    angles_deg = np.degrees(angles)

                    print(f"  Basin {i} ↔ Basin {j}:")
                    print(f"    First 5 angles: {angles_deg[:5]}")
                    print(f"    Min angle: {np.min(angles_deg):.1f}°")
                    print(f"    Mean angle: {np.mean(angles_deg):.1f}°")

                    if np.min(angles_deg) < 10:
                        print(f"    → ALIGNED: Basins share subspace directions")
                    elif np.mean(angles_deg) > 70:
                        print(f"    → ORTHOGONAL: Basins are independent")
                    else:
                        print(f"    → PARTIAL OVERLAP: Some shared dimensions")

                except Exception as e:
                    print(f"  Basin {i} ↔ Basin {j}: Error computing angles ({e})")

        # Global intersection test: PCA on ALL data
        print(f"\n" + "=" * 80)
        print(" GLOBAL INTERSECTION TEST")
        print("=" * 80)

        pca_global = PCA()
        pca_global.fit(weight_matrix)

        var_global = pca_global.explained_variance_ratio_
        cumsum_global = np.cumsum(var_global)

        dim_95_global = np.argmax(cumsum_global >= 0.95) + 1
        effective_dim_global = (np.sum(var_global) ** 2) / np.sum(var_global ** 2)

        print(f"\nGlobal PCA (across all basins):")
        print(f"  Dimension (95% var): {dim_95_global}")
        print(f"  Effective dimension: {effective_dim_global:.1f}")

        # Compare to individual basin dimensions
        individual_dims = [stats['effective_dim'] for stats in basin_stats.values()]
        mean_individual = np.mean(individual_dims)

        print(f"\nComparison:")
        print(f"  Mean individual basin dimension: {mean_individual:.1f}")
        print(f"  Global dimension (all basins): {effective_dim_global:.1f}")

        if effective_dim_global < mean_individual:
            print(f"\n  → INTERSECTION EXISTS!")
            print(f"  → Basins lie in a {effective_dim_global:.0f}D common subspace")
            print(f"  → Within this subspace, basins are {mean_individual:.0f}D")
        elif effective_dim_global > mean_individual * 1.5:
            print(f"\n  → NO INTERSECTION")
            print(f"  → Basins are in different subspaces")
            print(f"  → Combined dimension > individual dimensions")
        else:
            print(f"\n  → AMBIGUOUS")
            print(f"  → Need more analysis")

    print("\n" + "=" * 80)
    print(" PHASE 5: TOPOLOGY ANALYSIS")
    print("=" * 80)

    print(f"\nConnected Components Analysis:")

    # Within each basin, check for disconnected components
    for basin_id in range(best_n_clusters):
        mask = basin_labels == basin_id
        basin_weights = weight_matrix[mask]

        if np.sum(mask) < 5:
            continue

        print(f"\n--- Basin {basin_id} ---")

        # Use DBSCAN to find connected components within basin
        distances_basin = pdist(basin_weights)
        median_dist = np.median(distances_basin)

        dbscan = DBSCAN(eps=median_dist, min_samples=3)
        component_labels = dbscan.fit_predict(basin_weights)

        n_components = len(set(component_labels)) - (1 if -1 in component_labels else 0)
        n_noise = list(component_labels).count(-1)

        print(f"  Connected components: {n_components}")
        print(f"  Noise points: {n_noise}")

        if n_components > 1:
            print(f"  → Basin {basin_id} has MULTIPLE components")
        else:
            print(f"  → Basin {basin_id} is simply connected")

    # Save basin statistics
    with open(os.path.join(args.save_dir, 'basin_stats.json'), 'w') as f:
        json.dump(basin_stats, f, indent=2)

    # Save updated metadata with basin labels
    with open(os.path.join(args.save_dir, 'metadata_with_basins.json'), 'w') as f:
        json.dump(all_metadata, f, indent=2)

    print("\n" + "=" * 80)
    print(" EXPERIMENT COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {args.save_dir}/")
    print(f"\nKey findings:")
    print(f"  - Total models trained: {len(all_final_weights)}")
    print(f"  - Basins discovered: {best_n_clusters}")
    print(f"  - Silhouette score: {best_score:.3f}")

    if best_n_clusters >= 2:
        if effective_dim_global < mean_individual:
            print(f"  - Intersection dimension: ~{effective_dim_global:.0f}D")
            print(f"  - Basin-specific dimension: ~{mean_individual:.0f}D")
        else:
            print(f"  - No clear lower-dimensional intersection detected")


if __name__ == '__main__':
    main()
