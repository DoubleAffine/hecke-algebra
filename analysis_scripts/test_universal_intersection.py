#!/usr/bin/env python3
"""
CORRECT EXPERIMENT: Universal Subspace Intersection Test

The right question:
- Train models on MULTIPLE DIFFERENT datasets
- Find the manifold for each dataset
- Compute the INTERSECTION of all manifolds
- Check if intersection is finite-dimensional or empty

This tests TRUE universality across tasks!
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from scipy.linalg import subspace_angles
import argparse
import os
import json

from src.trainer import Trainer
from src.datasets import DatasetManager

sns.set_style('whitegrid')

def compute_pca_subspace(weight_matrix, n_components=50):
    """Get principal subspace for a set of models."""
    pca = PCA(n_components=n_components)
    pca.fit(weight_matrix)

    # Return principal components (column vectors)
    return pca.components_.T, pca.explained_variance_ratio_

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models-per-dataset', type=int, default=50,
                       help='Models to train per dataset')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Training epochs')
    parser.add_argument('--save-dir', type=str, default='results_intersection',
                       help='Save directory')

    args = parser.parse_args()

    print("=" * 80)
    print(" UNIVERSAL SUBSPACE INTERSECTION TEST")
    print(" Testing if different tasks share a common low-D subspace")
    print("=" * 80)

    # CRITICAL: Use datasets with SAME input/output dimensions
    # (Otherwise we can't compare weight vectors!)
    datasets = [
        'binary_classification_synthetic',  # Clean binary classification
        'binary_random_labels',             # Same dims, random labels (noise)
        # Can't add more without matching dimensions!
    ]

    print(f"\nConfiguration:")
    print(f"  Models per dataset: {args.models_per_dataset}")
    print(f"  Datasets: {len(datasets)}")
    for ds in datasets:
        print(f"    - {ds}")
    print(f"  Total models: {len(datasets) * args.models_per_dataset}")

    os.makedirs(args.save_dir, exist_ok=True)

    # Storage: one entry per dataset
    dataset_manifolds = {}

    print(f"\n{'='*80}")
    print(f" PHASE 1: TRAIN MODELS ON EACH DATASET")
    print(f"{'='*80}")

    for dataset_idx, dataset_name in enumerate(datasets):
        print(f"\n[{dataset_idx+1}/{len(datasets)}] Dataset: {dataset_name}")
        print("-" * 80)

        # Load dataset
        train_loader, test_loader, dataset_metadata = DatasetManager.load_dataset(
            dataset_name
        )

        print(f"  Task: {dataset_metadata['task_type']}")
        print(f"  Dims: {dataset_metadata['input_dim']} → {dataset_metadata['output_dim']}")

        # Create trainer
        trainer = Trainer(
            hidden_dims=[16, 16],
            learning_rate=0.001,
            epochs=args.epochs,
            patience=20
        )

        # Train multiple models on THIS dataset
        weights_for_dataset = []
        metadata_for_dataset = []

        for model_idx in range(args.models_per_dataset):
            final_weights, train_stats = trainer.train_single_model(
                train_loader, test_loader, dataset_metadata
            )

            weights_for_dataset.append(final_weights)
            metadata_for_dataset.append(train_stats)

            if (model_idx + 1) % 10 == 0:
                print(f"    [{model_idx+1}/{args.models_per_dataset}] "
                      f"Loss: {train_stats['best_test_loss']:.4f}")

        # Cleanup dataset
        DatasetManager.cleanup()

        # Store
        weight_matrix = np.array(weights_for_dataset)
        dataset_manifolds[dataset_name] = {
            'weights': weight_matrix,
            'metadata': metadata_for_dataset,
            'dataset_info': dataset_metadata
        }

        print(f"  ✓ Trained {len(weights_for_dataset)} models")
        print(f"  Weight matrix: {weight_matrix.shape}")

    print(f"\n{'='*80}")
    print(f" PHASE 2: ANALYZE EACH DATASET'S MANIFOLD")
    print(f"{'='*80}")

    dataset_subspaces = {}

    for dataset_name, data in dataset_manifolds.items():
        print(f"\n--- {dataset_name} ---")

        weight_matrix = data['weights']

        # PCA analysis
        pca = PCA()
        pca.fit(weight_matrix)

        var_ratios = pca.explained_variance_ratio_
        cumsum = np.cumsum(var_ratios)

        dim_95 = np.argmax(cumsum >= 0.95) + 1
        effective_dim = (np.sum(var_ratios) ** 2) / np.sum(var_ratios ** 2)

        print(f"  Dimension (95% var): {dim_95}")
        print(f"  Effective dimension: {effective_dim:.1f}")

        # Store principal subspace (top components that explain 95% variance)
        subspace, var_explained = compute_pca_subspace(weight_matrix, n_components=dim_95)
        dataset_subspaces[dataset_name] = {
            'subspace': subspace,
            'dimension': dim_95,
            'effective_dim': effective_dim,
            'variance_explained': var_explained
        }

    print(f"\n{'='*80}")
    print(f" PHASE 3: COMPUTE INTERSECTION")
    print(f"{'='*80}")

    # Get all dataset names
    dataset_names = list(dataset_subspaces.keys())

    if len(dataset_names) < 2:
        print("\n⚠ Need at least 2 datasets to compute intersection!")
        return

    print(f"\nComparing subspaces between datasets:")

    # Compute pairwise principal angles
    for i in range(len(dataset_names)):
        for j in range(i + 1, len(dataset_names)):
            ds_i = dataset_names[i]
            ds_j = dataset_names[j]

            subspace_i = dataset_subspaces[ds_i]['subspace']
            subspace_j = dataset_subspaces[ds_j]['subspace']

            # Compute principal angles
            angles = subspace_angles(subspace_i, subspace_j)
            angles_deg = np.degrees(angles)

            print(f"\n{ds_i} ↔ {ds_j}:")
            print(f"  Dimensions: {subspace_i.shape[1]}D × {subspace_j.shape[1]}D")
            print(f"  First 10 principal angles: {angles_deg[:10]}")
            print(f"  Min angle: {np.min(angles_deg):.1f}°")
            print(f"  Mean angle: {np.mean(angles_deg):.1f}°")
            print(f"  Max angle: {np.max(angles_deg):.1f}°")

            # Count aligned dimensions (angle < 10°)
            aligned = np.sum(angles_deg < 10)
            print(f"  Aligned dimensions (< 10°): {aligned}")

            if aligned > 10:
                print(f"    → STRONG INTERSECTION (~{aligned}D shared subspace)")
            elif aligned > 5:
                print(f"    → MODERATE INTERSECTION (~{aligned}D shared)")
            elif aligned > 0:
                print(f"    → WEAK INTERSECTION (~{aligned}D shared)")
            else:
                print(f"    → NO INTERSECTION (orthogonal subspaces)")

    # Global intersection: PCA on ALL models from ALL datasets
    print(f"\n{'='*80}")
    print(f" PHASE 4: GLOBAL ANALYSIS")
    print(f"{'='*80}")

    # Concatenate all weights
    all_weights = []
    all_labels = []

    for dataset_name, data in dataset_manifolds.items():
        all_weights.append(data['weights'])
        all_labels.extend([dataset_name] * len(data['weights']))

    combined_weights = np.vstack(all_weights)

    print(f"\nCombined weight matrix: {combined_weights.shape}")
    print(f"  Total models: {len(combined_weights)}")

    # PCA on combined data
    pca_global = PCA()
    pca_global.fit(combined_weights)

    var_global = pca_global.explained_variance_ratio_
    cumsum_global = np.cumsum(var_global)

    dim_95_global = np.argmax(cumsum_global >= 0.95) + 1
    effective_dim_global = (np.sum(var_global) ** 2) / np.sum(var_global ** 2)

    print(f"\nGlobal PCA (all datasets combined):")
    print(f"  Dimension (95% var): {dim_95_global}")
    print(f"  Effective dimension: {effective_dim_global:.1f}")

    # Compare to individual dimensions
    individual_dims = [info['effective_dim'] for info in dataset_subspaces.values()]
    mean_individual = np.mean(individual_dims)

    print(f"\nComparison:")
    print(f"  Individual dataset dimensions: {individual_dims}")
    print(f"  Mean individual: {mean_individual:.1f}D")
    print(f"  Global (combined): {effective_dim_global:.1f}D")

    if effective_dim_global < mean_individual * 0.8:
        print(f"\n  ✓ INTERSECTION EXISTS!")
        print(f"  → Global dimension < individual dimensions")
        print(f"  → Datasets share a ~{effective_dim_global:.0f}D subspace")
        intersection_dim = effective_dim_global
    elif effective_dim_global > mean_individual * 1.2:
        print(f"\n  ✗ NO INTERSECTION")
        print(f"  → Global dimension > individual dimensions")
        print(f"  → Datasets occupy DIFFERENT subspaces")
        intersection_dim = 0
    else:
        print(f"\n  ? AMBIGUOUS")
        print(f"  → Global ≈ individual dimensions")
        print(f"  → Weak evidence for intersection")
        intersection_dim = effective_dim_global * 0.5  # Rough estimate

    # Visualization
    print(f"\n{'='*80}")
    print(f" PHASE 5: VISUALIZATION")
    print(f"{'='*80}")

    fig = plt.figure(figsize=(18, 10))

    # Plot 1: PCA colored by dataset
    ax1 = plt.subplot(2, 3, 1)
    pca_transformed = pca_global.transform(combined_weights)

    for dataset_name in dataset_names:
        mask = np.array(all_labels) == dataset_name
        ax1.scatter(pca_transformed[mask, 0], pca_transformed[mask, 1],
                   label=dataset_name, alpha=0.6, s=50)

    ax1.set_xlabel(f'Global PC1 ({var_global[0]*100:.1f}%)')
    ax1.set_ylabel(f'Global PC2 ({var_global[1]*100:.1f}%)')
    ax1.set_title('Global PCA: All Datasets\n(Overlap = Intersection)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: 3D PCA
    ax2 = plt.subplot(2, 3, 2, projection='3d')
    for dataset_name in dataset_names:
        mask = np.array(all_labels) == dataset_name
        ax2.scatter(pca_transformed[mask, 0],
                   pca_transformed[mask, 1],
                   pca_transformed[mask, 2],
                   label=dataset_name, alpha=0.6, s=30)
    ax2.set_xlabel('PC1')
    ax2.set_ylabel('PC2')
    ax2.set_zlabel('PC3')
    ax2.set_title('3D View')
    ax2.legend()

    # Plot 3: Dimension comparison
    ax3 = plt.subplot(2, 3, 3)
    dims_to_plot = [info['effective_dim'] for info in dataset_subspaces.values()]
    dims_to_plot.append(effective_dim_global)
    labels_plot = list(dataset_names) + ['Global (Intersection)']
    colors = ['blue'] * len(dataset_names) + ['red']

    ax3.bar(range(len(dims_to_plot)), dims_to_plot, color=colors, alpha=0.7)
    ax3.set_xticks(range(len(dims_to_plot)))
    ax3.set_xticklabels(labels_plot, rotation=45, ha='right')
    ax3.set_ylabel('Effective Dimension')
    ax3.set_title('Dimension Comparison\n(Red = Intersection)')
    ax3.grid(True, alpha=0.3, axis='y')

    # Plot 4: Variance spectrum
    ax4 = plt.subplot(2, 3, 4)
    for dataset_name, info in dataset_subspaces.items():
        var_ratios = info['variance_explained'][:50]
        ax4.plot(range(1, len(var_ratios)+1), var_ratios,
                label=dataset_name, linewidth=2, alpha=0.7)
    ax4.set_xlabel('Principal Component')
    ax4.set_ylabel('Explained Variance Ratio')
    ax4.set_title('Variance Spectra by Dataset')
    ax4.set_yscale('log')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Plot 5: Distance distributions
    ax5 = plt.subplot(2, 3, 5)
    from scipy.spatial.distance import pdist

    for dataset_name, data in dataset_manifolds.items():
        distances = pdist(data['weights'])
        ax5.hist(distances, bins=30, alpha=0.5, label=dataset_name)

    ax5.set_xlabel('Pairwise Distance')
    ax5.set_ylabel('Count')
    ax5.set_title('Distance Distributions')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Plot 6: Summary
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')

    summary = f"""
UNIVERSAL INTERSECTION SUMMARY

Datasets tested: {len(dataset_names)}
Models per dataset: {args.models_per_dataset}

Individual dimensions:
{chr(10).join(f'  {name}: {dataset_subspaces[name]["effective_dim"]:.1f}D' for name in dataset_names)}

Global dimension: {effective_dim_global:.1f}D

Intersection estimate: ~{intersection_dim:.0f}D

Status: {'INTERSECTION EXISTS' if effective_dim_global < mean_individual * 0.8 else 'NO CLEAR INTERSECTION'}
"""

    ax6.text(0.1, 0.5, summary, fontsize=10, family='monospace',
             verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    plt.savefig(f"{args.save_dir}/intersection_analysis.png", dpi=300, bbox_inches='tight')
    print(f"Saved: {args.save_dir}/intersection_analysis.png")

    # Save results
    results = {
        'datasets': dataset_names,
        'models_per_dataset': args.models_per_dataset,
        'individual_dimensions': {name: float(info['effective_dim'])
                                 for name, info in dataset_subspaces.items()},
        'global_dimension': float(effective_dim_global),
        'intersection_dimension_estimate': float(intersection_dim)
    }

    with open(f"{args.save_dir}/results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*80}")
    print(f" CONCLUSION")
    print(f"{'='*80}")

    if effective_dim_global < mean_individual * 0.8:
        print(f"\n✓ UNIVERSAL SUBSPACE EXISTS")
        print(f"  Dimension: ~{intersection_dim:.0f}D")
        print(f"  Compression: {465 / intersection_dim:.1f}× from ambient space")
    else:
        print(f"\n✗ NO UNIVERSAL SUBSPACE")
        print(f"  Different tasks occupy different regions")
        print(f"  No shared low-dimensional structure")

    print(f"\nResults saved to: {args.save_dir}/")

if __name__ == '__main__':
    main()
