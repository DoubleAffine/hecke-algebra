#!/usr/bin/env python3
"""
Multi-Dataset Manifold Intersection Experiment

Goal: For each of 100 datasets, estimate the attractor manifold from models
      initialized near zero, then intersect ALL manifolds to find the
      universal subspace.

Design:
- 100 different datasets
- For each dataset: train N models from near-zero initializations
- Estimate manifold dimension for each dataset
- Compute intersection of all 100 manifolds
- Compare with PCA-based global analysis (as in the paper)
"""
import numpy as np
import argparse
import os
import sys
import time
from datetime import datetime
import json

from src.trainer import Trainer
from src.datasets import DatasetManager

class ExperimentProgress:
    """Visual progress tracker for multi-dataset experiment."""

    def __init__(self, n_datasets, models_per_dataset):
        self.n_datasets = n_datasets
        self.models_per_dataset = models_per_dataset
        self.total_models = n_datasets * models_per_dataset

        self.current_dataset = 0
        self.current_model = 0
        self.total_completed = 0

        self.start_time = time.time()

    def start_dataset(self, dataset_idx, dataset_name):
        self.current_dataset = dataset_idx
        self.dataset_name = dataset_name
        self.dataset_start = time.time()

        print(f'\n{"="*80}')
        print(f' DATASET {dataset_idx+1}/{self.n_datasets}: {dataset_name}')
        print(f'{"="*80}')

    def update_model(self, model_idx, loss, acc=None):
        self.current_model = model_idx
        self.total_completed += 1

        # Progress bar
        progress = self.total_completed / self.total_models
        bar_width = 50
        filled = int(bar_width * progress)
        bar = '█' * filled + '░' * (bar_width - filled)

        # ETA
        elapsed = time.time() - self.start_time
        if progress > 0:
            total_time = elapsed / progress
            remaining = total_time - elapsed
            eta_min = remaining / 60
        else:
            eta_min = 0

        sys.stdout.write('\r')
        sys.stdout.write(f'[{bar}] {progress*100:.1f}% ')
        sys.stdout.write(f'| Overall: {self.total_completed}/{self.total_models} ')
        sys.stdout.write(f'| Dataset: {model_idx+1}/{self.models_per_dataset} ')
        sys.stdout.write(f'| Loss: {loss:.4f} ')
        if acc is not None:
            sys.stdout.write(f'| Acc: {acc:.1f}% ')
        sys.stdout.write(f'| ETA: {eta_min:.1f}min ')
        sys.stdout.flush()

    def finish_dataset(self, dimension):
        elapsed = (time.time() - self.dataset_start) / 60
        print(f'\n  ✓ Dataset complete: {dimension:.1f}D manifold ({elapsed:.1f} min)')


def get_dataset_list(n_datasets):
    """
    Generate a list of diverse datasets to test.

    We want varied tasks to test universality:
    - Different data distributions
    - Different pattern types
    - Different complexities
    """
    datasets = []

    # Binary classification tasks with different characteristics
    base_tasks = [
        'binary_classification_synthetic',
        'binary_random_labels',
    ]

    # We'll use the same base tasks but with different random seeds
    # This gives us different data distributions while keeping architecture consistent
    for i in range(n_datasets):
        # Alternate between signal and noise tasks
        base = base_tasks[i % len(base_tasks)]
        datasets.append({
            'name': f'{base}_seed{i}',
            'base_task': base,
            'seed': i
        })

    return datasets


def main():
    parser = argparse.ArgumentParser(
        description='Multi-dataset manifold intersection experiment'
    )
    parser.add_argument('--datasets', type=int, default=100,
                       help='Number of datasets to test (default: 100)')
    parser.add_argument('--models-per-dataset', type=int, default=50,
                       help='Models per dataset (default: 50)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Training epochs (default: 100)')
    parser.add_argument('--save-dir', type=str,
                       default='experiments/current/multi_dataset_intersection',
                       help='Save directory')

    args = parser.parse_args()

    print("=" * 80)
    print(" MULTI-DATASET MANIFOLD INTERSECTION")
    print(" Estimating attractor manifolds and computing intersection")
    print("=" * 80)

    print(f"\nConfiguration:")
    print(f"  Datasets: {args.datasets}")
    print(f"  Models per dataset: {args.models_per_dataset}")
    print(f"  Total models: {args.datasets * args.models_per_dataset}")
    print(f"  Epochs per model: {args.epochs}")
    print(f"  Initialization: Near zero (small random)")

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Configuration
    config = {
        'n_datasets': args.datasets,
        'models_per_dataset': args.models_per_dataset,
        'epochs': args.epochs,
        'initialization': 'near_zero',
        'started': datetime.now().isoformat()
    }

    with open(f'{args.save_dir}/config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # Get dataset list
    datasets = get_dataset_list(args.datasets)

    # Progress tracker
    progress = ExperimentProgress(args.datasets, args.models_per_dataset)

    # Storage for manifold data
    all_manifolds = []
    dataset_metadata = []

    print(f'\n{"="*80}')
    print(f' PHASE 1: TRAINING MODELS ON EACH DATASET')
    print(f'{"="*80}')

    # Process each dataset
    for dataset_idx, dataset_info in enumerate(datasets):
        progress.start_dataset(dataset_idx, dataset_info['name'])

        # Load dataset with specific seed
        np.random.seed(dataset_info['seed'])
        train_loader, test_loader, ds_metadata = DatasetManager.load_dataset(
            dataset_info['base_task']
        )

        # Create trainer
        trainer = Trainer(
            hidden_dims=[16, 16],
            learning_rate=0.001,
            epochs=args.epochs,
            patience=20
        )

        # Train models from near-zero initializations
        dataset_weights = []

        for model_idx in range(args.models_per_dataset):
            # Train model (initialization is handled in trainer)
            final_weights, train_stats = trainer.train_single_model(
                train_loader, test_loader, ds_metadata
            )

            # Update progress
            progress.update_model(
                model_idx,
                train_stats['best_test_loss'],
                train_stats.get('best_test_accuracy')
            )

            # Store weights
            dataset_weights.append(final_weights)

        # Convert to array and estimate dimension
        weights_array = np.array(dataset_weights)

        # PCA for dimension estimation
        from sklearn.decomposition import PCA
        pca = PCA()
        pca.fit(weights_array)

        var_ratios = pca.explained_variance_ratio_
        cumsum = np.cumsum(var_ratios)

        dim_95 = int(np.argmax(cumsum >= 0.95) + 1)
        effective_dim = float((np.sum(var_ratios) ** 2) / np.sum(var_ratios ** 2))

        # Save manifold data
        manifold_data = {
            'dataset_idx': dataset_idx,
            'dataset_name': dataset_info['name'],
            'weights': weights_array,
            'pca': pca,
            'dim_95': dim_95,
            'effective_dim': effective_dim,
            'variance_ratios': var_ratios
        }

        all_manifolds.append(manifold_data)

        # Save individual dataset results incrementally
        np.save(
            f'{args.save_dir}/weights_dataset_{dataset_idx:03d}.npy',
            weights_array
        )

        dataset_metadata.append({
            'dataset_idx': dataset_idx,
            'dataset_name': dataset_info['name'],
            'dim_95': dim_95,
            'effective_dim': effective_dim
        })

        # Cleanup
        del train_loader, test_loader
        DatasetManager.cleanup()

        progress.finish_dataset(effective_dim)

    print(f'\n{"="*80}')
    print(f' PHASE 2: MULTI-METHOD INTERSECTION ANALYSIS')
    print(f'{"="*80}')

    # Analyze individual dimensions
    dims = [m['effective_dim'] for m in all_manifolds]

    print(f"\nIndividual manifold dimensions:")
    print(f"  Mean: {np.mean(dims):.1f}D")
    print(f"  Std: {np.std(dims):.1f}D")
    print(f"  Min: {np.min(dims):.1f}D")
    print(f"  Max: {np.max(dims):.1f}D")
    print(f"  Median: {np.median(dims):.1f}D")

    # Prepare data for intersection analysis
    from src.intersection_methods import IntersectionAnalyzer

    weights_list = [m['weights'] for m in all_manifolds]
    dims_list = [m['dim_95'] for m in all_manifolds]

    analyzer = IntersectionAnalyzer(
        manifolds=weights_list,
        manifold_dims=dims_list,
        variance_threshold=0.95
    )

    # Run all intersection methods
    intersection_results = analyzer.analyze_all_methods()

    print(f'\n{"="*80}')
    print(f' LEGACY: GLOBAL PCA (as in paper)')
    print(f'{"="*80}')

    # Combine all weights
    print(f"\nCombining all {len(all_manifolds)} manifolds...")
    all_weights = np.vstack([m['weights'] for m in all_manifolds])
    print(f"  Combined shape: {all_weights.shape}")

    # Global PCA
    print(f"\nComputing global PCA...")
    pca_global = PCA()
    pca_global.fit(all_weights)

    var_global = pca_global.explained_variance_ratio_
    cumsum_global = np.cumsum(var_global)

    dim_95_global = int(np.argmax(cumsum_global >= 0.95) + 1)
    effective_global = float((np.sum(var_global) ** 2) / np.sum(var_global ** 2))

    print(f"\nGlobal analysis (all models combined):")
    print(f"  Dimension (95% var): {dim_95_global}D")
    print(f"  Effective dimension: {effective_global:.1f}D")

    print(f'\n{"="*80}')
    print(f' PHASE 4: INTERSECTION ESTIMATE')
    print(f'{"="*80}')

    mean_individual = np.mean(dims)
    ratio = effective_global / mean_individual

    print(f"\nComparison:")
    print(f"  Mean individual: {mean_individual:.1f}D")
    print(f"  Global: {effective_global:.1f}D")
    print(f"  Ratio: {ratio:.2f}")

    # If ratio ≈ 1, strong overlap (shared subspace)
    # If ratio ≈ n_datasets, orthogonal (no intersection)
    # In between: partial overlap

    if ratio < 1.2:
        conclusion = "STRONG OVERLAP - Universal subspace exists"
        intersection_dim = effective_global
        print(f"\n  ✓ {conclusion}")
        print(f"  Intersection dimension: ~{intersection_dim:.0f}D")
    elif ratio > args.datasets * 0.5:
        conclusion = "NEARLY ORTHOGONAL - No universal subspace"
        intersection_dim = 0
        print(f"\n  ✗ {conclusion}")
    else:
        conclusion = "PARTIAL OVERLAP"
        # Estimate intersection using ratio
        # intersection ≈ global - (ratio - 1) * mean_individual
        intersection_dim = max(0, effective_global - (ratio - 1) * mean_individual)
        print(f"\n  ? {conclusion}")
        print(f"  Estimated intersection: ~{intersection_dim:.0f}D")

    # Save results
    results = {
        'config': config,
        'individual_dimensions': {
            'mean': mean_individual,
            'std': float(np.std(dims)),
            'min': float(np.min(dims)),
            'max': float(np.max(dims)),
            'all': [float(d) for d in dims]
        },
        'global': {
            'dimension_95': dim_95_global,
            'effective_dimension': effective_global
        },
        'intersection_methods': {
            method: {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                     for k, v in result.items() if k != 'method'}
            for method, result in intersection_results.items()
            if result is not None
        },
        'legacy_pca_intersection': {
            'ratio': ratio,
            'conclusion': conclusion,
            'intersection_dimension': intersection_dim
        },
        'dataset_metadata': dataset_metadata,
        'completed': datetime.now().isoformat()
    }

    print(f"\nSaving results...")
    with open(f'{args.save_dir}/results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Save global PCA
    np.save(f'{args.save_dir}/global_pca_components.npy', pca_global.components_)
    np.save(f'{args.save_dir}/global_pca_variance.npy', var_global)

    print(f'\n{"="*80}')
    print(f' EXPERIMENT COMPLETE')
    print(f'{"="*80}')

    total_time = (time.time() - progress.start_time) / 60
    print(f"\nTotal time: {total_time:.1f} minutes")
    print(f"Results saved to: {args.save_dir}/")
    print(f'\n{"="*80}')


if __name__ == '__main__':
    main()
