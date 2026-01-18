#!/usr/bin/env python3
"""
10-Dataset Manifold Intersection Experiment

Goal: Test Universal Subspace Hypothesis across 10 diverse datasets
- All use same architecture [16, 16]
- All binary classification with same input dim (via preprocessing)
- Target >90% accuracy
- 50 models per dataset for robust manifold estimation
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
    """Visual progress tracker."""

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

    def finish_dataset(self, dimension, mean_acc):
        elapsed = (time.time() - self.dataset_start) / 60
        print(f'\n  ✓ Complete: {dimension:.1f}D manifold, {mean_acc:.1f}% mean accuracy ({elapsed:.1f} min)')


def get_10_datasets():
    """
    Return 10 diverse datasets for testing universality.

    IMPORTANT: All must have SAME input dimension (10D) for weight vector compatibility.
    We use only synthetic datasets which can be generated with 10 features.

    Mix of:
    - Synthetic separable with different parameters
    - Random labels (noise baseline)
    - Different random seeds for variety
    """
    datasets = [
        # Synthetic separable with varying difficulty
        {'name': 'synthetic_easy_1', 'base': 'binary_classification_synthetic', 'seed': 1},
        {'name': 'synthetic_easy_2', 'base': 'binary_classification_synthetic', 'seed': 2},
        {'name': 'synthetic_easy_3', 'base': 'binary_classification_synthetic', 'seed': 3},
        {'name': 'synthetic_easy_4', 'base': 'binary_classification_synthetic', 'seed': 4},
        {'name': 'synthetic_easy_5', 'base': 'binary_classification_synthetic', 'seed': 5},
        {'name': 'synthetic_easy_6', 'base': 'binary_classification_synthetic', 'seed': 6},
        {'name': 'synthetic_easy_7', 'base': 'binary_classification_synthetic', 'seed': 7},
        {'name': 'synthetic_easy_8', 'base': 'binary_classification_synthetic', 'seed': 8},

        # Noise baselines (cannot be learned)
        {'name': 'random_labels_1', 'base': 'binary_random_labels', 'seed': 1},
        {'name': 'random_labels_2', 'base': 'binary_random_labels', 'seed': 2},
    ]

    return datasets


def main():
    parser = argparse.ArgumentParser(
        description='10-dataset manifold intersection with same architecture'
    )
    parser.add_argument('--models-per-dataset', type=int, default=50,
                       help='Models per dataset (default: 50)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Training epochs (default: 100)')
    parser.add_argument('--save-dir', type=str,
                       default='experiments/current/10_dataset_intersection',
                       help='Save directory')

    args = parser.parse_args()

    print("=" * 80)
    print(" 10-DATASET MANIFOLD INTERSECTION")
    print(" Testing Universal Subspace Hypothesis")
    print("=" * 80)

    datasets = get_10_datasets()

    print(f"\nConfiguration:")
    print(f"  Datasets: {len(datasets)}")
    for i, ds in enumerate(datasets):
        print(f"    {i+1}. {ds['name']:25s} ({ds['base']})")
    print(f"  Models per dataset: {args.models_per_dataset}")
    print(f"  Total models: {len(datasets) * args.models_per_dataset}")
    print(f"  Epochs per model: {args.epochs}")
    print(f"  Architecture: [16, 16]")
    print(f"  Target accuracy: >90%")

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Configuration
    config = {
        'n_datasets': len(datasets),
        'datasets': [ds['name'] for ds in datasets],
        'models_per_dataset': args.models_per_dataset,
        'epochs': args.epochs,
        'architecture': [16, 16],
        'started': datetime.now().isoformat()
    }

    with open(f'{args.save_dir}/config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # Progress tracker
    progress = ExperimentProgress(len(datasets), args.models_per_dataset)

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
            dataset_info['base']
        )

        # Create trainer
        trainer = Trainer(
            hidden_dims=[16, 16],
            learning_rate=0.001,
            epochs=args.epochs,
            patience=20
        )

        # Train models
        dataset_weights = []
        dataset_accuracies = []

        for model_idx in range(args.models_per_dataset):
            # Train model
            final_weights, train_stats = trainer.train_single_model(
                train_loader, test_loader, ds_metadata
            )

            # Update progress
            progress.update_model(
                model_idx,
                train_stats['best_test_loss'],
                train_stats.get('best_test_accuracy')
            )

            # Store weights and accuracy
            dataset_weights.append(final_weights)
            if 'best_test_accuracy' in train_stats:
                dataset_accuracies.append(train_stats['best_test_accuracy'])

        # Convert to array and estimate dimension
        weights_array = np.array(dataset_weights)
        mean_accuracy = np.mean(dataset_accuracies) if dataset_accuracies else None

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
            'variance_ratios': var_ratios,
            'mean_accuracy': mean_accuracy
        }

        all_manifolds.append(manifold_data)

        # Save individual dataset results
        np.save(
            f'{args.save_dir}/weights_{dataset_info["name"]}.npy',
            weights_array
        )

        dataset_metadata.append({
            'dataset_idx': dataset_idx,
            'dataset_name': dataset_info['name'],
            'base_task': dataset_info['base'],
            'dim_95': dim_95,
            'effective_dim': effective_dim,
            'mean_accuracy': mean_accuracy
        })

        # Cleanup
        del train_loader, test_loader
        DatasetManager.cleanup()

        progress.finish_dataset(effective_dim, mean_accuracy if mean_accuracy else 0)

    print(f'\n{"="*80}')
    print(f' PHASE 2: MULTI-METHOD INTERSECTION ANALYSIS')
    print(f'{"="*80}')

    # Analyze individual dimensions
    dims = [m['effective_dim'] for m in all_manifolds]
    accs = [m['mean_accuracy'] for m in all_manifolds if m['mean_accuracy'] is not None]

    print(f"\nIndividual manifold dimensions:")
    print(f"  Mean: {np.mean(dims):.1f}D")
    print(f"  Std: {np.std(dims):.1f}D")
    print(f"  Min: {np.min(dims):.1f}D")
    print(f"  Max: {np.max(dims):.1f}D")

    print(f"\nAccuracy summary:")
    print(f"  Mean: {np.mean(accs):.1f}%")
    print(f"  Min: {np.min(accs):.1f}%")
    print(f"  Max: {np.max(accs):.1f}%")

    # Run multi-method intersection analysis
    from src.intersection_methods import IntersectionAnalyzer

    weights_list = [m['weights'] for m in all_manifolds]
    dims_list = [m['dim_95'] for m in all_manifolds]

    analyzer = IntersectionAnalyzer(
        manifolds=weights_list,
        manifold_dims=dims_list,
        variance_threshold=0.95
    )

    intersection_results = analyzer.analyze_all_methods()

    # Save results
    results = {
        'config': config,
        'individual_dimensions': {
            'mean': float(np.mean(dims)),
            'std': float(np.std(dims)),
            'min': float(np.min(dims)),
            'max': float(np.max(dims)),
            'all': [float(d) for d in dims]
        },
        'accuracies': {
            'mean': float(np.mean(accs)) if accs else None,
            'min': float(np.min(accs)) if accs else None,
            'max': float(np.max(accs)) if accs else None,
        },
        'intersection_methods': {
            method: {k: float(v) if isinstance(v, (np.floating, np.integer))
                     else (v.tolist() if isinstance(v, np.ndarray) else v)
                     for k, v in result.items() if k != 'method'}
            for method, result in intersection_results.items()
            if result is not None
        },
        'dataset_metadata': dataset_metadata,
        'completed': datetime.now().isoformat()
    }

    print(f"\nSaving results...")
    with open(f'{args.save_dir}/results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f'\n{"="*80}')
    print(f' EXPERIMENT COMPLETE')
    print(f'{"="*80}')

    total_time = (time.time() - progress.start_time) / 60
    print(f"\nTotal time: {total_time:.1f} minutes")
    print(f"Results saved to: {args.save_dir}/")

    # Summary
    print(f"\nKey findings:")
    for method, result in intersection_results.items():
        if result and 'intersection_dim' in result:
            print(f"  {method}: {result['intersection_dim']:.1f}D intersection")

    print(f'\n{"="*80}')


if __name__ == '__main__':
    main()
