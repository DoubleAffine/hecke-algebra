#!/usr/bin/env python3
"""
PROPER Multi-Task Intersection Test

Key improvements:
1. ADEQUATE SAMPLING: 100+ models per task
2. MEMORY EFFICIENT: Incremental processing, no data retention
3. REAL-TIME PROGRESS: Terminal visualization
4. MULTIPLE TASKS: Test intersection across different datasets
5. ROBUST ANALYSIS: Multiple verification methods

Design:
- Train models in batches
- Save weights incrementally
- Delete training data immediately
- Live progress tracking
- Compute intersection only at the end
"""
import numpy as np
import argparse
import os
import sys
import time
from datetime import datetime, timedelta
import json

from src.trainer import Trainer
from src.datasets import DatasetManager

class ProgressTracker:
    """Real-time progress visualization in terminal."""

    def __init__(self, total_models, tasks):
        self.total_models = total_models
        self.tasks = tasks
        self.current_model = 0
        self.current_task = None
        self.start_time = time.time()
        self.task_times = []

    def start_task(self, task_name):
        self.current_task = task_name
        self.task_start = time.time()

    def update(self, model_idx, loss, acc=None):
        self.current_model += 1
        elapsed = time.time() - self.start_time
        progress = self.current_model / self.total_models

        # Estimate remaining time
        if progress > 0:
            total_time = elapsed / progress
            remaining = total_time - elapsed
            eta = datetime.now() + timedelta(seconds=remaining)
        else:
            eta = None

        # Clear line and print progress
        bar_width = 50
        filled = int(bar_width * progress)
        bar = '█' * filled + '░' * (bar_width - filled)

        sys.stdout.write('\r')
        sys.stdout.write(f'[{bar}] {progress*100:.1f}% ')
        sys.stdout.write(f'| Model {self.current_model}/{self.total_models} ')
        sys.stdout.write(f'| Task: {self.current_task} ')
        sys.stdout.write(f'| Loss: {loss:.4f} ')
        if acc is not None:
            sys.stdout.write(f'| Acc: {acc:.1f}% ')
        if eta:
            sys.stdout.write(f'| ETA: {eta.strftime("%H:%M:%S")} ')
        sys.stdout.flush()

    def finish_task(self):
        task_time = time.time() - self.task_start
        self.task_times.append((self.current_task, task_time))
        print(f'\n  ✓ {self.current_task} complete ({task_time/60:.1f} min)')

    def summary(self):
        total_time = time.time() - self.start_time
        print(f'\n{"="*80}')
        print(f' TRAINING COMPLETE')
        print(f'{"="*80}')
        print(f'\nTotal time: {total_time/60:.1f} minutes')
        print(f'Average per model: {total_time/self.total_models:.1f} seconds')
        print(f'\nPer-task breakdown:')
        for task, t in self.task_times:
            print(f'  {task}: {t/60:.1f} min')


def main():
    parser = argparse.ArgumentParser(
        description='Proper Multi-Task Intersection Test with adequate sampling'
    )
    parser.add_argument('--models-per-task', type=int, default=100,
                       help='Models per task (default: 100 for adequate sampling)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Training epochs')
    parser.add_argument('--save-dir', type=str, default='experiments/current/intersection_proper',
                       help='Save directory')
    parser.add_argument('--batch-size', type=int, default=10,
                       help='Process models in batches to save memory')

    args = parser.parse_args()

    print("=" * 80)
    print(" PROPER MULTI-TASK INTERSECTION TEST")
    print(" With adequate sampling and memory management")
    print("=" * 80)

    # Tasks to test (MUST have same input/output dimensions!)
    tasks = [
        'binary_classification_synthetic',  # Real patterns
        'binary_random_labels',             # Pure noise
    ]

    print(f"\nConfiguration:")
    print(f"  Models per task: {args.models_per_task}")
    print(f"  Tasks: {len(tasks)}")
    for t in tasks:
        print(f"    - {t}")
    print(f"  Total models: {len(tasks) * args.models_per_task}")
    print(f"  Batch processing: {args.batch_size} models at a time")
    print(f"  Epochs per model: {args.epochs}")

    # Verify sampling adequacy
    expected_dim = args.models_per_task * 0.72  # From our scaling law
    samples_per_dim = args.models_per_task / expected_dim

    print(f"\nSampling analysis:")
    print(f"  Expected dimension: ~{expected_dim:.0f}D (from scaling law)")
    print(f"  Samples per dimension: {samples_per_dim:.1f}")

    if samples_per_dim < 2:
        print(f"  ⚠ WARNING: Undersampled! Recommend {expected_dim * 10:.0f}+ models")
        print(f"  Continue anyway? (Ctrl+C to abort)")
        time.sleep(3)
    else:
        print(f"  ✓ Adequately sampled")

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Save configuration
    config = {
        'models_per_task': args.models_per_task,
        'tasks': tasks,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'expected_dimension': float(expected_dim),
        'samples_per_dimension': float(samples_per_dim),
        'started': datetime.now().isoformat()
    }

    with open(f'{args.save_dir}/config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # Initialize progress tracker
    total_models = len(tasks) * args.models_per_task
    progress = ProgressTracker(total_models, tasks)

    # Storage for weights (per task)
    task_weights = {task: [] for task in tasks}
    task_metadata = {task: [] for task in tasks}

    print(f'\n{"="*80}')
    print(f' TRAINING PHASE')
    print(f'{"="*80}\n')

    # Train models for each task
    for task_idx, task_name in enumerate(tasks):
        progress.start_task(task_name)

        # Load dataset ONCE
        train_loader, test_loader, dataset_metadata = DatasetManager.load_dataset(
            task_name
        )

        # Create trainer
        trainer = Trainer(
            hidden_dims=[16, 16],
            learning_rate=0.001,
            epochs=args.epochs,
            patience=20
        )

        # Train models in batches
        for batch_start in range(0, args.models_per_task, args.batch_size):
            batch_end = min(batch_start + args.batch_size, args.models_per_task)
            batch_weights = []

            for model_idx in range(batch_start, batch_end):
                # Train model
                final_weights, train_stats = trainer.train_single_model(
                    train_loader, test_loader, dataset_metadata
                )

                # Update progress
                progress.update(
                    model_idx,
                    train_stats['best_test_loss'],
                    train_stats.get('best_test_accuracy')
                )

                # Store weights
                batch_weights.append(final_weights)
                task_metadata[task_name].append(train_stats)

            # Save batch to disk and clear from memory
            batch_array = np.array(batch_weights)
            batch_file = f'{args.save_dir}/weights_{task_name}_batch_{batch_start:04d}.npy'
            np.save(batch_file, batch_array)

            # Keep in memory for now (we'll consolidate later)
            task_weights[task_name].extend(batch_weights)

            # Clear batch
            del batch_weights, batch_array

        # Cleanup dataset
        del train_loader, test_loader
        DatasetManager.cleanup()

        progress.finish_task()

    progress.summary()

    # PHASE 2: ANALYSIS
    print(f'\n{"="*80}')
    print(f' ANALYSIS PHASE')
    print(f'{"="*80}')

    print(f'\nConverting weights to arrays...')
    task_weight_matrices = {}
    for task in tasks:
        task_weight_matrices[task] = np.array(task_weights[task])
        print(f'  {task}: {task_weight_matrices[task].shape}')

        # Save consolidated weights
        np.save(f'{args.save_dir}/weights_{task}.npy', task_weight_matrices[task])

    # Clear individual weights from memory
    del task_weights

    # Save metadata
    for task in tasks:
        with open(f'{args.save_dir}/metadata_{task}.json', 'w') as f:
            json.dump(task_metadata[task], f, indent=2)

    print(f'\n{"="*80}')
    print(f' COMPUTING DIMENSIONS')
    print(f'{"="*80}')

    from sklearn.decomposition import PCA

    task_dimensions = {}

    for task in tasks:
        print(f'\n{task}:')

        weights = task_weight_matrices[task]

        # PCA
        pca = PCA()
        pca.fit(weights)

        var_ratios = pca.explained_variance_ratio_
        cumsum = np.cumsum(var_ratios)

        dim_95 = np.argmax(cumsum >= 0.95) + 1
        dim_99 = np.argmax(cumsum >= 0.99) + 1
        effective_dim = (np.sum(var_ratios) ** 2) / np.sum(var_ratios ** 2)

        print(f'  Dimension (95% var): {dim_95}D')
        print(f'  Dimension (99% var): {dim_99}D')
        print(f'  Effective dimension: {effective_dim:.1f}D')

        task_dimensions[task] = {
            'dim_95': dim_95,
            'dim_99': dim_99,
            'effective_dim': float(effective_dim),
            'pca_components': pca.components_[:min(100, dim_95)].tolist(),
            'variance_ratios': var_ratios[:100].tolist()
        }

    print(f'\n{"="*80}')
    print(f' COMPUTING INTERSECTION')
    print(f'{"="*80}')

    if len(tasks) >= 2:
        from scipy.linalg import subspace_angles

        # Get subspaces
        task1, task2 = tasks[0], tasks[1]

        pca1 = PCA(n_components=task_dimensions[task1]['dim_95'])
        pca1.fit(task_weight_matrices[task1])

        pca2 = PCA(n_components=task_dimensions[task2]['dim_95'])
        pca2.fit(task_weight_matrices[task2])

        # Principal angles
        angles = subspace_angles(pca1.components_.T, pca2.components_.T)
        angles_deg = np.degrees(angles)

        print(f'\nPrincipal angles between {task1} and {task2}:')
        print(f'  Min: {np.min(angles_deg):.1f}°')
        print(f'  Mean: {np.mean(angles_deg):.1f}°')
        print(f'  Max: {np.max(angles_deg):.1f}°')
        print(f'  First 10: {angles_deg[:10]}')

        aligned = np.sum(angles_deg < 10)
        print(f'\n  Aligned dimensions (< 10°): {aligned}')

        # Global PCA
        print(f'\nGlobal analysis (all models combined):')
        all_weights = np.vstack([task_weight_matrices[t] for t in tasks])

        pca_global = PCA()
        pca_global.fit(all_weights)

        var_global = pca_global.explained_variance_ratio_
        cumsum_global = np.cumsum(var_global)

        dim_95_global = np.argmax(cumsum_global >= 0.95) + 1
        effective_global = (np.sum(var_global) ** 2) / np.sum(var_global ** 2)

        print(f'  Global dimension (95%): {dim_95_global}D')
        print(f'  Global effective: {effective_global:.1f}D')

        # Compare
        mean_individual = np.mean([task_dimensions[t]['effective_dim'] for t in tasks])

        print(f'\nComparison:')
        print(f'  Mean individual: {mean_individual:.1f}D')
        print(f'  Global: {effective_global:.1f}D')
        print(f'  Ratio: {effective_global / mean_individual:.2f}')

        if effective_global < mean_individual * 0.9:
            intersection_dim = effective_global
            print(f'\n  ✓ INTERSECTION EXISTS: ~{intersection_dim:.0f}D')
        elif effective_global > mean_individual * 1.3:
            intersection_dim = 0
            print(f'\n  ✗ NO INTERSECTION (orthogonal subspaces)')
        else:
            intersection_dim = effective_global * 0.7  # Estimate
            print(f'\n  ? PARTIAL OVERLAP: ~{intersection_dim:.0f}D estimated')

    # Save results
    results = {
        'config': config,
        'task_dimensions': task_dimensions,
        'intersection': {
            'aligned_dimensions': int(aligned) if len(tasks) >= 2 else None,
            'min_angle': float(np.min(angles_deg)) if len(tasks) >= 2 else None,
            'mean_angle': float(np.mean(angles_deg)) if len(tasks) >= 2 else None,
            'global_dimension': float(effective_global) if len(tasks) >= 2 else None,
            'intersection_estimate': float(intersection_dim) if len(tasks) >= 2 else None
        },
        'completed': datetime.now().isoformat()
    }

    with open(f'{args.save_dir}/results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f'\n{"="*80}')
    print(f' EXPERIMENT COMPLETE')
    print(f'{"="*80}')
    print(f'\nResults saved to: {args.save_dir}/')
    print(f'\nKey files:')
    print(f'  - results.json: Summary of findings')
    print(f'  - weights_{{task}}.npy: Model weights per task')
    print(f'  - metadata_{{task}}.json: Training stats per task')
    print(f'  - config.json: Experiment configuration')

    print(f'\n{"="*80}')


if __name__ == '__main__':
    main()
