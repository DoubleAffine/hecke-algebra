#!/usr/bin/env python3
"""
Large-scale sampling: Train 100 models to properly characterize the manifold.
"""
import numpy as np
import argparse
import os

from src.trainer import Trainer
from src.geometry_analysis import GeometricAnalyzer
from src.visualization import Visualizer


def main():
    parser = argparse.ArgumentParser(
        description='Large-scale manifold sampling'
    )
    parser.add_argument('--n-models', type=int, default=100,
                       help='Number of models to train (default: 100)')
    parser.add_argument('--hidden-dims', type=int, nargs='+', default=[16, 16],
                       help='Hidden layer dimensions')
    parser.add_argument('--dataset', type=str, default='binary_classification_synthetic',
                       help='Dataset to use')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Training epochs per model')
    parser.add_argument('--save-dir', type=str, default='results_large_scale',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print(" LARGE-SCALE MANIFOLD SAMPLING")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Models to train: {args.n_models}")
    print(f"  Architecture: {args.hidden_dims}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Epochs per model: {args.epochs}")
    print(f"\nEstimated time: ~{args.n_models * 0.5 / 60:.1f} - {args.n_models * 1.5 / 60:.1f} hours")
    
    # Create trainer
    trainer = Trainer(
        hidden_dims=args.hidden_dims,
        learning_rate=0.001,
        epochs=args.epochs,
        patience=20
    )
    
    # Train many models (all with different random initializations)
    datasets = [args.dataset] * args.n_models
    
    print(f"\n{'=' * 80}")
    print(f" TRAINING {args.n_models} MODELS")
    print(f"{'=' * 80}\n")
    
    weight_matrix, metadata_list = trainer.train_on_all_datasets(
        dataset_names=datasets,
        save_dir=args.save_dir
    )
    
    print(f"\n{'=' * 80}")
    print(" GEOMETRIC ANALYSIS")
    print(f"{'=' * 80}\n")
    
    # Analyze
    analyzer = GeometricAnalyzer(weight_matrix, metadata_list)
    results = analyzer.full_analysis()
    
    # Save
    np.savez(
        os.path.join(args.save_dir, 'geometry_analysis.npz'),
        weight_matrix=weight_matrix,
        pca_transformed=results['pca']['transformed'],
        pca_variance=results['pca']['explained_variance_ratio'],
        umap_2d=results['umap_2d']['embedding'],
        intrinsic_dim=results['intrinsic_dim']['intrinsic_dimension_mean']
    )
    
    # Visualize
    visualizer = Visualizer(results, metadata_list,
                           save_dir=os.path.join(args.save_dir, 'figures'))
    visualizer.create_all_plots()
    
    # Statistical summary
    print(f"\n{'=' * 80}")
    print(" STATISTICAL SUMMARY")
    print(f"{'=' * 80}")
    
    intrinsic_dim = results['intrinsic_dim']['intrinsic_dimension_mean']
    intrinsic_std = results['intrinsic_dim']['intrinsic_dimension_std']
    pca_dim_95 = results['pca']['effective_dim_95']
    
    print(f"\nSample size: {args.n_models} models")
    print(f"Ambient dimension: {weight_matrix.shape[1]}")
    print(f"\nIntrinsic dimension (MLE): {intrinsic_dim:.2f} ± {intrinsic_std:.2f}")
    print(f"PCA dimension (95% var): {pca_dim_95}")
    print(f"Compression ratio: {weight_matrix.shape[1] / pca_dim_95:.1f}×")
    
    # Confidence assessment
    samples_per_dim = args.n_models / pca_dim_95
    print(f"\nSamples per dimension: {samples_per_dim:.1f}")
    
    if samples_per_dim >= 10:
        print("✓ WELL-SAMPLED: High confidence in dimension estimate")
    elif samples_per_dim >= 5:
        print("~ ADEQUATELY SAMPLED: Moderate confidence")
    else:
        print("⚠ UNDERSAMPLED: Low confidence, need more samples")
    
    print(f"\n{'=' * 80}")
    print(f" RESULTS SAVED TO: {args.save_dir}/")
    print(f"{'=' * 80}")


if __name__ == '__main__':
    main()
