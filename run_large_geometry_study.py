#!/usr/bin/env python3
"""
Large-scale geometry study of the neural network weight manifold.

This experiment trains many models (100+) to properly sample the manifold
and perform sophisticated geometric analysis.
"""
import numpy as np
import argparse
import os

from src.trainer import Trainer
from src.geometry_analysis import GeometricAnalyzer
from src.visualization import Visualizer
from src.model_persistence import WeightStorage


def main():
    parser = argparse.ArgumentParser(
        description='Large-scale geometric analysis of weight manifold'
    )
    parser.add_argument('--hidden-dims', type=int, nargs='+', default=[16, 16],
                       help='Hidden layer dimensions (default: [16, 16])')
    parser.add_argument('--n-models', type=int, default=100,
                       help='Number of models to train (default: 100)')
    parser.add_argument('--dataset', type=str, default='binary_classification_synthetic',
                       help='Dataset to use (all models same architecture)')
    parser.add_argument('--save-dir', type=str, default='results_geometry',
                       help='Directory to save results')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Training epochs per model')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print(" LARGE-SCALE WEIGHT MANIFOLD GEOMETRY STUDY")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Dataset: {args.dataset}")
    print(f"  Architecture: {args.hidden_dims}")
    print(f"  Number of models: {args.n_models}")
    print(f"  Epochs per model: {args.epochs}")
    print(f"  Purpose: Sample the weight manifold to study its geometry")
    
    # Create trainer
    trainer = Trainer(
        hidden_dims=args.hidden_dims,
        learning_rate=0.001,
        epochs=args.epochs,
        patience=20
    )
    
    # Create weight storage
    storage = WeightStorage(storage_dir=args.save_dir)
    
    print(f"\n{'=' * 80}")
    print(f" TRAINING {args.n_models} MODELS")
    print(f"{'=' * 80}\n")
    
    # Train many models with different initializations
    datasets = [args.dataset] * args.n_models
    
    weight_matrix, metadata_list = trainer.train_on_datasets(
        dataset_names=datasets,
        save_dir=args.save_dir
    )
    
    print(f"\n{'=' * 80}")
    print(" GEOMETRIC ANALYSIS")
    print(f"{'=' * 80}\n")
    
    # Perform geometric analysis
    analyzer = GeometricAnalyzer(weight_matrix, metadata_list)
    results = analyzer.full_analysis()
    
    # Save results
    np.savez(
        os.path.join(args.save_dir, 'geometry_analysis.npz'),
        **{k: v for result in results.values() for k, v in result.items() if isinstance(v, np.ndarray)}
    )
    
    # Visualize
    visualizer = Visualizer(results, metadata_list,
                           save_dir=os.path.join(args.save_dir, 'figures'))
    visualizer.create_all_plots()
    
    # Summary report
    print(f"\n{'=' * 80}")
    print(" SUMMARY")
    print(f"{'=' * 80}")
    print(f"Models trained: {args.n_models}")
    print(f"Total parameters per model: {weight_matrix.shape[1]}")
    print(f"Intrinsic dimension: {results['intrinsic_dim']['intrinsic_dimension_mean']:.2f}")
    print(f"PCA 95% dimension: {results['pca']['effective_dim_95']}")
    print(f"Correlation dimension: {results['fractal_correlation']['correlation_dimension']:.2f}")
    print(f"\nInterpretation:")
    print(f"  - Manifold dimension: ~{int(results['intrinsic_dim']['intrinsic_dimension_mean'])}")
    print(f"  - Compression ratio: {weight_matrix.shape[1] / results['intrinsic_dim']['intrinsic_dimension_mean']:.1f}x")
    print(f"  - The {args.n_models} models live on a ~{int(results['intrinsic_dim']['intrinsic_dimension_mean'])}-D manifold")
    print(f"    embedded in {weight_matrix.shape[1]}-D space")
    
    print(f"\nResults saved to: {args.save_dir}/")


if __name__ == '__main__':
    main()
