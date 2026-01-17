#!/usr/bin/env python3
"""
Main script to run the Universal Subspace Hypothesis investigation.

This script:
1. Trains small neural networks on diverse datasets
2. Extracts weight vectors from each trained model
3. Analyzes the geometry of the weight space
4. Tests whether weights cluster around a low-dimensional manifold
5. Investigates potential fractal structure
"""
import numpy as np
import argparse
import os
import json

from src.datasets import ALL_DATASETS, DatasetManager
from src.trainer import Trainer
from src.geometry_analysis import GeometricAnalyzer
from src.visualization import Visualizer


def main():
    parser = argparse.ArgumentParser(
        description='Universal Subspace Hypothesis Investigation'
    )
    parser.add_argument('--hidden-dims', type=int, nargs='+', default=[16, 16],
                       help='Hidden layer dimensions (default: [16, 16])')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Max training epochs (default: 100)')
    parser.add_argument('--patience', type=int, default=15,
                       help='Early stopping patience (default: 15)')
    parser.add_argument('--datasets', type=str, nargs='+', default=None,
                       help='Specific datasets to use (default: all)')
    parser.add_argument('--save-dir', type=str, default='results',
                       help='Directory to save results (default: results)')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip training and load existing results')
    parser.add_argument('--skip-analysis', action='store_true',
                       help='Only train, skip geometric analysis')

    args = parser.parse_args()

    # Setup
    print("=" * 80)
    print(" UNIVERSAL SUBSPACE HYPOTHESIS INVESTIGATION")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Hidden dimensions: {args.hidden_dims}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Max epochs: {args.epochs}")
    print(f"  Early stopping patience: {args.patience}")
    print(f"  Save directory: {args.save_dir}")

    # Determine datasets to use
    if args.datasets is None:
        datasets = ALL_DATASETS
    else:
        datasets = args.datasets

    print(f"  Datasets: {len(datasets)} total")
    for ds in datasets:
        print(f"    - {ds}")

    # PHASE 1: Training
    if not args.skip_training:
        print("\n" + "=" * 80)
        print(" PHASE 1: TRAINING MODELS")
        print("=" * 80)

        trainer = Trainer(
            hidden_dims=args.hidden_dims,
            learning_rate=args.lr,
            epochs=args.epochs,
            patience=args.patience
        )

        weight_matrix, metadata_list = trainer.train_on_all_datasets(
            dataset_names=datasets,
            save_dir=args.save_dir
        )

        print(f"\nTraining complete. Results saved to {args.save_dir}/")

    else:
        print("\nSkipping training, loading existing results...")
        weight_matrix = np.load(os.path.join(args.save_dir, 'weight_matrix.npy'))
        with open(os.path.join(args.save_dir, 'metadata.json'), 'r') as f:
            metadata_list = json.load(f)

        print(f"Loaded weight matrix: {weight_matrix.shape}")

    # PHASE 2: Geometric Analysis
    if not args.skip_analysis:
        print("\n" + "=" * 80)
        print(" PHASE 2: GEOMETRIC ANALYSIS")
        print("=" * 80)

        analyzer = GeometricAnalyzer(weight_matrix, metadata_list)
        results = analyzer.full_analysis()

        # Save analysis results
        analysis_file = os.path.join(args.save_dir, 'analysis_results.npz')
        np.savez(analysis_file,
                 pca_transformed=results['pca']['transformed'],
                 pca_variance=results['pca']['explained_variance_ratio'],
                 umap_2d=results['umap_2d']['embedding'],
                 umap_3d=results['umap_3d']['embedding'],
                 cluster_labels=results['clustering']['labels'],
                 fractal_dim_bc=results['fractal_boxcount']['fractal_dimension'],
                 fractal_dim_corr=results['fractal_correlation']['correlation_dimension'],
                 intrinsic_dim_mean=results['intrinsic_dim']['intrinsic_dimension_mean'])

        print(f"\nAnalysis results saved to {analysis_file}")

        # PHASE 3: Visualization
        print("\n" + "=" * 80)
        print(" PHASE 3: VISUALIZATION")
        print("=" * 80)

        visualizer = Visualizer(results, metadata_list,
                               save_dir=os.path.join(args.save_dir, 'figures'))
        visualizer.create_all_plots()

        # Save summary report
        summary_file = os.path.join(args.save_dir, 'summary_report.txt')
        with open(summary_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write(" UNIVERSAL SUBSPACE HYPOTHESIS - EXPERIMENT SUMMARY\n")
            f.write("=" * 80 + "\n\n")

            f.write("ARCHITECTURE:\n")
            f.write(f"  Hidden dimensions: {args.hidden_dims}\n")
            f.write(f"  Total parameters: {weight_matrix.shape[1]}\n")
            f.write(f"  Models trained: {weight_matrix.shape[0]}\n\n")

            f.write("DIMENSIONALITY ANALYSIS:\n")
            f.write(f"  PCA effective dimension (95% var): {results['pca']['effective_dim_95']}\n")
            f.write(f"  Intrinsic dimension (MLE): {results['intrinsic_dim']['intrinsic_dimension_mean']:.2f} ± {results['intrinsic_dim']['intrinsic_dimension_std']:.2f}\n")
            f.write(f"  Fractal dimension (box-counting): {results['fractal_boxcount']['fractal_dimension']:.2f} (R²={results['fractal_boxcount']['r_squared']:.3f})\n")
            f.write(f"  Correlation dimension: {results['fractal_correlation']['correlation_dimension']:.2f} (R²={results['fractal_correlation']['r_squared']:.3f})\n\n")

            f.write("CLUSTERING:\n")
            f.write(f"  Number of clusters: {results['clustering']['n_clusters']}\n")
            if results['clustering']['silhouette_score'] is not None:
                f.write(f"  Silhouette score: {results['clustering']['silhouette_score']:.3f}\n\n")

            f.write("INTERPRETATION:\n")
            intrinsic = results['intrinsic_dim']['intrinsic_dimension_mean']
            fractal = results['fractal_correlation']['correlation_dimension']
            pca_dim = results['pca']['effective_dim_95']

            if pca_dim < 10:
                f.write(f"  ✓ Very low effective dimensionality ({pca_dim}) - STRONG support for universal subspace hypothesis\n")
            elif pca_dim < 30:
                f.write(f"  ~ Moderate dimensionality reduction ({pca_dim}) - PARTIAL support for hypothesis\n")
            else:
                f.write(f"  ✗ High dimensionality ({pca_dim}) - WEAK support for universal subspace\n")

            if abs(intrinsic - fractal) < 1.0:
                f.write("  ✓ Fractal and intrinsic dimensions agree - suggests SMOOTH low-dimensional manifold\n")
            else:
                f.write("  ⚠ Dimension estimates differ - possible FRACTAL or IRREGULAR structure\n")
                f.write(f"    This supports the hypothesis that the manifold may have fractal-like properties\n")

            f.write("\n" + "=" * 80 + "\n")

        print(f"\nSummary report saved to {summary_file}")

    print("\n" + "=" * 80)
    print(" EXPERIMENT COMPLETE")
    print("=" * 80)
    print(f"\nAll results saved in: {args.save_dir}/")
    print("\nKey files:")
    print(f"  - weight_matrix.npy: Trained model weights")
    print(f"  - metadata.json: Training statistics")
    print(f"  - analysis_results.npz: Geometric analysis results")
    print(f"  - summary_report.txt: Human-readable summary")
    print(f"  - figures/: All visualizations")


if __name__ == '__main__':
    main()
