#!/usr/bin/env python3
"""
Run Universal Subspace Hypothesis experiment with downloaded datasets.

This version downloads datasets from public repositories:
- UCI Machine Learning Repository
- OpenML
- TorchVision (MNIST, Fashion-MNIST)

Workflow: Download → Train → Save weights → Delete data → Repeat
"""
import numpy as np
import argparse
import os

from src.trainer_with_download import TrainerWithDownload
from src.dataset_downloader import DOWNLOADABLE_DATASETS
from src.geometry_analysis import GeometricAnalyzer
from src.visualization import Visualizer


def main():
    parser = argparse.ArgumentParser(
        description='Universal Subspace Hypothesis - With Downloaded Datasets'
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
                       help='Specific datasets to use (default: recommended subset)')
    parser.add_argument('--list-datasets', action='store_true',
                       help='List all available datasets and exit')
    parser.add_argument('--save-dir', type=str, default='results',
                       help='Directory to save results (default: results)')
    parser.add_argument('--cache-dir', type=str, default='./data_cache',
                       help='Directory for dataset cache (default: ./data_cache)')
    parser.add_argument('--incremental', action='store_true',
                       help='Use incremental storage (for very large experiments)')
    parser.add_argument('--skip-analysis', action='store_true',
                       help='Only train, skip geometric analysis')

    args = parser.parse_args()

    # Create trainer
    trainer = TrainerWithDownload(
        hidden_dims=args.hidden_dims,
        learning_rate=args.lr,
        epochs=args.epochs,
        patience=args.patience,
        cache_dir=args.cache_dir,
        storage_dir=os.path.join(args.save_dir, 'weights'),
        incremental_storage=args.incremental
    )

    # List datasets if requested
    if args.list_datasets:
        trainer.list_available_datasets()
        return

    print("=" * 80)
    print(" UNIVERSAL SUBSPACE HYPOTHESIS - DOWNLOADED DATASETS")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Hidden dimensions: {args.hidden_dims}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Max epochs: {args.epochs}")
    print(f"  Early stopping patience: {args.patience}")
    print(f"  Save directory: {args.save_dir}")
    print(f"  Cache directory: {args.cache_dir}")
    print(f"  Incremental storage: {args.incremental}")

    # Determine datasets to use
    if args.datasets is None:
        # Recommended subset for quick experiments
        datasets = [
            'adult_income',      # UCI, binary classification, ~48K samples
            'bank_marketing',    # UCI, binary classification, ~45K samples
            'iris',              # UCI, multi-class (3), 150 samples
            'mnist_binary',      # TorchVision, binary (0 vs 1), ~13K samples
            'fashion_mnist_5class',  # TorchVision, multi-class (5), ~30K samples
            'openml_credit',     # OpenML, binary classification
            'openml_blood',      # OpenML, binary classification
        ]
        print(f"\nUsing recommended dataset subset ({len(datasets)} datasets)")
    else:
        datasets = args.datasets
        print(f"\nUsing custom dataset selection ({len(datasets)} datasets)")

    # Verify datasets exist
    valid_datasets = []
    for ds in datasets:
        if ds in DOWNLOADABLE_DATASETS:
            valid_datasets.append(ds)
        else:
            print(f"  ⚠ Warning: Unknown dataset '{ds}', skipping...")

    if not valid_datasets:
        print("\n✗ No valid datasets specified. Use --list-datasets to see available options.")
        return

    print("\nDatasets to process:")
    for ds in valid_datasets:
        source = DOWNLOADABLE_DATASETS[ds]['source']
        print(f"  - {ds} ({source})")

    # PHASE 1: Training with downloads
    print("\n" + "=" * 80)
    print(" PHASE 1: DOWNLOAD & TRAIN")
    print("=" * 80)

    weight_matrix, metadata_list = trainer.train_on_downloadable_datasets(
        dataset_names=valid_datasets,
        save_dir=args.save_dir
    )

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

            f.write("DATASETS:\n")
            for meta in metadata_list:
                f.write(f"  - {meta['name']} ({meta['source']}): {meta['task_type']}\n")
            f.write("\n")

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
    if not args.skip_analysis:
        print(f"  - analysis_results.npz: Geometric analysis results")
        print(f"  - summary_report.txt: Human-readable summary")
        print(f"  - figures/: All visualizations")


if __name__ == '__main__':
    main()
