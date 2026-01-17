#!/usr/bin/env python3
"""
Optimization Dynamics Investigation: Signal vs Noise

This experiment tests whether the Universal Subspace is:
1. Task-dependent (models learning different tasks go to different places)
2. Dynamics-dependent (optimization process determines location, not task)

Experiments:
- Train models on clean data (learning signal)
- Train models on random labels (learning pure noise)
- Compare their locations in weight space

If they cluster together → manifold is about optimization dynamics
If they separate → task semantics matter
"""
import numpy as np
import argparse
import os

from src.trainer_with_trajectories import TrajectoryTrainer
from src.geometry_analysis import GeometricAnalyzer
from src.visualization import Visualizer


def main():
    parser = argparse.ArgumentParser(
        description='Dynamics Investigation: Signal vs Noise'
    )
    parser.add_argument('--hidden-dims', type=int, nargs='+', default=[16, 16],
                       help='Hidden layer dimensions (default: [16, 16])')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--epochs', type=int, default=150,
                       help='Max training epochs (default: 150)')
    parser.add_argument('--patience', type=int, default=20,
                       help='Early stopping patience (default: 20)')
    parser.add_argument('--track-every', type=int, default=10,
                       help='Save weight snapshot every N epochs (default: 10)')
    parser.add_argument('--save-dir', type=str, default='results_dynamics',
                       help='Directory to save results (default: results_dynamics)')
    parser.add_argument('--n-replicates', type=int, default=5,
                       help='Number of replicate models per task (default: 5)')

    args = parser.parse_args()

    print("=" * 80)
    print(" OPTIMIZATION DYNAMICS INVESTIGATION")
    print(" Signal vs Noise: Does the task matter?")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Hidden dimensions: {args.hidden_dims}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Max epochs: {args.epochs}")
    print(f"  Tracking interval: every {args.track_every} epochs")
    print(f"  Replicates per task: {args.n_replicates}")

    # Define experimental datasets
    # IMPORTANT: All datasets must have IDENTICAL input/output dims for geometric analysis
    # Using only binary_classification_synthetic (10 features, 1 output) and its noise version
    signal_datasets = [
        'binary_classification_synthetic',    # Clean binary task (10 input, 1 output)
    ]

    noise_datasets = [
        'binary_random_labels',               # Same dims, random labels (10 input, 1 output)
    ]

    print(f"\nSignal datasets (learning real patterns):")
    for ds in signal_datasets:
        print(f"  - {ds}")

    print(f"\nNoise datasets (learning pure noise):")
    for ds in noise_datasets:
        print(f"  - {ds}")

    # Create trainer
    trainer = TrajectoryTrainer(
        hidden_dims=args.hidden_dims,
        learning_rate=args.lr,
        epochs=args.epochs,
        patience=args.patience,
        track_every=args.track_every
    )

    # PHASE 1: Train and collect trajectories
    print("\n" + "=" * 80)
    print(" PHASE 1: TRAINING WITH TRAJECTORY TRACKING")
    print("=" * 80)

    # Train on signal datasets (multiple replicates)
    all_datasets = []
    for _ in range(args.n_replicates):
        all_datasets.extend(signal_datasets)
        all_datasets.extend(noise_datasets)

    trajectories, metadata_list = trainer.train_on_datasets(
        dataset_names=all_datasets,
        save_dir=args.save_dir
    )

    # PHASE 2: Analyze final convergence points
    print("\n" + "=" * 80)
    print(" PHASE 2: CONVERGENCE ANALYSIS")
    print("=" * 80)

    # Extract final weights (end of each trajectory)
    final_weights = []
    for dataset_name, traj in trajectories.items():
        final_weights.append(traj[-1])  # Last snapshot = converged weights

    weight_matrix = np.array(final_weights)
    print(f"\nWeight matrix shape: {weight_matrix.shape}")

    # Geometric analysis
    try:
        analyzer = GeometricAnalyzer(weight_matrix, metadata_list)
        results = analyzer.full_analysis()

        # Save analysis
        np.savez(os.path.join(args.save_dir, 'convergence_analysis.npz'),
                 pca_transformed=results['pca']['transformed'],
                 pca_variance=results['pca']['explained_variance_ratio'],
                 umap_2d=results['umap_2d']['embedding'],
                 umap_3d=results['umap_3d']['embedding'],
                 cluster_labels=results['clustering']['labels'])

        # PHASE 3: Visualize - Signal vs Noise
        print("\n" + "=" * 80)
        print(" PHASE 3: VISUALIZATION")
        print("=" * 80)

        # Create custom visualization highlighting signal vs noise
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Color by signal vs noise
        colors = []
        labels_list = []
        for meta in metadata_list:
            if meta.get('is_noise', False):
                colors.append('red')
                labels_list.append('Noise')
            else:
                colors.append('blue')
                labels_list.append('Signal')

        # Plot 1: PCA 2D
        pca_data = results['pca']['transformed']
        for color, label in zip(['blue', 'red'], ['Signal', 'Noise']):
            mask = np.array(colors) == color
            ax1.scatter(pca_data[mask, 0], pca_data[mask, 1],
                       c=color, label=label, alpha=0.6, s=100)
        ax1.set_xlabel(f"PC1 ({results['pca']['explained_variance_ratio'][0]:.1%})")
        ax1.set_ylabel(f"PC2 ({results['pca']['explained_variance_ratio'][1]:.1%})")
        ax1.set_title('PCA: Signal vs Noise Models')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: UMAP 2D
        umap_data = results['umap_2d']['embedding']
        for color, label in zip(['blue', 'red'], ['Signal', 'Noise']):
            mask = np.array(colors) == color
            ax2.scatter(umap_data[mask, 0], umap_data[mask, 1],
                       c=color, label=label, alpha=0.6, s=100)
        ax2.set_xlabel('UMAP 1')
        ax2.set_ylabel('UMAP 2')
        ax2.set_title('UMAP: Signal vs Noise Models')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{args.save_dir}/signal_vs_noise.png", dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved: {args.save_dir}/signal_vs_noise.png")

        # Create standard visualizations
        visualizer = Visualizer(results, metadata_list,
                               save_dir=os.path.join(args.save_dir, 'figures'))
        visualizer.create_all_plots()

    except ValueError as e:
        print(f"\n⚠ Analysis error: {e}")
        print("This may be due to different parameter counts. Check that all models have same architecture.")

    # PHASE 4: Trajectory Analysis
    print("\n" + "=" * 80)
    print(" PHASE 4: TRAJECTORY ANALYSIS")
    print("=" * 80)

    # Analyze convergence paths
    print("\nAnalyzing optimization trajectories...")

    # Calculate trajectory lengths (distance traveled in weight space)
    trajectory_lengths = {}
    for dataset_name, traj in trajectories.items():
        total_distance = 0
        for i in range(len(traj) - 1):
            dist = np.linalg.norm(traj[i+1] - traj[i])
            total_distance += dist
        trajectory_lengths[dataset_name] = total_distance

    # Report
    print("\nTrajectory lengths (total distance traveled in weight space):")
    signal_lengths = [trajectory_lengths[ds] for ds in signal_datasets if ds in trajectory_lengths]
    noise_lengths = [trajectory_lengths[ds] for ds in noise_datasets if ds in trajectory_lengths]

    if signal_lengths:
        print(f"  Signal tasks: {np.mean(signal_lengths):.2f} ± {np.std(signal_lengths):.2f}")
    if noise_lengths:
        print(f"  Noise tasks:  {np.mean(noise_lengths):.2f} ± {np.std(noise_lengths):.2f}")

    # Save summary report
    with open(os.path.join(args.save_dir, 'dynamics_report.txt'), 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(" OPTIMIZATION DYNAMICS INVESTIGATION - SUMMARY\n")
        f.write("=" * 80 + "\n\n")

        f.write("HYPOTHESIS TEST:\n")
        f.write("  Does the task (signal) matter, or only optimization dynamics?\n\n")

        f.write("EXPERIMENTAL SETUP:\n")
        f.write(f"  Signal tasks: {len(signal_datasets)} datasets x {args.n_replicates} replicates\n")
        f.write(f"  Noise tasks: {len(noise_datasets)} datasets x {args.n_replicates} replicates\n")
        f.write(f"  Architecture: {args.hidden_dims}\n\n")

        if 'results' in locals():
            f.write("CONVERGENCE GEOMETRY:\n")
            f.write(f"  PCA effective dimension: {results['pca']['effective_dim_95']}\n")
            f.write(f"  Intrinsic dimension: {results['intrinsic_dim']['intrinsic_dimension_mean']:.2f}\n")
            f.write(f"  Fractal dimension: {results['fractal_correlation']['correlation_dimension']:.2f}\n")
            f.write(f"  Number of clusters: {results['clustering']['n_clusters']}\n\n")

            f.write("INTERPRETATION:\n")
            # Check if signal and noise models cluster together or separately
            f.write("  See signal_vs_noise.png for visual analysis\n")
            f.write("  Blue = Signal (learning real patterns)\n")
            f.write("  Red = Noise (learning random labels)\n\n")

            f.write("  If they CLUSTER TOGETHER:\n")
            f.write("    → Manifold is determined by OPTIMIZATION DYNAMICS\n")
            f.write("    → Task semantics don't matter\n")
            f.write("    → Universal subspace is a property of the loss landscape\n\n")

            f.write("  If they SEPARATE:\n")
            f.write("    → Task semantics matter\n")
            f.write("    → Different tasks converge to different regions\n")
            f.write("    → Universal subspace is task-specific\n\n")

        f.write("=" * 80 + "\n")

    print(f"\nSummary report saved to: {args.save_dir}/dynamics_report.txt")

    print("\n" + "=" * 80)
    print(" EXPERIMENT COMPLETE")
    print("=" * 80)
    print(f"\nResults saved in: {args.save_dir}/")
    print("\nKey files:")
    print(f"  - trajectories.npz: Weight evolution during training")
    print(f"  - signal_vs_noise.png: Visualization of main result")
    print(f"  - dynamics_report.txt: Interpretation and conclusions")
    print(f"  - figures/: All geometric analysis plots")


if __name__ == '__main__':
    main()
