"""
Visualization tools for geometric analysis of weight space.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, List, Optional
import os


class Visualizer:
    """Create visualizations for weight space geometry."""

    def __init__(self, results: Dict, metadata_list: List[Dict] = None,
                 save_dir: str = 'results/figures'):
        """
        Args:
            results: Results dictionary from GeometricAnalyzer
            metadata_list: List of model metadata
            save_dir: Directory to save figures
        """
        self.results = results
        self.metadata_list = metadata_list
        self.save_dir = save_dir

        os.makedirs(save_dir, exist_ok=True)

        # Set style
        sns.set_style('whitegrid')
        plt.rcParams['figure.figsize'] = (10, 6)

    def plot_pca_variance(self):
        """Plot explained variance from PCA."""
        pca_results = self.results['pca']

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Individual variance
        ax1.bar(range(len(pca_results['explained_variance_ratio'])),
                pca_results['explained_variance_ratio'])
        ax1.set_xlabel('Principal Component')
        ax1.set_ylabel('Explained Variance Ratio')
        ax1.set_title('PCA: Explained Variance by Component')
        ax1.grid(True, alpha=0.3)

        # Cumulative variance
        ax2.plot(pca_results['cumulative_variance'], marker='o')
        ax2.axhline(y=0.95, color='r', linestyle='--', label='95% threshold')
        ax2.axvline(x=pca_results['effective_dim_95']-1, color='g',
                   linestyle='--', label=f"{pca_results['effective_dim_95']} components")
        ax2.set_xlabel('Number of Components')
        ax2.set_ylabel('Cumulative Explained Variance')
        ax2.set_title('PCA: Cumulative Explained Variance')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/pca_variance.png", dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved: {self.save_dir}/pca_variance.png")

    def plot_fractal_dimension(self):
        """Plot fractal dimension analysis."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Box-counting
        bc = self.results['fractal_boxcount']
        ax1.scatter(bc['log_eps'], bc['log_counts'], alpha=0.6, s=50)
        ax1.plot(bc['log_eps'],
                bc['fit_slope'] * bc['log_eps'] + bc['fit_intercept'],
                'r--', label=f"Fit: D={bc['fractal_dimension']:.2f}")
        ax1.set_xlabel('log(ε)')
        ax1.set_ylabel('log(N(ε))')
        ax1.set_title(f"Box-Counting Method (R²={bc['r_squared']:.3f})")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Correlation dimension
        cd = self.results['fractal_correlation']
        ax2.scatter(cd['log_radii'], cd['log_counts'], alpha=0.6, s=50)
        ax2.plot(cd['log_radii'],
                cd['fit_slope'] * cd['log_radii'] + cd['fit_intercept'],
                'r--', label=f"Fit: D={cd['correlation_dimension']:.2f}")
        ax2.set_xlabel('log(r)')
        ax2.set_ylabel('log(C(r))')
        ax2.set_title(f"Correlation Dimension (R²={cd['r_squared']:.3f})")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/fractal_dimension.png", dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved: {self.save_dir}/fractal_dimension.png")

    def plot_intrinsic_dimension(self):
        """Plot intrinsic dimension estimates."""
        id_results = self.results['intrinsic_dim']

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.hist(id_results['filtered_estimates'], bins=30, alpha=0.7, edgecolor='black')
        ax.axvline(id_results['intrinsic_dimension_mean'], color='r',
                  linestyle='--', linewidth=2,
                  label=f"Mean: {id_results['intrinsic_dimension_mean']:.2f}")
        ax.axvline(id_results['intrinsic_dimension_median'], color='g',
                  linestyle='--', linewidth=2,
                  label=f"Median: {id_results['intrinsic_dimension_median']:.2f}")

        ax.set_xlabel('Estimated Intrinsic Dimension')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Local Intrinsic Dimension Estimates')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/intrinsic_dimension.png", dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved: {self.save_dir}/intrinsic_dimension.png")

    def plot_umap_2d(self):
        """Plot 2D UMAP embedding."""
        umap_data = self.results['umap_2d']['embedding']

        fig, ax = plt.subplots(figsize=(10, 8))

        if self.metadata_list is not None:
            # Color by task type
            task_types = [meta['task_type'] for meta in self.metadata_list]
            unique_tasks = list(set(task_types))
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_tasks)))
            task_to_color = {task: colors[i] for i, task in enumerate(unique_tasks)}

            for task in unique_tasks:
                mask = np.array([t == task for t in task_types])
                ax.scatter(umap_data[mask, 0], umap_data[mask, 1],
                          label=task, alpha=0.7, s=100,
                          c=[task_to_color[task]])

            ax.legend(title='Task Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            ax.scatter(umap_data[:, 0], umap_data[:, 1], alpha=0.7, s=100)

        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.set_title('2D UMAP Embedding of Weight Space')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/umap_2d.png", dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved: {self.save_dir}/umap_2d.png")

    def plot_umap_3d(self):
        """Plot 3D UMAP embedding."""
        umap_data = self.results['umap_3d']['embedding']

        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')

        if self.metadata_list is not None:
            # Color by task type
            task_types = [meta['task_type'] for meta in self.metadata_list]
            unique_tasks = list(set(task_types))
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_tasks)))
            task_to_color = {task: colors[i] for i, task in enumerate(unique_tasks)}

            for task in unique_tasks:
                mask = np.array([t == task for t in task_types])
                ax.scatter(umap_data[mask, 0], umap_data[mask, 1], umap_data[mask, 2],
                          label=task, alpha=0.7, s=100,
                          c=[task_to_color[task]])

            ax.legend(title='Task Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            ax.scatter(umap_data[:, 0], umap_data[:, 1], umap_data[:, 2],
                      alpha=0.7, s=100)

        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.set_zlabel('UMAP 3')
        ax.set_title('3D UMAP Embedding of Weight Space')

        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/umap_3d.png", dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved: {self.save_dir}/umap_3d.png")

    def plot_pca_3d(self):
        """Plot 3D PCA projection."""
        pca_data = self.results['pca']['transformed']

        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')

        if self.metadata_list is not None:
            # Color by task type
            task_types = [meta['task_type'] for meta in self.metadata_list]
            unique_tasks = list(set(task_types))
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_tasks)))
            task_to_color = {task: colors[i] for i, task in enumerate(unique_tasks)}

            for task in unique_tasks:
                mask = np.array([t == task for t in task_types])
                ax.scatter(pca_data[mask, 0], pca_data[mask, 1], pca_data[mask, 2],
                          label=task, alpha=0.7, s=100,
                          c=[task_to_color[task]])

            ax.legend(title='Task Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            ax.scatter(pca_data[:, 0], pca_data[:, 1], pca_data[:, 2],
                      alpha=0.7, s=100)

        variance = self.results['pca']['explained_variance_ratio']
        ax.set_xlabel(f'PC1 ({variance[0]:.1%})')
        ax.set_ylabel(f'PC2 ({variance[1]:.1%})')
        ax.set_zlabel(f'PC3 ({variance[2]:.1%})')
        ax.set_title('3D PCA Projection of Weight Space')

        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/pca_3d.png", dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved: {self.save_dir}/pca_3d.png")

    def plot_clustering(self):
        """Plot clustering results on UMAP embedding."""
        umap_data = self.results['umap_2d']['embedding']
        labels = self.results['clustering']['labels']

        fig, ax = plt.subplots(figsize=(10, 8))

        # Color by cluster
        unique_labels = np.unique(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

        for i, label in enumerate(unique_labels):
            mask = labels == label
            if label == -1:
                # Noise points
                ax.scatter(umap_data[mask, 0], umap_data[mask, 1],
                          c='gray', alpha=0.3, s=50, label='Noise')
            else:
                ax.scatter(umap_data[mask, 0], umap_data[mask, 1],
                          c=[colors[i]], alpha=0.7, s=100,
                          label=f'Cluster {label}')

        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.set_title(f"Clustering on UMAP Embedding ({self.results['clustering']['n_clusters']} clusters)")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/clustering.png", dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved: {self.save_dir}/clustering.png")

    def plot_dimension_comparison(self):
        """Compare different dimension estimates."""
        pca_dim = self.results['pca']['effective_dim_95']
        intrinsic_dim = self.results['intrinsic_dim']['intrinsic_dimension_mean']
        fractal_bc = self.results['fractal_boxcount']['fractal_dimension']
        fractal_corr = self.results['fractal_correlation']['correlation_dimension']

        fig, ax = plt.subplots(figsize=(10, 6))

        methods = ['PCA\n(95% var)', 'Intrinsic\n(MLE)', 'Fractal\n(Box-count)',
                  'Fractal\n(Correlation)']
        dimensions = [pca_dim, intrinsic_dim, fractal_bc, fractal_corr]

        bars = ax.bar(methods, dimensions, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],
                     alpha=0.7, edgecolor='black', linewidth=1.5)

        # Add value labels on bars
        for bar, dim in zip(bars, dimensions):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{dim:.2f}',
                   ha='center', va='bottom', fontsize=12, fontweight='bold')

        ax.set_ylabel('Estimated Dimension', fontsize=12)
        ax.set_title('Comparison of Dimensionality Estimates', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/dimension_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved: {self.save_dir}/dimension_comparison.png")

    def create_all_plots(self):
        """Generate all visualizations."""
        print("\n" + "=" * 60)
        print("GENERATING VISUALIZATIONS")
        print("=" * 60 + "\n")

        self.plot_pca_variance()
        self.plot_fractal_dimension()
        self.plot_intrinsic_dimension()
        self.plot_dimension_comparison()
        self.plot_pca_3d()
        self.plot_umap_2d()
        self.plot_umap_3d()
        self.plot_clustering()

        print(f"\nAll figures saved to: {self.save_dir}/")
