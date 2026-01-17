"""
Geometric and topological analysis of weight space.
Includes: PCA, fractal dimension estimation, manifold learning, clustering.
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import umap
from scipy.spatial.distance import pdist, squareform
from scipy.stats import linregress
from typing import Dict, Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')


class GeometricAnalyzer:
    """
    Analyze the geometric structure of trained model weights in weight space.
    """

    def __init__(self, weight_matrix: np.ndarray, metadata_list: List[Dict] = None):
        """
        Args:
            weight_matrix: Matrix where each row is a weight vector (n_models x n_params)
            metadata_list: Optional list of metadata for each model
        """
        self.weight_matrix = weight_matrix
        self.metadata_list = metadata_list

        # Handle object arrays (models with different parameter counts)
        if weight_matrix.dtype == object:
            raise ValueError(
                "Weight matrix contains models with different parameter counts!\n"
                "For Universal Subspace Hypothesis analysis, all models must have the SAME architecture.\n"
                "Parameter counts found: " + str(set(len(w) for w in weight_matrix)) + "\n"
                "Please train models with identical architectures (same input/output dims)."
            )

        self.n_models = weight_matrix.shape[0]
        self.n_params = weight_matrix.shape[1]

        print(f"Analyzing {self.n_models} models with {self.n_params} parameters each")

    def pca_analysis(self, n_components: Optional[int] = None) -> Dict:
        """
        Standard PCA analysis as baseline.

        Returns:
            Dictionary with PCA results including explained variance
        """
        if n_components is None:
            n_components = min(self.n_models - 1, 20)

        pca = PCA(n_components=n_components)
        transformed = pca.fit_transform(self.weight_matrix)

        # Calculate cumulative explained variance
        cumsum_variance = np.cumsum(pca.explained_variance_ratio_)

        # Find effective dimensionality (e.g., 95% variance explained)
        n_dim_95 = np.argmax(cumsum_variance >= 0.95) + 1

        results = {
            'transformed': transformed,
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'cumulative_variance': cumsum_variance,
            'n_components': n_components,
            'effective_dim_95': n_dim_95,
            'total_variance_explained': cumsum_variance[-1]
        }

        print(f"\nPCA Analysis:")
        print(f"  First 3 components explain: {cumsum_variance[2]:.2%} of variance")
        print(f"  95% variance captured in: {n_dim_95} dimensions")

        return results

    def estimate_fractal_dimension_boxcount(self, max_boxes: int = 20) -> Dict:
        """
        Estimate fractal dimension using box-counting method.
        This checks if the weight manifold has fractal-like structure.

        Returns:
            Dictionary with fractal dimension estimate and scaling data
        """
        print("\nEstimating fractal dimension (box-counting)...")

        # Normalize data to unit hypercube
        normalized = (self.weight_matrix - self.weight_matrix.min(axis=0)) / \
                     (self.weight_matrix.max(axis=0) - self.weight_matrix.min(axis=0) + 1e-10)

        # Different box sizes (epsilon values)
        epsilons = np.logspace(-2, 0, max_boxes)
        counts = []

        for eps in epsilons:
            # Discretize space into boxes of size eps
            if eps > 0:
                discretized = np.floor(normalized / eps).astype(int)
                # Count unique boxes containing points
                unique_boxes = len(np.unique(discretized, axis=0))
                counts.append(unique_boxes)
            else:
                counts.append(self.n_models)

        # Remove any invalid counts
        valid_idx = np.array(counts) > 0
        epsilons = epsilons[valid_idx]
        counts = np.array(counts)[valid_idx]

        # Fit log-log relationship: log(N) ~ -D * log(epsilon)
        log_eps = np.log(epsilons)
        log_counts = np.log(counts)

        # Linear regression in log-log space
        slope, intercept, r_value, p_value, std_err = linregress(log_eps, log_counts)

        fractal_dim = -slope

        results = {
            'fractal_dimension': fractal_dim,
            'r_squared': r_value**2,
            'p_value': p_value,
            'epsilons': epsilons,
            'counts': counts,
            'log_eps': log_eps,
            'log_counts': log_counts,
            'fit_slope': slope,
            'fit_intercept': intercept
        }

        print(f"  Estimated fractal dimension: {fractal_dim:.3f}")
        print(f"  R² of fit: {r_value**2:.3f}")

        return results

    def estimate_fractal_dimension_correlation(self, n_samples: int = None) -> Dict:
        """
        Estimate fractal dimension using correlation dimension method.
        More robust for high-dimensional data.

        Returns:
            Dictionary with correlation dimension estimate
        """
        print("\nEstimating correlation dimension...")

        # For computational efficiency, sample if we have many models
        if n_samples is None or n_samples > self.n_models:
            n_samples = min(self.n_models, 1000)

        if n_samples < self.n_models:
            indices = np.random.choice(self.n_models, n_samples, replace=False)
            sample = self.weight_matrix[indices]
        else:
            sample = self.weight_matrix

        # Compute pairwise distances
        distances = pdist(sample, metric='euclidean')

        # Different radius values
        radii = np.logspace(np.log10(distances.min() + 1e-10),
                           np.log10(distances.max()), 20)

        counts = []
        for r in radii:
            # Count pairs with distance < r
            count = np.sum(distances < r)
            counts.append(count)

        # Remove zeros
        valid_idx = np.array(counts) > 0
        radii = radii[valid_idx]
        counts = np.array(counts)[valid_idx]

        # Fit log-log: log(C(r)) ~ D * log(r)
        log_r = np.log(radii)
        log_counts = np.log(counts)

        slope, intercept, r_value, p_value, std_err = linregress(log_r, log_counts)

        correlation_dim = slope

        results = {
            'correlation_dimension': correlation_dim,
            'r_squared': r_value**2,
            'p_value': p_value,
            'radii': radii,
            'counts': counts,
            'log_radii': log_r,
            'log_counts': log_counts,
            'fit_slope': slope,
            'fit_intercept': intercept
        }

        print(f"  Estimated correlation dimension: {correlation_dim:.3f}")
        print(f"  R² of fit: {r_value**2:.3f}")

        return results

    def umap_embedding(self, n_components: int = 3,
                       n_neighbors: int = 15,
                       min_dist: float = 0.1) -> Dict:
        """
        UMAP manifold learning to visualize intrinsic geometry.
        Better than t-SNE for preserving global structure.

        Args:
            n_components: Dimension of embedding (2 or 3 for visualization)
            n_neighbors: UMAP parameter controlling local vs global structure
            min_dist: Minimum distance between points in embedding

        Returns:
            Dictionary with UMAP embedding and parameters
        """
        print(f"\nComputing UMAP embedding to {n_components}D...")

        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric='euclidean',
            random_state=42
        )

        embedding = reducer.fit_transform(self.weight_matrix)

        results = {
            'embedding': embedding,
            'n_components': n_components,
            'n_neighbors': n_neighbors,
            'min_dist': min_dist
        }

        print(f"  UMAP embedding shape: {embedding.shape}")

        return results

    def estimate_intrinsic_dimension_mle(self, k: int = 20) -> Dict:
        """
        Estimate intrinsic dimension using maximum likelihood estimation.
        Based on distances to k nearest neighbors.

        Args:
            k: Number of nearest neighbors to consider

        Returns:
            Dictionary with intrinsic dimension estimate
        """
        print(f"\nEstimating intrinsic dimension (MLE, k={k})...")

        # Compute distance matrix
        dist_matrix = squareform(pdist(self.weight_matrix, metric='euclidean'))

        # For each point, get k nearest neighbors
        intrinsic_dims = []

        for i in range(self.n_models):
            # Get sorted distances (excluding self)
            dists = np.sort(dist_matrix[i])[1:k+1]

            # MLE estimator: d = (k-1) / sum(log(r_k / r_i))
            if dists[-1] > 0:
                log_ratios = np.log(dists[-1] / dists[:-1])
                log_ratios = log_ratios[log_ratios > 0]  # Remove invalid values

                if len(log_ratios) > 0:
                    dim_estimate = len(log_ratios) / np.sum(log_ratios)
                    intrinsic_dims.append(dim_estimate)

        intrinsic_dims = np.array(intrinsic_dims)

        # Remove outliers (e.g., > 3 std devs)
        mean_dim = np.mean(intrinsic_dims)
        std_dim = np.std(intrinsic_dims)
        mask = np.abs(intrinsic_dims - mean_dim) < 3 * std_dim
        filtered_dims = intrinsic_dims[mask]

        results = {
            'intrinsic_dimension_mean': np.mean(filtered_dims),
            'intrinsic_dimension_std': np.std(filtered_dims),
            'intrinsic_dimension_median': np.median(filtered_dims),
            'all_estimates': intrinsic_dims,
            'filtered_estimates': filtered_dims,
            'k': k
        }

        print(f"  Estimated intrinsic dimension: {results['intrinsic_dimension_mean']:.2f} ± {results['intrinsic_dimension_std']:.2f}")
        print(f"  Median: {results['intrinsic_dimension_median']:.2f}")

        return results

    def clustering_analysis(self, method: str = 'dbscan', **kwargs) -> Dict:
        """
        Cluster analysis to find structure in weight space.

        Args:
            method: 'dbscan' or 'hierarchical'
            **kwargs: Parameters for clustering algorithm

        Returns:
            Dictionary with cluster labels and metrics
        """
        print(f"\nClustering analysis ({method})...")

        # Use PCA-reduced data for clustering (more stable)
        pca = PCA(n_components=min(20, self.n_models - 1))
        reduced_data = pca.fit_transform(self.weight_matrix)

        if method == 'dbscan':
            eps = kwargs.get('eps', 0.5)
            min_samples = kwargs.get('min_samples', 3)

            clusterer = DBSCAN(eps=eps, min_samples=min_samples)
            labels = clusterer.fit_predict(reduced_data)

        elif method == 'hierarchical':
            n_clusters = kwargs.get('n_clusters', 5)

            clusterer = AgglomerativeClustering(n_clusters=n_clusters)
            labels = clusterer.fit_predict(reduced_data)

        else:
            raise ValueError(f"Unknown clustering method: {method}")

        # Compute silhouette score (if we have more than 1 cluster)
        n_clusters = len(np.unique(labels[labels >= 0]))  # Exclude noise (-1)

        if n_clusters > 1 and n_clusters < self.n_models:
            score = silhouette_score(reduced_data, labels)
        else:
            score = None

        results = {
            'labels': labels,
            'n_clusters': n_clusters,
            'silhouette_score': score,
            'method': method,
            'params': kwargs
        }

        print(f"  Found {n_clusters} clusters")
        if score is not None:
            print(f"  Silhouette score: {score:.3f}")

        # If we have metadata, analyze clusters by task type
        if self.metadata_list is not None:
            self._analyze_clusters_by_task(labels)

        return results

    def _analyze_clusters_by_task(self, labels: np.ndarray):
        """Analyze how clusters align with task types."""
        task_types = [meta['task_type'] for meta in self.metadata_list]

        print("\n  Cluster composition by task type:")
        for cluster_id in np.unique(labels):
            if cluster_id < 0:  # Skip noise
                continue

            cluster_mask = labels == cluster_id
            cluster_tasks = [task_types[i] for i in range(len(task_types)) if cluster_mask[i]]

            task_counts = {}
            for task in cluster_tasks:
                task_counts[task] = task_counts.get(task, 0) + 1

            print(f"    Cluster {cluster_id}: {task_counts}")

    def full_analysis(self) -> Dict:
        """
        Run complete geometric analysis pipeline.

        Returns:
            Dictionary with all analysis results
        """
        print("\n" + "=" * 60)
        print("GEOMETRIC ANALYSIS OF WEIGHT SPACE")
        print("=" * 60)

        results = {}

        # 1. PCA baseline
        results['pca'] = self.pca_analysis()

        # 2. Fractal dimension estimates
        results['fractal_boxcount'] = self.estimate_fractal_dimension_boxcount()
        results['fractal_correlation'] = self.estimate_fractal_dimension_correlation()

        # 3. Intrinsic dimension
        results['intrinsic_dim'] = self.estimate_intrinsic_dimension_mle()

        # 4. UMAP embedding
        results['umap_3d'] = self.umap_embedding(n_components=3)
        results['umap_2d'] = self.umap_embedding(n_components=2)

        # 5. Clustering
        results['clustering'] = self.clustering_analysis(method='dbscan', eps=2.0, min_samples=2)

        print("\n" + "=" * 60)
        print("SUMMARY OF FINDINGS")
        print("=" * 60)
        print(f"Total parameters per model: {self.n_params}")
        print(f"PCA effective dimension (95% var): {results['pca']['effective_dim_95']}")
        print(f"Intrinsic dimension (MLE): {results['intrinsic_dim']['intrinsic_dimension_mean']:.2f}")
        print(f"Fractal dimension (box-counting): {results['fractal_boxcount']['fractal_dimension']:.2f}")
        print(f"Correlation dimension: {results['fractal_correlation']['correlation_dimension']:.2f}")
        print(f"Number of clusters found: {results['clustering']['n_clusters']}")

        # Interpretation
        print("\nINTERPRETATION:")
        intrinsic = results['intrinsic_dim']['intrinsic_dimension_mean']
        fractal = results['fractal_correlation']['correlation_dimension']

        if abs(intrinsic - fractal) < 1.0:
            print("  ✓ Fractal and intrinsic dimensions agree - suggests smooth low-dim manifold")
        else:
            print("  ⚠ Dimension estimates differ - possible fractal/irregular structure")

        if results['pca']['effective_dim_95'] < 10:
            print(f"  ✓ Very low effective dimensionality - strong universal subspace hypothesis")
        elif results['pca']['effective_dim_95'] < 30:
            print(f"  ~ Moderate dimensionality reduction - partial support for hypothesis")
        else:
            print(f"  ✗ High dimensionality - weak support for universal subspace")

        return results
