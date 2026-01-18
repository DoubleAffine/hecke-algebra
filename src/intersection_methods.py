"""
Multiple methods for computing manifold intersections.

We compare different approaches:
1. PCA-based (global vs individual dimensions)
2. Linear algebra (direct subspace intersection)
3. Clustering (separate shared vs task-specific dimensions)
4. Alternative dimension estimators (MLE, correlation dimension, etc.)
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.linalg import svd, orth
from scipy.spatial.distance import pdist, squareform


class IntersectionAnalyzer:
    """Compute manifold intersections using multiple methods."""

    def __init__(self, manifolds, manifold_dims=None, variance_threshold=0.95):
        """
        Args:
            manifolds: List of weight matrices, one per dataset
            manifold_dims: List of estimated dimensions (if known)
            variance_threshold: Variance threshold for PCA
        """
        self.manifolds = manifolds
        self.n_manifolds = len(manifolds)
        self.variance_threshold = variance_threshold

        # Estimate dimensions if not provided
        if manifold_dims is None:
            self.manifold_dims = []
            for weights in manifolds:
                pca = PCA()
                pca.fit(weights)
                var_ratios = pca.explained_variance_ratio_
                cumsum = np.cumsum(var_ratios)
                dim_95 = int(np.argmax(cumsum >= variance_threshold) + 1)
                self.manifold_dims.append(dim_95)
        else:
            self.manifold_dims = manifold_dims

        # Compute subspace bases for each manifold
        self.subspace_bases = []
        for weights, dim in zip(manifolds, self.manifold_dims):
            pca = PCA(n_components=dim)
            pca.fit(weights)
            self.subspace_bases.append(pca.components_.T)  # Column vectors

        print(f"Initialized with {self.n_manifolds} manifolds")
        print(f"Dimensions: {self.manifold_dims}")

    def method_1_pca_ratio(self):
        """
        Method 1: Global PCA ratio

        Compare global dimension (all data) vs mean individual dimension.
        If ratio ≈ 1: strong overlap
        If ratio ≈ n: orthogonal
        """
        print("\n" + "="*80)
        print(" METHOD 1: PCA RATIO ANALYSIS")
        print("="*80)

        # Combine all weights
        all_weights = np.vstack(self.manifolds)
        print(f"\nCombined shape: {all_weights.shape}")

        # Global PCA
        pca_global = PCA()
        pca_global.fit(all_weights)

        var_global = pca_global.explained_variance_ratio_
        cumsum_global = np.cumsum(var_global)

        dim_global = int(np.argmax(cumsum_global >= self.variance_threshold) + 1)
        effective_global = float((np.sum(var_global) ** 2) / np.sum(var_global ** 2))

        mean_individual = np.mean(self.manifold_dims)
        ratio = effective_global / mean_individual

        print(f"\nGlobal dimension: {effective_global:.1f}D")
        print(f"Mean individual: {mean_individual:.1f}D")
        print(f"Ratio: {ratio:.2f}")

        if ratio < 1.2:
            intersection_estimate = effective_global
            conclusion = "Strong overlap (universal subspace)"
        elif ratio > self.n_manifolds * 0.3:
            intersection_estimate = 0
            conclusion = "Nearly orthogonal (no intersection)"
        else:
            intersection_estimate = max(0, effective_global - (ratio - 1) * mean_individual)
            conclusion = "Partial overlap"

        print(f"\nConclusion: {conclusion}")
        print(f"Intersection estimate: ~{intersection_estimate:.0f}D")

        return {
            'method': 'pca_ratio',
            'global_dim': effective_global,
            'mean_individual': mean_individual,
            'ratio': ratio,
            'intersection_dim': intersection_estimate,
            'conclusion': conclusion
        }

    def method_2_subspace_intersection(self):
        """
        Method 2: Direct subspace intersection via linear algebra

        Compute the intersection of subspaces using:
        - For two subspaces U and V, intersection dim = rank(U) + rank(V) - rank([U V])
        - For multiple subspaces, compute iteratively
        """
        print("\n" + "="*80)
        print(" METHOD 2: LINEAR ALGEBRA SUBSPACE INTERSECTION")
        print("="*80)

        if self.n_manifolds < 2:
            print("\nNeed at least 2 manifolds for intersection")
            return None

        # Start with first two subspaces
        U = self.subspace_bases[0]
        V = self.subspace_bases[1]

        print(f"\nComputing intersection of {self.n_manifolds} subspaces...")
        print(f"Subspace 1: {U.shape[1]}D")
        print(f"Subspace 2: {V.shape[1]}D")

        # Intersection dimension: dim(U ∩ V) = dim(U) + dim(V) - dim(U + V)
        # where dim(U + V) = rank([U V])

        combined = np.hstack([U, V])
        rank_sum = np.linalg.matrix_rank(combined, tol=1e-10)

        intersection_dim = U.shape[1] + V.shape[1] - rank_sum

        print(f"Rank([U V]): {rank_sum}")
        print(f"Intersection dim (U ∩ V): {intersection_dim}D")

        # For multiple subspaces, compute intersection iteratively
        if self.n_manifolds > 2:
            print(f"\nExtending to {self.n_manifolds} subspaces...")

            # Compute pairwise intersections
            pairwise_dims = []
            for i in range(self.n_manifolds):
                for j in range(i+1, self.n_manifolds):
                    U_i = self.subspace_bases[i]
                    U_j = self.subspace_bases[j]
                    combined_ij = np.hstack([U_i, U_j])
                    rank_ij = np.linalg.matrix_rank(combined_ij, tol=1e-10)
                    int_dim_ij = U_i.shape[1] + U_j.shape[1] - rank_ij
                    pairwise_dims.append(int_dim_ij)

            print(f"Pairwise intersections: min={np.min(pairwise_dims):.0f}D, "
                  f"mean={np.mean(pairwise_dims):.0f}D, max={np.max(pairwise_dims):.0f}D")

            # Lower bound on full intersection
            intersection_dim = np.min(pairwise_dims)
            print(f"\nFull intersection (lower bound): ≥{intersection_dim}D")

        return {
            'method': 'subspace_intersection',
            'intersection_dim': intersection_dim,
            'rank_sum_first_two': rank_sum,
            'pairwise_min': np.min(pairwise_dims) if self.n_manifolds > 2 else None,
            'pairwise_mean': np.mean(pairwise_dims) if self.n_manifolds > 2 else None
        }

    def method_3_clustering(self):
        """
        Method 3: Clustering-based dimension separation

        Cluster the principal components from all manifolds.
        Shared dimensions: components that cluster together across manifolds
        Task-specific: components unique to individual manifolds
        """
        print("\n" + "="*80)
        print(" METHOD 3: CLUSTERING-BASED ANALYSIS")
        print("="*80)

        # Collect all principal components
        all_components = []
        component_labels = []  # Which manifold each component comes from

        for idx, basis in enumerate(self.subspace_bases):
            # Each column is a principal component (direction in weight space)
            for i in range(basis.shape[1]):
                all_components.append(basis[:, i])
                component_labels.append(idx)

        all_components = np.array(all_components)
        component_labels = np.array(component_labels)

        print(f"\nTotal principal components: {len(all_components)}")
        print(f"From {self.n_manifolds} manifolds")

        # Compute pairwise angles between components
        # Two components are "aligned" if their angle is small
        print(f"\nComputing pairwise component similarities...")

        similarities = []
        for i in range(len(all_components)):
            for j in range(i+1, len(all_components)):
                # Skip components from same manifold
                if component_labels[i] == component_labels[j]:
                    continue

                # Cosine similarity (abs to handle sign ambiguity)
                sim = abs(np.dot(all_components[i], all_components[j]))
                similarities.append(sim)

        similarities = np.array(similarities)

        print(f"Cross-manifold similarities: min={np.min(similarities):.3f}, "
              f"mean={np.mean(similarities):.3f}, max={np.max(similarities):.3f}")

        # Count highly aligned components (similarity > 0.9)
        aligned_threshold = 0.9
        highly_aligned = np.sum(similarities > aligned_threshold)

        print(f"\nHighly aligned component pairs (>{aligned_threshold}): {highly_aligned}")

        # Estimate shared dimensions
        # Each highly aligned pair suggests one shared dimension
        # (This is a rough estimate)
        shared_dims_estimate = int(highly_aligned / (self.n_manifolds - 1))

        print(f"Estimated shared dimensions: ~{shared_dims_estimate}D")

        return {
            'method': 'clustering',
            'intersection_dim': shared_dims_estimate,
            'mean_similarity': np.mean(similarities),
            'max_similarity': np.max(similarities),
            'highly_aligned_pairs': int(highly_aligned)
        }

    def method_4_mle_dimension(self):
        """
        Method 4: Maximum Likelihood Estimation of intrinsic dimension

        Compare MLE dimension of combined data vs individual manifolds.
        """
        print("\n" + "="*80)
        print(" METHOD 4: MLE DIMENSION ESTIMATION")
        print("="*80)

        def mle_dimension(data, k=20):
            """
            Estimate intrinsic dimension using MLE method.
            Based on Levina & Bickel (2004).
            """
            n_samples = data.shape[0]

            # For each point, find k nearest neighbors
            distances = squareform(pdist(data, metric='euclidean'))

            # Sort distances for each point
            sorted_dists = np.sort(distances, axis=1)

            # Use distances to k-th neighbor (exclude self at index 0)
            r_k = sorted_dists[:, k]  # Distance to k-th neighbor
            r = sorted_dists[:, 1:k+1]  # Distances to all k neighbors (exclude self)

            # MLE estimate for each point
            dim_estimates = []
            for i in range(n_samples):
                if r_k[i] > 0:
                    # Estimate: dim = (k-1) / sum(log(r_k / r_j))
                    log_ratios = np.log(r_k[i] / (r[i, :] + 1e-10))
                    dim_i = (k - 1) / np.sum(log_ratios)
                    if np.isfinite(dim_i) and dim_i > 0:
                        dim_estimates.append(dim_i)

            return np.median(dim_estimates) if dim_estimates else np.nan

        # Estimate dimension for each manifold
        print(f"\nEstimating MLE dimensions for individual manifolds...")
        individual_mle_dims = []

        for idx, weights in enumerate(self.manifolds):
            dim_mle = mle_dimension(weights, k=min(20, weights.shape[0]//2))
            individual_mle_dims.append(dim_mle)
            print(f"  Manifold {idx+1}: {dim_mle:.1f}D")

        # Estimate for combined data
        all_weights = np.vstack(self.manifolds)
        global_mle_dim = mle_dimension(all_weights, k=min(20, all_weights.shape[0]//2))

        mean_individual_mle = np.nanmean(individual_mle_dims)

        print(f"\nGlobal MLE dimension: {global_mle_dim:.1f}D")
        print(f"Mean individual MLE: {mean_individual_mle:.1f}D")
        print(f"Ratio: {global_mle_dim / mean_individual_mle:.2f}")

        return {
            'method': 'mle',
            'global_dim': global_mle_dim,
            'mean_individual': mean_individual_mle,
            'individual_dims': individual_mle_dims
        }

    def analyze_all_methods(self):
        """Run all intersection methods and compare results."""
        print("="*80)
        print(" COMPREHENSIVE INTERSECTION ANALYSIS")
        print("="*80)

        results = {}

        # Method 1: PCA ratio
        results['pca_ratio'] = self.method_1_pca_ratio()

        # Method 2: Subspace intersection
        results['subspace_intersection'] = self.method_2_subspace_intersection()

        # Method 3: Clustering
        results['clustering'] = self.method_3_clustering()

        # Method 4: MLE
        try:
            results['mle'] = self.method_4_mle_dimension()
        except Exception as e:
            print(f"\nMLE method failed: {e}")
            results['mle'] = None

        # Summary comparison
        print("\n" + "="*80)
        print(" COMPARISON OF METHODS")
        print("="*80)

        print("\nIntersection dimension estimates:")
        for method_name, result in results.items():
            if result and 'intersection_dim' in result:
                print(f"  {method_name:25s}: {result['intersection_dim']:.1f}D")

        return results
