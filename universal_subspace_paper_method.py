#!/usr/bin/env python3
"""
Replicating the Universal Weight Subspace Hypothesis methodology
from Kaushik et al. (2024) on our data.

Paper's approach:
1. Stack all model weights into a tensor
2. Apply HOSVD (Higher-Order SVD) / mode-wise SVD
3. Analyze spectral decay
4. Find minimal k that captures τ% variance
5. Reconstruct models from low-rank basis
6. Measure reconstruction accuracy and compression

Reference: https://arxiv.org/abs/2512.05117
"""
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from datetime import datetime

# For reconstruction testing
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class UniversalSubspaceAnalyzer:
    """
    Implements the paper's methodology for finding universal weight subspaces.
    """

    def __init__(self, weight_matrices, dataset_names=None):
        """
        Args:
            weight_matrices: List of (n_models, n_params) arrays, one per dataset
            dataset_names: Optional names for each dataset
        """
        self.weight_matrices = weight_matrices
        self.n_datasets = len(weight_matrices)
        self.dataset_names = dataset_names or [f"dataset_{i}" for i in range(self.n_datasets)]

        # Stack all weights
        self.all_weights = np.vstack(weight_matrices)
        self.n_models, self.n_params = self.all_weights.shape

        print(f"Loaded {self.n_datasets} datasets")
        print(f"Total models: {self.n_models}")
        print(f"Parameter dimension: {self.n_params}")

    def step1_spectral_decomposition(self):
        """
        Step 1: Spectral decomposition of combined weight matrix.

        Following the paper: "Zero-center, then perform thin SVD"
        """
        print("\n" + "=" * 80)
        print(" STEP 1: SPECTRAL DECOMPOSITION")
        print("=" * 80)

        # Zero-center (as per paper)
        self.mean_weights = np.mean(self.all_weights, axis=0)
        weights_centered = self.all_weights - self.mean_weights

        # Thin SVD
        print("\nPerforming SVD on centered weight matrix...")
        self.U, self.S, self.Vt = np.linalg.svd(weights_centered, full_matrices=False)

        print(f"  U shape: {self.U.shape} (model coefficients)")
        print(f"  S shape: {self.S.shape} (singular values)")
        print(f"  Vt shape: {self.Vt.shape} (weight space basis)")

        # Variance explained
        total_var = np.sum(self.S ** 2)
        self.var_explained = (self.S ** 2) / total_var
        self.var_cumsum = np.cumsum(self.var_explained)

        return self.U, self.S, self.Vt

    def step2_spectral_decay_analysis(self):
        """
        Step 2: Analyze spectral decay (scree plot).

        Paper finding: "Sharp spectral decay, k ≤ 16 captures majority of variance"
        """
        print("\n" + "=" * 80)
        print(" STEP 2: SPECTRAL DECAY ANALYSIS")
        print("=" * 80)

        # Find k for different variance thresholds
        thresholds = [0.50, 0.80, 0.90, 0.95, 0.99]
        self.k_values = {}

        print("\nVariance thresholds:")
        for tau in thresholds:
            k = np.argmax(self.var_cumsum >= tau) + 1
            self.k_values[tau] = k
            print(f"  τ = {tau*100:.0f}%: k = {k} components")

        # Spectral decay rate
        print("\nSpectral decay (first 30 components):")
        print(f"  {'k':<6} {'σ_k':<12} {'Var %':<10} {'Cumulative %':<12}")
        print(f"  {'-'*40}")
        for k in range(min(30, len(self.S))):
            print(f"  {k+1:<6} {self.S[k]:<12.4f} {self.var_explained[k]*100:<10.2f} {self.var_cumsum[k]*100:<12.2f}")

        # Effective rank (participation ratio)
        self.effective_rank = (np.sum(self.S) ** 2) / np.sum(self.S ** 2)
        print(f"\nEffective rank: {self.effective_rank:.1f}")

        return self.k_values

    def step3_extract_universal_basis(self, k=None, tau=0.95):
        """
        Step 3: Extract the universal k-dimensional basis.

        The top k right singular vectors form the universal subspace.
        """
        print("\n" + "=" * 80)
        print(" STEP 3: EXTRACT UNIVERSAL BASIS")
        print("=" * 80)

        if k is None:
            k = self.k_values.get(tau, self.k_values[0.95])

        self.k = k
        self.universal_basis = self.Vt[:k, :].T  # (n_params, k) - columns are basis vectors

        print(f"\nUniversal subspace dimension: {k}")
        print(f"Basis shape: {self.universal_basis.shape}")
        print(f"Compression ratio: {self.n_params / k:.1f}x")

        return self.universal_basis

    def step4_project_and_reconstruct(self):
        """
        Step 4: Project models onto universal subspace and reconstruct.

        w_reconstructed = mean + V_k @ V_k.T @ (w - mean)
        """
        print("\n" + "=" * 80)
        print(" STEP 4: PROJECTION AND RECONSTRUCTION")
        print("=" * 80)

        # Project each model onto the universal subspace
        weights_centered = self.all_weights - self.mean_weights

        # Coefficients in the universal basis
        self.coefficients = weights_centered @ self.universal_basis  # (n_models, k)

        # Reconstruct
        self.weights_reconstructed = self.mean_weights + self.coefficients @ self.universal_basis.T

        # Reconstruction error
        errors = np.linalg.norm(self.all_weights - self.weights_reconstructed, axis=1)
        original_norms = np.linalg.norm(self.all_weights, axis=1)
        relative_errors = errors / original_norms

        print(f"\nReconstruction error (using k={self.k} components):")
        print(f"  Mean absolute error: {np.mean(errors):.6f}")
        print(f"  Mean relative error: {np.mean(relative_errors)*100:.2f}%")
        print(f"  Max relative error: {np.max(relative_errors)*100:.2f}%")

        self.reconstruction_errors = relative_errors

        return self.weights_reconstructed, self.coefficients

    def step5_test_reconstruction_accuracy(self, test_data_fn):
        """
        Step 5: Test if reconstructed models maintain accuracy.

        This is the key test: do models projected onto the universal
        subspace still perform well on their original tasks?
        """
        print("\n" + "=" * 80)
        print(" STEP 5: RECONSTRUCTION ACCURACY TEST")
        print("=" * 80)

        results = []
        start_idx = 0

        for i, (weights, name) in enumerate(zip(self.weight_matrices, self.dataset_names)):
            n_models = weights.shape[0]
            end_idx = start_idx + n_models

            # Get original and reconstructed weights for this dataset
            original_weights = weights
            reconstructed_weights = self.weights_reconstructed[start_idx:end_idx]

            # Test on fresh data
            test_loader = test_data_fn(name)

            # Evaluate original models
            orig_accs = []
            for w in original_weights[:5]:  # Sample 5 models
                acc = self._evaluate_model(w, test_loader)
                orig_accs.append(acc)

            # Evaluate reconstructed models
            recon_accs = []
            for w in reconstructed_weights[:5]:
                acc = self._evaluate_model(w, test_loader)
                recon_accs.append(acc)

            mean_orig = np.mean(orig_accs)
            mean_recon = np.mean(recon_accs)
            acc_drop = mean_orig - mean_recon

            print(f"\n{name}:")
            print(f"  Original accuracy: {mean_orig:.1f}%")
            print(f"  Reconstructed accuracy: {mean_recon:.1f}%")
            print(f"  Accuracy drop: {acc_drop:.1f}%")

            results.append({
                'dataset': name,
                'original_acc': mean_orig,
                'reconstructed_acc': mean_recon,
                'accuracy_drop': acc_drop
            })

            start_idx = end_idx

        return results

    def _evaluate_model(self, weights, test_loader):
        """Evaluate a single model given its flattened weights."""
        # Reconstruct model from weights
        model = self._weights_to_model(weights)
        model.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for X, y in test_loader:
                outputs = model(X)
                predictions = (outputs > 0.5).float()
                correct += (predictions == y).sum().item()
                total += y.size(0)

        return 100 * correct / total

    def _weights_to_model(self, weights):
        """Convert flattened weight vector back to PyTorch model."""
        model = nn.Sequential(
            nn.Linear(10, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

        # Unflatten weights
        idx = 0
        for name, param in model.named_parameters():
            shape = param.shape
            size = param.numel()
            param.data = torch.FloatTensor(weights[idx:idx+size].reshape(shape))
            idx += size

        return model

    def step6_compression_analysis(self):
        """
        Step 6: Analyze compression achieved.

        Paper claims: "100x memory reduction"
        """
        print("\n" + "=" * 80)
        print(" STEP 6: COMPRESSION ANALYSIS")
        print("=" * 80)

        # Original storage: n_models × n_params
        original_size = self.n_models * self.n_params

        # Compressed storage: mean (n_params) + basis (n_params × k) + coefficients (n_models × k)
        compressed_size = self.n_params + (self.n_params * self.k) + (self.n_models * self.k)

        compression_ratio = original_size / compressed_size

        print(f"\nStorage analysis:")
        print(f"  Original: {self.n_models} models × {self.n_params} params = {original_size:,} floats")
        print(f"  Compressed: mean({self.n_params}) + basis({self.n_params}×{self.k}) + coef({self.n_models}×{self.k})")
        print(f"            = {compressed_size:,} floats")
        print(f"  Compression ratio: {compression_ratio:.1f}x")

        # Memory in MB (assuming float32)
        original_mb = original_size * 4 / (1024 * 1024)
        compressed_mb = compressed_size * 4 / (1024 * 1024)

        print(f"\n  Original memory: {original_mb:.2f} MB")
        print(f"  Compressed memory: {compressed_mb:.2f} MB")
        print(f"  Savings: {original_mb - compressed_mb:.2f} MB ({(1 - compressed_mb/original_mb)*100:.1f}%)")

        return compression_ratio

    def create_visualizations(self, save_dir):
        """Create paper-style visualizations."""
        os.makedirs(save_dir, exist_ok=True)

        # 1. Scree plot (spectral decay)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Individual variance
        ax1 = axes[0]
        ax1.bar(range(1, min(51, len(self.var_explained)+1)),
                self.var_explained[:50] * 100, color='steelblue')
        ax1.set_xlabel('Principal Component', fontsize=12)
        ax1.set_ylabel('Variance Explained (%)', fontsize=12)
        ax1.set_title('Spectral Decay (Scree Plot)', fontsize=14)
        ax1.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='1% threshold')
        ax1.legend()

        # Cumulative variance
        ax2 = axes[1]
        ax2.plot(range(1, len(self.var_cumsum)+1), self.var_cumsum * 100,
                 color='steelblue', linewidth=2)
        ax2.axhline(y=95, color='red', linestyle='--', alpha=0.7, label='95% threshold')
        ax2.axhline(y=99, color='orange', linestyle='--', alpha=0.7, label='99% threshold')
        ax2.axvline(x=self.k_values[0.95], color='red', linestyle=':', alpha=0.7)
        ax2.axvline(x=self.k_values[0.99], color='orange', linestyle=':', alpha=0.7)
        ax2.set_xlabel('Number of Components', fontsize=12)
        ax2.set_ylabel('Cumulative Variance (%)', fontsize=12)
        ax2.set_title('Cumulative Variance Explained', fontsize=14)
        ax2.legend()
        ax2.set_xlim(0, min(300, len(self.var_cumsum)))

        plt.tight_layout()
        plt.savefig(f'{save_dir}/spectral_analysis.png', dpi=150)
        plt.close()

        # 2. Reconstruction error vs k
        fig, ax = plt.subplots(figsize=(8, 5))

        k_values = list(range(10, min(300, len(self.S)), 10))
        recon_errors = []

        for k in k_values:
            basis_k = self.Vt[:k, :].T
            centered = self.all_weights - self.mean_weights
            coef = centered @ basis_k
            recon = self.mean_weights + coef @ basis_k.T
            error = np.mean(np.linalg.norm(self.all_weights - recon, axis=1) /
                          np.linalg.norm(self.all_weights, axis=1))
            recon_errors.append(error * 100)

        ax.plot(k_values, recon_errors, 'o-', color='steelblue', linewidth=2)
        ax.axhline(y=5, color='red', linestyle='--', alpha=0.7, label='5% error threshold')
        ax.axhline(y=1, color='orange', linestyle='--', alpha=0.7, label='1% error threshold')
        ax.set_xlabel('Universal Subspace Dimension (k)', fontsize=12)
        ax.set_ylabel('Mean Reconstruction Error (%)', fontsize=12)
        ax.set_title('Reconstruction Error vs Subspace Dimension', fontsize=14)
        ax.legend()
        ax.set_yscale('log')

        plt.tight_layout()
        plt.savefig(f'{save_dir}/reconstruction_error.png', dpi=150)
        plt.close()

        print(f"\nVisualizations saved to {save_dir}/")


def create_test_data(dataset_name):
    """Create test data for accuracy evaluation."""
    if 'random_labels' in dataset_name:
        # Random labels - use random test set
        X, _ = make_classification(n_samples=200, n_features=10, n_informative=5,
                                   n_redundant=2, random_state=999)
        y = np.random.randint(0, 2, size=200)
    else:
        # Synthetic data - use consistent test set
        X, y = make_classification(n_samples=200, n_features=10, n_informative=5,
                                   n_redundant=2, random_state=42)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    y = y.reshape(-1, 1).astype(np.float32)

    dataset = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y))
    return DataLoader(dataset, batch_size=32)


def main():
    print("=" * 80)
    print(" UNIVERSAL WEIGHT SUBSPACE ANALYSIS")
    print(" Following Kaushik et al. (2024) methodology")
    print("=" * 80)

    # Load our experimental data
    results_dir = 'experiments/current/10_dataset_intersection'

    dataset_names = ['synthetic_easy_1', 'synthetic_easy_2', 'synthetic_easy_3',
                     'synthetic_easy_4', 'synthetic_easy_5', 'synthetic_easy_6',
                     'synthetic_easy_7', 'synthetic_easy_8',
                     'random_labels_1', 'random_labels_2']

    weight_matrices = []
    for name in dataset_names:
        path = f'{results_dir}/weights_{name}.npy'
        if os.path.exists(path):
            weight_matrices.append(np.load(path))

    if len(weight_matrices) == 0:
        print("No weight files found! Run the 10-dataset experiment first.")
        return

    # Create analyzer
    analyzer = UniversalSubspaceAnalyzer(weight_matrices, dataset_names)

    # Run the paper's pipeline
    analyzer.step1_spectral_decomposition()
    analyzer.step2_spectral_decay_analysis()
    analyzer.step3_extract_universal_basis(tau=0.95)
    analyzer.step4_project_and_reconstruct()

    # Test reconstruction accuracy
    print("\nTesting reconstruction accuracy on fresh data...")
    accuracy_results = analyzer.step5_test_reconstruction_accuracy(create_test_data)

    # Compression analysis
    compression = analyzer.step6_compression_analysis()

    # Create visualizations
    save_dir = f'{results_dir}/paper_method_analysis'
    analyzer.create_visualizations(save_dir)

    # Save results
    results = {
        'methodology': 'Universal Weight Subspace (Kaushik et al. 2024)',
        'n_datasets': analyzer.n_datasets,
        'n_models': analyzer.n_models,
        'n_params': analyzer.n_params,
        'k_values': {str(k): int(v) for k, v in analyzer.k_values.items()},
        'effective_rank': float(analyzer.effective_rank),
        'compression_ratio': float(compression),
        'mean_reconstruction_error': float(np.mean(analyzer.reconstruction_errors) * 100),
        'accuracy_results': accuracy_results,
        'timestamp': datetime.now().isoformat()
    }

    with open(f'{save_dir}/results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Summary
    print("\n" + "=" * 80)
    print(" SUMMARY")
    print("=" * 80)

    print(f"""
Universal Subspace Analysis Results:

1. SPECTRAL ANALYSIS:
   - Total models: {analyzer.n_models}
   - Parameter dimension: {analyzer.n_params}
   - Effective rank: {analyzer.effective_rank:.1f}

2. UNIVERSAL SUBSPACE:
   - k for 95% variance: {analyzer.k_values[0.95]}
   - k for 99% variance: {analyzer.k_values[0.99]}
   - Compression ratio: {compression:.1f}x

3. RECONSTRUCTION:
   - Mean error: {np.mean(analyzer.reconstruction_errors)*100:.2f}%
   - Mean accuracy drop: {np.mean([r['accuracy_drop'] for r in accuracy_results]):.1f}%

4. COMPARISON WITH PAPER:
   - Paper (ViT, 500 models): k ≈ 100 for 95% variance
   - Paper (Mistral LoRA): k ≈ 16 for majority variance
   - Our data (MLP, 500 models): k = {analyzer.k_values[0.95]} for 95% variance

Results saved to: {save_dir}/
""")

    print("=" * 80)


if __name__ == '__main__':
    main()
