#!/usr/bin/env python3
"""
Dropout Spread Experiment

Question: Does dropout spread out solutions in weight space or constrain them?

Design:
- 4 architectures: Small, Medium, Large, XLarge
- 3 dropout rates: 0.0, 0.3, 0.5
- 50 models per condition
- Measure: effective dimension, pairwise distance, spectral decay, accuracy
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist
import sys
import os
import json
import time
from datetime import datetime

# Architectures to test
ARCHITECTURES = {
    'small':  [10, 16, 16, 1],      # 465 params
    'medium': [10, 64, 64, 1],      # 5,249 params
    'large':  [10, 256, 256, 1],    # 69,377 params
    'xlarge': [10, 512, 512, 1],    # 268,289 params
}

DROPOUT_RATES = [0.0, 0.3, 0.5]
MODELS_PER_CONDITION = 50
EPOCHS = 100


class MLPWithDropout(nn.Module):
    """MLP with configurable dropout."""

    def __init__(self, layer_sizes, dropout_rate=0.0):
        super().__init__()

        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))

            if i < len(layer_sizes) - 2:  # Not last layer
                layers.append(nn.ReLU())
                if dropout_rate > 0:
                    layers.append(nn.Dropout(dropout_rate))

        layers.append(nn.Sigmoid())
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

    def get_flat_weights(self):
        """Return all weights as a flat vector."""
        weights = []
        for param in self.parameters():
            weights.append(param.data.cpu().numpy().flatten())
        return np.concatenate(weights)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


def create_dataset(seed=42):
    """Create binary classification dataset."""
    np.random.seed(seed)
    X, y = make_classification(
        n_samples=1000, n_features=10, n_informative=5,
        n_redundant=2, n_classes=2, random_state=seed
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train.reshape(-1, 1))
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test),
        torch.FloatTensor(y_test.reshape(-1, 1))
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, test_loader


def train_model(layer_sizes, dropout_rate, train_loader, test_loader, epochs=100):
    """Train a single model and return final weights and accuracy."""
    model = MLPWithDropout(layer_sizes, dropout_rate)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_acc = 0
    patience = 20
    patience_counter = 0

    for epoch in range(epochs):
        # Train
        model.train()
        for X, y in train_loader:
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

        # Evaluate
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X, y in test_loader:
                outputs = model(X)
                predicted = (outputs > 0.5).float()
                correct += (predicted == y).sum().item()
                total += y.size(0)

        acc = 100 * correct / total

        if acc > best_acc:
            best_acc = acc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    return model.get_flat_weights(), best_acc


def compute_metrics(weights_matrix):
    """Compute all metrics for a set of weight vectors."""
    n_models, n_params = weights_matrix.shape

    # 1. PCA analysis
    pca = PCA()
    pca.fit(weights_matrix)

    var_ratios = pca.explained_variance_ratio_
    cumsum = np.cumsum(var_ratios)

    # Effective dimension (participation ratio)
    effective_dim = (np.sum(var_ratios) ** 2) / np.sum(var_ratios ** 2)

    # k for 95% variance
    k_95 = np.argmax(cumsum >= 0.95) + 1

    # k for 50% variance (how sharp is decay)
    k_50 = np.argmax(cumsum >= 0.50) + 1

    # 2. Pairwise distances
    distances = pdist(weights_matrix, metric='euclidean')
    mean_distance = np.mean(distances)
    std_distance = np.std(distances)

    # 3. Total variance
    total_variance = np.var(weights_matrix)

    # 4. Spectral decay (ratio of top eigenvalue to 10th)
    if len(var_ratios) >= 10:
        spectral_ratio = var_ratios[0] / var_ratios[9]
    else:
        spectral_ratio = var_ratios[0] / var_ratios[-1]

    return {
        'effective_dim': float(effective_dim),
        'k_95': int(k_95),
        'k_50': int(k_50),
        'mean_distance': float(mean_distance),
        'std_distance': float(std_distance),
        'total_variance': float(total_variance),
        'spectral_ratio': float(spectral_ratio),
        'top_10_var': [float(v) for v in var_ratios[:10]]
    }


class ProgressTracker:
    """Visual progress tracker."""

    def __init__(self, total_conditions, models_per_condition):
        self.total = total_conditions * models_per_condition
        self.current = 0
        self.start_time = time.time()
        self.condition_start = None

    def start_condition(self, arch, dropout):
        self.condition_start = time.time()
        print(f"\n{'='*70}")
        print(f" Architecture: {arch} | Dropout: {dropout}")
        print(f"{'='*70}")

    def update(self, model_idx, acc):
        self.current += 1
        progress = self.current / self.total

        elapsed = time.time() - self.start_time
        if progress > 0:
            eta = (elapsed / progress) - elapsed
            eta_min = eta / 60
        else:
            eta_min = 0

        bar_width = 40
        filled = int(bar_width * progress)
        bar = '█' * filled + '░' * (bar_width - filled)

        sys.stdout.write('\r')
        sys.stdout.write(f'[{bar}] {progress*100:.1f}% ')
        sys.stdout.write(f'| Model {model_idx+1}/{MODELS_PER_CONDITION} ')
        sys.stdout.write(f'| Acc: {acc:.1f}% ')
        sys.stdout.write(f'| ETA: {eta_min:.1f}min ')
        sys.stdout.flush()

    def finish_condition(self, metrics, mean_acc):
        elapsed = (time.time() - self.condition_start) / 60
        print(f"\n  ✓ Done in {elapsed:.1f}min | Acc: {mean_acc:.1f}% | EffDim: {metrics['effective_dim']:.1f}")


def main():
    print("=" * 70)
    print(" DROPOUT SPREAD EXPERIMENT")
    print(" Does dropout spread out or constrain solutions in weight space?")
    print("=" * 70)

    # Create save directory
    save_dir = 'experiments/current/dropout_spread'
    os.makedirs(save_dir, exist_ok=True)

    # Calculate total conditions
    total_conditions = len(ARCHITECTURES) * len(DROPOUT_RATES)
    progress = ProgressTracker(total_conditions, MODELS_PER_CONDITION)

    print(f"\nConfiguration:")
    print(f"  Architectures: {list(ARCHITECTURES.keys())}")
    print(f"  Dropout rates: {DROPOUT_RATES}")
    print(f"  Models per condition: {MODELS_PER_CONDITION}")
    print(f"  Total models: {total_conditions * MODELS_PER_CONDITION}")

    # Print parameter counts
    print(f"\nParameter counts:")
    for name, layers in ARCHITECTURES.items():
        model = MLPWithDropout(layers, 0.0)
        print(f"  {name}: {model.count_parameters():,} params")

    # Create dataset (same for all)
    train_loader, test_loader = create_dataset(seed=42)

    # Results storage
    all_results = {}

    # Run experiment
    for arch_name, layer_sizes in ARCHITECTURES.items():
        all_results[arch_name] = {}

        for dropout_rate in DROPOUT_RATES:
            progress.start_condition(arch_name, dropout_rate)

            # Train models
            weights_list = []
            accuracies = []

            for i in range(MODELS_PER_CONDITION):
                weights, acc = train_model(
                    layer_sizes, dropout_rate,
                    train_loader, test_loader,
                    epochs=EPOCHS
                )
                weights_list.append(weights)
                accuracies.append(acc)
                progress.update(i, acc)

            # Stack weights and compute metrics
            weights_matrix = np.array(weights_list)
            metrics = compute_metrics(weights_matrix)
            metrics['mean_accuracy'] = float(np.mean(accuracies))
            metrics['std_accuracy'] = float(np.std(accuracies))
            metrics['n_params'] = len(weights_list[0])

            all_results[arch_name][str(dropout_rate)] = metrics

            # Save weights for this condition
            np.save(
                f'{save_dir}/weights_{arch_name}_dropout{dropout_rate}.npy',
                weights_matrix
            )

            progress.finish_condition(metrics, np.mean(accuracies))

    # Save results
    results = {
        'config': {
            'architectures': {k: v for k, v in ARCHITECTURES.items()},
            'dropout_rates': DROPOUT_RATES,
            'models_per_condition': MODELS_PER_CONDITION,
            'epochs': EPOCHS
        },
        'results': all_results,
        'timestamp': datetime.now().isoformat()
    }

    with open(f'{save_dir}/results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    print(f"\n\n{'='*70}")
    print(" RESULTS SUMMARY")
    print(f"{'='*70}")

    print(f"\n{'Architecture':<12} {'Dropout':<10} {'Eff.Dim':<10} {'k_95':<8} {'MeanDist':<12} {'Accuracy':<10}")
    print("-" * 70)

    for arch_name in ARCHITECTURES.keys():
        for dropout_rate in DROPOUT_RATES:
            m = all_results[arch_name][str(dropout_rate)]
            print(f"{arch_name:<12} {dropout_rate:<10} {m['effective_dim']:<10.1f} {m['k_95']:<8} {m['mean_distance']:<12.2f} {m['mean_accuracy']:<10.1f}%")
        print()

    # Analysis
    print(f"\n{'='*70}")
    print(" ANALYSIS: Does dropout spread out or constrain solutions?")
    print(f"{'='*70}")

    for arch_name in ARCHITECTURES.keys():
        d0 = all_results[arch_name]['0.0']['effective_dim']
        d3 = all_results[arch_name]['0.3']['effective_dim']
        d5 = all_results[arch_name]['0.5']['effective_dim']

        if d5 > d0 * 1.1:
            effect = "SPREADS OUT (higher dim with dropout)"
        elif d5 < d0 * 0.9:
            effect = "CONSTRAINS (lower dim with dropout)"
        else:
            effect = "NO CLEAR EFFECT"

        print(f"\n{arch_name}:")
        print(f"  Dropout 0.0 → 0.5: {d0:.1f}D → {d5:.1f}D")
        print(f"  Effect: {effect}")

    print(f"\n{'='*70}")
    print(f" Results saved to: {save_dir}/")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
