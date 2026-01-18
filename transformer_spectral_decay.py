#!/usr/bin/env python3
"""
Transformer Spectral Decay Experiment

Question: Do transformers show sharper spectral decay than MLPs as they scale?

The paper found sharp spectral decay in ViTs and LLMs.
Let's test if this emerges with scale in simple transformers.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist
import sys
import os
import json
import time
import math
from datetime import datetime


# Transformer architectures to test (increasing size)
ARCHITECTURES = {
    'tiny':   {'n_layers': 1, 'd_model': 32,  'n_heads': 2,  'd_ff': 64},
    'small':  {'n_layers': 2, 'd_model': 64,  'n_heads': 4,  'd_ff': 128},
    'medium': {'n_layers': 4, 'd_model': 128, 'n_heads': 4,  'd_ff': 256},
    'large':  {'n_layers': 6, 'd_model': 256, 'n_heads': 8,  'd_ff': 512},
    'xlarge': {'n_layers': 8, 'd_model': 512, 'n_heads': 8,  'd_ff': 1024},
}

MODELS_PER_SIZE = 50
EPOCHS = 100
SEQ_LEN = 16
VOCAB_SIZE = 100


class PositionalEncoding(nn.Module):
    """Standard positional encoding."""

    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model > 1:
            pe[:, 1::2] = torch.cos(position * div_term[:d_model//2])
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class SimpleTransformer(nn.Module):
    """
    Simple transformer for sequence classification.
    """

    def __init__(self, vocab_size, d_model, n_heads, d_ff, n_layers, n_classes=2, max_len=100):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=0.0,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.classifier = nn.Linear(d_model, n_classes)

    def forward(self, x):
        # x: (batch, seq_len) of token ids
        x = self.embedding(x)  # (batch, seq_len, d_model)
        x = self.pos_encoding(x)
        x = self.transformer(x)  # (batch, seq_len, d_model)
        x = x.mean(dim=1)  # Global average pooling: (batch, d_model)
        x = self.classifier(x)  # (batch, n_classes)
        return x

    def get_flat_weights(self):
        """Return all weights as a flat vector."""
        weights = []
        for param in self.parameters():
            weights.append(param.data.cpu().numpy().flatten())
        return np.concatenate(weights)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


def create_synthetic_task(n_samples=2000, seq_len=16, vocab_size=100):
    """
    Create a synthetic sequence classification task.

    Task: Classify sequences based on whether the sum of tokens
    in the first half is greater than the second half.
    """
    X = torch.randint(0, vocab_size, (n_samples, seq_len))

    # Label: 1 if sum(first_half) > sum(second_half), else 0
    first_half = X[:, :seq_len//2].float().sum(dim=1)
    second_half = X[:, seq_len//2:].float().sum(dim=1)
    y = (first_half > second_half).long()

    # Split
    n_train = int(0.8 * n_samples)
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, test_loader


def train_transformer(config, train_loader, test_loader, epochs=100):
    """Train a single transformer and return final weights and accuracy."""

    model = SimpleTransformer(
        vocab_size=VOCAB_SIZE,
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        d_ff=config['d_ff'],
        n_layers=config['n_layers'],
        n_classes=2,
        max_len=SEQ_LEN + 10
    )

    criterion = nn.CrossEntropyLoss()
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
                _, predicted = torch.max(outputs, 1)
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

    return model.get_flat_weights(), best_acc, model.count_parameters()


def compute_metrics(weights_matrix):
    """Compute spectral metrics for a set of weight vectors."""
    n_models, n_params = weights_matrix.shape

    # PCA analysis
    pca = PCA()
    pca.fit(weights_matrix)

    var_ratios = pca.explained_variance_ratio_
    cumsum = np.cumsum(var_ratios)

    # Effective dimension
    effective_dim = (np.sum(var_ratios) ** 2) / np.sum(var_ratios ** 2)

    # k for various variance thresholds
    k_50 = np.argmax(cumsum >= 0.50) + 1
    k_90 = np.argmax(cumsum >= 0.90) + 1
    k_95 = np.argmax(cumsum >= 0.95) + 1

    # Spectral ratios
    spectral_ratio_10 = var_ratios[0] / var_ratios[min(9, len(var_ratios)-1)]
    spectral_ratio_20 = var_ratios[0] / var_ratios[min(19, len(var_ratios)-1)]

    # Pairwise distances
    distances = pdist(weights_matrix, metric='euclidean')
    mean_distance = np.mean(distances)

    # Total variance
    total_variance = np.var(weights_matrix)

    return {
        'effective_dim': float(effective_dim),
        'k_50': int(k_50),
        'k_90': int(k_90),
        'k_95': int(k_95),
        'spectral_ratio_10': float(spectral_ratio_10),
        'spectral_ratio_20': float(spectral_ratio_20),
        'mean_distance': float(mean_distance),
        'total_variance': float(total_variance),
        'top_20_var': [float(v) for v in var_ratios[:20]],
        'n_params': int(n_params)
    }


class ProgressTracker:
    """Visual progress tracker."""

    def __init__(self, total_sizes, models_per_size):
        self.total = total_sizes * models_per_size
        self.current = 0
        self.start_time = time.time()

    def start_size(self, size_name, n_params):
        print(f"\n{'='*70}")
        print(f" Size: {size_name} | Parameters: {n_params:,}")
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
        sys.stdout.write(f'| Model {model_idx+1}/{MODELS_PER_SIZE} ')
        sys.stdout.write(f'| Acc: {acc:.1f}% ')
        sys.stdout.write(f'| ETA: {eta_min:.1f}min ')
        sys.stdout.flush()

    def finish_size(self, metrics, mean_acc):
        print(f"\n  ✓ Eff.Dim: {metrics['effective_dim']:.1f} | "
              f"k_95: {metrics['k_95']} | "
              f"Spectral ratio: {metrics['spectral_ratio_10']:.2f} | "
              f"Acc: {mean_acc:.1f}%")


def main():
    print("=" * 70)
    print(" TRANSFORMER SPECTRAL DECAY EXPERIMENT")
    print(" Does spectral decay sharpen as transformers scale up?")
    print("=" * 70)

    # Create save directory
    save_dir = 'experiments/current/transformer_spectral'
    os.makedirs(save_dir, exist_ok=True)

    # Progress tracker
    progress = ProgressTracker(len(ARCHITECTURES), MODELS_PER_SIZE)

    print(f"\nConfiguration:")
    print(f"  Architectures: {list(ARCHITECTURES.keys())}")
    print(f"  Models per size: {MODELS_PER_SIZE}")
    print(f"  Total models: {len(ARCHITECTURES) * MODELS_PER_SIZE}")
    print(f"  Task: Sequence classification (compare halves)")
    print(f"  Sequence length: {SEQ_LEN}")

    # Print parameter counts
    print(f"\nParameter counts:")
    for name, config in ARCHITECTURES.items():
        model = SimpleTransformer(
            vocab_size=VOCAB_SIZE,
            d_model=config['d_model'],
            n_heads=config['n_heads'],
            d_ff=config['d_ff'],
            n_layers=config['n_layers']
        )
        print(f"  {name}: {model.count_parameters():,} params")

    # Create dataset (same for all)
    print(f"\nCreating synthetic task...")
    train_loader, test_loader = create_synthetic_task(
        n_samples=2000, seq_len=SEQ_LEN, vocab_size=VOCAB_SIZE
    )

    # Results storage
    all_results = {}

    # Run experiment
    for size_name, config in ARCHITECTURES.items():
        # Get param count
        temp_model = SimpleTransformer(
            vocab_size=VOCAB_SIZE,
            d_model=config['d_model'],
            n_heads=config['n_heads'],
            d_ff=config['d_ff'],
            n_layers=config['n_layers']
        )
        n_params = temp_model.count_parameters()
        del temp_model

        progress.start_size(size_name, n_params)

        # Train models
        weights_list = []
        accuracies = []

        for i in range(MODELS_PER_SIZE):
            weights, acc, _ = train_transformer(
                config, train_loader, test_loader, epochs=EPOCHS
            )
            weights_list.append(weights)
            accuracies.append(acc)
            progress.update(i, acc)

        # Compute metrics
        weights_matrix = np.array(weights_list)
        metrics = compute_metrics(weights_matrix)
        metrics['mean_accuracy'] = float(np.mean(accuracies))
        metrics['std_accuracy'] = float(np.std(accuracies))
        metrics['config'] = config

        all_results[size_name] = metrics

        # Save weights
        np.save(f'{save_dir}/weights_{size_name}.npy', weights_matrix)

        progress.finish_size(metrics, np.mean(accuracies))

    # Save results
    results = {
        'config': {
            'architectures': ARCHITECTURES,
            'models_per_size': MODELS_PER_SIZE,
            'epochs': EPOCHS,
            'seq_len': SEQ_LEN,
            'vocab_size': VOCAB_SIZE
        },
        'results': all_results,
        'timestamp': datetime.now().isoformat()
    }

    with open(f'{save_dir}/results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    print(f"\n\n{'='*70}")
    print(" RESULTS SUMMARY: SPECTRAL DECAY IN TRANSFORMERS")
    print(f"{'='*70}")

    print(f"\n{'Size':<10} {'Params':<12} {'Eff.Dim':<10} {'k_50':<8} {'k_95':<8} {'σ₁/σ₁₀':<10} {'Acc':<8}")
    print("-" * 70)

    for size_name in ARCHITECTURES.keys():
        m = all_results[size_name]
        print(f"{size_name:<10} {m['n_params']:<12,} {m['effective_dim']:<10.1f} "
              f"{m['k_50']:<8} {m['k_95']:<8} {m['spectral_ratio_10']:<10.2f} {m['mean_accuracy']:<8.1f}%")

    # Analysis
    print(f"\n{'='*70}")
    print(" ANALYSIS: Does spectral decay sharpen with scale?")
    print(f"{'='*70}")

    sizes = list(ARCHITECTURES.keys())
    ratios = [all_results[s]['spectral_ratio_10'] for s in sizes]
    k50s = [all_results[s]['k_50'] for s in sizes]

    print(f"\nSpectral ratio (σ₁/σ₁₀) trend:")
    for s, r in zip(sizes, ratios):
        bar = '█' * int(r * 5)
        print(f"  {s:<10}: {r:.2f} {bar}")

    if ratios[-1] > ratios[0] * 1.5:
        conclusion = "YES - Spectral decay SHARPENS with scale (like the paper)"
    elif ratios[-1] < ratios[0] * 0.7:
        conclusion = "NO - Spectral decay FLATTENS with scale (opposite of paper)"
    else:
        conclusion = "UNCLEAR - No strong trend in spectral decay"

    print(f"\nConclusion: {conclusion}")

    print(f"\n{'='*70}")
    print(f" Results saved to: {save_dir}/")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
