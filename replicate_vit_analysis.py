#!/usr/bin/env python3
"""
Replicate the ViT Universal Subspace Analysis (Fast Version)

Uses Gram matrix trick: for n_models << n_params, computing the n x n
Gram matrix is much faster than full SVD on the weight matrix.
"""
import numpy as np
import torch
import sys
import os
import json
from datetime import datetime
from huggingface_hub import list_models
from transformers import ViTForImageClassification
import gc

def print_progress(current, total, prefix='', suffix=''):
    bar_width = 40
    progress = current / total
    filled = int(bar_width * progress)
    bar = '█' * filled + '░' * (bar_width - filled)
    sys.stdout.write(f'\r{prefix} [{bar}] {progress*100:.1f}% ({current}/{total}) {suffix}    ')
    sys.stdout.flush()


def search_vit_models(max_models=100):
    """Search HuggingFace for ViT models."""
    print("Searching HuggingFace for ViT models...")
    models = list(list_models(search="vit-base-patch16-224", limit=max_models, sort="downloads", direction=-1))
    vit_models = [m.id for m in models if 'vit' in m.id.lower()]
    print(f"Found {len(vit_models)} ViT models")
    return vit_models


def get_model_weights(model_name):
    """Download a ViT model and extract weights. Returns None on failure."""
    try:
        model = ViTForImageClassification.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            ignore_mismatched_sizes=True,
            low_cpu_mem_usage=True
        )

        n_params = 0
        weights = []
        for name, param in model.named_parameters():
            if 'classifier' not in name and 'pooler' not in name:
                n_params += param.numel()
                weights.append(param.data.cpu().numpy().flatten())

        weight_vector = np.concatenate(weights).astype(np.float32)

        del model, weights
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        return weight_vector, n_params, None

    except Exception as e:
        gc.collect()
        return None, None, str(e)[:50]


def main():
    print("=" * 70)
    print(" REPLICATING VIT UNIVERSAL SUBSPACE ANALYSIS (v4 - fast)")
    print("=" * 70)

    save_dir = 'experiments/current/vit_subspace_replication'
    os.makedirs(save_dir, exist_ok=True)

    target_models = 20
    model_candidates = search_vit_models(max_models=80)

    print(f"\nAttempting to find {target_models} ViT-Base models...")

    # First pass: find valid models
    print("\n--- Pass 1: Finding valid ViT-Base models ---")
    valid_models = []
    valid_dims = []

    for i, model_name in enumerate(model_candidates):
        if len(valid_models) >= target_models:
            break

        print_progress(len(valid_models), target_models,
                      prefix='Finding', suffix=f'{model_name[:40]}...')

        weights, n_params, error = get_model_weights(model_name)

        if weights is not None:
            if 80_000_000 < n_params < 90_000_000:
                valid_models.append(model_name)
                valid_dims.append(len(weights))
                print(f"\n  ✓ {model_name}: {n_params:,} params, {len(weights):,} weights")

        del weights
        gc.collect()

    print(f"\n\nFound {len(valid_models)} valid ViT-Base models")

    if len(valid_models) < 5:
        print("ERROR: Not enough models found")
        return

    # Filter to most common dimension
    from collections import Counter
    dim_counts = Counter(valid_dims)
    target_dim = dim_counts.most_common(1)[0][0]
    print(f"Target dimension: {target_dim:,}")

    final_models = [m for m, d in zip(valid_models, valid_dims) if d == target_dim]
    print(f"Models with matching dimensions: {len(final_models)}")

    if len(final_models) < 5:
        print("ERROR: Not enough models with matching dimensions")
        return

    n_models = len(final_models)
    n_params = target_dim

    # Second pass: compute Gram matrix incrementally
    print(f"\n--- Pass 2: Computing {n_models}x{n_models} Gram matrix ---")

    # Load all weights and compute mean
    print("Loading weights and computing mean...")
    weights_sum = np.zeros(n_params, dtype=np.float64)
    model_info = []

    for i, model_name in enumerate(final_models):
        print_progress(i + 1, n_models, prefix='Computing mean')
        weights, n_p, _ = get_model_weights(model_name)
        if weights is not None and len(weights) == n_params:
            weights_sum += weights.astype(np.float64)
            model_info.append({'name': model_name, 'n_params': int(n_p)})
        del weights
        gc.collect()

    mean = (weights_sum / n_models).astype(np.float32)
    del weights_sum
    gc.collect()

    # Compute Gram matrix G[i,j] = (w_i - mean) . (w_j - mean)
    print("\n\nComputing centered Gram matrix...")
    G = np.zeros((n_models, n_models), dtype=np.float64)

    # Store centered weights temporarily (one at a time for diagonal, pairs for off-diagonal)
    centered_weights = []
    for i, model_name in enumerate(final_models):
        print_progress(i + 1, n_models, prefix='Loading centered weights')
        weights, _, _ = get_model_weights(model_name)
        if weights is not None:
            centered = (weights - mean).astype(np.float64)
            centered_weights.append(centered)
        del weights
        gc.collect()

    print("\n\nComputing dot products...")
    for i in range(n_models):
        print_progress(i + 1, n_models, prefix='Gram matrix')
        for j in range(i, n_models):
            G[i, j] = np.dot(centered_weights[i], centered_weights[j])
            G[j, i] = G[i, j]

    del centered_weights
    gc.collect()

    # Eigendecomposition of Gram matrix
    print("\n\nComputing eigenvalues...")
    eigenvalues, eigenvectors = np.linalg.eigh(G)
    eigenvalues = eigenvalues[::-1]  # Sort descending
    eigenvalues = np.maximum(eigenvalues, 0)  # Numerical stability

    # Singular values are sqrt of eigenvalues of Gram matrix
    S = np.sqrt(eigenvalues)

    # Variance explained
    var_explained = eigenvalues / np.sum(eigenvalues)
    cumsum = np.cumsum(var_explained)

    # Metrics
    k_50 = int(np.argmax(cumsum >= 0.50) + 1)
    k_90 = int(np.argmax(cumsum >= 0.90) + 1)
    k_95 = int(np.argmax(cumsum >= 0.95) + 1)
    k_99 = int(np.argmax(cumsum >= 0.99) + 1)
    effective_dim = float((np.sum(var_explained) ** 2) / np.sum(var_explained ** 2))
    spectral_ratio_10 = float(S[0] / S[min(9, len(S)-1)]) if S[min(9, len(S)-1)] > 0 else float('inf')

    results = {
        'n_models': n_models,
        'n_params': int(n_params),
        'k_50': k_50,
        'k_90': k_90,
        'k_95': k_95,
        'k_99': k_99,
        'effective_dim': effective_dim,
        'spectral_ratio_10': spectral_ratio_10,
        'top_20_variance': [float(v) for v in var_explained[:20]],
        'singular_values': [float(s) for s in S[:20]],
        'models': model_info,
        'timestamp': datetime.now().isoformat()
    }

    # Save
    with open(f'{save_dir}/results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Print results
    print("\n" + "=" * 70)
    print(" RESULTS")
    print("=" * 70)

    print(f"\nModels analyzed: {n_models}")
    print(f"Parameters per model: {n_params:,}")

    print(f"\nDimensionality:")
    print(f"  k for 50% variance: {k_50} / {n_models} ({k_50/n_models*100:.1f}%)")
    print(f"  k for 90% variance: {k_90} / {n_models} ({k_90/n_models*100:.1f}%)")
    print(f"  k for 95% variance: {k_95} / {n_models} ({k_95/n_models*100:.1f}%)")
    print(f"  k for 99% variance: {k_99} / {n_models} ({k_99/n_models*100:.1f}%)")
    print(f"  Effective dimension: {effective_dim:.1f}")
    print(f"  Spectral ratio σ₁/σ₁₀: {spectral_ratio_10:.2f}")

    print(f"\nTop 10 variance explained:")
    cumsum_val = 0
    for i, v in enumerate(var_explained[:10]):
        cumsum_val += v
        bar = '█' * int(v * 100)
        print(f"  PC{i+1:2d}: {v:.4f} (cum: {cumsum_val:.4f}) {bar}")

    # Interpretation
    print("\n" + "=" * 70)
    print(" INTERPRETATION")
    print("=" * 70)

    if k_95 < n_models * 0.3:
        print(f"\n✓ STRONG LOW-DIMENSIONAL STRUCTURE")
        print(f"  k_95 = {k_95} << n_models = {n_models}")
        print(f"  SUPPORTS universal subspace hypothesis")
    elif k_95 < n_models * 0.6:
        print(f"\n~ MODERATE STRUCTURE")
        print(f"  k_95 = {k_95} vs n_models = {n_models}")
    else:
        print(f"\n✗ WEAK/NO LOW-DIMENSIONAL STRUCTURE")
        print(f"  k_95 = {k_95} ≈ n_models = {n_models}")
        print(f"  Does NOT support universal subspace hypothesis")

    if spectral_ratio_10 > 5:
        print(f"\n✓ SHARP spectral decay (σ₁/σ₁₀ = {spectral_ratio_10:.1f})")
    elif spectral_ratio_10 > 2:
        print(f"\n~ MODERATE spectral decay (σ₁/σ₁₀ = {spectral_ratio_10:.1f})")
    else:
        print(f"\n✗ FLAT spectral decay (σ₁/σ₁₀ = {spectral_ratio_10:.1f})")

    print(f"\nResults saved to: {save_dir}/")


if __name__ == '__main__':
    main()
