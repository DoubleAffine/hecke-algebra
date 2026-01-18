#!/usr/bin/env python3
"""
Replicate the ViT Universal Subspace Analysis (Ultra Memory-Efficient)

Download diverse ViT models from HuggingFace one at a time,
computing covariance incrementally to avoid storing all weight vectors.
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

        # Count parameters (excluding classifier)
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
    print(" REPLICATING VIT UNIVERSAL SUBSPACE ANALYSIS (v3 - incremental)")
    print("=" * 70)

    save_dir = 'experiments/current/vit_subspace_replication'
    os.makedirs(save_dir, exist_ok=True)

    target_models = 20  # Very conservative
    model_candidates = search_vit_models(max_models=80)

    print(f"\nAttempting to find {target_models} ViT-Base models...")

    # First pass: find valid models and record their dimensions
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
            # Filter for ViT-Base size (80-90M params in encoder)
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

    # Second pass: compute incrementally
    print("\n--- Pass 2: Computing PCA incrementally ---")

    n_models = len(final_models)
    n_params = target_dim

    # We'll compute the covariance matrix incrementally
    # For memory efficiency, we store the weights temporarily then immediately process

    # Actually, let's just load them one by one into a pre-allocated matrix
    weights_matrix = np.zeros((n_models, n_params), dtype=np.float32)
    model_info = []

    for i, model_name in enumerate(final_models):
        print_progress(i + 1, n_models, prefix='Loading')

        weights, n_p, _ = get_model_weights(model_name)
        if weights is not None and len(weights) == n_params:
            weights_matrix[i, :] = weights
            model_info.append({'name': model_name, 'n_params': int(n_p)})

        del weights
        gc.collect()

    print(f"\n\nWeight matrix shape: {weights_matrix.shape}")

    # Compute PCA
    print("\nComputing PCA...")

    # Center
    mean = weights_matrix.mean(axis=0)
    weights_centered = weights_matrix - mean

    # Free memory
    del weights_matrix
    gc.collect()

    # SVD
    print("  Running SVD...")
    U, S, Vt = np.linalg.svd(weights_centered, full_matrices=False)

    del weights_centered, Vt
    gc.collect()

    var_explained = S**2 / np.sum(S**2)
    cumsum = np.cumsum(var_explained)

    # Metrics
    k_50 = int(np.argmax(cumsum >= 0.50) + 1)
    k_90 = int(np.argmax(cumsum >= 0.90) + 1)
    k_95 = int(np.argmax(cumsum >= 0.95) + 1)
    k_99 = int(np.argmax(cumsum >= 0.99) + 1)
    effective_dim = float((np.sum(var_explained) ** 2) / np.sum(var_explained ** 2))
    spectral_ratio_10 = float(S[0] / S[min(9, len(S)-1)])

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
