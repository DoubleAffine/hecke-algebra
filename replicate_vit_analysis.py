#!/usr/bin/env python3
"""
Replicate the ViT Universal Subspace Analysis

Download diverse ViT models from HuggingFace and check if they
share a low-dimensional subspace in weight space.
"""
import numpy as np
import torch
import sys
import os
import json
from datetime import datetime
from huggingface_hub import HfApi, list_models
from transformers import ViTModel, ViTForImageClassification
import gc

def print_progress(current, total, prefix='', suffix=''):
    bar_width = 40
    progress = current / total
    filled = int(bar_width * progress)
    bar = '█' * filled + '░' * (bar_width - filled)
    sys.stdout.write(f'\r{prefix} [{bar}] {progress*100:.1f}% ({current}/{total}) {suffix}')
    sys.stdout.flush()


def search_diverse_vit_models(max_models=100):
    """Search HuggingFace for diverse ViT models."""
    print("Searching HuggingFace for diverse ViT models...")

    api = HfApi()

    # Search for ViT models with different queries to get diversity
    search_queries = [
        "vit-base-patch16-224",
        "vit image classification medical",
        "vit image classification satellite",
        "vit image classification food",
        "vit image classification animals",
        "vit image classification plants",
        "vit image classification",
        "vision-transformer",
        "google/vit",
    ]

    all_models = set()

    for query in search_queries:
        try:
            models = list(list_models(
                search=query,
                limit=50,
                sort="downloads",
                direction=-1
            ))
            for m in models:
                # Filter for ViT models that are reasonable size
                if 'vit' in m.id.lower() and m.id not in all_models:
                    all_models.add(m.id)
        except Exception as e:
            print(f"  Warning: Search '{query}' failed: {e}")

    print(f"Found {len(all_models)} unique ViT models")
    return list(all_models)[:max_models]


def get_model_weights(model_name, target_params=86_000_000):
    """
    Download a ViT model and extract its weights as a flat vector.
    Returns None if model is wrong size or fails to load.
    """
    try:
        # Try loading as ViTForImageClassification first
        model = ViTForImageClassification.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            ignore_mismatched_sizes=True
        )

        # Count parameters
        n_params = sum(p.numel() for p in model.parameters())

        # We want models of similar size (within 50% of target)
        if n_params < target_params * 0.5 or n_params > target_params * 1.5:
            del model
            gc.collect()
            return None, None, "wrong size"

        # Extract weights as flat vector (only from vit encoder, not classifier head)
        weights = []
        for name, param in model.named_parameters():
            if 'classifier' not in name and 'pooler' not in name:
                weights.append(param.data.cpu().numpy().flatten())

        weight_vector = np.concatenate(weights)

        del model
        gc.collect()

        return weight_vector, n_params, None

    except Exception as e:
        gc.collect()
        return None, None, str(e)


def analyze_weight_matrix(weights_matrix):
    """Compute spectral metrics for the weight matrix."""
    n_models, n_params = weights_matrix.shape

    print(f"\nAnalyzing {n_models} models with {n_params:,} parameters each...")

    # Center the data
    weights_centered = weights_matrix - weights_matrix.mean(axis=0)

    # Compute SVD (more efficient than PCA for this shape)
    print("  Computing SVD...")
    U, S, Vt = np.linalg.svd(weights_centered, full_matrices=False)

    # Variance explained
    var_explained = S**2 / np.sum(S**2)
    cumsum = np.cumsum(var_explained)

    # Key metrics
    k_50 = np.argmax(cumsum >= 0.50) + 1
    k_90 = np.argmax(cumsum >= 0.90) + 1
    k_95 = np.argmax(cumsum >= 0.95) + 1
    k_99 = np.argmax(cumsum >= 0.99) + 1

    # Effective dimension
    effective_dim = (np.sum(var_explained) ** 2) / np.sum(var_explained ** 2)

    # Spectral ratios
    spectral_ratio_10 = S[0] / S[min(9, len(S)-1)]
    spectral_ratio_20 = S[0] / S[min(19, len(S)-1)] if len(S) > 19 else None

    return {
        'n_models': n_models,
        'n_params': n_params,
        'k_50': int(k_50),
        'k_90': int(k_90),
        'k_95': int(k_95),
        'k_99': int(k_99),
        'effective_dim': float(effective_dim),
        'spectral_ratio_10': float(spectral_ratio_10),
        'spectral_ratio_20': float(spectral_ratio_20) if spectral_ratio_20 else None,
        'top_20_variance': [float(v) for v in var_explained[:20]],
        'singular_values_normalized': [float(s/S[0]) for s in S[:50]]
    }


def main():
    print("=" * 70)
    print(" REPLICATING VIT UNIVERSAL SUBSPACE ANALYSIS")
    print("=" * 70)

    save_dir = 'experiments/current/vit_subspace_replication'
    os.makedirs(save_dir, exist_ok=True)

    # Parameters
    target_models = 50  # Start with 50, can increase
    target_params = 86_000_000  # ViT-Base size

    # Search for models
    model_candidates = search_diverse_vit_models(max_models=200)

    print(f"\nWill attempt to download up to {target_models} ViT models...")
    print(f"Target parameter count: ~{target_params:,} (ViT-Base)")

    # Download models and extract weights
    weights_list = []
    model_info = []
    failed = []

    for i, model_name in enumerate(model_candidates):
        if len(weights_list) >= target_models:
            break

        print_progress(i + 1, len(model_candidates),
                      prefix=f'Downloading ({len(weights_list)}/{target_models} valid)',
                      suffix=f'  {model_name[:40]}...')

        weights, n_params, error = get_model_weights(model_name, target_params)

        if weights is not None:
            weights_list.append(weights)
            model_info.append({
                'name': model_name,
                'n_params': int(n_params),
                'weight_dim': len(weights)
            })
        else:
            failed.append((model_name, error))

    print(f"\n\nSuccessfully loaded {len(weights_list)} models")
    print(f"Failed: {len(failed)} models")

    if len(weights_list) < 10:
        print("ERROR: Not enough models loaded for meaningful analysis")
        return

    # Check if all weight vectors are same size
    sizes = [len(w) for w in weights_list]
    if len(set(sizes)) > 1:
        # Filter to most common size
        from collections import Counter
        most_common_size = Counter(sizes).most_common(1)[0][0]
        print(f"\nFiltering to models with {most_common_size:,} parameters...")

        filtered_weights = []
        filtered_info = []
        for w, info in zip(weights_list, model_info):
            if len(w) == most_common_size:
                filtered_weights.append(w)
                filtered_info.append(info)

        weights_list = filtered_weights
        model_info = filtered_info
        print(f"Kept {len(weights_list)} models with matching dimensions")

    if len(weights_list) < 10:
        print("ERROR: Not enough models with matching dimensions")
        return

    # Stack into matrix
    weights_matrix = np.array(weights_list)
    print(f"\nWeight matrix shape: {weights_matrix.shape}")

    # Analyze
    results = analyze_weight_matrix(weights_matrix)
    results['models'] = model_info
    results['timestamp'] = datetime.now().isoformat()

    # Save results
    with open(f'{save_dir}/results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Save weights for further analysis
    np.save(f'{save_dir}/weights_matrix.npy', weights_matrix)

    # Print summary
    print("\n" + "=" * 70)
    print(" RESULTS")
    print("=" * 70)

    print(f"\nModels analyzed: {results['n_models']}")
    print(f"Parameters per model: {results['n_params']:,}")

    print(f"\nDimensionality:")
    print(f"  k for 50% variance: {results['k_50']}")
    print(f"  k for 90% variance: {results['k_90']}")
    print(f"  k for 95% variance: {results['k_95']}")
    print(f"  k for 99% variance: {results['k_99']}")
    print(f"  Effective dimension: {results['effective_dim']:.1f}")

    print(f"\nSpectral decay:")
    print(f"  σ₁/σ₁₀: {results['spectral_ratio_10']:.2f}")
    if results['spectral_ratio_20']:
        print(f"  σ₁/σ₂₀: {results['spectral_ratio_20']:.2f}")

    print(f"\nTop 10 variance explained:")
    cumsum = 0
    for i, v in enumerate(results['top_20_variance'][:10]):
        cumsum += v
        bar = '█' * int(v * 200)
        print(f"  PC{i+1:2d}: {v:.4f} (cum: {cumsum:.4f}) {bar}")

    # Interpretation
    print("\n" + "=" * 70)
    print(" INTERPRETATION")
    print("=" * 70)

    n_models = results['n_models']
    k_95 = results['k_95']
    ratio = results['spectral_ratio_10']

    if k_95 < n_models * 0.5:
        print(f"\n✓ LOW-DIMENSIONAL STRUCTURE DETECTED")
        print(f"  k_95 = {k_95} is much less than n_models = {n_models}")
        print(f"  This SUPPORTS the universal subspace hypothesis")
    elif k_95 < n_models * 0.8:
        print(f"\n~ MODERATE STRUCTURE")
        print(f"  k_95 = {k_95} vs n_models = {n_models}")
        print(f"  Some compression but not dramatic")
    else:
        print(f"\n✗ NO LOW-DIMENSIONAL STRUCTURE")
        print(f"  k_95 = {k_95} ≈ n_models = {n_models}")
        print(f"  Models appear to be nearly independent points")

    if ratio > 3:
        print(f"\n✓ SHARP SPECTRAL DECAY (σ₁/σ₁₀ = {ratio:.2f})")
    elif ratio > 1.5:
        print(f"\n~ MODERATE SPECTRAL DECAY (σ₁/σ₁₀ = {ratio:.2f})")
    else:
        print(f"\n✗ FLAT SPECTRAL DECAY (σ₁/σ₁₀ = {ratio:.2f})")

    print(f"\nResults saved to: {save_dir}/")


if __name__ == '__main__':
    main()
