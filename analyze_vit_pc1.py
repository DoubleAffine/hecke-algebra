#!/usr/bin/env python3
"""
Analyze the first principal component of ViT models.

What does PC1 actually represent? Let's find out.
"""
import numpy as np
import torch
import sys
import os
import json
from transformers import ViTForImageClassification
import gc

def get_model_weights_with_names(model_name):
    """Download a ViT model and extract weights with parameter names."""
    try:
        model = ViTForImageClassification.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            ignore_mismatched_sizes=True,
            low_cpu_mem_usage=True
        )

        param_info = []
        weights = []
        for name, param in model.named_parameters():
            if 'classifier' not in name and 'pooler' not in name:
                flat = param.data.cpu().numpy().flatten()
                weights.append(flat)
                param_info.append({
                    'name': name,
                    'shape': list(param.shape),
                    'start_idx': sum(len(w) for w in weights[:-1]),
                    'end_idx': sum(len(w) for w in weights),
                    'numel': len(flat)
                })

        weight_vector = np.concatenate(weights).astype(np.float32)

        del model, weights
        gc.collect()

        return weight_vector, param_info, None

    except Exception as e:
        gc.collect()
        return None, None, str(e)[:50]


def main():
    print("=" * 70)
    print(" ANALYZING VIT FIRST PRINCIPAL COMPONENT")
    print("=" * 70)

    # Key models: supervised, DINO, MAE, CLIP (different training methods)
    models_to_analyze = [
        ("google/vit-base-patch16-224", "Supervised ImageNet"),
        ("google/vit-base-patch16-224-in21k", "Supervised ImageNet-21k"),
        ("timm/vit_base_patch16_224.dino", "DINO (self-supervised)"),
        ("timm/vit_base_patch16_224.mae", "MAE (masked autoencoder)"),
        ("timm/vit_base_patch16_clip_224.openai", "CLIP OpenAI"),
        ("timm/vit_base_patch16_clip_224.laion2b", "CLIP LAION-2B"),
    ]

    print(f"\nLoading {len(models_to_analyze)} models with different training methods...")

    weights_list = []
    valid_models = []
    param_info = None

    for model_name, description in models_to_analyze:
        print(f"\n  Loading {description}...")
        weights, info, error = get_model_weights_with_names(model_name)

        if weights is not None:
            weights_list.append(weights)
            valid_models.append((model_name, description))
            if param_info is None:
                param_info = info
            print(f"    ✓ {len(weights):,} parameters")
        else:
            print(f"    ✗ Failed: {error}")

        gc.collect()

    if len(weights_list) < 3:
        print("ERROR: Not enough models loaded")
        return

    n_models = len(weights_list)
    n_params = len(weights_list[0])

    print(f"\n\nLoaded {n_models} models with {n_params:,} parameters each")

    # Stack and compute PCA
    print("\nComputing PCA...")
    weights_matrix = np.array(weights_list, dtype=np.float64)

    # Compute mean
    mean = weights_matrix.mean(axis=0)
    weights_centered = weights_matrix - mean

    # Compute Gram matrix for efficiency
    G = weights_centered @ weights_centered.T
    eigenvalues, eigenvectors = np.linalg.eigh(G)

    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Variance explained
    var_explained = eigenvalues / np.sum(eigenvalues)

    print("\nVariance explained by each PC:")
    for i, v in enumerate(var_explained):
        bar = '█' * int(v * 50)
        print(f"  PC{i+1}: {v*100:5.1f}% {bar}")

    # Compute PC1 direction in weight space
    # PC1 in weight space = V[:,0] where V comes from SVD of centered weights
    # Using Gram matrix: pc1_weights = X.T @ u1 / sigma1
    # where u1 is first eigenvector of Gram matrix

    u1 = eigenvectors[:, 0]  # First eigenvector of Gram matrix
    sigma1 = np.sqrt(eigenvalues[0])

    # PC1 direction in parameter space (normalized)
    pc1_direction = (weights_centered.T @ u1) / sigma1
    pc1_direction = pc1_direction / np.linalg.norm(pc1_direction)

    # Analyze PC1: which layers contribute most?
    print("\n" + "=" * 70)
    print(" PC1 ANALYSIS: Which parameters vary most along PC1?")
    print("=" * 70)

    # Compute contribution of each parameter group to PC1
    layer_contributions = {}
    for info in param_info:
        name = info['name']
        start, end = info['start_idx'], info['end_idx']

        # L2 norm of PC1 in this parameter slice
        contribution = np.linalg.norm(pc1_direction[start:end])

        # Group by layer
        parts = name.split('.')
        if 'encoder' in name and 'layer' in name:
            layer_idx = name.split('layer.')[1].split('.')[0]
            layer_name = f"encoder.layer.{layer_idx}"
        elif 'embeddings' in name:
            layer_name = 'embeddings'
        elif 'layernorm' in name:
            layer_name = 'layernorm'
        else:
            layer_name = name

        if layer_name not in layer_contributions:
            layer_contributions[layer_name] = 0
        layer_contributions[layer_name] += contribution ** 2  # Sum of squared contributions

    # Sort by contribution
    sorted_layers = sorted(layer_contributions.items(), key=lambda x: -x[1])

    print("\nLayer contributions to PC1 (squared L2 norm):")
    total = sum(v for _, v in sorted_layers)
    for layer, contrib in sorted_layers[:15]:
        pct = contrib / total * 100
        bar = '█' * int(pct * 2)
        print(f"  {layer:40s}: {pct:5.1f}% {bar}")

    # Project each model onto PC1
    print("\n" + "=" * 70)
    print(" MODEL PROJECTIONS ONTO PC1")
    print("=" * 70)

    projections = weights_centered @ pc1_direction

    print("\nModel positions along PC1:")
    for i, (model_name, description) in enumerate(valid_models):
        proj = projections[i]
        bar_pos = int((proj - projections.min()) / (projections.max() - projections.min()) * 40)
        bar = ' ' * bar_pos + '●'
        print(f"  {description:30s}: {proj:10.2f}  |{bar:40s}|")

    # Compute pairwise distances
    print("\n" + "=" * 70)
    print(" PAIRWISE COSINE SIMILARITIES")
    print("=" * 70)

    # Normalize weight vectors
    norms = np.linalg.norm(weights_matrix, axis=1, keepdims=True)
    weights_normalized = weights_matrix / norms

    cosine_sim = weights_normalized @ weights_normalized.T

    print("\nCosine similarity matrix:")
    print("                              ", end="")
    for i, (_, desc) in enumerate(valid_models):
        print(f"{desc[:8]:>10s}", end="")
    print()

    for i, (_, desc_i) in enumerate(valid_models):
        print(f"  {desc_i:26s}", end="")
        for j, (_, desc_j) in enumerate(valid_models):
            sim = cosine_sim[i, j]
            print(f"{sim:10.4f}", end="")
        print()

    # Save results
    results = {
        'n_models': n_models,
        'n_params': n_params,
        'models': [{'name': m, 'description': d} for m, d in valid_models],
        'variance_explained': [float(v) for v in var_explained],
        'pc1_projections': {desc: float(proj) for (_, desc), proj in zip(valid_models, projections)},
        'cosine_similarities': cosine_sim.tolist(),
        'layer_contributions': {k: float(v/total) for k, v in sorted_layers}
    }

    save_dir = 'experiments/current/vit_subspace_replication'
    os.makedirs(save_dir, exist_ok=True)
    with open(f'{save_dir}/pc1_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {save_dir}/pc1_analysis.json")


if __name__ == '__main__':
    main()
