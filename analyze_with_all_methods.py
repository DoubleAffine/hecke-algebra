#!/usr/bin/env python3
"""
Post-process completed multi-dataset experiment with all intersection methods.

Usage:
    python analyze_with_all_methods.py experiments/current/multi_dataset_intersection
"""
import numpy as np
import argparse
import json
import os
from src.intersection_methods import IntersectionAnalyzer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('results_dir', help='Directory with experiment results')
    args = parser.parse_args()

    print("=" * 80)
    print(" MULTI-METHOD INTERSECTION ANALYSIS")
    print("=" * 80)

    # Load configuration
    with open(f'{args.results_dir}/config.json', 'r') as f:
        config = json.load(f)

    n_datasets = config['n_datasets']
    print(f"\nLoading {n_datasets} datasets...")

    # Load all weight matrices
    manifolds = []
    manifold_dims = []

    for i in range(n_datasets):
        weights_file = f'{args.results_dir}/weights_dataset_{i:03d}.npy'
        if not os.path.exists(weights_file):
            print(f"Warning: Missing {weights_file}, skipping...")
            continue

        weights = np.load(weights_file)
        manifolds.append(weights)

        # Quick PCA to get dimension
        from sklearn.decomposition import PCA
        pca = PCA()
        pca.fit(weights)
        var_ratios = pca.explained_variance_ratio_
        cumsum = np.cumsum(var_ratios)
        dim_95 = int(np.argmax(cumsum >= 0.95) + 1)
        manifold_dims.append(dim_95)

    print(f"Loaded {len(manifolds)} manifolds")
    print(f"Dimension range: {np.min(manifold_dims)}D - {np.max(manifold_dims)}D")

    # Create analyzer
    analyzer = IntersectionAnalyzer(
        manifolds=manifolds,
        manifold_dims=manifold_dims,
        variance_threshold=0.95
    )

    # Run all methods
    results = analyzer.analyze_all_methods()

    # Save enhanced results
    enhanced_results = {
        'original_config': config,
        'n_manifolds_analyzed': len(manifolds),
        'intersection_methods': {
            method: {k: float(v) if isinstance(v, (np.floating, np.integer))
                     else (v.tolist() if isinstance(v, np.ndarray) else v)
                     for k, v in result.items() if k != 'method'}
            for method, result in results.items()
            if result is not None
        }
    }

    output_file = f'{args.results_dir}/multi_method_analysis.json'
    with open(output_file, 'w') as f:
        json.dump(enhanced_results, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
