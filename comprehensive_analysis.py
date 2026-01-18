#!/usr/bin/env python3
"""
Comprehensive Analysis of Universal Subspace Experiments

Re-examine all our experimental results to understand:
1. How does the manifold dimension scale with model size?
2. What is the relationship between effective dimension and parameter count?
3. Is there evidence for a universal subspace?
4. How do different tasks' manifolds relate to each other?
"""
import numpy as np
import json
import os

def load_results():
    """Load all experimental results."""
    results = {}

    # 10-dataset intersection
    with open('experiments/current/10_dataset_intersection/results.json') as f:
        results['10_dataset'] = json.load(f)

    # Paper method analysis
    with open('experiments/current/10_dataset_intersection/paper_method_analysis/results.json') as f:
        results['paper_method'] = json.load(f)

    # Dropout spread experiment
    with open('experiments/current/dropout_spread/results.json') as f:
        results['dropout'] = json.load(f)

    return results


def analyze_dimension_vs_parameters():
    """Analyze how effective dimension relates to parameter count."""
    results = load_results()
    dropout = results['dropout']['results']

    print("=" * 70)
    print(" ANALYSIS 1: Effective Dimension vs Parameter Count")
    print("=" * 70)
    print("\nMLPs (no dropout):")
    print(f"{'Size':<10} {'Params':<10} {'Eff.Dim':<12} {'k_95':<8} {'Dim/Params':<12} {'k_95/n_models':<12}")
    print("-" * 70)

    for size in ['small', 'medium', 'large', 'xlarge']:
        d = dropout[size]['0.0']
        n_params = d['n_params']
        eff_dim = d['effective_dim']
        k_95 = d['k_95']
        n_models = 50

        print(f"{size:<10} {n_params:<10} {eff_dim:<12.2f} {k_95:<8} {eff_dim/n_params:<12.6f} {k_95/n_models:<12.2f}")

    print("\nKey observation: Effective dimension approaches n_models (50) as params increase")
    print("This suggests we're measuring the manifold of 50 random points, not a constrained subspace")


def analyze_spectral_decay():
    """Analyze spectral decay patterns across scales."""
    results = load_results()
    dropout = results['dropout']['results']

    print("\n" + "=" * 70)
    print(" ANALYSIS 2: Spectral Decay (σ₁/σ₁₀) vs Scale")
    print("=" * 70)

    print("\nMLP spectral ratios (no dropout):")
    print(f"{'Size':<10} {'Params':<10} {'σ₁/σ₁₀':<12} {'Interpretation':<30}")
    print("-" * 70)

    for size in ['small', 'medium', 'large', 'xlarge']:
        d = dropout[size]['0.0']
        n_params = d['n_params']
        ratio = d['spectral_ratio']

        if ratio > 1.5:
            interp = "Some concentration"
        elif ratio > 1.2:
            interp = "Mild concentration"
        else:
            interp = "Nearly uniform (flat)"

        print(f"{size:<10} {n_params:<10} {ratio:<12.3f} {interp:<30}")

    print("\nTrend: Spectral decay gets FLATTER with more parameters")
    print("       Small (465 params): σ₁/σ₁₀ = 1.62")
    print("       XLarge (269K params): σ₁/σ₁₀ = 1.06")


def analyze_variance_distribution():
    """Look at the actual variance explained by top components."""
    results = load_results()
    dropout = results['dropout']['results']

    print("\n" + "=" * 70)
    print(" ANALYSIS 3: Variance Distribution (Top 10 Components)")
    print("=" * 70)

    for size in ['small', 'xlarge']:
        d = dropout[size]['0.0']
        variances = d['top_10_var']

        print(f"\n{size.upper()} ({d['n_params']} params):")
        cumsum = np.cumsum(variances)
        for i, (v, c) in enumerate(zip(variances, cumsum)):
            bar = '█' * int(v * 200)
            print(f"  PC{i+1:2d}: {v:.4f} (cum: {c:.4f}) {bar}")

    print("\nObservation: Small models show more variation in eigenvalues")
    print("             Large models have nearly equal eigenvalues (uniform)")


def analyze_dataset_intersection():
    """Analyze the intersection of different task manifolds."""
    results = load_results()
    ten_ds = results['10_dataset']
    paper = results['paper_method']

    print("\n" + "=" * 70)
    print(" ANALYSIS 4: Multi-Dataset Intersection")
    print("=" * 70)

    print("\nDatasets used:")
    for meta in ten_ds['dataset_metadata']:
        print(f"  {meta['dataset_name']}: {meta['base_task']}, acc={meta['mean_accuracy']:.1f}%")

    print(f"\nIndividual manifold dimensions:")
    print(f"  Mean effective dim: {ten_ds['individual_dimensions']['mean']:.2f}")
    print(f"  Std: {ten_ds['individual_dimensions']['std']:.2f}")

    print(f"\nGlobal analysis (all 500 models):")
    print(f"  k for 95% variance: {paper['k_values']['0.95']}")
    print(f"  k for 99% variance: {paper['k_values']['0.99']}")
    print(f"  Effective rank: {paper['effective_rank']:.1f}")

    print(f"\nIntersection analysis:")
    inter = ten_ds['intersection_methods']
    print(f"  PCA ratio method: {inter['pca_ratio']['intersection_dim']}D intersection")
    print(f"  Subspace intersection: {inter['subspace_intersection']['intersection_dim']}D")
    print(f"  Mean cosine similarity: {inter['clustering']['mean_similarity']:.4f}")

    # Compute expected dimension if independent
    n_datasets = 10
    mean_individual = ten_ds['individual_dimensions']['mean']
    n_params = 465
    n_models = 50

    print(f"\nDimensional analysis:")
    print(f"  Each dataset: ~{mean_individual:.0f}D manifold in {n_params}D space")
    print(f"  10 datasets × ~{mean_individual:.0f}D = ~{10*mean_individual:.0f}D if orthogonal")
    print(f"  But parameter space is only {n_params}D")
    print(f"  Global dim (k_95): {paper['k_values']['0.95']}D")

    compression = paper['k_values']['0.95'] / (n_datasets * mean_individual)
    print(f"  Compression: {compression:.2f}x (vs orthogonal assumption)")


def analyze_dropout_effect():
    """Analyze how dropout affects the solution manifold."""
    results = load_results()
    dropout = results['dropout']['results']

    print("\n" + "=" * 70)
    print(" ANALYSIS 5: Dropout Effect on Solution Manifold")
    print("=" * 70)

    print("\nMean pairwise distance between solutions:")
    print(f"{'Size':<10} {'No Dropout':<15} {'Dropout 0.3':<15} {'Dropout 0.5':<15} {'Effect':<20}")
    print("-" * 75)

    for size in ['small', 'medium', 'large', 'xlarge']:
        d0 = dropout[size]['0.0']['mean_distance']
        d3 = dropout[size]['0.3']['mean_distance']
        d5 = dropout[size]['0.5']['mean_distance']

        if d5 < d0:
            effect = "CONSTRAINS (smaller)"
        else:
            effect = "SPREADS (larger)"

        print(f"{size:<10} {d0:<15.2f} {d3:<15.2f} {d5:<15.2f} {effect:<20}")

    print("\nConclusion: Dropout CONSTRAINS solutions to a smaller region")
    print("            (opposite of the intuition that it 'spreads' learning)")


def analyze_transformer_data():
    """Analyze saved transformer weight matrices."""
    print("\n" + "=" * 70)
    print(" ANALYSIS 6: Transformer Weight Manifolds")
    print("=" * 70)

    transformer_dir = 'experiments/current/transformer_spectral'

    results = {}
    for fname in ['weights_tiny.npy', 'weights_small.npy', 'weights_medium.npy']:
        path = os.path.join(transformer_dir, fname)
        if os.path.exists(path):
            size = fname.replace('weights_', '').replace('.npy', '')
            weights = np.load(path)

            # Compute PCA metrics
            from sklearn.decomposition import PCA
            pca = PCA()
            pca.fit(weights)
            var_ratios = pca.explained_variance_ratio_

            # Metrics
            effective_dim = (np.sum(var_ratios) ** 2) / np.sum(var_ratios ** 2)
            k_95 = np.argmax(np.cumsum(var_ratios) >= 0.95) + 1
            spectral_ratio = var_ratios[0] / var_ratios[9] if len(var_ratios) > 9 else None

            results[size] = {
                'n_models': weights.shape[0],
                'n_params': weights.shape[1],
                'effective_dim': effective_dim,
                'k_95': k_95,
                'spectral_ratio': spectral_ratio,
                'top_10_var': var_ratios[:10].tolist()
            }

    if results:
        print(f"\n{'Size':<10} {'Params':<12} {'Eff.Dim':<10} {'k_95':<8} {'σ₁/σ₁₀':<10}")
        print("-" * 55)
        for size in ['tiny', 'small', 'medium']:
            if size in results:
                r = results[size]
                print(f"{size:<10} {r['n_params']:<12,} {r['effective_dim']:<10.2f} {r['k_95']:<8} {r['spectral_ratio']:<10.3f}")

        print("\nTransformer variance distribution (tiny vs medium):")
        for size in ['tiny', 'medium']:
            if size in results:
                print(f"\n  {size.upper()}:")
                for i, v in enumerate(results[size]['top_10_var'][:5]):
                    bar = '█' * int(v * 200)
                    print(f"    PC{i+1}: {v:.4f} {bar}")


def key_findings():
    """Summarize key findings."""
    print("\n" + "=" * 70)
    print(" KEY FINDINGS")
    print("=" * 70)

    print("""
1. MANIFOLD DIMENSION SATURATES AT n_models
   - Effective dimension ≈ 40-49 for 50 models
   - This is ~80-98% of the maximum possible (50)
   - Larger models approach this limit faster

2. SPECTRAL DECAY FLATTENS WITH SCALE (opposite of paper's claim)
   - Small models: σ₁/σ₁₀ ≈ 1.6 (some structure)
   - Large models: σ₁/σ₁₀ ≈ 1.06 (nearly uniform)
   - This suggests LESS low-rank structure at scale, not more

3. NO UNIVERSAL SUBSPACE INTERSECTION
   - 10 different tasks show 0D strict intersection
   - Mean cosine similarity ≈ 0.04 (nearly orthogonal)
   - Each task uses different directions in weight space

4. GLOBAL COMPRESSION EXISTS BUT IS WEAK
   - k=260 for 95% variance (out of 465 params)
   - This is ~56% compression, not the dramatic reduction the paper claims

5. DROPOUT CONSTRAINS SOLUTIONS
   - Smaller mean pairwise distance with dropout
   - Solutions cluster more tightly, not spread out

INTERPRETATION:
Our small models (465 - 269K params) don't show the phenomena the paper
describes for large models (100M+ params). The "universal subspace" may
only emerge at much larger scales, possibly related to double descent
and emergent capabilities.
""")


def main():
    print("=" * 70)
    print(" COMPREHENSIVE ANALYSIS OF UNIVERSAL SUBSPACE EXPERIMENTS")
    print("=" * 70)

    analyze_dimension_vs_parameters()
    analyze_spectral_decay()
    analyze_variance_distribution()
    analyze_dataset_intersection()
    analyze_dropout_effect()
    analyze_transformer_data()
    key_findings()


if __name__ == '__main__':
    main()
