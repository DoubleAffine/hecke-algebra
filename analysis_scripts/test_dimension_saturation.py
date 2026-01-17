#!/usr/bin/env python3
"""
Test if dimension saturates or keeps growing with sample size.

Critical question: Is 72D real, or will it keep growing to fill 465D?

Strategy:
- Train models in batches: 50, 100, 200, 400
- Track effective dimension at each batch size
- If it saturates → dimension is real
- If it keeps growing → we're just measuring sampling dimension
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import argparse
import os
import json

from src.trainer import Trainer
from src.datasets import DatasetManager

sns.set_style('whitegrid')

def compute_dimension_metrics(weight_matrix):
    """Compute various dimension estimates."""
    pca = PCA()
    pca.fit(weight_matrix)

    var_ratios = pca.explained_variance_ratio_
    cumsum = np.cumsum(var_ratios)

    dim_95 = np.argmax(cumsum >= 0.95) + 1
    dim_99 = np.argmax(cumsum >= 0.99) + 1
    effective_dim = (np.sum(var_ratios) ** 2) / np.sum(var_ratios ** 2)

    return {
        'dim_95': dim_95,
        'dim_99': dim_99,
        'effective_dim': effective_dim,
        'first_pc_var': var_ratios[0],
        'variance_ratios': var_ratios
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-models', type=int, default=400,
                       help='Maximum number of models to train')
    parser.add_argument('--checkpoint-every', type=int, default=50,
                       help='Analyze dimension every N models')
    parser.add_argument('--epochs', type=int, default=80,
                       help='Training epochs per model')
    parser.add_argument('--save-dir', type=str, default='results_saturation',
                       help='Save directory')

    args = parser.parse_args()

    print("=" * 80)
    print(" DIMENSION SATURATION TEST")
    print(" Does dimension keep growing or saturate?")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Max models: {args.max_models}")
    print(f"  Checkpoints: every {args.checkpoint_every} models")
    print(f"  Training epochs: {args.epochs}")

    os.makedirs(args.save_dir, exist_ok=True)

    # Load dataset
    print(f"\nLoading dataset...")
    train_loader, test_loader, dataset_metadata = DatasetManager.load_dataset(
        'binary_classification_synthetic'
    )

    # Create trainer
    trainer = Trainer(
        hidden_dims=[16, 16],
        learning_rate=0.001,
        epochs=args.epochs,
        patience=15
    )

    # Storage
    all_weights = []
    all_metadata = []

    # Track dimension vs sample size
    checkpoints = []

    print(f"\n{'='*80}")
    print(f" TRAINING PHASE")
    print(f"{'='*80}")

    for i in range(args.max_models):
        # Train model
        final_weights, train_stats = trainer.train_single_model(
            train_loader, test_loader, dataset_metadata
        )

        all_weights.append(final_weights)
        all_metadata.append(train_stats)

        # Progress
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{args.max_models}] Loss: {train_stats['best_test_loss']:.4f}, "
                  f"Acc: {train_stats.get('best_test_accuracy', 0):.1f}%")

        # Checkpoint analysis
        if (i + 1) % args.checkpoint_every == 0:
            n_models = i + 1
            weight_matrix = np.array(all_weights)

            print(f"\n--- Checkpoint: {n_models} models ---")
            metrics = compute_dimension_metrics(weight_matrix)

            print(f"  Dim (95% var): {metrics['dim_95']}")
            print(f"  Dim (99% var): {metrics['dim_99']}")
            print(f"  Effective dim: {metrics['effective_dim']:.1f}")
            print(f"  First PC var: {metrics['first_pc_var']*100:.2f}%")

            checkpoints.append({
                'n_models': n_models,
                **metrics
            })

            # Save checkpoint
            np.save(f"{args.save_dir}/weights_{n_models}.npy", weight_matrix)

    # Cleanup
    DatasetManager.cleanup()

    print(f"\n{'='*80}")
    print(f" SATURATION ANALYSIS")
    print(f"{'='*80}")

    # Extract data for plotting
    sample_sizes = [c['n_models'] for c in checkpoints]
    dims_95 = [c['dim_95'] for c in checkpoints]
    dims_99 = [c['dim_99'] for c in checkpoints]
    effective_dims = [c['effective_dim'] for c in checkpoints]

    # Check for saturation
    print(f"\nDimension growth:")
    print(f"  Sample size | Dim (95%) | Dim (99%) | Effective")
    print(f"  " + "-" * 50)
    for i, cp in enumerate(checkpoints):
        print(f"  {cp['n_models']:10d} | {cp['dim_95']:9d} | {cp['dim_99']:9d} | {cp['effective_dim']:9.1f}")

    # Compute growth rate (last half vs first half)
    mid_idx = len(checkpoints) // 2
    if len(checkpoints) >= 4:
        early_dim = np.mean(effective_dims[:mid_idx])
        late_dim = np.mean(effective_dims[mid_idx:])
        growth_rate = (late_dim - early_dim) / early_dim

        print(f"\nGrowth analysis:")
        print(f"  Early mean (n={sample_sizes[0]}-{sample_sizes[mid_idx-1]}): {early_dim:.1f}D")
        print(f"  Late mean (n={sample_sizes[mid_idx]}-{sample_sizes[-1]}): {late_dim:.1f}D")
        print(f"  Growth rate: {growth_rate*100:+.1f}%")

        if abs(growth_rate) < 0.05:
            print(f"\n  ✓ SATURATED")
            print(f"  → Dimension is STABLE (< 5% change)")
            print(f"  → The ~{late_dim:.0f}D is the TRUE manifold dimension")
        elif growth_rate > 0.2:
            print(f"\n  ✗ STILL GROWING")
            print(f"  → Dimension increases {growth_rate*100:.0f}% from early to late")
            print(f"  → May need more samples to find true dimension")
            print(f"  → Or dimension = sample size (no manifold!)")
        else:
            print(f"\n  ? UNCERTAIN")
            print(f"  → Moderate growth ({growth_rate*100:.0f}%)")
            print(f"  → Need more samples to determine saturation")

    # Check if approaching full space
    final_dim = effective_dims[-1]
    ambient_dim = 465
    compression_ratio = ambient_dim / final_dim

    print(f"\nCompression analysis:")
    print(f"  Ambient space: {ambient_dim}D")
    print(f"  Final effective dim: {final_dim:.1f}D")
    print(f"  Compression ratio: {compression_ratio:.1f}×")

    if compression_ratio < 2:
        print(f"\n  ⚠ DANGER: Nearly filling full space!")
        print(f"  → Compression < 2× suggests no real manifold structure")
        print(f"  → Might just be high-dimensional noise")
    elif compression_ratio < 5:
        print(f"\n  ⚠ WARNING: Modest compression")
        print(f"  → Less structure than expected")
    else:
        print(f"\n  ✓ GOOD: Clear compression")
        print(f"  → Manifold structure confirmed")

    # Visualizations
    print(f"\n{'='*80}")
    print(f" CREATING VISUALIZATIONS")
    print(f"{'='*80}")

    fig = plt.figure(figsize=(18, 10))

    # Plot 1: Dimension vs sample size
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(sample_sizes, dims_95, 'b.-', linewidth=2, markersize=8, label='95% variance')
    ax1.plot(sample_sizes, dims_99, 'r.-', linewidth=2, markersize=8, label='99% variance')
    ax1.plot(sample_sizes, effective_dims, 'g.-', linewidth=2, markersize=8, label='Effective (participation)')

    # Add reference lines
    ax1.axhline(y=72, color='purple', linestyle='--', alpha=0.5, label='Previous estimate (72D)')
    ax1.axhline(y=465, color='black', linestyle='--', alpha=0.3, label='Ambient space (465D)')

    ax1.set_xlabel('Number of Models')
    ax1.set_ylabel('Dimension')
    ax1.set_title('Dimension vs Sample Size\n(Flat = Saturated)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Dimension growth rate
    ax2 = plt.subplot(2, 3, 2)
    if len(sample_sizes) > 1:
        growth_rates = [0] + [
            (effective_dims[i] - effective_dims[i-1]) / effective_dims[i-1] * 100
            for i in range(1, len(effective_dims))
        ]
        ax2.plot(sample_sizes, growth_rates, 'ro-', linewidth=2, markersize=6)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax2.axhline(y=5, color='orange', linestyle='--', alpha=0.5, label='5% threshold')
        ax2.axhline(y=-5, color='orange', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Number of Models')
        ax2.set_ylabel('Growth Rate (%)')
        ax2.set_title('Dimension Growth Rate\n(Near zero = Saturated)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    # Plot 3: Compression ratio
    ax3 = plt.subplot(2, 3, 3)
    compression_ratios = [ambient_dim / d for d in effective_dims]
    ax3.plot(sample_sizes, compression_ratios, 'g.-', linewidth=2, markersize=8)
    ax3.axhline(y=2, color='red', linestyle='--', alpha=0.5, label='Danger threshold (2×)')
    ax3.axhline(y=5, color='orange', linestyle='--', alpha=0.5, label='Warning threshold (5×)')
    ax3.set_xlabel('Number of Models')
    ax3.set_ylabel('Compression Ratio (ambient/effective)')
    ax3.set_title('Compression Ratio\n(Higher = More Structure)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Variance spectrum evolution
    ax4 = plt.subplot(2, 3, 4)
    for i, cp in enumerate(checkpoints[::2]):  # Every other checkpoint
        n = cp['n_models']
        var_ratios = cp['variance_ratios'][:50]
        alpha = 0.3 + 0.7 * (i / (len(checkpoints[::2]) - 1))
        ax4.plot(range(1, len(var_ratios)+1), var_ratios,
                alpha=alpha, linewidth=2, label=f'n={n}')
    ax4.set_xlabel('Principal Component')
    ax4.set_ylabel('Explained Variance Ratio')
    ax4.set_title('Variance Spectrum Evolution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')

    # Plot 5: Cumulative variance
    ax5 = plt.subplot(2, 3, 5)
    for i, cp in enumerate(checkpoints[::2]):
        n = cp['n_models']
        var_ratios = cp['variance_ratios']
        cumsum = np.cumsum(var_ratios)[:100]
        alpha = 0.3 + 0.7 * (i / (len(checkpoints[::2]) - 1))
        ax5.plot(range(1, len(cumsum)+1), cumsum,
                alpha=alpha, linewidth=2, label=f'n={n}')
    ax5.axhline(y=0.95, color='red', linestyle='--', alpha=0.5)
    ax5.axhline(y=0.99, color='orange', linestyle='--', alpha=0.5)
    ax5.set_xlabel('Number of Components')
    ax5.set_ylabel('Cumulative Variance')
    ax5.set_title('Cumulative Variance Evolution')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Plot 6: Summary box
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')

    if len(checkpoints) >= 4:
        saturation_status = "SATURATED ✓" if abs(growth_rate) < 0.05 else ("GROWING ✗" if growth_rate > 0.2 else "UNCERTAIN ?")
        compression_status = "GOOD ✓" if compression_ratio >= 5 else ("WARNING ⚠" if compression_ratio >= 2 else "DANGER ⚠")

        summary = f"""
SATURATION TEST SUMMARY

Sample sizes: {sample_sizes[0]} → {sample_sizes[-1]}
Checkpoints: {len(checkpoints)}

Dimension Evolution:
  Start: {effective_dims[0]:.1f}D
  End: {effective_dims[-1]:.1f}D
  Growth: {growth_rate*100:+.1f}%

Status: {saturation_status}

Compression:
  Final: {compression_ratio:.1f}×
  Status: {compression_status}

Conclusion:
{"  TRUE MANIFOLD" if abs(growth_rate) < 0.05 and compression_ratio >= 5 else "  UNCLEAR - need more data"}
{"  Dimension ≈ " + f"{effective_dims[-1]:.0f}D" if abs(growth_rate) < 0.05 else "  Still changing..."}
"""
    else:
        summary = f"\nNeed more checkpoints for analysis\n(have {len(checkpoints)}, need 4+)"

    ax6.text(0.1, 0.5, summary, fontsize=11, family='monospace',
             verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    plt.savefig(f"{args.save_dir}/saturation_analysis.png", dpi=300, bbox_inches='tight')
    print(f"Saved: {args.save_dir}/saturation_analysis.png")

    # Save data
    with open(f"{args.save_dir}/checkpoints.json", 'w') as f:
        # Convert numpy arrays to lists for JSON
        checkpoints_json = []
        for cp in checkpoints:
            cp_copy = cp.copy()
            cp_copy['variance_ratios'] = cp_copy['variance_ratios'].tolist()
            checkpoints_json.append(cp_copy)
        json.dump(checkpoints_json, f, indent=2)

    print(f"\n{'='*80}")
    print(f" COMPLETE")
    print(f"{'='*80}")
    print(f"\nResults saved to: {args.save_dir}/")

if __name__ == '__main__':
    main()
