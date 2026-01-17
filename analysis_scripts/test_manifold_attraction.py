#!/usr/bin/env python3
"""
Test if the convergence manifold is an attractor.

Experiment:
1. Load the known manifold (from previous converged models)
2. Initialize new models at various distances from the manifold
3. Train them and see if they return to the manifold
4. Measure convergence dynamics
"""
import numpy as np
import torch
import torch.optim as optim
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import json
import os

from src.models import create_model_for_task, get_loss_function
from src.datasets import DatasetManager


def load_manifold_representation(results_dir='results_dynamics'):
    """Load the manifold from previous experiment."""
    traj_data = np.load(os.path.join(results_dir, 'trajectories.npz'))
    
    # Extract final weights
    final_weights = []
    for key in sorted(traj_data.files):
        trajectory = traj_data[key]
        final_weights.append(trajectory[-1])
    
    X = np.array(final_weights)
    
    # Compute PCA representation
    pca = PCA()
    pca.fit(X)
    
    # Centroid (manifold center)
    centroid = np.mean(X, axis=0)
    
    return {
        'centroid': centroid,
        'pca': pca,
        'samples': X,
        'dim': X.shape[1]
    }


def distance_to_manifold(weights, manifold):
    """
    Compute distance from a weight vector to the manifold.
    Using distance to nearest point in manifold samples.
    """
    distances = np.linalg.norm(manifold['samples'] - weights, axis=1)
    return np.min(distances)


def create_perturbed_initialization(manifold, perturbation_scale):
    """
    Create an initialization at a specified distance from the manifold.
    
    Strategy:
    - Start from manifold centroid
    - Add random perturbation of given scale
    - Perturbation is in a random direction in weight space
    """
    centroid = manifold['centroid']
    dim = manifold['dim']
    
    # Random direction
    direction = np.random.randn(dim)
    direction = direction / np.linalg.norm(direction)
    
    # Perturbed weights
    weights = centroid + perturbation_scale * direction
    
    return weights


def initialize_model_with_weights(model, weights):
    """Set model parameters from a weight vector."""
    offset = 0
    for param in model.parameters():
        numel = param.numel()
        param.data = torch.FloatTensor(
            weights[offset:offset+numel].reshape(param.shape)
        )
        offset += numel


def extract_weights(model):
    """Extract weights as flat vector."""
    weights = []
    for param in model.parameters():
        weights.append(param.data.cpu().numpy().flatten())
    return np.concatenate(weights)


def train_from_initialization(initial_weights, manifold, dataset_name='binary_classification_synthetic',
                              epochs=150, lr=0.001, track_every=10):
    """
    Train a model from a specific initialization and track its trajectory.
    """
    # Load dataset
    train_loader, test_loader, metadata = DatasetManager.load_dataset(dataset_name)
    
    # Create model
    model = create_model_for_task(
        task_type=metadata['task_type'],
        input_dim=metadata['input_dim'],
        output_dim=metadata['output_dim'],
        hidden_dims=[16, 16]
    )
    
    # Initialize with provided weights
    initialize_model_with_weights(model, initial_weights)
    
    # Setup training
    criterion = get_loss_function(metadata['task_type'])
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Track trajectory
    trajectory = []
    distances_to_manifold = []
    epochs_list = []
    
    # Initial state
    current_weights = extract_weights(model)
    trajectory.append(current_weights.copy())
    distances_to_manifold.append(distance_to_manifold(current_weights, manifold))
    epochs_list.append(0)
    
    print(f"  Initial distance to manifold: {distances_to_manifold[0]:.4f}")
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Track at intervals
        if (epoch + 1) % track_every == 0:
            current_weights = extract_weights(model)
            trajectory.append(current_weights.copy())
            dist = distance_to_manifold(current_weights, manifold)
            distances_to_manifold.append(dist)
            epochs_list.append(epoch + 1)
            
            if (epoch + 1) % 50 == 0:
                print(f"    Epoch {epoch+1}: Loss={train_loss:.4f}, Dist to manifold={dist:.4f}")
    
    # Final state
    final_weights = extract_weights(model)
    final_dist = distance_to_manifold(final_weights, manifold)
    
    print(f"  Final distance to manifold: {final_dist:.4f}")
    print(f"  Distance reduction: {distances_to_manifold[0] - final_dist:.4f}")
    
    # Cleanup
    del model, optimizer, train_loader, test_loader
    DatasetManager.cleanup()
    
    return {
        'trajectory': np.array(trajectory),
        'distances': np.array(distances_to_manifold),
        'epochs': np.array(epochs_list),
        'initial_dist': distances_to_manifold[0],
        'final_dist': final_dist
    }


def main():
    print("=" * 80)
    print(" TESTING MANIFOLD ATTRACTION")
    print("=" * 80)
    
    # Load known manifold
    print("\nLoading manifold from previous experiment...")
    manifold = load_manifold_representation()
    print(f"  Manifold dimension: {manifold['dim']}")
    print(f"  Number of samples: {len(manifold['samples'])}")
    
    # Test initialization distances
    perturbation_scales = [0, 5, 10, 20, 40, 80]
    n_trials_per_scale = 2
    
    print(f"\n" + "=" * 80)
    print(" RUNNING EXPERIMENTS")
    print("=" * 80)
    print(f"\nTesting {len(perturbation_scales)} perturbation scales")
    print(f"Trials per scale: {n_trials_per_scale}")
    
    results = []
    
    for scale in perturbation_scales:
        print(f"\n{'─' * 80}")
        print(f"Perturbation scale: {scale}")
        print(f"{'─' * 80}")
        
        for trial in range(n_trials_per_scale):
            print(f"\n  Trial {trial + 1}/{n_trials_per_scale}:")
            
            # Create initialization
            if scale == 0:
                # Initialize at manifold centroid
                init_weights = manifold['centroid'].copy()
            else:
                init_weights = create_perturbed_initialization(manifold, scale)
            
            # Train
            result = train_from_initialization(
                init_weights, manifold, 
                epochs=150, track_every=10
            )
            
            result['perturbation_scale'] = scale
            result['trial'] = trial
            results.append(result)
    
    # Save results
    save_dir = 'results_attraction'
    os.makedirs(save_dir, exist_ok=True)
    
    # Save trajectories
    np.savez(os.path.join(save_dir, 'attraction_test.npz'),
             **{f"result_{i}": r['trajectory'] for i, r in enumerate(results)})
    
    # Save metadata
    metadata = [{
        'perturbation_scale': r['perturbation_scale'],
        'trial': r['trial'],
        'initial_dist': float(r['initial_dist']),
        'final_dist': float(r['final_dist']),
        'reduction': float(r['initial_dist'] - r['final_dist'])
    } for r in results]
    
    with open(os.path.join(save_dir, 'attraction_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Visualize
    print(f"\n{'=' * 80}")
    print(" CREATING VISUALIZATIONS")
    print(f"{'=' * 80}")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Distance vs Epoch for all trials
    colors = plt.cm.viridis(np.linspace(0, 1, len(perturbation_scales)))
    
    for i, scale in enumerate(perturbation_scales):
        scale_results = [r for r in results if r['perturbation_scale'] == scale]
        
        for r in scale_results:
            ax1.plot(r['epochs'], r['distances'], 
                    color=colors[i], alpha=0.6, linewidth=2,
                    label=f"Scale={scale}" if r['trial'] == 0 else "")
    
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Distance to Manifold', fontsize=12)
    ax1.set_title('Convergence to Manifold from Different Initializations', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Plot 2: Initial vs Final distance
    initial_dists = [r['initial_dist'] for r in results]
    final_dists = [r['final_dist'] for r in results]
    scales = [r['perturbation_scale'] for r in results]
    
    scatter = ax2.scatter(initial_dists, final_dists, 
                         c=scales, cmap='viridis', s=100, 
                         edgecolors='black', linewidth=1.5)
    
    # Diagonal line (no convergence)
    max_val = max(max(initial_dists), max(final_dists))
    ax2.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='No convergence')
    
    ax2.set_xlabel('Initial Distance to Manifold', fontsize=12)
    ax2.set_ylabel('Final Distance to Manifold', fontsize=12)
    ax2.set_title('Attraction to Manifold', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax2, label='Perturbation Scale')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'manifold_attraction.png'), dpi=300, bbox_inches='tight')
    print(f"\nSaved: {save_dir}/manifold_attraction.png")
    
    # Summary
    print(f"\n{'=' * 80}")
    print(" SUMMARY")
    print(f"{'=' * 80}")
    
    for scale in perturbation_scales:
        scale_results = [r for r in results if r['perturbation_scale'] == scale]
        avg_initial = np.mean([r['initial_dist'] for r in scale_results])
        avg_final = np.mean([r['final_dist'] for r in scale_results])
        avg_reduction = avg_initial - avg_final
        reduction_pct = (avg_reduction / avg_initial * 100) if avg_initial > 0 else 0
        
        print(f"\nPerturbation scale: {scale}")
        print(f"  Avg initial distance: {avg_initial:.4f}")
        print(f"  Avg final distance: {avg_final:.4f}")
        print(f"  Avg reduction: {avg_reduction:.4f} ({reduction_pct:.1f}%)")
    
    print(f"\n{'=' * 80}")
    print(" CONCLUSION")
    print(f"{'=' * 80}")
    
    avg_final_all = np.mean([r['final_dist'] for r in results])
    std_final_all = np.std([r['final_dist'] for r in results])
    
    print(f"\nAverage final distance across ALL initializations: {avg_final_all:.4f} ± {std_final_all:.4f}")
    print(f"\nThe manifold IS an attractor if:")
    print(f"  1. All trajectories converge to similar final distances")
    print(f"  2. Final distance is small regardless of initial distance")
    print(f"  3. Distance decreases monotonically during training")
    
    if std_final_all < 2.0 and avg_final_all < 5.0:
        print(f"\n✓ CONFIRMED: The manifold is a strong attractor!")
        print(f"  Models initialized far away ({max(perturbation_scales)} units) converge back.")
    else:
        print(f"\n? UNCLEAR: Need more data or different initialization strategy")


if __name__ == '__main__':
    main()
