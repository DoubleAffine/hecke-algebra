#!/usr/bin/env python3
"""
Extended experiment:
A) Train distant initializations for much longer (500 epochs)
B) Check if they converge to a different manifold than the original
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


def load_original_manifold(results_dir='results_dynamics'):
    """Load the original manifold we discovered."""
    traj_data = np.load(os.path.join(results_dir, 'trajectories.npz'))
    
    final_weights = []
    for key in sorted(traj_data.files):
        trajectory = traj_data[key]
        final_weights.append(trajectory[-1])
    
    return np.array(final_weights)


def extract_weights(model):
    """Extract weights as flat vector."""
    weights = []
    for param in model.parameters():
        weights.append(param.data.cpu().numpy().flatten())
    return np.concatenate(weights)


def initialize_model_with_weights(model, weights):
    """Set model parameters from a weight vector."""
    offset = 0
    for param in model.parameters():
        numel = param.numel()
        param.data = torch.FloatTensor(
            weights[offset:offset+numel].reshape(param.shape)
        )
        offset += numel


def train_from_distant_init(initial_weights, epochs=500, lr=0.001, 
                            track_every=25, dataset_name='binary_classification_synthetic'):
    """Train from a distant initialization for extended time."""
    
    # Load dataset
    train_loader, test_loader, metadata = DatasetManager.load_dataset(dataset_name)
    
    # Create model
    model = create_model_for_task(
        task_type=metadata['task_type'],
        input_dim=metadata['input_dim'],
        output_dim=metadata['output_dim'],
        hidden_dims=[16, 16]
    )
    
    # Initialize
    initialize_model_with_weights(model, initial_weights)
    
    # Setup
    criterion = get_loss_function(metadata['task_type'])
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Track
    trajectory = []
    losses = []
    epochs_list = []
    
    # Initial
    trajectory.append(extract_weights(model).copy())
    epochs_list.append(0)
    
    print(f"  Initial L2 norm: {np.linalg.norm(initial_weights):.4f}")
    
    # Training
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
        
        # Track
        if (epoch + 1) % track_every == 0 or epoch == epochs - 1:
            current_weights = extract_weights(model)
            trajectory.append(current_weights.copy())
            losses.append(train_loss)
            epochs_list.append(epoch + 1)
            
            if (epoch + 1) % 100 == 0:
                print(f"    Epoch {epoch+1}: Loss={train_loss:.4f}")
    
    final_weights = extract_weights(model)
    print(f"  Final L2 norm: {np.linalg.norm(final_weights):.4f}")
    
    # Cleanup
    del model, optimizer, train_loader, test_loader
    DatasetManager.cleanup()
    
    return {
        'trajectory': np.array(trajectory),
        'losses': np.array(losses),
        'epochs': np.array(epochs_list),
        'initial_weights': initial_weights,
        'final_weights': final_weights
    }


def main():
    print("=" * 80)
    print(" EXTENDED CONVERGENCE TEST")
    print(" A) Long training from distant initializations")
    print(" B) Check if they form a different manifold")
    print("=" * 80)
    
    # Load original manifold
    print("\nLoading original manifold...")
    original_manifold = load_original_manifold()
    original_centroid = np.mean(original_manifold, axis=0)
    print(f"  Original manifold: {len(original_manifold)} models")
    print(f"  Centroid L2 norm: {np.linalg.norm(original_centroid):.4f}")
    
    # Create distant initializations
    perturbation_scales = [40, 80, 120]
    n_trials = 3
    
    print(f"\n{'=' * 80}")
    print(f" TRAINING FROM DISTANT INITIALIZATIONS")
    print(f"{'=' * 80}")
    print(f"Scales: {perturbation_scales}")
    print(f"Trials per scale: {n_trials}")
    print(f"Epochs: 500 (vs 150 in previous experiment)")
    
    all_results = []
    
    for scale in perturbation_scales:
        print(f"\n{'─' * 80}")
        print(f"Perturbation scale: {scale}")
        print(f"{'─' * 80}")
        
        for trial in range(n_trials):
            print(f"\n  Trial {trial + 1}/{n_trials}:")
            
            # Create random initialization
            direction = np.random.randn(len(original_centroid))
            direction = direction / np.linalg.norm(direction)
            init_weights = original_centroid + scale * direction
            
            # Train
            result = train_from_distant_init(
                init_weights, 
                epochs=500,
                track_every=25
            )
            
            result['scale'] = scale
            result['trial'] = trial
            all_results.append(result)
    
    # Save
    save_dir = 'results_distant_convergence'
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract final weights from distant training
    distant_final_weights = np.array([r['final_weights'] for r in all_results])
    
    print(f"\n{'=' * 80}")
    print(" ANALYSIS: DO DISTANT MODELS FORM A DIFFERENT MANIFOLD?")
    print(f"{'=' * 80}")
    
    # Combine original and distant manifolds
    print(f"\nOriginal manifold: {len(original_manifold)} models")
    print(f"Distant converged: {len(distant_final_weights)} models")
    
    # Compute centroids
    original_centroid = np.mean(original_manifold, axis=0)
    distant_centroid = np.mean(distant_final_weights, axis=0)
    
    centroid_distance = np.linalg.norm(original_centroid - distant_centroid)
    print(f"\nDistance between centroids: {centroid_distance:.4f}")
    
    # PCA on both
    combined = np.vstack([original_manifold, distant_final_weights])
    
    pca = PCA(n_components=min(10, len(combined)))
    combined_pca = pca.fit_transform(combined)
    
    original_pca = combined_pca[:len(original_manifold)]
    distant_pca = combined_pca[len(original_manifold):]
    
    # Check separation
    from scipy.spatial.distance import cdist
    
    # Average distance within groups vs between groups
    original_dists = cdist(original_manifold, original_manifold)
    distant_dists = cdist(distant_final_weights, distant_final_weights)
    cross_dists = cdist(original_manifold, distant_final_weights)
    
    avg_original = np.mean(original_dists[np.triu_indices_from(original_dists, k=1)])
    avg_distant = np.mean(distant_dists[np.triu_indices_from(distant_dists, k=1)])
    avg_cross = np.mean(cross_dists)
    
    print(f"\nAverage pairwise distances:")
    print(f"  Within original manifold: {avg_original:.4f}")
    print(f"  Within distant manifold: {avg_distant:.4f}")
    print(f"  Between manifolds: {avg_cross:.4f}")
    
    separation_ratio = avg_cross / avg_original
    print(f"\nSeparation ratio: {separation_ratio:.2f}")
    
    # Visualize
    fig = plt.figure(figsize=(20, 12))
    
    # Plot 1: Training curves
    ax1 = plt.subplot(2, 3, 1)
    colors = plt.cm.viridis(np.linspace(0, 1, len(perturbation_scales)))
    
    for i, scale in enumerate(perturbation_scales):
        scale_results = [r for r in all_results if r['scale'] == scale]
        for r in scale_results:
            ax1.plot(r['epochs'], r['losses'], color=colors[i], alpha=0.5)
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Training Curves from Distant Initializations')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: PCA - original vs distant (2D)
    ax2 = plt.subplot(2, 3, 2)
    ax2.scatter(original_pca[:, 0], original_pca[:, 1], 
               c='blue', label='Original manifold', s=100, alpha=0.7)
    ax2.scatter(distant_pca[:, 0], distant_pca[:, 1], 
               c='red', label='Distant convergence', s=100, alpha=0.7)
    ax2.set_xlabel('PC1')
    ax2.set_ylabel('PC2')
    ax2.set_title('Original vs Distant Manifolds (PCA)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: PCA 3D
    ax3 = plt.subplot(2, 3, 3, projection='3d')
    ax3.scatter(original_pca[:, 0], original_pca[:, 1], original_pca[:, 2],
               c='blue', label='Original', s=100, alpha=0.7)
    ax3.scatter(distant_pca[:, 0], distant_pca[:, 1], distant_pca[:, 2],
               c='red', label='Distant', s=100, alpha=0.7)
    ax3.set_xlabel('PC1')
    ax3.set_ylabel('PC2')
    ax3.set_zlabel('PC3')
    ax3.set_title('3D View')
    ax3.legend()
    
    # Plot 4: Trajectory example
    ax4 = plt.subplot(2, 3, 4)
    
    # Show one trajectory in PCA space
    example = all_results[0]
    traj_pca = pca.transform(example['trajectory'])
    
    ax4.plot(traj_pca[:, 0], traj_pca[:, 1], 'r-', alpha=0.5, linewidth=2)
    ax4.scatter(traj_pca[0, 0], traj_pca[0, 1], c='green', s=200, 
               marker='*', edgecolors='black', linewidth=2, label='Start', zorder=10)
    ax4.scatter(traj_pca[-1, 0], traj_pca[-1, 1], c='red', s=200,
               marker='o', edgecolors='black', linewidth=2, label='End', zorder=10)
    
    # Original manifold
    ax4.scatter(original_pca[:, 0], original_pca[:, 1], 
               c='blue', s=50, alpha=0.3, label='Original manifold')
    
    ax4.set_xlabel('PC1')
    ax4.set_ylabel('PC2')
    ax4.set_title(f'Example Trajectory (scale={example["scale"]})')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Distance matrix heatmap
    ax5 = plt.subplot(2, 3, 5)
    
    # Create combined distance matrix
    all_points = np.vstack([original_manifold, distant_final_weights])
    dist_matrix = cdist(all_points, all_points)
    
    im = ax5.imshow(dist_matrix, cmap='viridis', aspect='auto')
    ax5.axhline(len(original_manifold) - 0.5, color='red', linewidth=2)
    ax5.axvline(len(original_manifold) - 0.5, color='red', linewidth=2)
    ax5.set_xlabel('Model Index')
    ax5.set_ylabel('Model Index')
    ax5.set_title('Distance Matrix (Red line separates groups)')
    plt.colorbar(im, ax=ax5, label='Distance')
    
    # Plot 6: Summary statistics
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    summary_text = f"""
    CONVERGENCE ANALYSIS SUMMARY
    
    Original Manifold:
      - Models: {len(original_manifold)}
      - Avg pairwise distance: {avg_original:.2f}
    
    Distant Convergence (500 epochs):
      - Models: {len(distant_final_weights)}
      - Avg pairwise distance: {avg_distant:.2f}
    
    Between Manifolds:
      - Centroid distance: {centroid_distance:.2f}
      - Avg cross-distance: {avg_cross:.2f}
      - Separation ratio: {separation_ratio:.2f}x
    
    CONCLUSION:
    """
    
    if separation_ratio > 2.0:
        summary_text += "\n    ✓ DIFFERENT MANIFOLDS!\n    Distant initializations converge\n    to a separate region."
    elif separation_ratio > 1.3:
        summary_text += "\n    ? PARTIALLY SEPARATED\n    Some overlap but distinct regions."
    else:
        summary_text += "\n    ✓ SAME MANIFOLD!\n    All converge to same region\n    regardless of initialization."
    
    ax6.text(0.1, 0.5, summary_text, transform=ax6.transAxes,
            fontsize=11, verticalalignment='center',
            family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'distant_convergence_analysis.png'), 
                dpi=300, bbox_inches='tight')
    print(f"\nSaved: {save_dir}/distant_convergence_analysis.png")
    
    # Save data
    np.savez(os.path.join(save_dir, 'manifolds.npz'),
             original=original_manifold,
             distant=distant_final_weights)
    
    print(f"\n{'=' * 80}")
    print(" FINAL CONCLUSION")
    print(f"{'=' * 80}")
    
    if separation_ratio > 2.0:
        print("\nDistant initializations converge to a DIFFERENT manifold!")
        print("This suggests:")
        print("  - Multiple basins of attraction exist")
        print("  - Standard initialization matters for finding the 'good' basin")
        print("  - The 8D manifold is LOCAL, not GLOBAL")
    else:
        print("\nAll initializations converge to the SAME manifold!")
        print("This suggests:")
        print("  - The manifold is a GLOBAL attractor")
        print("  - Initialization doesn't matter (given enough training)")
        print("  - The 8D structure is universal for this architecture")


if __name__ == '__main__':
    main()
