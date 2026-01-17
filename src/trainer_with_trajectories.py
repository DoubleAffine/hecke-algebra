"""
Enhanced trainer that tracks weight trajectories during training.
This allows analysis of optimization dynamics and convergence behavior.
"""
import torch
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
import os

from src.models import SmallMLP, create_model_for_task, get_loss_function
from src.datasets import DatasetManager


class TrajectoryTrainer:
    """
    Trains models while tracking weight evolution over time.
    Useful for studying optimization dynamics, attractors, and convergence patterns.
    """

    def __init__(self, hidden_dims: List[int] = [16, 16],
                 learning_rate: float = 0.001,
                 epochs: int = 100,
                 device: str = None,
                 patience: int = 15,
                 track_every: int = 10):
        """
        Args:
            hidden_dims: Architecture of hidden layers
            learning_rate: Learning rate for optimizer
            epochs: Maximum training epochs
            device: Device to train on (cuda/cpu)
            patience: Early stopping patience
            track_every: Save weight snapshot every N epochs
        """
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.patience = patience
        self.track_every = track_every

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")
        print(f"Tracking weight snapshots every {track_every} epochs")

    def train_with_trajectory(self, train_loader, test_loader, metadata: Dict) -> Tuple[List[np.ndarray], Dict]:
        """
        Train a single model and track its weight trajectory.

        Returns:
            trajectory: List of weight vectors at different points in training
            training_stats: Dictionary with training statistics
        """
        # Create model
        model = create_model_for_task(
            task_type=metadata['task_type'],
            input_dim=metadata['input_dim'],
            output_dim=metadata['output_dim'],
            hidden_dims=self.hidden_dims
        ).to(self.device)

        # Loss and optimizer
        criterion = get_loss_function(metadata['task_type'])
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)

        # Training statistics
        train_losses = []
        test_losses = []
        test_accuracies = []
        trajectory = []  # Weight snapshots
        trajectory_epochs = []  # Which epochs were saved

        best_test_loss = float('inf')
        patience_counter = 0

        # Save initial weights
        trajectory.append(self._extract_weights(model))
        trajectory_epochs.append(0)

        # Training loop
        for epoch in range(self.epochs):
            # Train
            model.train()
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)
            train_losses.append(train_loss)

            # Test
            model.eval()
            test_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs = model(batch_X)

                    if metadata['task_type'] == 'multi_class':
                        loss = criterion(outputs, batch_y)
                        _, predicted = torch.max(outputs.data, 1)
                        total += batch_y.size(0)
                        correct += (predicted == batch_y).sum().item()
                    elif metadata['task_type'] == 'binary_classification':
                        loss = criterion(outputs, batch_y)
                        predicted = (torch.sigmoid(outputs) > 0.5).float()
                        total += batch_y.size(0)
                        correct += (predicted == batch_y).sum().item()
                    else:
                        loss = criterion(outputs, batch_y)

                    test_loss += loss.item()

            test_loss /= len(test_loader)
            test_losses.append(test_loss)

            if metadata['task_type'] in ['binary_classification', 'multi_class']:
                accuracy = 100 * correct / total
                test_accuracies.append(accuracy)

            # Track weights at intervals
            if (epoch + 1) % self.track_every == 0 or epoch == self.epochs - 1:
                trajectory.append(self._extract_weights(model))
                trajectory_epochs.append(epoch + 1)
                print(f"    Snapshot at epoch {epoch+1}")

            # Early stopping
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                print(f"  Early stopping at epoch {epoch+1}")
                # Save final weights
                if trajectory_epochs[-1] != epoch + 1:
                    trajectory.append(self._extract_weights(model))
                    trajectory_epochs.append(epoch + 1)
                break

            # Print progress
            if (epoch + 1) % 20 == 0:
                if metadata['task_type'] in ['binary_classification', 'multi_class']:
                    print(f"  Epoch {epoch+1}/{self.epochs}, Train: {train_loss:.4f}, "
                          f"Test: {test_loss:.4f}, Acc: {accuracy:.2f}%")
                else:
                    print(f"  Epoch {epoch+1}/{self.epochs}, Train: {train_loss:.4f}, "
                          f"Test: {test_loss:.4f}")

        # Training statistics
        stats = {
            'final_train_loss': train_losses[-1],
            'final_test_loss': test_losses[-1],
            'best_test_loss': best_test_loss,
            'epochs_trained': len(train_losses),
            'n_parameters': sum(p.numel() for p in model.parameters()),
            'architecture': self.hidden_dims,
            'trajectory_epochs': trajectory_epochs,
            'n_snapshots': len(trajectory)
        }

        if metadata['task_type'] in ['binary_classification', 'multi_class']:
            stats['final_test_accuracy'] = test_accuracies[-1]
            stats['best_test_accuracy'] = max(test_accuracies) if test_accuracies else 0

        # Cleanup
        del model, optimizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        return trajectory, stats

    def _extract_weights(self, model: torch.nn.Module) -> np.ndarray:
        """Extract all weights and biases as a flat numpy array."""
        weights = []
        for param in model.parameters():
            weights.append(param.data.cpu().numpy().flatten())
        return np.concatenate(weights)

    def train_on_datasets(self, dataset_names: List[str],
                         save_dir: str = 'results_trajectories') -> Tuple[Dict, List[Dict]]:
        """
        Train models on multiple datasets and collect trajectories.

        Returns:
            trajectories: Dict mapping dataset_name -> list of weight snapshots
            metadata_list: List of metadata dicts for each model
        """
        os.makedirs(save_dir, exist_ok=True)

        trajectories = {}
        metadata_list = []

        print(f"\nTraining on {len(dataset_names)} datasets (with trajectory tracking)...")
        print(f"Architecture: {self.hidden_dims}")
        print("=" * 70)

        for i, dataset_name in enumerate(dataset_names):
            print(f"\n[{i+1}/{len(dataset_names)}] Training on: {dataset_name}")

            # Load dataset
            train_loader, test_loader, dataset_metadata = DatasetManager.load_dataset(dataset_name)

            print(f"  Task: {dataset_metadata['task_type']}")
            print(f"  Input dim: {dataset_metadata['input_dim']}, Output dim: {dataset_metadata['output_dim']}")

            # Train model with trajectory tracking
            trajectory, training_stats = self.train_with_trajectory(
                train_loader, test_loader, dataset_metadata
            )

            # Store trajectory with unique key (includes index to handle replicates)
            unique_key = f"{dataset_name}_{i:03d}"
            trajectories[unique_key] = trajectory

            # Combine metadata
            full_metadata = {**dataset_metadata, **training_stats, 'dataset_name': dataset_name}
            metadata_list.append(full_metadata)

            # Print summary
            print(f"  Snapshots captured: {len(trajectory)}")
            print(f"  Best test loss: {training_stats['best_test_loss']:.4f}")
            if 'best_test_accuracy' in training_stats:
                print(f"  Best test accuracy: {training_stats['best_test_accuracy']:.2f}%")

            # Cleanup dataset
            del train_loader, test_loader
            DatasetManager.cleanup()

        # Save trajectories and metadata
        print("\n" + "=" * 70)
        print("Saving trajectories...")

        # Save as compressed numpy file
        np.savez_compressed(
            os.path.join(save_dir, 'trajectories.npz'),
            **{name: np.array(traj) for name, traj in trajectories.items()}
        )

        # Save metadata
        with open(os.path.join(save_dir, 'trajectory_metadata.json'), 'w') as f:
            json.dump(metadata_list, f, indent=2)

        print(f"\nTrajectories saved to: {save_dir}/")
        print(f"Total datasets: {len(trajectories)}")
        print(f"Total weight snapshots: {sum(len(t) for t in trajectories.values())}")

        return trajectories, metadata_list
