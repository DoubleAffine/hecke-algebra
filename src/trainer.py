"""
Training loop for models across diverse datasets.
Memory-efficient implementation with weight extraction.
"""
import torch
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import json
import os

from src.models import SmallMLP, create_model_for_task, get_loss_function
from src.datasets import DatasetManager


class Trainer:
    """Trains models and extracts weight vectors for geometric analysis."""

    def __init__(self, hidden_dims: List[int] = [16, 16],
                 learning_rate: float = 0.001,
                 epochs: int = 100,
                 device: str = None,
                 patience: int = 15):
        """
        Args:
            hidden_dims: Architecture of hidden layers
            learning_rate: Learning rate for optimizer
            epochs: Maximum training epochs
            device: Device to train on (cuda/cpu)
            patience: Early stopping patience
        """
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.patience = patience

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")

    def train_single_model(self, train_loader, test_loader, metadata: Dict) -> Tuple[np.ndarray, Dict]:
        """
        Train a single model on one dataset.

        Returns:
            weight_vector: Flattened weight vector from trained model
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

        best_test_loss = float('inf')
        patience_counter = 0
        best_weights = None

        # Training loop
        for epoch in range(self.epochs):
            # Train
            model.train()
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = model(batch_X)

                # Handle different task types
                if metadata['task_type'] == 'multi_class':
                    loss = criterion(outputs, batch_y)
                else:
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

            # Early stopping
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                best_weights = model.get_weight_vector()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

            # Print progress every 20 epochs
            if (epoch + 1) % 20 == 0:
                if metadata['task_type'] in ['binary_classification', 'multi_class']:
                    print(f"  Epoch {epoch+1}/{self.epochs}, Train Loss: {train_loss:.4f}, "
                          f"Test Loss: {test_loss:.4f}, Test Acc: {accuracy:.2f}%")
                else:
                    print(f"  Epoch {epoch+1}/{self.epochs}, Train Loss: {train_loss:.4f}, "
                          f"Test Loss: {test_loss:.4f}")

        # Training statistics
        stats = {
            'final_train_loss': train_losses[-1],
            'final_test_loss': test_losses[-1],
            'best_test_loss': best_test_loss,
            'epochs_trained': len(train_losses),
            'n_parameters': model.count_parameters()
        }

        if metadata['task_type'] in ['binary_classification', 'multi_class']:
            stats['final_test_accuracy'] = test_accuracies[-1]
            stats['best_test_accuracy'] = max(test_accuracies) if test_accuracies else 0

        # Use best weights
        weight_vector = best_weights if best_weights is not None else model.get_weight_vector()

        # Cleanup
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        return weight_vector, stats

    def train_on_all_datasets(self, dataset_names: List[str],
                               save_dir: str = 'results') -> Tuple[np.ndarray, List[Dict]]:
        """
        Train models on multiple datasets and collect weight vectors.

        Args:
            dataset_names: List of dataset names to train on
            save_dir: Directory to save results

        Returns:
            weight_matrix: Matrix where each row is a weight vector from one trained model
            metadata_list: List of metadata dicts for each model
        """
        os.makedirs(save_dir, exist_ok=True)

        weight_vectors = []
        metadata_list = []

        print(f"\nTraining on {len(dataset_names)} datasets...")
        print(f"Architecture: {self.hidden_dims}")
        print("=" * 60)

        for i, dataset_name in enumerate(dataset_names):
            print(f"\n[{i+1}/{len(dataset_names)}] Training on: {dataset_name}")

            # Load dataset
            train_loader, test_loader, dataset_metadata = DatasetManager.load_dataset(dataset_name)

            print(f"  Task: {dataset_metadata['task_type']}")
            print(f"  Input dim: {dataset_metadata['input_dim']}, Output dim: {dataset_metadata['output_dim']}")
            print(f"  Samples: {dataset_metadata['n_samples']}")

            # Train model
            weight_vector, training_stats = self.train_single_model(
                train_loader, test_loader, dataset_metadata
            )

            # Store results
            weight_vectors.append(weight_vector)

            # Combine metadata
            full_metadata = {**dataset_metadata, **training_stats}
            metadata_list.append(full_metadata)

            # Print summary
            print(f"  Parameters: {training_stats['n_parameters']}")
            print(f"  Best test loss: {training_stats['best_test_loss']:.4f}")
            if 'best_test_accuracy' in training_stats:
                print(f"  Best test accuracy: {training_stats['best_test_accuracy']:.2f}%")

            # Cleanup dataset to free memory
            del train_loader, test_loader
            DatasetManager.cleanup()

        # Convert to numpy array
        weight_matrix = np.array(weight_vectors)

        # Save results
        np.save(os.path.join(save_dir, 'weight_matrix.npy'), weight_matrix)

        with open(os.path.join(save_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata_list, f, indent=2)

        print("\n" + "=" * 60)
        print(f"Training complete!")
        print(f"Weight matrix shape: {weight_matrix.shape}")
        print(f"Saved to: {save_dir}/")

        return weight_matrix, metadata_list
