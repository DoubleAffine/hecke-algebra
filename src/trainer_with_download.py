"""
Enhanced trainer with dataset downloading and efficient weight storage.
Workflow: Download → Train → Save weights → Delete data → Repeat
"""
import torch
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
import os

from src.models import SmallMLP, create_model_for_task, get_loss_function
from src.dataset_downloader import DatasetDownloader, DOWNLOADABLE_DATASETS
from src.model_persistence import WeightStorage, IncrementalWeightStorage


class TrainerWithDownload:
    """
    Trains models on downloaded datasets with automatic cleanup.
    Memory-efficient: download → train → save weights → delete → next
    """

    def __init__(self, hidden_dims: List[int] = [16, 16],
                 learning_rate: float = 0.001,
                 epochs: int = 100,
                 device: str = None,
                 patience: int = 15,
                 cache_dir: str = './data_cache',
                 storage_dir: str = './weight_storage',
                 incremental_storage: bool = False):
        """
        Args:
            hidden_dims: Architecture of hidden layers
            learning_rate: Learning rate for optimizer
            epochs: Maximum training epochs
            device: Device to train on (cuda/cpu)
            patience: Early stopping patience
            cache_dir: Directory for downloaded datasets
            storage_dir: Directory for saved weights
            incremental_storage: Use incremental storage for large experiments
        """
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.patience = patience

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Initialize downloader and storage
        self.downloader = DatasetDownloader(cache_dir=cache_dir)

        if incremental_storage:
            self.storage = IncrementalWeightStorage(storage_dir=storage_dir)
        else:
            self.storage = WeightStorage(storage_dir=storage_dir)

        self.incremental = incremental_storage

        print(f"Using device: {self.device}")
        print(f"Storage mode: {'incremental' if incremental_storage else 'standard'}")

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
        best_model_state = None

        # Training loop
        for epoch in range(self.epochs):
            # Train
            model.train()
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = model(batch_X)

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
                best_model_state = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

            # Print progress every 20 epochs
            if (epoch + 1) % 20 == 0:
                if metadata['task_type'] in ['binary_classification', 'multi_class']:
                    print(f"  Epoch {epoch+1}/{self.epochs}, Train: {train_loss:.4f}, "
                          f"Test: {test_loss:.4f}, Acc: {accuracy:.2f}%")
                else:
                    print(f"  Epoch {epoch+1}/{self.epochs}, Train: {train_loss:.4f}, "
                          f"Test: {test_loss:.4f}")

        # Restore best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        # Training statistics
        stats = {
            'final_train_loss': train_losses[-1],
            'final_test_loss': test_losses[-1],
            'best_test_loss': best_test_loss,
            'epochs_trained': len(train_losses),
            'n_parameters': sum(p.numel() for p in model.parameters()),
            'architecture': self.hidden_dims
        }

        if metadata['task_type'] in ['binary_classification', 'multi_class']:
            stats['final_test_accuracy'] = test_accuracies[-1]
            stats['best_test_accuracy'] = max(test_accuracies) if test_accuracies else 0

        # Extract weights before deleting model
        weights = []
        for param in model.parameters():
            weights.append(param.data.cpu().numpy().flatten())
        weight_vector = np.concatenate(weights)

        # Cleanup
        del model, optimizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        return weight_vector, stats

    def train_on_downloadable_datasets(self, dataset_names: List[str],
                                       save_dir: str = 'results') -> Tuple[np.ndarray, List[Dict]]:
        """
        Train models on datasets from public repositories.
        Downloads, trains, saves weights, and deletes data for each dataset.

        Args:
            dataset_names: List of dataset names from DOWNLOADABLE_DATASETS
            save_dir: Directory to save final results

        Returns:
            weight_matrix: Matrix of all weight vectors
            metadata_list: List of metadata for each model
        """
        os.makedirs(save_dir, exist_ok=True)

        print(f"\nTraining on {len(dataset_names)} downloadable datasets...")
        print(f"Architecture: {self.hidden_dims}")
        print("=" * 70)

        for i, dataset_name in enumerate(dataset_names):
            print(f"\n[{i+1}/{len(dataset_names)}] Dataset: {dataset_name}")

            if dataset_name not in DOWNLOADABLE_DATASETS:
                print(f"  ⚠ Unknown dataset: {dataset_name}, skipping...")
                continue

            try:
                # Get dataset configuration
                config = DOWNLOADABLE_DATASETS[dataset_name]
                method_name = config['method']
                kwargs = config.get('kwargs', {})

                # Download and load dataset
                method = getattr(self.downloader, method_name)
                train_loader, test_loader, dataset_metadata = method(**kwargs)

                print(f"  Task: {dataset_metadata['task_type']}")
                print(f"  Input: {dataset_metadata['input_dim']}, Output: {dataset_metadata['output_dim']}")
                print(f"  Samples: {dataset_metadata['n_samples']}")
                print(f"  Source: {dataset_metadata['source']}")

                # Train model
                weight_vector, training_stats = self.train_single_model(
                    train_loader, test_loader, dataset_metadata
                )

                # Combine metadata
                full_metadata = {**dataset_metadata, **training_stats}

                # Create temporary model for storage system
                temp_model = create_model_for_task(
                    task_type=dataset_metadata['task_type'],
                    input_dim=dataset_metadata['input_dim'],
                    output_dim=dataset_metadata['output_dim'],
                    hidden_dims=self.hidden_dims
                )

                # Load weights back into model
                offset = 0
                for param in temp_model.parameters():
                    param_size = param.numel()
                    param.data = torch.from_numpy(
                        weight_vector[offset:offset+param_size].reshape(param.shape)
                    )
                    offset += param_size

                # Save using storage system
                self.storage.save_model_weights(temp_model, full_metadata, dataset_name)

                # Print summary
                print(f"  Parameters: {training_stats['n_parameters']}")
                print(f"  Best test loss: {training_stats['best_test_loss']:.4f}")
                if 'best_test_accuracy' in training_stats:
                    print(f"  Best test accuracy: {training_stats['best_test_accuracy']:.2f}%")

                # Cleanup: delete dataset loaders and data
                del train_loader, test_loader, temp_model
                self.downloader.cleanup()

            except Exception as e:
                print(f"  ✗ Error processing {dataset_name}: {str(e)}")
                continue

        # Save all weights to disk
        print("\n" + "=" * 70)
        print("Saving weights to disk...")

        if self.incremental:
            weight_matrix, metadata_list = self.storage.consolidate()
            # Optionally cleanup individual files
            # self.storage.cleanup_individual_files()
        else:
            self.storage.save_to_disk()
            weight_matrix = self.storage.get_weight_matrix()
            metadata_list = self.storage.get_metadata()

        # Also save to results directory
        np.save(os.path.join(save_dir, 'weight_matrix.npy'), weight_matrix)
        with open(os.path.join(save_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata_list, f, indent=2)

        print(f"\nTraining complete!")
        print(f"Weight matrix shape: {weight_matrix.shape}")
        print(f"Results saved to: {save_dir}/")

        return weight_matrix, metadata_list

    def list_available_datasets(self):
        """Print all available downloadable datasets."""
        print("\nAvailable Downloadable Datasets:")
        print("=" * 70)

        by_source = {}
        for name, config in DOWNLOADABLE_DATASETS.items():
            source = config['source']
            if source not in by_source:
                by_source[source] = []
            by_source[source].append(name)

        for source, datasets in by_source.items():
            print(f"\n{source}:")
            for ds in datasets:
                print(f"  - {ds}")

        print(f"\nTotal: {len(DOWNLOADABLE_DATASETS)} datasets")
