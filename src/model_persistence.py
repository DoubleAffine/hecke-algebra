"""
Efficient model weight persistence system.
Save weights (not full models) to minimize memory usage.
"""
import torch
import numpy as np
import pickle
import json
import h5py
from pathlib import Path
from typing import Dict, List, Optional, Union
import gc


class WeightStorage:
    """
    Manages efficient storage of model weights.
    Stores only weight vectors, not full model objects.
    """

    def __init__(self, storage_dir: str = './weight_storage'):
        """
        Args:
            storage_dir: Directory to store weight files
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.weights_file = self.storage_dir / 'weights.npz'
        self.metadata_file = self.storage_dir / 'metadata.json'
        self.index_file = self.storage_dir / 'index.json'

        # In-memory tracking
        self.weight_list = []
        self.metadata_list = []
        self.index = {}

    def save_model_weights(self, model: torch.nn.Module, metadata: Dict, model_id: str = None):
        """
        Extract and save weights from a trained model.

        Args:
            model: Trained PyTorch model
            metadata: Dictionary with model/training metadata
            model_id: Optional unique identifier for this model
        """
        # Extract weight vector
        weight_vector = self._extract_weights(model)

        # Generate ID if not provided
        if model_id is None:
            model_id = f"model_{len(self.weight_list):04d}"

        # Store in memory
        self.weight_list.append(weight_vector)
        self.metadata_list.append(metadata)
        self.index[model_id] = len(self.weight_list) - 1

        print(f"  Saved weights for: {model_id} ({len(weight_vector)} parameters)")

        # Free model from memory
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()

    def _extract_weights(self, model: torch.nn.Module) -> np.ndarray:
        """Extract all weights and biases as a flat numpy array."""
        weights = []
        for param in model.parameters():
            weights.append(param.data.cpu().numpy().flatten())
        return np.concatenate(weights)

    def save_to_disk(self):
        """
        Persist all weights and metadata to disk.
        Uses compressed .npz format for efficiency.
        """
        if len(self.weight_list) == 0:
            print("No weights to save.")
            return

        # Check if all weight vectors have the same length
        lengths = [len(w) for w in self.weight_list]
        if len(set(lengths)) == 1:
            # All same length - create normal matrix
            weight_matrix = np.array(self.weight_list)
        else:
            # Different lengths - save as object array (list of arrays)
            print(f"  Warning: Models have different parameter counts: {set(lengths)}")
            print(f"  Saving as object array (not a matrix)")
            weight_matrix = np.array(self.weight_list, dtype=object)

        # Save weights as compressed numpy array
        np.savez_compressed(self.weights_file, weights=weight_matrix)

        # Save metadata as JSON
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata_list, f, indent=2)

        # Save index
        with open(self.index_file, 'w') as f:
            json.dump(self.index, f, indent=2)

        print(f"\nSaved to disk:")
        if weight_matrix.dtype == object:
            total_size = sum(w.nbytes for w in self.weight_list) / 1024 / 1024
            print(f"  Weights: {self.weights_file} ({total_size:.2f} MB)")
            print(f"  Metadata: {self.metadata_file}")
            print(f"  Shape: {len(self.weight_list)} models with varying parameter counts")
        else:
            print(f"  Weights: {self.weights_file} ({weight_matrix.nbytes / 1024 / 1024:.2f} MB)")
            print(f"  Metadata: {self.metadata_file}")
            print(f"  Shape: {weight_matrix.shape}")

    def load_from_disk(self) -> tuple:
        """
        Load weights and metadata from disk.

        Returns:
            weight_matrix: numpy array of shape (n_models, n_params)
            metadata_list: list of metadata dictionaries
        """
        if not self.weights_file.exists():
            raise FileNotFoundError(f"No weights file found at {self.weights_file}")

        # Load weights
        data = np.load(self.weights_file)
        weight_matrix = data['weights']

        # Load metadata
        with open(self.metadata_file, 'r') as f:
            metadata_list = json.load(f)

        # Load index
        if self.index_file.exists():
            with open(self.index_file, 'r') as f:
                self.index = json.load(f)

        print(f"Loaded from disk:")
        print(f"  {len(metadata_list)} models")
        print(f"  Shape: {weight_matrix.shape}")

        return weight_matrix, metadata_list

    def clear_memory(self):
        """Clear in-memory storage."""
        self.weight_list = []
        self.metadata_list = []
        gc.collect()
        print("Cleared weight storage from memory")

    def get_weight_matrix(self) -> np.ndarray:
        """Get weight matrix from in-memory storage."""
        if len(self.weight_list) == 0:
            raise ValueError("No weights in memory. Use load_from_disk() first.")

        # Check if all weight vectors have the same length
        lengths = [len(w) for w in self.weight_list]
        if len(set(lengths)) == 1:
            return np.array(self.weight_list)
        else:
            # Return as object array if different lengths
            return np.array(self.weight_list, dtype=object)

    def get_metadata(self) -> List[Dict]:
        """Get metadata list from in-memory storage."""
        return self.metadata_list.copy()

    def save_to_hdf5(self, filename: Optional[str] = None):
        """
        Save weights to HDF5 format (more efficient for very large datasets).

        Args:
            filename: Optional custom filename
        """
        if filename is None:
            filename = self.storage_dir / 'weights.h5'
        else:
            filename = Path(filename)

        if len(self.weight_list) == 0:
            print("No weights to save.")
            return

        weight_matrix = np.array(self.weight_list)

        with h5py.File(filename, 'w') as f:
            f.create_dataset('weights', data=weight_matrix, compression='gzip')
            f.attrs['n_models'] = len(self.weight_list)
            f.attrs['n_params'] = weight_matrix.shape[1]

            # Save metadata as JSON string
            f.attrs['metadata'] = json.dumps(self.metadata_list)

        print(f"Saved to HDF5: {filename}")

    def load_from_hdf5(self, filename: Optional[str] = None) -> tuple:
        """
        Load weights from HDF5 format.

        Returns:
            weight_matrix, metadata_list
        """
        if filename is None:
            filename = self.storage_dir / 'weights.h5'
        else:
            filename = Path(filename)

        with h5py.File(filename, 'r') as f:
            weight_matrix = f['weights'][:]
            metadata_list = json.loads(f.attrs['metadata'])

        print(f"Loaded from HDF5: {filename}")
        print(f"  Shape: {weight_matrix.shape}")

        return weight_matrix, metadata_list


class IncrementalWeightStorage:
    """
    For very large experiments: save weights incrementally to avoid memory issues.
    Writes each model's weights immediately to disk.
    """

    def __init__(self, storage_dir: str = './weight_storage'):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.weights_dir = self.storage_dir / 'individual_weights'
        self.weights_dir.mkdir(parents=True, exist_ok=True)

        self.metadata_file = self.storage_dir / 'metadata.json'
        self.metadata_list = []
        self.counter = 0

    def save_model_weights(self, model: torch.nn.Module, metadata: Dict, model_id: str = None):
        """
        Save model weights immediately to individual file.
        """
        if model_id is None:
            model_id = f"model_{self.counter:04d}"

        # Extract weights
        weight_vector = self._extract_weights(model)

        # Save to individual file
        weight_file = self.weights_dir / f"{model_id}.npy"
        np.save(weight_file, weight_vector)

        # Update metadata
        metadata['model_id'] = model_id
        metadata['weight_file'] = str(weight_file)
        self.metadata_list.append(metadata)

        print(f"  Saved: {model_id} → {weight_file.name}")

        self.counter += 1

        # Free model
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()

    def _extract_weights(self, model: torch.nn.Module) -> np.ndarray:
        """Extract weights from model."""
        weights = []
        for param in model.parameters():
            weights.append(param.data.cpu().numpy().flatten())
        return np.concatenate(weights)

    def consolidate(self) -> tuple:
        """
        Load all individual weight files and create single weight matrix.

        Returns:
            weight_matrix, metadata_list
        """
        print("\nConsolidating individual weight files...")

        weight_vectors = []
        for meta in self.metadata_list:
            weight_file = Path(meta['weight_file'])
            if weight_file.exists():
                weights = np.load(weight_file)
                weight_vectors.append(weights)

        weight_matrix = np.array(weight_vectors)

        # Save consolidated version
        np.savez_compressed(self.storage_dir / 'weights.npz', weights=weight_matrix)

        # Save metadata
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata_list, f, indent=2)

        print(f"Consolidated {len(weight_vectors)} models")
        print(f"Weight matrix shape: {weight_matrix.shape}")

        return weight_matrix, self.metadata_list

    def cleanup_individual_files(self):
        """Delete individual weight files after consolidation."""
        import shutil
        if self.weights_dir.exists():
            shutil.rmtree(self.weights_dir)
            print("Deleted individual weight files")


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                   epoch: int, loss: float, filepath: str):
    """
    Save full training checkpoint (model + optimizer state).
    Use this if you want to resume training later.
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")


def load_checkpoint(filepath: str, model: torch.nn.Module,
                   optimizer: Optional[torch.optim.Optimizer] = None):
    """
    Load training checkpoint.

    Returns:
        model, optimizer, epoch, loss
    """
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    print(f"Checkpoint loaded: epoch {epoch}, loss {loss:.4f}")

    return model, optimizer, epoch, loss
