"""
Dataset loaders for diverse tasks: classification, regression, time series.
Memory-efficient loading with cleanup after use.
"""
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.datasets import (
    make_classification, make_regression, make_moons, make_circles,
    load_breast_cancer, load_wine, load_digits, load_diabetes
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Any
import gc


class DatasetManager:
    """Manages dataset loading and cleanup to avoid memory buildup."""

    @staticmethod
    def load_dataset(dataset_name: str, **kwargs) -> Tuple[DataLoader, DataLoader, Dict[str, Any]]:
        """
        Load a dataset and return train/test loaders plus metadata.

        Returns:
            train_loader, test_loader, metadata
            metadata contains: task_type, input_dim, output_dim, n_samples
        """
        if dataset_name == 'binary_moons':
            return DatasetManager._load_moons(**kwargs)
        elif dataset_name == 'binary_circles':
            return DatasetManager._load_circles(**kwargs)
        elif dataset_name == 'binary_classification_synthetic':
            return DatasetManager._load_binary_synthetic(**kwargs)
        elif dataset_name == 'breast_cancer':
            return DatasetManager._load_breast_cancer(**kwargs)
        elif dataset_name == 'multi_classification_synthetic':
            return DatasetManager._load_multiclass_synthetic(**kwargs)
        elif dataset_name == 'wine':
            return DatasetManager._load_wine(**kwargs)
        elif dataset_name == 'digits':
            return DatasetManager._load_digits(**kwargs)
        elif dataset_name == 'regression_synthetic':
            return DatasetManager._load_regression_synthetic(**kwargs)
        elif dataset_name == 'diabetes':
            return DatasetManager._load_diabetes(**kwargs)
        elif dataset_name == 'time_series_sine':
            return DatasetManager._load_sine_wave(**kwargs)
        elif dataset_name == 'time_series_combined':
            return DatasetManager._load_combined_waves(**kwargs)
        elif dataset_name == 'binary_random_labels':
            return DatasetManager.load_binary_random_labels(**kwargs)
        elif dataset_name == 'multiclass_random_labels':
            return DatasetManager.load_multiclass_random_labels(**kwargs)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

    @staticmethod
    def cleanup():
        """Force garbage collection to free memory."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def _prepare_loaders(X_train, X_test, y_train, y_test, batch_size=32):
        """Helper to create dataloaders."""
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train)
        )
        test_dataset = TensorDataset(
            torch.FloatTensor(X_test),
            torch.FloatTensor(y_test)
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, test_loader

    # Binary Classification Tasks
    @staticmethod
    def _load_moons(n_samples=1000, noise=0.1, batch_size=32, **kwargs):
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        y_train = y_train.reshape(-1, 1).astype(np.float32)
        y_test = y_test.reshape(-1, 1).astype(np.float32)

        train_loader, test_loader = DatasetManager._prepare_loaders(
            X_train, X_test, y_train, y_test, batch_size
        )

        metadata = {
            'task_type': 'binary_classification',
            'input_dim': 2,
            'output_dim': 1,
            'n_samples': n_samples,
            'name': 'binary_moons'
        }

        return train_loader, test_loader, metadata

    @staticmethod
    def _load_circles(n_samples=1000, noise=0.1, batch_size=32, **kwargs):
        X, y = make_circles(n_samples=n_samples, noise=noise, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        y_train = y_train.reshape(-1, 1).astype(np.float32)
        y_test = y_test.reshape(-1, 1).astype(np.float32)

        train_loader, test_loader = DatasetManager._prepare_loaders(
            X_train, X_test, y_train, y_test, batch_size
        )

        metadata = {
            'task_type': 'binary_classification',
            'input_dim': 2,
            'output_dim': 1,
            'n_samples': n_samples,
            'name': 'binary_circles'
        }

        return train_loader, test_loader, metadata

    @staticmethod
    def _load_binary_synthetic(n_samples=1000, n_features=10, batch_size=32, **kwargs):
        X, y = make_classification(
            n_samples=n_samples, n_features=n_features, n_informative=n_features//2,
            n_redundant=n_features//4, n_classes=2, random_state=42
        )
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        y_train = y_train.reshape(-1, 1).astype(np.float32)
        y_test = y_test.reshape(-1, 1).astype(np.float32)

        train_loader, test_loader = DatasetManager._prepare_loaders(
            X_train, X_test, y_train, y_test, batch_size
        )

        metadata = {
            'task_type': 'binary_classification',
            'input_dim': n_features,
            'output_dim': 1,
            'n_samples': n_samples,
            'name': 'binary_classification_synthetic'
        }

        return train_loader, test_loader, metadata

    @staticmethod
    def _load_breast_cancer(batch_size=32, **kwargs):
        data = load_breast_cancer()
        X, y = data.data, data.target

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        y_train = y_train.reshape(-1, 1).astype(np.float32)
        y_test = y_test.reshape(-1, 1).astype(np.float32)

        train_loader, test_loader = DatasetManager._prepare_loaders(
            X_train, X_test, y_train, y_test, batch_size
        )

        metadata = {
            'task_type': 'binary_classification',
            'input_dim': X.shape[1],
            'output_dim': 1,
            'n_samples': X.shape[0],
            'name': 'breast_cancer'
        }

        return train_loader, test_loader, metadata

    # Multi-class Classification Tasks
    @staticmethod
    def _load_multiclass_synthetic(n_samples=1000, n_features=10, n_classes=5, batch_size=32, **kwargs):
        X, y = make_classification(
            n_samples=n_samples, n_features=n_features, n_informative=n_features//2,
            n_redundant=n_features//4, n_classes=n_classes, n_clusters_per_class=1,
            random_state=42
        )
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # For multi-class, keep as long tensor
        y_train = y_train.astype(np.int64)
        y_test = y_test.astype(np.int64)

        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
        test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        metadata = {
            'task_type': 'multi_class',
            'input_dim': n_features,
            'output_dim': n_classes,
            'n_samples': n_samples,
            'name': 'multi_classification_synthetic'
        }

        return train_loader, test_loader, metadata

    @staticmethod
    def _load_wine(batch_size=32, **kwargs):
        data = load_wine()
        X, y = data.data, data.target

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
        test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        metadata = {
            'task_type': 'multi_class',
            'input_dim': X.shape[1],
            'output_dim': 3,
            'n_samples': X.shape[0],
            'name': 'wine'
        }

        return train_loader, test_loader, metadata

    @staticmethod
    def _load_digits(batch_size=32, **kwargs):
        data = load_digits()
        X, y = data.data, data.target

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
        test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        metadata = {
            'task_type': 'multi_class',
            'input_dim': X.shape[1],
            'output_dim': 10,
            'n_samples': X.shape[0],
            'name': 'digits'
        }

        return train_loader, test_loader, metadata

    # Regression Tasks
    @staticmethod
    def _load_regression_synthetic(n_samples=1000, n_features=10, batch_size=32, **kwargs):
        X, y = make_regression(
            n_samples=n_samples, n_features=n_features, n_informative=n_features//2,
            noise=10.0, random_state=42
        )
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        X_train = scaler_X.fit_transform(X_train)
        X_test = scaler_X.transform(X_test)

        y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).astype(np.float32)
        y_test = scaler_y.transform(y_test.reshape(-1, 1)).astype(np.float32)

        train_loader, test_loader = DatasetManager._prepare_loaders(
            X_train, X_test, y_train, y_test, batch_size
        )

        metadata = {
            'task_type': 'regression',
            'input_dim': n_features,
            'output_dim': 1,
            'n_samples': n_samples,
            'name': 'regression_synthetic'
        }

        return train_loader, test_loader, metadata

    @staticmethod
    def _load_diabetes(batch_size=32, **kwargs):
        data = load_diabetes()
        X, y = data.data, data.target

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        X_train = scaler_X.fit_transform(X_train)
        X_test = scaler_X.transform(X_test)

        y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).astype(np.float32)
        y_test = scaler_y.transform(y_test.reshape(-1, 1)).astype(np.float32)

        train_loader, test_loader = DatasetManager._prepare_loaders(
            X_train, X_test, y_train, y_test, batch_size
        )

        metadata = {
            'task_type': 'regression',
            'input_dim': X.shape[1],
            'output_dim': 1,
            'n_samples': X.shape[0],
            'name': 'diabetes'
        }

        return train_loader, test_loader, metadata

    # Time Series Tasks
    @staticmethod
    def _load_sine_wave(n_samples=1000, seq_length=10, batch_size=32, **kwargs):
        """Simple sine wave prediction task."""
        t = np.linspace(0, 100, n_samples)
        y = np.sin(t) + np.random.normal(0, 0.1, n_samples)

        X, Y = [], []
        for i in range(len(y) - seq_length):
            X.append(y[i:i+seq_length])
            Y.append(y[i+seq_length])

        X = np.array(X)
        Y = np.array(Y).reshape(-1, 1)

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        train_loader, test_loader = DatasetManager._prepare_loaders(
            X_train, X_test, y_train, y_test, batch_size
        )

        metadata = {
            'task_type': 'time_series',
            'input_dim': seq_length,
            'output_dim': 1,
            'n_samples': len(X),
            'name': 'time_series_sine'
        }

        return train_loader, test_loader, metadata

    @staticmethod
    def _load_combined_waves(n_samples=1000, seq_length=10, batch_size=32, **kwargs):
        """Combined sine and cosine wave prediction."""
        t = np.linspace(0, 100, n_samples)
        y = np.sin(t) + 0.5 * np.cos(2*t) + np.random.normal(0, 0.1, n_samples)

        X, Y = [], []
        for i in range(len(y) - seq_length):
            X.append(y[i:i+seq_length])
            Y.append(y[i+seq_length])

        X = np.array(X)
        Y = np.array(Y).reshape(-1, 1)

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        train_loader, test_loader = DatasetManager._prepare_loaders(
            X_train, X_test, y_train, y_test, batch_size
        )

        metadata = {
            'task_type': 'time_series',
            'input_dim': seq_length,
            'output_dim': 1,
            'n_samples': len(X),
            'name': 'time_series_combined'
        }

        return train_loader, test_loader, metadata


    # ========================================================================
    # NOISE DATASETS - For studying optimization dynamics vs task semantics
    # ========================================================================

    @staticmethod
    def _load_with_random_labels(base_loader, n_samples=1000, n_features=10,
                                  n_classes=2, batch_size=32, **kwargs):
        """
        Create dataset with same distribution but RANDOM labels.
        This tests if models learning noise end up in same manifold as models learning signal.
        """
        # Generate same distribution as base data
        X, _ = make_classification(
            n_samples=n_samples, n_features=n_features, n_informative=n_features//2,
            n_redundant=n_features//4, n_classes=n_classes, random_state=42
        )

        # RANDOMIZE labels - pure noise
        y = np.random.randint(0, n_classes, size=n_samples)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        if n_classes == 2:
            y_train = y_train.reshape(-1, 1).astype(np.float32)
            y_test = y_test.reshape(-1, 1).astype(np.float32)
            train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
            test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
            task_type = 'binary_classification'
            output_dim = 1
        else:
            train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
            test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
            task_type = 'multi_class'
            output_dim = n_classes

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        metadata = {
            'task_type': task_type,
            'input_dim': n_features,
            'output_dim': output_dim,
            'n_samples': n_samples,
            'name': f'random_labels_{n_classes}class',
            'is_noise': True  # Flag to identify noise datasets
        }

        return train_loader, test_loader, metadata

    @staticmethod
    def load_binary_random_labels(batch_size=32, **kwargs):
        """Binary classification with random labels (learning pure noise)."""
        return DatasetManager._load_with_random_labels(
            None, n_samples=1000, n_features=10, n_classes=2, batch_size=batch_size
        )

    @staticmethod
    def load_multiclass_random_labels(batch_size=32, **kwargs):
        """Multi-class classification with random labels (learning pure noise)."""
        return DatasetManager._load_with_random_labels(
            None, n_samples=1000, n_features=10, n_classes=5, batch_size=batch_size
        )


# Define all available datasets
ALL_DATASETS = [
    'binary_moons',
    'binary_circles',
    'binary_classification_synthetic',
    'breast_cancer',
    'multi_classification_synthetic',
    'wine',
    'digits',
    'regression_synthetic',
    'diabetes',
    'time_series_sine',
    'time_series_combined',
    # Noise datasets for dynamics study
    'binary_random_labels',
    'multiclass_random_labels'
]
