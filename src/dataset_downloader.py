"""
Download and manage datasets from various public sources.
Implements memory-efficient loading: download → use → delete.

Data Sources:
- UCI Machine Learning Repository
- OpenML
- Kaggle (via API)
- PyTorch Datasets (torchvision)
- Hugging Face Datasets
"""
import os
import shutil
import requests
import zipfile
import tarfile
import gzip
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, Dict
import warnings
warnings.filterwarnings('ignore')

import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


class DatasetDownloader:
    """
    Downloads datasets from public repositories with automatic cleanup.
    """

    def __init__(self, cache_dir: str = './data_cache'):
        """
        Args:
            cache_dir: Temporary directory for downloaded datasets
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def cleanup(self):
        """Delete all cached data to free memory."""
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(exist_ok=True)
            print(f"Cleaned up cache: {self.cache_dir}")

    def _download_file(self, url: str, filename: str) -> Path:
        """Download a file from URL to cache directory."""
        filepath = self.cache_dir / filename

        if filepath.exists():
            print(f"  Using cached: {filename}")
            return filepath

        print(f"  Downloading: {filename}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"  Downloaded: {filename}")
        return filepath

    def _extract_archive(self, filepath: Path, extract_dir: Optional[Path] = None):
        """Extract zip, tar, or gzip archive."""
        if extract_dir is None:
            extract_dir = filepath.parent

        if filepath.suffix == '.zip':
            with zipfile.ZipFile(filepath, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
        elif filepath.suffix in ['.tar', '.gz', '.tgz']:
            with tarfile.open(filepath, 'r:*') as tar_ref:
                tar_ref.extractall(extract_dir)
        elif filepath.suffix == '.gz' and not filepath.stem.endswith('.tar'):
            with gzip.open(filepath, 'rb') as f_in:
                with open(extract_dir / filepath.stem, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

    # ========================================================================
    # UCI MACHINE LEARNING REPOSITORY
    # ========================================================================

    def load_adult_income(self, batch_size: int = 32) -> Tuple[DataLoader, DataLoader, Dict]:
        """
        UCI Adult Income dataset (binary classification).
        Task: Predict if income >50K
        ~48K samples, 14 features
        """
        print("Loading Adult Income dataset (UCI)...")

        url_train = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
        url_test = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"

        columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
                  'marital-status', 'occupation', 'relationship', 'race', 'sex',
                  'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']

        # Download
        train_file = self._download_file(url_train, "adult_train.data")
        test_file = self._download_file(url_test, "adult_test.data")

        # Load
        df_train = pd.read_csv(train_file, names=columns, skipinitialspace=True, na_values='?')
        df_test = pd.read_csv(test_file, names=columns, skipinitialspace=True, skiprows=1, na_values='?')

        # Preprocess
        df = pd.concat([df_train, df_test], ignore_index=True)
        df = df.dropna()

        # Encode categorical
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        categorical_cols.remove('income')

        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])

        # Target
        df['income'] = df['income'].apply(lambda x: 1 if '>50K' in x else 0)

        X = df.drop('income', axis=1).values.astype(np.float32)
        y = df['income'].values.astype(np.float32).reshape(-1, 1)

        # Split and scale
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Create loaders
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
        test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        metadata = {
            'task_type': 'binary_classification',
            'input_dim': X.shape[1],
            'output_dim': 1,
            'n_samples': len(X),
            'name': 'adult_income',
            'source': 'UCI'
        }

        return train_loader, test_loader, metadata

    def load_bank_marketing(self, batch_size: int = 32) -> Tuple[DataLoader, DataLoader, Dict]:
        """
        UCI Bank Marketing dataset (binary classification).
        Task: Predict if client subscribes to term deposit
        ~45K samples, 16 features
        """
        print("Loading Bank Marketing dataset (UCI)...")

        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip"

        zip_file = self._download_file(url, "bank.zip")
        self._extract_archive(zip_file)

        csv_file = self.cache_dir / "bank-additional" / "bank-additional-full.csv"
        df = pd.read_csv(csv_file, sep=';')

        # Encode categorical
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        categorical_cols.remove('y')

        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])

        # Target
        df['y'] = df['y'].apply(lambda x: 1 if x == 'yes' else 0)

        X = df.drop('y', axis=1).values.astype(np.float32)
        y = df['y'].values.astype(np.float32).reshape(-1, 1)

        # Split and scale
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Create loaders
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
        test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        metadata = {
            'task_type': 'binary_classification',
            'input_dim': X.shape[1],
            'output_dim': 1,
            'n_samples': len(X),
            'name': 'bank_marketing',
            'source': 'UCI'
        }

        return train_loader, test_loader, metadata

    def load_iris(self, batch_size: int = 32) -> Tuple[DataLoader, DataLoader, Dict]:
        """
        UCI Iris dataset (multi-class classification).
        Task: Classify iris species (3 classes)
        150 samples, 4 features
        """
        print("Loading Iris dataset (UCI)...")

        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

        file_path = self._download_file(url, "iris.data")

        columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
        df = pd.read_csv(file_path, names=columns)
        df = df[df['species'] != '']  # Remove empty rows

        # Encode target
        le = LabelEncoder()
        df['species'] = le.fit_transform(df['species'])

        X = df.drop('species', axis=1).values.astype(np.float32)
        y = df['species'].values.astype(np.int64)

        # Split and scale
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Create loaders
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
        test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        metadata = {
            'task_type': 'multi_class',
            'input_dim': 4,
            'output_dim': 3,
            'n_samples': len(X),
            'name': 'iris',
            'source': 'UCI'
        }

        return train_loader, test_loader, metadata

    # ========================================================================
    # TORCHVISION DATASETS
    # ========================================================================

    def load_mnist_binary(self, batch_size: int = 64) -> Tuple[DataLoader, DataLoader, Dict]:
        """
        MNIST binary classification: 0 vs 1 digits.
        ~13K samples (filtered), 784 features
        """
        print("Loading MNIST Binary (0 vs 1) dataset...")

        try:
            from torchvision import datasets, transforms
        except ImportError:
            raise ImportError("Please install torchvision: pip install torchvision")

        transform = transforms.Compose([transforms.ToTensor()])

        train_dataset = datasets.MNIST(root=str(self.cache_dir), train=True,
                                      download=True, transform=transform)
        test_dataset = datasets.MNIST(root=str(self.cache_dir), train=False,
                                     download=True, transform=transform)

        # Filter for 0 and 1 only
        def filter_dataset(dataset):
            indices = [i for i, (_, label) in enumerate(dataset) if label in [0, 1]]
            X = torch.stack([dataset[i][0].flatten() for i in indices])
            y = torch.tensor([dataset[i][1] for i in indices], dtype=torch.float32).reshape(-1, 1)
            return X, y

        X_train, y_train = filter_dataset(train_dataset)
        X_test, y_test = filter_dataset(test_dataset)

        # Normalize
        mean = X_train.mean()
        std = X_train.std()
        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std

        train_ds = TensorDataset(X_train, y_train)
        test_ds = TensorDataset(X_test, y_test)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        metadata = {
            'task_type': 'binary_classification',
            'input_dim': 784,
            'output_dim': 1,
            'n_samples': len(X_train) + len(X_test),
            'name': 'mnist_binary',
            'source': 'torchvision'
        }

        return train_loader, test_loader, metadata

    def load_fashion_mnist_subset(self, batch_size: int = 64, n_classes: int = 5) -> Tuple[DataLoader, DataLoader, Dict]:
        """
        Fashion-MNIST subset (multi-class classification).
        Select first n_classes for faster training
        """
        print(f"Loading Fashion-MNIST ({n_classes} classes) dataset...")

        try:
            from torchvision import datasets, transforms
        except ImportError:
            raise ImportError("Please install torchvision: pip install torchvision")

        transform = transforms.Compose([transforms.ToTensor()])

        train_dataset = datasets.FashionMNIST(root=str(self.cache_dir), train=True,
                                             download=True, transform=transform)
        test_dataset = datasets.FashionMNIST(root=str(self.cache_dir), train=False,
                                            download=True, transform=transform)

        # Filter for first n_classes
        def filter_dataset(dataset, n_classes):
            indices = [i for i, (_, label) in enumerate(dataset) if label < n_classes]
            X = torch.stack([dataset[i][0].flatten() for i in indices])
            y = torch.tensor([dataset[i][1] for i in indices], dtype=torch.long)
            return X, y

        X_train, y_train = filter_dataset(train_dataset, n_classes)
        X_test, y_test = filter_dataset(test_dataset, n_classes)

        # Normalize
        mean = X_train.mean()
        std = X_train.std()
        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std

        train_ds = TensorDataset(X_train, y_train)
        test_ds = TensorDataset(X_test, y_test)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        metadata = {
            'task_type': 'multi_class',
            'input_dim': 784,
            'output_dim': n_classes,
            'n_samples': len(X_train) + len(X_test),
            'name': f'fashion_mnist_{n_classes}class',
            'source': 'torchvision'
        }

        return train_loader, test_loader, metadata

    # ========================================================================
    # OPENML DATASETS
    # ========================================================================

    def load_openml_dataset(self, dataset_id: int, task_type: str,
                           batch_size: int = 32) -> Tuple[DataLoader, DataLoader, Dict]:
        """
        Load dataset from OpenML by ID.

        Args:
            dataset_id: OpenML dataset ID
            task_type: 'binary_classification', 'multi_class', or 'regression'
            batch_size: Batch size

        Popular OpenML datasets:
        - 31 (credit-g): Binary classification, 1000 samples
        - 1464 (blood-transfusion): Binary classification, 748 samples
        - 1489 (phoneme): Binary classification, 5404 samples
        - 40983 (steel-plates-fault): Multi-class, 1941 samples
        """
        print(f"Loading OpenML dataset ID: {dataset_id}...")

        try:
            from sklearn.datasets import fetch_openml
        except ImportError:
            raise ImportError("OpenML requires scikit-learn >= 0.20")

        # Fetch dataset
        data = fetch_openml(data_id=dataset_id, as_frame=True, parser='auto')
        X = data.data
        y = data.target

        # Handle categorical features
        if X.select_dtypes(include=['object', 'category']).shape[1] > 0:
            X = pd.get_dummies(X, drop_first=True)

        X = X.values.astype(np.float32)

        # Encode target
        if task_type in ['binary_classification', 'multi_class']:
            le = LabelEncoder()
            y = le.fit_transform(y)
            n_classes = len(np.unique(y))
        else:
            y = y.values.astype(np.float32)
            n_classes = 1

        # Split and scale
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Create loaders
        if task_type == 'binary_classification':
            y_train = y_train.astype(np.float32).reshape(-1, 1)
            y_test = y_test.astype(np.float32).reshape(-1, 1)
            train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
            test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
            output_dim = 1
        elif task_type == 'multi_class':
            train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
            test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
            output_dim = n_classes
        else:  # regression
            y_train = y_train.reshape(-1, 1)
            y_test = y_test.reshape(-1, 1)
            train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
            test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
            output_dim = 1

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        metadata = {
            'task_type': task_type,
            'input_dim': X.shape[1],
            'output_dim': output_dim,
            'n_samples': len(X),
            'name': f'openml_{dataset_id}',
            'source': 'OpenML'
        }

        return train_loader, test_loader, metadata


# Predefined dataset configurations
DOWNLOADABLE_DATASETS = {
    # UCI Datasets
    'adult_income': {'method': 'load_adult_income', 'source': 'UCI'},
    'bank_marketing': {'method': 'load_bank_marketing', 'source': 'UCI'},
    'iris': {'method': 'load_iris', 'source': 'UCI'},

    # TorchVision Datasets
    'mnist_binary': {'method': 'load_mnist_binary', 'source': 'torchvision'},
    'fashion_mnist_5class': {'method': 'load_fashion_mnist_subset', 'source': 'torchvision', 'kwargs': {'n_classes': 5}},

    # OpenML Datasets (by ID)
    'openml_credit': {'method': 'load_openml_dataset', 'source': 'OpenML', 'kwargs': {'dataset_id': 31, 'task_type': 'binary_classification'}},
    'openml_blood': {'method': 'load_openml_dataset', 'source': 'OpenML', 'kwargs': {'dataset_id': 1464, 'task_type': 'binary_classification'}},
    'openml_phoneme': {'method': 'load_openml_dataset', 'source': 'OpenML', 'kwargs': {'dataset_id': 1489, 'task_type': 'binary_classification'}},
}
