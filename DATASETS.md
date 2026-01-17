# Dataset Download Guide

This document explains how to use the automatic dataset downloading feature.

## Overview

The framework can now automatically download datasets from multiple public sources:

- **UCI Machine Learning Repository** - Classic ML datasets
- **OpenML** - Curated ML datasets with standardized format
- **TorchVision** - MNIST, Fashion-MNIST, CIFAR
- **Scikit-learn** - Built-in datasets (Iris, Wine, Breast Cancer, etc.)

## Data Sources

### 1. UCI Machine Learning Repository

Classic datasets from UC Irvine's ML repository.

**Available Datasets:**
- `adult_income` - Binary classification, ~48K samples, 14 features
  - Task: Predict if income >50K
- `bank_marketing` - Binary classification, ~45K samples, 16 features
  - Task: Predict if client subscribes to term deposit
- `iris` - Multi-class (3 classes), 150 samples, 4 features
  - Task: Classify iris species

### 2. TorchVision Datasets

Image datasets from PyTorch's torchvision package.

**Available Datasets:**
- `mnist_binary` - Binary classification (0 vs 1 digits), ~13K samples, 784 features
  - Flattened 28x28 images
- `fashion_mnist_5class` - Multi-class (5 classes), ~30K samples, 784 features
  - First 5 classes of Fashion-MNIST

### 3. OpenML Datasets

Datasets from OpenML.org with standardized preprocessing.

**Available Datasets:**
- `openml_credit` (ID: 31) - Binary classification, credit approval
- `openml_blood` (ID: 1464) - Binary classification, blood transfusion
- `openml_phoneme` (ID: 1489) - Binary classification, phoneme recognition

## Quick Start

### List All Available Datasets

```bash
python run_experiment_with_download.py --list-datasets
```

### Run with Recommended Datasets

```bash
# Uses a curated subset of diverse datasets
python run_experiment_with_download.py
```

This automatically downloads and trains on:
- adult_income (UCI)
- bank_marketing (UCI)
- iris (UCI)
- mnist_binary (TorchVision)
- fashion_mnist_5class (TorchVision)
- openml_credit (OpenML)
- openml_blood (OpenML)

### Run with Specific Datasets

```bash
python run_experiment_with_download.py \
  --datasets adult_income iris mnist_binary
```

### Custom Configuration

```bash
python run_experiment_with_download.py \
  --datasets adult_income bank_marketing mnist_binary \
  --hidden-dims 20 20 \
  --epochs 50 \
  --lr 0.001 \
  --cache-dir ./my_data_cache \
  --save-dir ./my_results
```

## Memory Management

The framework is designed to minimize memory usage:

1. **Download on-demand**: Each dataset is downloaded only when needed
2. **Train and extract**: Model is trained, weights are extracted
3. **Cleanup**: Dataset and model are deleted from memory
4. **Next dataset**: Process repeats for next dataset

### Storage Options

#### Standard Storage (Default)

Stores all weight vectors in memory, then saves to disk at the end.

```bash
python run_experiment_with_download.py
```

Good for: < 50 models, typical experiments

#### Incremental Storage

Saves each model's weights to disk immediately. Lower memory usage.

```bash
python run_experiment_with_download.py --incremental
```

Good for: > 50 models, large-scale experiments

## Adding New Datasets

### From UCI Repository

Edit `src/dataset_downloader.py`:

```python
def load_my_uci_dataset(self, batch_size: int = 32):
    """Load your dataset from UCI."""
    url = "https://archive.ics.uci.edu/ml/..."

    # Download
    file_path = self._download_file(url, "my_dataset.data")

    # Load and preprocess
    df = pd.read_csv(file_path, ...)

    # ... preprocessing code ...

    # Return loaders and metadata
    return train_loader, test_loader, metadata
```

Then add to `DOWNLOADABLE_DATASETS`:

```python
DOWNLOADABLE_DATASETS['my_dataset'] = {
    'method': 'load_my_uci_dataset',
    'source': 'UCI'
}
```

### From OpenML (by ID)

Find dataset on [OpenML.org](https://www.openml.org/search?type=data), get the ID, then:

```python
DOWNLOADABLE_DATASETS['my_openml_dataset'] = {
    'method': 'load_openml_dataset',
    'source': 'OpenML',
    'kwargs': {
        'dataset_id': 12345,  # Your dataset ID
        'task_type': 'binary_classification'  # or 'multi_class', 'regression'
    }
}
```

### From TorchVision

```python
def load_my_torchvision_dataset(self, batch_size: int = 64):
    """Load dataset from torchvision."""
    from torchvision import datasets, transforms

    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = datasets.YourDataset(
        root=str(self.cache_dir),
        train=True,
        download=True,
        transform=transform
    )

    # ... process and return ...
```

## Programmatic Usage

### In Python Script

```python
from src.trainer_with_download import TrainerWithDownload
from src.geometry_analysis import GeometricAnalyzer

# Create trainer
trainer = TrainerWithDownload(
    hidden_dims=[16, 16],
    learning_rate=0.001,
    epochs=100,
    cache_dir='./data_cache'
)

# Train on specific datasets
datasets = ['adult_income', 'mnist_binary', 'iris']
weight_matrix, metadata = trainer.train_on_downloadable_datasets(datasets)

# Analyze
analyzer = GeometricAnalyzer(weight_matrix, metadata)
results = analyzer.full_analysis()
```

### Direct Dataset Access

```python
from src.dataset_downloader import DatasetDownloader

# Create downloader
downloader = DatasetDownloader(cache_dir='./data_cache')

# Load specific dataset
train_loader, test_loader, metadata = downloader.load_adult_income()

# Train your model...

# Cleanup when done
downloader.cleanup()
```

## File Structure

After running with downloads:

```
your-project/
├── data_cache/              # Downloaded datasets (auto-cleaned)
│   ├── adult_train.data
│   ├── bank.zip
│   └── MNIST/
├── results/
│   ├── weight_matrix.npy    # All model weights
│   ├── metadata.json        # Training info
│   ├── weights/             # Individual weight storage (if --incremental)
│   └── figures/             # Visualizations
└── run_experiment_with_download.py
```

## Data Caching

Datasets are cached in `data_cache/` directory:
- First run: Downloads from internet
- Subsequent runs: Uses cached version
- Manual cleanup: `rm -rf data_cache/`
- Auto cleanup: After each dataset is used (unless you disable it)

## Troubleshooting

### Download Fails

```
Error: Connection timeout
```

**Solution**: Check internet connection, or manually download and place in cache directory.

### Out of Memory

```
RuntimeError: CUDA out of memory
```

**Solution**: Use `--incremental` flag:

```bash
python run_experiment_with_download.py --incremental
```

### Dataset Not Found

```
Unknown dataset: my_dataset
```

**Solution**: Check available datasets with `--list-datasets` or verify spelling.

### TorchVision Import Error

```
ImportError: Please install torchvision
```

**Solution**:

```bash
pip install torchvision
```

## Advanced: Custom Preprocessing

Override dataset preprocessing in `DatasetDownloader`:

```python
class MyDatasetDownloader(DatasetDownloader):
    def load_adult_income(self, batch_size=32):
        # Custom preprocessing
        train_loader, test_loader, metadata = super().load_adult_income(batch_size)

        # Modify loaders or metadata
        # ...

        return train_loader, test_loader, metadata
```

## Performance Tips

1. **Use recommended subset first** - Test your pipeline with smaller datasets
2. **Enable incremental storage** - For experiments with >50 datasets
3. **Adjust batch size** - Larger batches for GPU, smaller for CPU
4. **Cache datasets** - Don't delete cache between runs for faster iteration

## Recommended Dataset Combinations

### Quick Test (< 5 minutes)
```bash
--datasets iris mnist_binary openml_credit
```

### Balanced Mix (10-20 minutes)
```bash
--datasets adult_income iris mnist_binary \
           fashion_mnist_5class openml_credit openml_blood
```

### Large Scale (> 1 hour)
```bash
# All available datasets
python run_experiment_with_download.py --incremental
```

## Dataset Statistics

| Dataset | Source | Task | Samples | Features | Classes |
|---------|--------|------|---------|----------|---------|
| adult_income | UCI | Binary | ~48K | 14 | 2 |
| bank_marketing | UCI | Binary | ~45K | 16 | 2 |
| iris | UCI | Multi-class | 150 | 4 | 3 |
| mnist_binary | TorchVision | Binary | ~13K | 784 | 2 |
| fashion_mnist_5class | TorchVision | Multi-class | ~30K | 784 | 5 |
| openml_credit | OpenML | Binary | 1000 | ~20 | 2 |
| openml_blood | OpenML | Binary | 748 | 4 | 2 |
| openml_phoneme | OpenML | Binary | 5404 | 5 | 2 |
