# Universal Subspace Hypothesis Investigation

An experimental framework to investigate whether trained neural network weights cluster around a low-dimensional manifold in weight space, and whether this manifold exhibits fractal-like properties.

## Background

The **Universal Subspace Hypothesis** suggests that well-trained neural networks of a given architecture may converge to weights that lie on or near a low-dimensional subspace or manifold within the high-dimensional parameter space. This project investigates:

1. Whether such a manifold exists across diverse tasks
2. What is the intrinsic dimensionality of this manifold
3. Whether the manifold has fractal-like structure (as suggested by the chaotic dynamics of backpropagation)

## Big Picture: How It Works

### The Workflow

```
┌─────────────────────────────────────────────────────────────────────┐
│                         PHASE 1: TRAINING                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  11 Diverse Datasets  →  Small MLP (16-16)  →  Weight Vectors       │
│  ─────────────────       ─────────────────      ─────────────        │
│  • Binary Classification     Input Layer          [w₁, w₂, ...]     │
│  • Multi-class                  ↓                      ↓             │
│  • Regression              Hidden (16)          Extract weights      │
│  • Time Series                  ↓                  as vectors        │
│                            Hidden (16)                ↓              │
│                                 ↓              (n_params dims)       │
│  [Train → Converge]        Output Layer                              │
│  [Delete dataset]          [Save weights]      Weight Matrix         │
│  [Next dataset...]         [Delete model]      (11 × n_params)      │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
                                   ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    PHASE 2: GEOMETRIC ANALYSIS                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│   Weight Matrix (11 models × ~600 params)                           │
│         ↓            ↓            ↓            ↓           ↓         │
│       ┌───┐        ┌───┐        ┌───┐       ┌───┐      ┌───┐       │
│       │PCA│        │MLE│        │Box│       │UMAP│     │DBSCAN│     │
│       └─┬─┘        └─┬─┘        └─┬─┘       └─┬─┘      └─┬─┘       │
│         │            │            │           │          │           │
│    Effective    Intrinsic   Fractal Dim   Manifold   Clusters       │
│    Dimension    Dimension    (Box-count   Embedding  (Task types)   │
│    (95% var)    (k-NN MLE)   + Corr dim)  (2D, 3D)                  │
│         │            │            │           │          │           │
│         └────────────┴────────────┴───────────┴──────────┘           │
│                              ↓                                        │
│                    Compare & Interpret:                              │
│                    • Is dim << n_params?  → Universal subspace       │
│                    • Fractal ≠ Intrinsic? → Fractal structure        │
│                    • Do tasks cluster?    → Task similarity          │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
                                   ↓
┌─────────────────────────────────────────────────────────────────────┐
│                     PHASE 3: VISUALIZATION                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  📊 PCA Variance       📈 Fractal Log-Log    📊 Dimension Compare    │
│  📊 UMAP 2D/3D         📊 Clustering         📊 Intrinsic Dim Dist   │
│                                                                       │
│  📄 Summary Report: Conclusions about manifold structure             │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

### Technology Stack by Component

```
┌──────────────────────────────────────────────────────────────────┐
│                    CORE PACKAGES & PURPOSES                       │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  🔥 PyTorch (Model Definition & Training)                        │
│     ├─ torch.nn.Module         → Neural network architecture     │
│     ├─ torch.nn.Linear         → Layer definitions              │
│     ├─ torch.optim.Adam        → Optimization                   │
│     ├─ torch.nn.*Loss          → Loss functions                 │
│     └─ torch.utils.data        → Data loading & batching        │
│                                                                   │
│  📊 NumPy (Numerical Computing)                                  │
│     ├─ Weight vector storage   → np.ndarray                     │
│     ├─ Matrix operations       → Linear algebra                 │
│     └─ Distance calculations   → pdist, norm                    │
│                                                                   │
│  🔬 Scikit-learn (Classical ML & Data Processing)                │
│     ├─ StandardScaler          → Data normalization             │
│     ├─ train_test_split        → Data splitting                 │
│     ├─ PCA                     → Dimensionality reduction        │
│     ├─ DBSCAN                  → Density-based clustering        │
│     ├─ AgglomerativeClustering → Hierarchical clustering        │
│     ├─ make_classification     → Synthetic datasets             │
│     ├─ make_regression         → Synthetic regression           │
│     └─ load_* datasets         → UCI datasets                   │
│                                                                   │
│  🗺️ UMAP (Manifold Learning)                                     │
│     └─ umap.UMAP               → Non-linear embedding            │
│                                  (Better than t-SNE for this)   │
│                                                                   │
│  📐 SciPy (Scientific Computing)                                 │
│     ├─ pdist, squareform       → Pairwise distances             │
│     └─ linregress              → Linear regression for fractal  │
│                                  dimension estimation            │
│                                                                   │
│  📈 Matplotlib & Seaborn (Visualization)                         │
│     ├─ 2D/3D scatter plots     → UMAP, PCA embeddings           │
│     ├─ Bar charts              → Variance explained             │
│     ├─ Log-log plots           → Fractal scaling                │
│     └─ Histograms              → Dimension distributions         │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### Data Flow Diagram

```
Dataset (sklearn/synthetic)
         ↓
    [NumPy arrays]
         ↓
  torch.FloatTensor  ←─────┐
         ↓                  │
  DataLoader (PyTorch)     │
         ↓                  │
  SmallMLP (PyTorch)       │  TRAINING LOOP
         ↓                  │  (PyTorch)
  Forward/Backward Pass    │
         ↓                  │
  Adam Optimizer  ─────────┘
         ↓
  Trained Model
         ↓
  model.parameters()  ←─── Extract weights
         ↓
  NumPy weight vector
         ↓
  [Stack all vectors]
         ↓
  Weight Matrix (NumPy)  ←─── 11 × n_params
         ↓
  ┌──────┴───────┐
  ↓              ↓
sklearn.PCA    umap.UMAP    ←─── ANALYSIS
  ↓              ↓               (NumPy/SciPy)
Results       Results
  ↓              ↓
matplotlib.pyplot  ←─── VISUALIZATION
  ↓
PNG figures
```

### Module Responsibilities

| Module | Primary Package | Purpose |
|--------|----------------|---------|
| `models.py` | **PyTorch** | Define MLP architecture, extract weights |
| `datasets.py` | **scikit-learn**, PyTorch | Load/generate diverse datasets |
| `trainer.py` | **PyTorch**, NumPy | Train models, manage GPU/memory |
| `geometry_analysis.py` | **scikit-learn**, SciPy, **UMAP** | Compute dimensions, manifold properties |
| `visualization.py` | **Matplotlib**, Seaborn | Create plots and figures |
| `run_experiment.py` | All of above | Orchestrate full pipeline |

## Project Structure

```
hecke-algebra/
├── src/
│   ├── models.py           # Small neural network architectures
│   ├── datasets.py         # Diverse dataset loaders (classification, regression, time series)
│   ├── trainer.py          # Memory-efficient training loop
│   ├── geometry_analysis.py # Geometric/topological analysis tools
│   └── visualization.py    # Plotting and visualization
├── run_experiment.py       # Main experiment script
├── requirements.txt        # Python dependencies
└── results/                # Output directory (created during run)
    ├── weight_matrix.npy
    ├── metadata.json
    ├── analysis_results.npz
    ├── summary_report.txt
    └── figures/
```

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

Run the complete experiment with default settings:

```bash
python run_experiment.py
```

This will:
1. Train small MLPs (2 hidden layers, 16 neurons each) on 11 diverse datasets
2. Extract weight vectors from each trained model
3. Perform geometric analysis (PCA, fractal dimension, manifold learning, clustering)
4. Generate visualizations
5. Save results to `results/`

## Usage

### Basic Usage

```bash
# Run with default architecture (16-16 hidden layers)
python run_experiment.py

# Custom architecture (10-20-10 hidden layers)
python run_experiment.py --hidden-dims 10 20 10

# Faster training (fewer epochs)
python run_experiment.py --epochs 50 --patience 10

# Train on specific datasets only
python run_experiment.py --datasets binary_moons wine digits regression_synthetic
```

### Advanced Options

```bash
# Skip training and analyze existing results
python run_experiment.py --skip-training

# Only train, skip analysis (useful for collecting more data)
python run_experiment.py --skip-analysis

# Custom save directory
python run_experiment.py --save-dir my_experiment

# Full custom run
python run_experiment.py \
  --hidden-dims 20 20 \
  --lr 0.0005 \
  --epochs 150 \
  --patience 20 \
  --save-dir results_20x20
```

## Available Datasets

The experiment includes 11 diverse datasets across 4 task types:

**Binary Classification:**
- `binary_moons` - Two moons dataset
- `binary_circles` - Concentric circles
- `binary_classification_synthetic` - Synthetic 10D binary classification
- `breast_cancer` - Wisconsin breast cancer dataset

**Multi-class Classification:**
- `multi_classification_synthetic` - Synthetic 5-class problem
- `wine` - Wine quality dataset (3 classes)
- `digits` - Handwritten digits (10 classes)

**Regression:**
- `regression_synthetic` - Synthetic 10D regression
- `diabetes` - Diabetes progression prediction

**Time Series:**
- `time_series_sine` - Sine wave prediction
- `time_series_combined` - Combined sine/cosine waves

## Analysis Methods

### 1. PCA (Baseline)
Standard linear dimensionality reduction to establish a baseline for comparison.

### 2. Fractal Dimension Estimation
- **Box-counting method**: Measures how the number of boxes needed to cover the manifold scales with box size
- **Correlation dimension**: More robust estimate based on pairwise distances

### 3. Intrinsic Dimension (MLE)
Maximum likelihood estimation of the true dimensionality of the manifold based on local neighborhoods.

### 4. Manifold Learning (UMAP)
Non-linear dimensionality reduction that preserves both local and global structure, better than t-SNE for understanding manifold topology.

### 5. Clustering Analysis
DBSCAN clustering to identify if models naturally group by task type or other characteristics.

## Interpreting Results

After running the experiment, check `results/summary_report.txt` for key findings:

**Strong Universal Subspace Evidence:**
- PCA effective dimension < 10
- Fractal and intrinsic dimensions agree (within ~1)
- High clustering of models in low-dimensional space

**Fractal Structure Evidence:**
- Fractal dimension significantly different from intrinsic dimension
- Non-integer fractal dimension estimates
- Complex structure visible in UMAP embeddings

**Key Visualizations:**
- `pca_variance.png` - How much variance is captured by principal components
- `dimension_comparison.png` - Comparison of different dimensionality estimates
- `fractal_dimension.png` - Log-log plots showing fractal scaling
- `umap_2d.png` / `umap_3d.png` - Low-dimensional embeddings colored by task type
- `clustering.png` - Discovered clusters in weight space

## Research Questions

This framework helps investigate:

1. **Does a universal subspace exist?** → Check PCA effective dimensionality
2. **Is it fractal?** → Compare fractal vs intrinsic dimension estimates
3. **Do different tasks cluster?** → Examine clustering results and UMAP embeddings
4. **How low-dimensional is it?** → Compare all dimension estimates
5. **Is PCA sufficient?** → Compare PCA vs UMAP embeddings

## Extending the Framework

### Add New Datasets

Edit `src/datasets.py`:

```python
@staticmethod
def _load_your_dataset(batch_size=32, **kwargs):
    # Your data loading code
    X, y = load_your_data()

    # Return train_loader, test_loader, metadata
    return train_loader, test_loader, metadata

# Add to ALL_DATASETS list
ALL_DATASETS.append('your_dataset')
```

### Modify Architecture

```bash
# Single hidden layer with 50 neurons
python run_experiment.py --hidden-dims 50

# Three hidden layers
python run_experiment.py --hidden-dims 32 16 8

# Larger network
python run_experiment.py --hidden-dims 64 64
```

### Add New Analysis Methods

Edit `src/geometry_analysis.py` to add methods to the `GeometricAnalyzer` class, then call them in `full_analysis()`.

## Technical Details

**Memory Management:**
- Datasets are loaded one at a time and deleted after training
- Models are deleted after weight extraction
- GPU cache is cleared after each training run

**Training:**
- Early stopping with patience (default: 15 epochs)
- Adam optimizer (default lr: 0.001)
- Task-appropriate loss functions (BCE, CrossEntropy, MSE)

**Computational Complexity:**
- Training: O(n_datasets × n_epochs × dataset_size)
- Geometric analysis: O(n_models²) for pairwise distances
- UMAP: O(n_models × log(n_models))

## Citation

If you use this framework in your research, please cite the relevant papers on the universal subspace hypothesis and fractal neural dynamics.

## License

MIT License