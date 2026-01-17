#!/usr/bin/env python3
"""
Quick basin discovery test - fewer models for faster results.
"""
import numpy as np
import os
import json

from src.trainer import Trainer
from src.datasets import DatasetManager

print("=" * 80)
print(" QUICK BASIN DISCOVERY TEST")
print("=" * 80)

# Config
distances = [0, 40, 80]  # Just 3 distances
n_models_per = 5  # 5 models per distance
epochs = 80  # Shorter training

print(f"\nTraining {len(distances) * n_models_per} models...")
print(f"Distances: {distances}")
print(f"Models per distance: {n_models_per}")

# Load dataset
train_loader, test_loader, dataset_metadata = DatasetManager.load_dataset(
    'binary_classification_synthetic'
)

trainer = Trainer(
    hidden_dims=[16, 16],
    learning_rate=0.001,
    epochs=epochs,
    patience=15
)

all_weights = []
all_metadata = []

for dist in distances:
    print(f"\n--- Distance {dist} ---")

    for i in range(n_models_per):
        final_weights, train_stats = trainer.train_single_model(
            train_loader, test_loader, dataset_metadata
        )

        all_weights.append(final_weights)
        all_metadata.append({
            'init_distance': dist,
            'model_idx': i,
            **train_stats
        })

        if (i + 1) % 2 == 0:
            print(f"  {i+1}/{n_models_per} done")

# Cleanup
DatasetManager.cleanup()

# Analyze
weight_matrix = np.array(all_weights)
print(f"\n{'='*80}")
print(f" ANALYSIS")
print(f"{'='*80}")
print(f"\nWeight matrix: {weight_matrix.shape}")

# Clustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

best_score = -1
best_n = 2

for n in range(2, 6):
    clustering = AgglomerativeClustering(n_clusters=n)
    labels = clustering.fit_predict(weight_matrix)
    score = silhouette_score(weight_matrix, labels)

    unique, counts = np.unique(labels, return_counts=True)
    print(f"{n} clusters: silhouette={score:.3f}, sizes={counts.tolist()}")

    if score > best_score:
        best_score = score
        best_n = n

print(f"\nBest: {best_n} clusters (score={best_score:.3f})")

# Use best
clustering = AgglomerativeClustering(n_clusters=best_n)
basin_labels = clustering.fit_predict(weight_matrix)

# Which distances in which basins?
print(f"\nBasin composition by init distance:")
for basin_id in range(best_n):
    mask = basin_labels == basin_id
    dists = [all_metadata[i]['init_distance'] for i in range(len(all_metadata)) if mask[i]]
    unique_d, counts_d = np.unique(dists, return_counts=True)
    print(f"  Basin {basin_id}: {dict(zip(unique_d, counts_d))}")

# Save
os.makedirs('results_quick_basin', exist_ok=True)
np.save('results_quick_basin/weights.npy', weight_matrix)

for i, meta in enumerate(all_metadata):
    meta['basin_id'] = int(basin_labels[i])

with open('results_quick_basin/metadata.json', 'w') as f:
    json.dump(all_metadata, f, indent=2)

print(f"\nSaved to: results_quick_basin/")
print(f"\nNext: python analyze_intersection_topology.py")
