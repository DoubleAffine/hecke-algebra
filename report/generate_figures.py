#!/usr/bin/env python3
"""
Generate figures for the Universal Subspace Hypothesis report.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import json
import os

# Create output directory
os.makedirs('figures', exist_ok=True)

# Set style
plt.rcParams['figure.figsize'] = (8, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11

# Load data
with open('../experiments/current/vit_subspace_replication/results.json') as f:
    vit_results = json.load(f)

with open('../experiments/current/vit_subspace_replication/pc1_analysis.json') as f:
    pc1_analysis = json.load(f)

# =============================================================================
# Figure 1: Variance explained by principal components (20 ViT models)
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

variance = vit_results['top_20_variance']
cumsum = np.cumsum(variance)
x = np.arange(1, len(variance) + 1)

ax.bar(x, variance, alpha=0.7, label='Individual variance', color='steelblue')
ax.plot(x, cumsum, 'ro-', linewidth=2, markersize=8, label='Cumulative variance')
ax.axhline(y=0.95, color='green', linestyle='--', linewidth=2, label='95% threshold')
ax.axhline(y=0.99, color='orange', linestyle='--', linewidth=2, label='99% threshold')

ax.set_xlabel('Principal Component')
ax.set_ylabel('Variance Explained')
ax.set_title(f'PCA of 20 HuggingFace ViT Models ({vit_results["n_params"]:,} params each)')
ax.legend(loc='right')
ax.set_xlim(0.5, 20.5)
ax.set_ylim(0, 1.05)

# Add annotations
ax.annotate(f'PC1: {variance[0]*100:.1f}%', xy=(1, variance[0]), xytext=(3, variance[0]+0.05),
            fontsize=12, arrowprops=dict(arrowstyle='->', color='black'))
ax.annotate(f'k₉₅ = {vit_results["k_95"]}', xy=(vit_results["k_95"], 0.95),
            xytext=(vit_results["k_95"]+2, 0.85), fontsize=12,
            arrowprops=dict(arrowstyle='->', color='green'))

plt.tight_layout()
plt.savefig('figures/variance_explained_20vit.pdf', dpi=300, bbox_inches='tight')
plt.savefig('figures/variance_explained_20vit.png', dpi=300, bbox_inches='tight')
plt.close()
print("Created: variance_explained_20vit.pdf")

# =============================================================================
# Figure 2: Variance explained by training method (6 models)
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

variance_6 = pc1_analysis['variance_explained']
x = np.arange(1, len(variance_6) + 1)

colors = ['#d62728' if i == 0 else 'steelblue' for i in range(len(variance_6))]
bars = ax.bar(x, [v*100 for v in variance_6], color=colors, alpha=0.8)

ax.set_xlabel('Principal Component')
ax.set_ylabel('Variance Explained (%)')
ax.set_title('PCA of 6 ViT Models with Different Training Methods')
ax.set_xlim(0.5, 6.5)
ax.set_xticks(x)

# Add percentage labels on bars
for i, (bar, v) in enumerate(zip(bars, variance_6)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{v*100:.1f}%', ha='center', va='bottom', fontsize=11)

ax.annotate('PC1 captures 88.6%\n(Supervised vs Self-supervised)',
            xy=(1, 88.6), xytext=(2.5, 70),
            fontsize=12, arrowprops=dict(arrowstyle='->', color='black'),
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('figures/variance_by_training_method.pdf', dpi=300, bbox_inches='tight')
plt.savefig('figures/variance_by_training_method.png', dpi=300, bbox_inches='tight')
plt.close()
print("Created: variance_by_training_method.pdf")

# =============================================================================
# Figure 3: PC1 projections showing clustering by training method
# =============================================================================
fig, ax = plt.subplots(figsize=(12, 5))

projections = pc1_analysis['pc1_projections']
models = list(projections.keys())
values = list(projections.values())

# Color by training method
colors = ['#d62728', '#d62728', '#2ca02c', '#2ca02c', '#1f77b4', '#1f77b4']
labels_short = ['Sup. IN', 'Sup. IN-21k', 'DINO', 'MAE', 'CLIP-OAI', 'CLIP-LAION']

y_pos = np.arange(len(models))
bars = ax.barh(y_pos, values, color=colors, alpha=0.8, height=0.6)

ax.set_yticks(y_pos)
ax.set_yticklabels(labels_short)
ax.set_xlabel('Projection onto PC1')
ax.set_title('Model Projections onto First Principal Component')
ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)

# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#d62728', alpha=0.8, label='Supervised'),
                   Patch(facecolor='#2ca02c', alpha=0.8, label='Self-supervised'),
                   Patch(facecolor='#1f77b4', alpha=0.8, label='CLIP')]
ax.legend(handles=legend_elements, loc='center right')

# Add annotations
ax.annotate('Supervised\ncluster', xy=(-517, 0.5), xytext=(-400, 2),
            fontsize=11, ha='center',
            arrowprops=dict(arrowstyle='->', color='black'))
ax.annotate('Self-supervised\n& CLIP cluster', xy=(258, 3.5), xytext=(100, 5),
            fontsize=11, ha='center',
            arrowprops=dict(arrowstyle='->', color='black'))

plt.tight_layout()
plt.savefig('figures/pc1_projections.pdf', dpi=300, bbox_inches='tight')
plt.savefig('figures/pc1_projections.png', dpi=300, bbox_inches='tight')
plt.close()
print("Created: pc1_projections.pdf")

# =============================================================================
# Figure 4: Cosine similarity heatmap
# =============================================================================
fig, ax = plt.subplots(figsize=(9, 8))

cosine_sim = np.array(pc1_analysis['cosine_similarities'])
labels = ['Supervised\nImageNet', 'Supervised\nIN-21k', 'DINO', 'MAE', 'CLIP\nOpenAI', 'CLIP\nLAION-2B']

im = ax.imshow(cosine_sim, cmap='RdYlBu_r', vmin=0, vmax=1)

ax.set_xticks(np.arange(len(labels)))
ax.set_yticks(np.arange(len(labels)))
ax.set_xticklabels(labels, fontsize=10)
ax.set_yticklabels(labels, fontsize=10)

# Rotate x labels
plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

# Add text annotations
for i in range(len(labels)):
    for j in range(len(labels)):
        val = cosine_sim[i, j]
        color = 'white' if val > 0.5 else 'black'
        ax.text(j, i, f'{val:.3f}', ha='center', va='center', color=color, fontsize=10)

ax.set_title('Pairwise Cosine Similarity Between ViT Models\n(Same Architecture, Different Training)')
cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label('Cosine Similarity')

# Add rectangles to highlight clusters
import matplotlib.patches as mpatches
rect1 = mpatches.Rectangle((-0.5, -0.5), 2, 2, linewidth=3, edgecolor='red', facecolor='none')
rect2 = mpatches.Rectangle((1.5, 1.5), 4, 4, linewidth=3, edgecolor='blue', facecolor='none')
ax.add_patch(rect1)
ax.add_patch(rect2)

plt.tight_layout()
plt.savefig('figures/cosine_similarity_heatmap.pdf', dpi=300, bbox_inches='tight')
plt.savefig('figures/cosine_similarity_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("Created: cosine_similarity_heatmap.pdf")

# =============================================================================
# Figure 5: Layer contributions to PC1
# =============================================================================
fig, ax = plt.subplots(figsize=(12, 6))

layer_contrib = pc1_analysis['layer_contributions']
# Sort by layer number for encoder layers
encoder_layers = {k: v for k, v in layer_contrib.items() if 'encoder.layer' in k}
other_layers = {k: v for k, v in layer_contrib.items() if 'encoder.layer' not in k}

# Sort encoder layers
sorted_encoder = sorted(encoder_layers.items(), key=lambda x: int(x[0].split('.')[-1]))
sorted_other = sorted(other_layers.items(), key=lambda x: -x[1])

all_sorted = sorted_encoder + sorted_other
names = [x[0].replace('encoder.layer.', 'Layer ') for x in all_sorted]
values = [x[1] * 100 for x in all_sorted]

colors = ['#ff7f0e' if 'Layer' in n and int(n.split()[-1]) >= 8 else 'steelblue' for n in names]

bars = ax.bar(names, values, color=colors, alpha=0.8)
ax.set_xlabel('Layer')
ax.set_ylabel('Contribution to PC1 (%)')
ax.set_title('Which Layers Contribute Most to the First Principal Component?')
plt.xticks(rotation=45, ha='right')

# Add annotation
ax.annotate('Later layers (8-11)\ncontribute most',
            xy=(9, values[9]), xytext=(6, 15),
            fontsize=12, arrowprops=dict(arrowstyle='->', color='black'),
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('figures/layer_contributions.pdf', dpi=300, bbox_inches='tight')
plt.savefig('figures/layer_contributions.png', dpi=300, bbox_inches='tight')
plt.close()
print("Created: layer_contributions.pdf")

# =============================================================================
# Figure 6: Spectral decay comparison
# =============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left: singular values
singular_values = vit_results['singular_values']
x = np.arange(1, len(singular_values) + 1)
ax1.semilogy(x, singular_values, 'bo-', markersize=8, linewidth=2)
ax1.set_xlabel('Component Index')
ax1.set_ylabel('Singular Value (log scale)')
ax1.set_title('Singular Value Decay (20 ViT Models)')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0.5, 20.5)

# Add spectral ratio annotation
ratio = singular_values[0] / singular_values[9]
ax1.annotate(f'σ₁/σ₁₀ = {ratio:.2f}', xy=(5, singular_values[4]),
             fontsize=12, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

# Right: normalized singular values
ax2.plot(x, np.array(singular_values) / singular_values[0], 'ro-', markersize=8, linewidth=2, label='20 HuggingFace ViTs')

# Add reference lines
ax2.axhline(y=0.1, color='gray', linestyle='--', alpha=0.5)
ax2.set_xlabel('Component Index')
ax2.set_ylabel('Normalized Singular Value (σᵢ/σ₁)')
ax2.set_title('Normalized Spectral Decay')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0.5, 20.5)
ax2.legend()

plt.tight_layout()
plt.savefig('figures/spectral_decay.pdf', dpi=300, bbox_inches='tight')
plt.savefig('figures/spectral_decay.png', dpi=300, bbox_inches='tight')
plt.close()
print("Created: spectral_decay.pdf")

# =============================================================================
# Figure 7: Summary diagram - what PC1 represents
# =============================================================================
fig, ax = plt.subplots(figsize=(12, 8))

# Create a schematic showing the two clusters
ax.set_xlim(-2, 2)
ax.set_ylim(-1.5, 1.5)

# Draw axis
ax.axhline(y=0, color='black', linewidth=1)
ax.axvline(x=0, color='gray', linewidth=0.5, linestyle='--')

# Supervised cluster (left)
sup_x = [-1.5, -1.45]
sup_y = [0.05, -0.05]
ax.scatter(sup_x, sup_y, s=300, c='#d62728', alpha=0.8, edgecolors='black', linewidths=2)
ax.annotate('Supervised\nModels', xy=(-1.5, 0), xytext=(-1.5, 0.6),
            fontsize=14, ha='center', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#d62728'))

# Self-supervised cluster (right)
ss_x = [1.2, 1.25, 1.3, 1.35]
ss_y = [0.1, -0.1, 0.05, -0.05]
colors_ss = ['#2ca02c', '#2ca02c', '#1f77b4', '#1f77b4']
ax.scatter(ss_x, ss_y, s=300, c=colors_ss, alpha=0.8, edgecolors='black', linewidths=2)
ax.annotate('DINO, MAE,\nCLIP Models', xy=(1.3, 0), xytext=(1.3, 0.6),
            fontsize=14, ha='center', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#2ca02c'))

# PC1 arrow
ax.annotate('', xy=(1.8, 0), xytext=(-1.8, 0),
            arrowprops=dict(arrowstyle='->', color='black', lw=2))
ax.text(0, -0.3, 'PC1 (88.6% of variance)', fontsize=14, ha='center', fontweight='bold')

# Cosine similarity annotations
ax.annotate('cos θ ≈ 0.05\n(nearly orthogonal)', xy=(0, 0), xytext=(0, -0.8),
            fontsize=12, ha='center', color='purple',
            bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.8))

ax.annotate('cos θ ≈ 0.999', xy=(-1.47, 0), xytext=(-1.47, -0.5),
            fontsize=11, ha='center', color='#d62728')

ax.annotate('cos θ ≈ 0.36', xy=(1.27, 0), xytext=(1.27, -0.5),
            fontsize=11, ha='center', color='#2ca02c')

ax.set_title('PC1 Separates Models by Training Objective\n(Not a "Universal" Subspace)', fontsize=16, fontweight='bold')
ax.axis('off')

plt.tight_layout()
plt.savefig('figures/pc1_schematic.pdf', dpi=300, bbox_inches='tight')
plt.savefig('figures/pc1_schematic.png', dpi=300, bbox_inches='tight')
plt.close()
print("Created: pc1_schematic.pdf")

print("\n✓ All figures generated successfully!")
