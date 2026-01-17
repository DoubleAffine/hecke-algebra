"""
Small neural network architectures for universal subspace hypothesis investigation.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple


class SmallMLP(nn.Module):
    """
    Small multi-layer perceptron with configurable architecture.
    Default: 2 hidden layers with 10-20 neurons each.
    """
    def __init__(self, input_dim: int, output_dim: int,
                 hidden_dims: List[int] = [16, 16],
                 activation: str = 'relu'):
        super(SmallMLP, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims

        # Build layers
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            # Don't add activation after last layer
            if i < len(dims) - 2:
                if activation == 'relu':
                    layers.append(nn.ReLU())
                elif activation == 'tanh':
                    layers.append(nn.Tanh())
                elif activation == 'sigmoid':
                    layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

    def get_weight_vector(self) -> np.ndarray:
        """
        Extract all weights and biases as a single flat vector.
        This represents a point in the high-dimensional weight space.
        """
        weights = []
        for param in self.parameters():
            weights.append(param.data.cpu().numpy().flatten())
        return np.concatenate(weights)

    def get_weight_dict(self) -> dict:
        """
        Get weights organized by layer for more detailed analysis.
        """
        weight_dict = {}
        for name, param in self.named_parameters():
            weight_dict[name] = param.data.cpu().numpy()
        return weight_dict

    def count_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model_for_task(task_type: str, input_dim: int, output_dim: int,
                          hidden_dims: List[int] = [16, 16]) -> SmallMLP:
    """
    Factory function to create appropriate model for different task types.
    """
    if task_type in ['binary_classification', 'multi_class', 'regression', 'time_series']:
        return SmallMLP(input_dim, output_dim, hidden_dims)
    else:
        raise ValueError(f"Unknown task type: {task_type}")


def get_loss_function(task_type: str):
    """Get appropriate loss function for task type."""
    if task_type == 'binary_classification':
        return nn.BCEWithLogitsLoss()
    elif task_type == 'multi_class':
        return nn.CrossEntropyLoss()
    elif task_type in ['regression', 'time_series']:
        return nn.MSELoss()
    else:
        raise ValueError(f"Unknown task type: {task_type}")
