import torch
import torch.nn as nn


def build_mlp(in_dim: int, hidden_dim: int, out_dim: int, num_layers: int = 2, dropout: float = 0.0) -> nn.Sequential:
    layers = []
    dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.SiLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)
