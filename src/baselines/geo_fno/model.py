from typing import Dict, Tuple

import torch
import torch.nn as nn

from src.models.mlp import build_mlp


class FourierFeatureEncoder(nn.Module):
    """Fixed Fourier features for unstructured coordinate inputs."""

    def __init__(self, num_features: int, scale: float = 1.0):
        super().__init__()
        freq = torch.randn(num_features, 3) * scale
        self.register_buffer("freq", freq)

    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        freq = self.freq.to(dtype=pos.dtype, device=pos.device)
        proj = 2.0 * torch.pi * pos @ freq.t()
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class GeoFNOProxy(nn.Module):
    """Fourier-feature MLP proxy for operator baselines on point clouds."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        fourier_features: int = 32,
        fourier_scale: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.fourier = FourierFeatureEncoder(fourier_features, scale=fourier_scale)
        in_features = in_dim + 3 + 2 * fourier_features
        self.trunk = build_mlp(in_features, hidden_dim, hidden_dim, num_layers=num_layers, dropout=dropout)
        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(self, batch) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        fourier = self.fourier(batch.pos)
        x_in = torch.cat([batch.x, batch.pos, fourier], dim=-1)
        h = self.trunk(x_in)
        pred = self.head(h)
        aux = {
            "latent": h,
            "flux_residual": torch.zeros(h.size(0), device=h.device, dtype=h.dtype),
        }
        return pred, aux
