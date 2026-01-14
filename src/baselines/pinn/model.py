from typing import Dict, Tuple

import torch
import torch.nn as nn

from src.models.mlp import build_mlp


def continuity_residual(pred: torch.Tensor, pos: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """Approximate divergence of velocity using edge differences."""
    if pred.size(1) < 3:
        return torch.zeros(pred.size(0), device=pred.device, dtype=pred.dtype)
    vel = pred[:, :3]
    src, dst = edge_index
    edge_vec = pos[dst] - pos[src]
    edge_norm_sq = edge_vec.pow(2).sum(dim=-1) + 1e-6
    vel_diff = vel[dst] - vel[src]
    proj = (vel_diff * edge_vec).sum(dim=-1) / edge_norm_sq

    residual = torch.zeros(pred.size(0), device=pred.device, dtype=pred.dtype)
    residual.index_add_(0, dst, proj)
    deg = torch.zeros(pred.size(0), device=pred.device, dtype=pred.dtype)
    deg.index_add_(0, dst, torch.ones(dst.size(0), device=pred.device, dtype=pred.dtype))
    residual = residual / (deg + 1e-6)
    return residual


class PINN(nn.Module):
    """Physics-Informed Neural Network baseline using a coordinate MLP."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.0,
        use_residual: bool = True,
    ):
        super().__init__()
        self.use_residual = use_residual
        self.trunk = build_mlp(in_dim + 3, hidden_dim, hidden_dim, num_layers=num_layers, dropout=dropout)
        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(self, batch) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        x_in = torch.cat([batch.x, batch.pos], dim=-1)
        h = self.trunk(x_in)
        pred = self.head(h)
        if self.use_residual:
            residual = continuity_residual(pred, batch.pos, batch.edge_index)
        else:
            residual = torch.zeros(pred.size(0), device=pred.device, dtype=pred.dtype)
        aux = {
            "latent": h,
            "flux_residual": residual,
        }
        return pred, aux
