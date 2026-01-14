from typing import Dict, Tuple

import torch
import torch.nn as nn

from src.models.mlp import build_mlp
from .layers import GCNLayer, GraphSAGELayer


class StandardGNN(nn.Module):
    """Standard message-passing GNN baseline (GraphSAGE or GCN)."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        layer_type: str = "sage",
        dropout: float = 0.0,
    ):
        super().__init__()
        layer_key = layer_type.lower()
        if layer_key not in {"sage", "gcn"}:
            raise ValueError(f"layer_type must be 'sage' or 'gcn', got {layer_type}")

        self.encoder = build_mlp(in_dim + 3, hidden_dim, hidden_dim, num_layers=2, dropout=dropout)
        layer_cls = GraphSAGELayer if layer_key == "sage" else GCNLayer
        self.layers = nn.ModuleList([layer_cls(hidden_dim, hidden_dim, dropout=dropout) for _ in range(num_layers)])
        self.decoder = build_mlp(hidden_dim, hidden_dim, out_dim, num_layers=2, dropout=dropout)

    def forward(self, batch) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        x_in = torch.cat([batch.x, batch.pos], dim=-1)
        h = self.encoder(x_in)
        for layer in self.layers:
            h = layer(h, batch.edge_index, batch.pos)
        pred = self.decoder(h)
        aux = {
            "latent": h,
            "flux_residual": torch.zeros(h.size(0), device=h.device, dtype=h.dtype),
        }
        return pred, aux
