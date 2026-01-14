from typing import Dict, Tuple

import torch
import torch.nn as nn

from src.models.mlp import build_mlp


class MPNNLayer(nn.Module):
    """Message passing layer with edge features (relative position + distance)."""

    def __init__(self, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.msg_mlp = build_mlp(hidden_dim * 2 + 4, hidden_dim, hidden_dim, num_layers=2, dropout=dropout)
        self.update_mlp = build_mlp(hidden_dim * 2, hidden_dim, hidden_dim, num_layers=2, dropout=dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index
        msg = self.msg_mlp(torch.cat([h[src], h[dst], edge_attr], dim=-1))

        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, msg)
        deg = torch.zeros(h.size(0), device=h.device, dtype=h.dtype)
        deg.index_add_(0, dst, torch.ones(dst.size(0), device=h.device, dtype=h.dtype))
        agg = agg / (deg.unsqueeze(-1) + 1e-6)

        update = self.update_mlp(torch.cat([h, agg], dim=-1))
        return self.norm(h + update)


class MPNN(nn.Module):
    """Deep message-passing network baseline (stacked MPNN)."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.encoder = build_mlp(in_dim + 3, hidden_dim, hidden_dim, num_layers=2, dropout=dropout)
        self.layers = nn.ModuleList([MPNNLayer(hidden_dim, dropout=dropout) for _ in range(num_layers)])
        self.decoder = build_mlp(hidden_dim, hidden_dim, out_dim, num_layers=2, dropout=dropout)

    def forward(self, batch) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        x_in = torch.cat([batch.x, batch.pos], dim=-1)
        h = self.encoder(x_in)

        src, dst = batch.edge_index
        rel_pos = batch.pos[dst] - batch.pos[src]
        dist = torch.norm(rel_pos, dim=-1, keepdim=True)
        edge_attr = torch.cat([rel_pos, dist], dim=-1)

        for layer in self.layers:
            h = layer(h, batch.edge_index, edge_attr)

        pred = self.decoder(h)
        aux = {
            "latent": h,
            "flux_residual": torch.zeros(h.size(0), device=h.device, dtype=h.dtype),
        }
        return pred, aux
