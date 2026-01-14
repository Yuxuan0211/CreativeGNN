from typing import Dict, Tuple

import torch
import torch.nn as nn

from src.models.mlp import build_mlp


class MeshGraphNetBlock(nn.Module):
    """Message passing block with edge updates and residual node updates."""

    def __init__(self, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.edge_mlp = build_mlp(hidden_dim * 3 + 4, hidden_dim, hidden_dim, num_layers=2, dropout=dropout)
        self.node_mlp = build_mlp(hidden_dim * 2, hidden_dim, hidden_dim, num_layers=2, dropout=dropout)
        self.edge_norm = nn.LayerNorm(hidden_dim)
        self.node_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        h: torch.Tensor,
        e: torch.Tensor,
        edge_index: torch.Tensor,
        pos: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        src, dst = edge_index
        rel_pos = pos[dst] - pos[src]
        dist = torch.norm(rel_pos, dim=-1, keepdim=True)
        edge_in = torch.cat([h[src], h[dst], e, rel_pos, dist], dim=-1)
        e_update = self.edge_mlp(edge_in)
        e = self.edge_norm(e + e_update)

        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, e)
        deg = torch.zeros(h.size(0), device=h.device, dtype=h.dtype)
        deg.index_add_(0, dst, torch.ones(dst.size(0), device=h.device, dtype=h.dtype))
        agg = agg / (deg.unsqueeze(-1) + 1e-6)

        node_in = torch.cat([h, agg], dim=-1)
        h_update = self.node_mlp(node_in)
        h = self.node_norm(h + h_update)
        return h, e


class MeshGraphNet(nn.Module):
    """MeshGraphNet-style baseline with encoder-processor-decoder blocks."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.node_encoder = build_mlp(in_dim + 3, hidden_dim, hidden_dim, num_layers=2, dropout=dropout)
        self.edge_encoder = build_mlp(4, hidden_dim, hidden_dim, num_layers=2, dropout=dropout)
        self.blocks = nn.ModuleList([MeshGraphNetBlock(hidden_dim, dropout=dropout) for _ in range(num_layers)])
        self.decoder = build_mlp(hidden_dim, hidden_dim, out_dim, num_layers=2, dropout=dropout)

    def forward(self, batch) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        x_in = torch.cat([batch.x, batch.pos], dim=-1)
        h = self.node_encoder(x_in)

        src, dst = batch.edge_index
        rel_pos = batch.pos[dst] - batch.pos[src]
        dist = torch.norm(rel_pos, dim=-1, keepdim=True)
        edge_feat = torch.cat([rel_pos, dist], dim=-1)
        e = self.edge_encoder(edge_feat)

        for block in self.blocks:
            h, e = block(h, e, batch.edge_index, batch.pos)

        pred = self.decoder(h)
        aux = {
            "latent": h,
            "flux_residual": torch.zeros(h.size(0), device=h.device, dtype=h.dtype),
        }
        return pred, aux
