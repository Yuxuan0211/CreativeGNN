from typing import Optional

import torch
import torch.nn as nn

from src.models.mlp import build_mlp


class GraphSAGELayer(nn.Module):
    """GraphSAGE-style mean aggregation with optional geometric messages."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.msg_mlp = build_mlp(in_dim * 2 + 3, out_dim, out_dim, num_layers=2, dropout=dropout)
        self.update_mlp = build_mlp(in_dim + out_dim, out_dim, out_dim, num_layers=2, dropout=dropout)
        self.norm = nn.LayerNorm(out_dim)

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        src, dst = edge_index
        if pos is None:
            rel_pos = torch.zeros((src.size(0), 3), device=h.device, dtype=h.dtype)
        else:
            rel_pos = pos[dst] - pos[src]
        msg = self.msg_mlp(torch.cat([h[src], h[dst], rel_pos], dim=-1))

        agg = torch.zeros(h.size(0), msg.size(-1), device=h.device, dtype=msg.dtype)
        agg.index_add_(0, dst, msg)
        deg = torch.zeros(h.size(0), device=h.device, dtype=msg.dtype)
        deg.index_add_(0, dst, torch.ones(dst.size(0), device=h.device, dtype=msg.dtype))
        agg = agg / (deg.unsqueeze(-1) + 1e-6)

        out = self.update_mlp(torch.cat([h, agg], dim=-1))
        if out.size(-1) == h.size(-1):
            out = out + h
        return self.norm(out)


class GCNLayer(nn.Module):
    """GCN-style mean aggregation without explicit edge features."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
        self.act = nn.SiLU()
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        src, dst = edge_index
        agg = torch.zeros(h.size(0), h.size(1), device=h.device, dtype=h.dtype)
        agg.index_add_(0, dst, h[src])
        deg = torch.zeros(h.size(0), device=h.device, dtype=h.dtype)
        deg.index_add_(0, dst, torch.ones(dst.size(0), device=h.device, dtype=h.dtype))
        agg = agg / (deg.unsqueeze(-1) + 1e-6)
        out = self.lin(agg)
        out = self.act(out)
        if self.dropout is not None:
            out = self.dropout(out)
        return self.norm(out)
