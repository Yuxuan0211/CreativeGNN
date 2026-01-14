from typing import Optional

import torch
import torch.nn as nn

from .mlp import build_mlp


class GraphNeuralODEFunc(nn.Module):
    """Continuous-time message passing dynamics."""

    def __init__(self, hidden_dim: int, message_dim: int, edge_feat_dim: int = 0, dropout: float = 0.0):
        super().__init__()
        self.msg_mlp = build_mlp(2 * hidden_dim + 3 + edge_feat_dim, message_dim, message_dim, num_layers=2, dropout=dropout)
        self.update_mlp = build_mlp(hidden_dim + message_dim, hidden_dim, hidden_dim, num_layers=2, dropout=dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        self.edge_attr: Optional[torch.Tensor] = None
        self.pos: Optional[torch.Tensor] = None
        self.edge_index: Optional[torch.Tensor] = None

    def set_graph(self, pos: torch.Tensor, edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor] = None) -> None:
        self.pos = pos
        self.edge_index = edge_index
        self.edge_attr = edge_attr

    def forward(self, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        if self.pos is None or self.edge_index is None:
            raise RuntimeError("Call set_graph(pos, edge_index, edge_attr) before odeint.")

        src, dst = self.edge_index
        rel_pos = self.pos[dst] - self.pos[src]
        pieces = [h[src], h[dst], rel_pos]
        if self.edge_attr is not None:
            pieces.append(self.edge_attr)
        msg = self.msg_mlp(torch.cat(pieces, dim=-1))

        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, msg)

        dh = self.update_mlp(torch.cat([h, agg], dim=-1))
        return self.norm(h + dh)
