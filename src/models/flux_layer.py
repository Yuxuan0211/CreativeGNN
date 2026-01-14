from typing import Tuple

import torch
import torch.nn as nn

from .mlp import build_mlp


class FluxLayer(nn.Module):
    """Predict edge-wise fluxes and correct node states to enforce conservation."""

    def __init__(self, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.flux_mlp = build_mlp(2 * hidden_dim + 3, hidden_dim, 1, num_layers=2, dropout=dropout)
        self.corr_mlp = build_mlp(hidden_dim + 1, hidden_dim, hidden_dim, num_layers=2, dropout=dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        h: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        src, dst = edge_index
        rel_pos = pos[dst] - pos[src]
        flux = self.flux_mlp(torch.cat([h[src], h[dst], rel_pos], dim=-1))  # (E, 1)

        # Net residual per node: outgoing - incoming
        outgoing = torch.zeros(h.size(0), 1, device=h.device)
        incoming = torch.zeros_like(outgoing)
        outgoing.index_add_(0, src, flux)
        incoming.index_add_(0, dst, flux)
        residual = outgoing - incoming  # (N, 1)

        correction = self.corr_mlp(torch.cat([h, residual], dim=-1))
        h_corrected = self.norm(h + correction)
        return h_corrected, flux.squeeze(-1), residual.squeeze(-1)
