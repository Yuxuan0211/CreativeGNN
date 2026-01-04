from typing import Optional, Tuple

import torch
import torch.nn as nn

from .mlp import build_mlp


class FluxLayer(nn.Module):
    """Predict edge-wise fluxes, project to nodal balance, and correct node states."""

    def __init__(self, hidden_dim: int, flux_dim: int = 5, dropout: float = 0.0, nproj: int = 1):
        super().__init__()
        self.flux_dim = flux_dim
        self.nproj = max(1, int(nproj))
        self.flux_mlp = build_mlp(2 * hidden_dim + 3, hidden_dim, flux_dim, num_layers=2, dropout=dropout)
        self.corr_mlp = build_mlp(hidden_dim + flux_dim, hidden_dim, hidden_dim, num_layers=2, dropout=dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        h: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        boundary_flux: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        src, dst = edge_index
        rel_pos = pos[dst] - pos[src]
        flux_raw = self.flux_mlp(torch.cat([h[src], h[dst], rel_pos], dim=-1))  # (E, F)

        # Residual before projection (matches sum_j F_ij = -F_ext)
        outgoing = torch.zeros(h.size(0), self.flux_dim, device=h.device)
        outgoing.index_add_(0, src, flux_raw)
        residual_pre = outgoing
        if boundary_flux is not None:
            residual_pre = residual_pre + boundary_flux

        # Project fluxes so each node is balanced: F_ij <- F_ij - Δ_i / |N(i)|
        deg = torch.zeros(h.size(0), 1, device=h.device)
        deg.index_add_(0, src, torch.ones_like(deg[src]))
        deg = deg.clamp_min(1.0)
        residual_pre_norm = residual_pre / deg
        flux_proj = flux_raw
        for _ in range(self.nproj):
            outgoing_iter = torch.zeros(h.size(0), self.flux_dim, device=h.device)
            outgoing_iter.index_add_(0, src, flux_proj)
            residual_iter = outgoing_iter
            if boundary_flux is not None:
                residual_iter = residual_iter + boundary_flux
            correction = residual_iter / deg
            flux_proj = flux_proj - correction[src]

        outgoing_proj = torch.zeros(h.size(0), self.flux_dim, device=h.device)
        outgoing_proj.index_add_(0, src, flux_proj)
        residual_post = outgoing_proj
        if boundary_flux is not None:
            residual_post = residual_post + boundary_flux
        residual_post_norm = residual_post / deg

        h_corr = self.corr_mlp(torch.cat([h, residual_pre], dim=-1))
        h_corrected = self.norm(h + h_corr)
        return h_corrected, flux_proj, residual_pre, residual_post, residual_pre_norm, residual_post_norm
