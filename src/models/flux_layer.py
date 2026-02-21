from typing import Optional, Tuple

import torch
import torch.nn as nn

from .mlp import build_mlp


class FluxLayer(nn.Module):
    """
    Predict interface fluxes and apply a lightweight conservative projection.

    The projection follows the paper's idea: solve (approximately) a minimum-change
    correction so node-wise flux imbalance is reduced over graph incidences.
    """

    def __init__(
        self,
        hidden_dim: int,
        flux_dim: int = 5,
        n_proj: int = 3,
        proj_relax: float = 1.0,
        eps: float = 1e-8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.flux_dim = int(flux_dim)
        self.n_proj = max(0, int(n_proj))
        self.proj_relax = float(proj_relax)
        self.eps = float(eps)

        self.flux_mlp = build_mlp(2 * hidden_dim + 3, hidden_dim, self.flux_dim, num_layers=2, dropout=dropout)
        self.corr_mlp = build_mlp(hidden_dim + self.flux_dim, hidden_dim, hidden_dim, num_layers=2, dropout=dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def _node_residual(
        self,
        flux: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        num_nodes: int,
        external_flux: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        outgoing = torch.zeros(num_nodes, self.flux_dim, device=flux.device, dtype=flux.dtype)
        incoming = torch.zeros_like(outgoing)
        outgoing.index_add_(0, src, flux)
        incoming.index_add_(0, dst, flux)
        residual = outgoing - incoming
        if external_flux is not None:
            residual = residual + external_flux
        return residual

    def _project_flux(
        self,
        flux_hat: torch.Tensor,
        residual_hat: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        """
        Jacobi iterations on (A A^T) lambda = residual_hat, then
        flux* = flux_hat - A^T lambda.
        """
        if self.n_proj <= 0:
            return flux_hat

        diag = torch.zeros(num_nodes, 1, device=flux_hat.device, dtype=flux_hat.dtype)
        ones = torch.ones(src.size(0), 1, device=flux_hat.device, dtype=flux_hat.dtype)
        diag.index_add_(0, src, ones)
        diag.index_add_(0, dst, ones)
        diag = diag.clamp_min(self.eps)

        lam = torch.zeros_like(residual_hat)
        for _ in range(self.n_proj):
            neigh = torch.zeros_like(lam)
            neigh.index_add_(0, src, lam[dst])
            neigh.index_add_(0, dst, lam[src])
            lam_new = (residual_hat + neigh) / diag
            lam = self.proj_relax * lam_new + (1.0 - self.proj_relax) * lam

        # A^T lambda for directed incidence (+ at src, - at dst)
        grad = lam[src] - lam[dst]
        return flux_hat - grad

    def forward(
        self,
        h: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        external_flux: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        src, dst = edge_index
        rel_pos = pos[dst] - pos[src]

        # Antisymmetric learned interface flux.
        f_ij = self.flux_mlp(torch.cat([h[src], h[dst], rel_pos], dim=-1))
        f_ji = self.flux_mlp(torch.cat([h[dst], h[src], -rel_pos], dim=-1))
        flux_hat = 0.5 * (f_ij - f_ji)

        residual_pre = self._node_residual(flux_hat, src, dst, h.size(0), external_flux=external_flux)
        flux_proj = self._project_flux(flux_hat, residual_pre, src, dst, h.size(0))
        residual_post = self._node_residual(flux_proj, src, dst, h.size(0), external_flux=external_flux)

        correction = self.corr_mlp(torch.cat([h, residual_post], dim=-1))
        h_corrected = self.norm(h + correction)
        return h_corrected, flux_proj, residual_pre, residual_post
