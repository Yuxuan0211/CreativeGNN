from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torchdiffeq import odeint

from .graph_ode import GraphNeuralODEFunc
from .local_solver import RegionLocalSolver
from .flux_layer import FluxLayer
from .mlp import build_mlp


class RBFEmbedding(nn.Module):
    """Learnable radial basis function embedding for coordinates."""

    def __init__(self, num_centers: int = 16, gamma: Optional[float] = None):
        super().__init__()
        self.num_centers = num_centers
        self.centers = nn.Parameter(torch.randn(num_centers, 3))
        if gamma is None:
            self.log_gamma = nn.Parameter(torch.zeros(1))
            self.fixed_gamma = None
        else:
            self.log_gamma = None
            self.fixed_gamma = float(gamma)

    @property
    def out_dim(self) -> int:
        return self.num_centers

    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        diff = pos.unsqueeze(1) - self.centers.unsqueeze(0)
        dist2 = (diff**2).sum(dim=-1)
        gamma = self.fixed_gamma if self.fixed_gamma is not None else torch.exp(self.log_gamma)
        return torch.exp(-gamma * dist2)


class PolynomialEmbedding(nn.Module):
    """Low-order polynomial embedding for coordinates (degree 2)."""

    def __init__(self):
        super().__init__()
        self._out_dim = 9

    @property
    def out_dim(self) -> int:
        return self._out_dim

    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        x, y, z = pos.unbind(dim=-1)
        feats = [
            x,
            y,
            z,
            x * x,
            y * y,
            z * z,
            x * y,
            x * z,
            y * z,
        ]
        return torch.stack(feats, dim=-1)


class GeometricPhysicalEncoder(nn.Module):
    """Combine physical inputs with geometric embedding into latent space."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        geom_type: str = "rbf",
        geom_dim: int = 16,
        geom_gamma: Optional[float] = None,
    ):
        super().__init__()
        if geom_type == "rbf":
            self.geom = RBFEmbedding(num_centers=geom_dim, gamma=geom_gamma)
        elif geom_type == "poly":
            self.geom = PolynomialEmbedding()
        else:
            raise ValueError(f"Unsupported geom_type: {geom_type}")
        geom_out = self.geom.out_dim
        self.fx = nn.Linear(in_dim, hidden_dim, bias=False)
        self.fg = nn.Linear(geom_out, hidden_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(hidden_dim))
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        g = self.geom(pos)
        h = self.fx(x) + self.fg(g) + self.bias
        return self.norm(h)


class CreativeGNN(nn.Module):
    """Encode → continuous graph dynamics → region local solver → flux correction → decode."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 128,
        num_regions: int = 4,
        ode_steps: int = 4,
        ode_method: str = "rk4",
        dropout: float = 0.0,
        geom_type: str = "rbf",
        geom_dim: int = 16,
        geom_gamma: Optional[float] = None,
        flux_dim: int = 5,
        local_beta: float = 1.0,
        local_depth: int = 2,
        flux_nproj: int = 1,
        ode_t: float = 1.0,
        ode_rtol: Optional[float] = None,
        ode_atol: Optional[float] = None,
    ):
        super().__init__()
        self.encoder = GeometricPhysicalEncoder(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            geom_type=geom_type,
            geom_dim=geom_dim,
            geom_gamma=geom_gamma,
        )
        self.ode_func = GraphNeuralODEFunc(hidden_dim, hidden_dim, dropout=dropout)
        self.local_solver = RegionLocalSolver(
            hidden_dim,
            num_regions=num_regions,
            dropout=dropout,
            beta=local_beta,
            depth=local_depth,
        )
        self.flux_layer = FluxLayer(hidden_dim, flux_dim=flux_dim, dropout=dropout, nproj=flux_nproj)
        self.decoder = build_mlp(hidden_dim, hidden_dim, out_dim, num_layers=3, dropout=dropout)
        self.ode_steps = ode_steps
        self.ode_method = ode_method
        self.ode_t = ode_t
        self.ode_rtol = ode_rtol
        self.ode_atol = ode_atol

        self.register_buffer("integration_times", torch.linspace(0, ode_t, steps=ode_steps + 1))

    def forward(
        self,
        batch,
        use_flux: bool = True,
        boundary_flux: Optional[torch.Tensor] = None,
        compute_flux_metrics: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        h0 = self.encoder(batch.x, batch.pos)

        self.ode_func.set_graph(batch.pos, batch.edge_index)
        times = self.integration_times.to(h0.device)
        ode_kwargs = {"method": self.ode_method}
        if self.ode_rtol is not None:
            ode_kwargs["rtol"] = self.ode_rtol
        if self.ode_atol is not None:
            ode_kwargs["atol"] = self.ode_atol
        h_traj = odeint(self.ode_func, h0, times, **ode_kwargs)
        h_latent = h_traj[-1]

        h_local = self.local_solver(h_latent, batch.edge_index, batch.region_mask)
        if use_flux:
            (
                h_flux,
                flux_edges,
                flux_residual_pre,
                flux_residual_post,
                flux_residual_pre_norm,
                flux_residual_post_norm,
            ) = self.flux_layer(
                h_local, batch.pos, batch.edge_index, boundary_flux=boundary_flux
            )
        else:
            h_flux = h_local
            if compute_flux_metrics:
                _, flux_edges, flux_residual_pre, flux_residual_post, flux_residual_pre_norm, flux_residual_post_norm = (
                    self.flux_layer(h_local, batch.pos, batch.edge_index, boundary_flux=boundary_flux)
                )
            else:
                flux_edges = torch.zeros(
                    batch.edge_index.size(1),
                    self.flux_layer.flux_dim,
                    device=h_local.device,
                )
                flux_residual_pre = torch.zeros(
                    h_local.size(0),
                    self.flux_layer.flux_dim,
                    device=h_local.device,
                )
                flux_residual_post = torch.zeros_like(flux_residual_pre)
                flux_residual_pre_norm = torch.zeros_like(flux_residual_pre)
                flux_residual_post_norm = torch.zeros_like(flux_residual_pre)

        pred = self.decoder(h_flux)
        aux = {
            "latent": h_flux,
            "flux_edges": flux_edges,
            "flux_residual_pre": flux_residual_pre,
            "flux_residual_post": flux_residual_post,
            "flux_residual_pre_norm": flux_residual_pre_norm,
            "flux_residual_post_norm": flux_residual_post_norm,
        }
        return pred, aux
