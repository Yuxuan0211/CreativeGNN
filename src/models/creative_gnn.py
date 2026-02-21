from typing import Dict, Tuple

import torch
import torch.nn as nn
from torchdiffeq import odeint

from .graph_ode import GraphNeuralODEFunc
from .local_solver import RegionLocalSolver
from .flux_layer import FluxLayer
from .mlp import build_mlp


class GeometricEmbedding(nn.Module):
    """Coordinate embedding used by the geometric-physical encoder."""

    def __init__(self, geom_type: str = "rbf", geom_dim: int = 16, geom_gamma: float = None):
        super().__init__()
        self.geom_type = str(geom_type)
        self.geom_dim = max(0, int(geom_dim))

        if self.geom_dim <= 0:
            self.register_buffer("_dummy", torch.zeros(1))
            return

        if self.geom_type == "rbf":
            centers = torch.empty(self.geom_dim, 3)
            nn.init.uniform_(centers, -1.0, 1.0)
            self.centers = nn.Parameter(centers)
            gamma_val = 1.0 if geom_gamma is None else float(geom_gamma)
            self.log_gamma = nn.Parameter(torch.log(torch.tensor(gamma_val, dtype=torch.float32)))
            self.poly_proj = None
        elif self.geom_type == "poly":
            # [x, y, z, x^2, y^2, z^2, xy, yz, zx]
            self.poly_proj = nn.Linear(9, self.geom_dim)
            self.centers = None
            self.log_gamma = None
        else:
            raise ValueError(f"Unsupported geom_type: {geom_type}")

    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        if self.geom_dim <= 0:
            return pos.new_zeros(pos.size(0), 0)
        if self.geom_type == "rbf":
            dist2 = (pos.unsqueeze(1) - self.centers.unsqueeze(0)).pow(2).sum(dim=-1)
            gamma = torch.exp(self.log_gamma).clamp_min(1e-6)
            return torch.exp(-gamma * dist2)

        x = pos[:, 0:1]
        y = pos[:, 1:2]
        z = pos[:, 2:3]
        poly = torch.cat([x, y, z, x * x, y * y, z * z, x * y, y * z, z * x], dim=-1)
        return self.poly_proj(poly)


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
        geom_type: str = "rbf",
        geom_dim: int = 16,
        geom_gamma: float = None,
        flux_dim: int = 5,
        flux_nproj: int = 3,
        local_beta: float = 1.0,
        local_depth: int = 2,
        dropout: float = 0.0,
        ode_t: float = 1.0,
        **_: object,
    ):
        super().__init__()
        self.geom_embed = GeometricEmbedding(geom_type=geom_type, geom_dim=geom_dim, geom_gamma=geom_gamma)
        encoder_in = in_dim + 3 + max(0, int(geom_dim))
        self.encoder = build_mlp(encoder_in, hidden_dim, hidden_dim, num_layers=3, dropout=dropout)
        self.ode_func = GraphNeuralODEFunc(hidden_dim, hidden_dim, dropout=dropout)
        self.local_solver = RegionLocalSolver(
            hidden_dim,
            num_regions=num_regions,
            local_depth=local_depth,
            local_beta=local_beta,
            dropout=dropout,
        )
        self.flux_layer = FluxLayer(hidden_dim, flux_dim=flux_dim, n_proj=flux_nproj, dropout=dropout)
        self.decoder = build_mlp(hidden_dim, hidden_dim, out_dim, num_layers=3, dropout=dropout)
        self.ode_steps = ode_steps
        self.ode_method = ode_method

        self.register_buffer("integration_times", torch.linspace(0, float(ode_t), steps=ode_steps + 1))

    def forward(
        self,
        batch,
        use_flux: bool = True,
        compute_flux_metrics: bool = False,
        **_: object,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        geom_feat = self.geom_embed(batch.pos)
        x_in = torch.cat([batch.x, batch.pos, geom_feat], dim=-1)
        h0 = self.encoder(x_in)

        self.ode_func.set_graph(batch.pos, batch.edge_index)
        times = self.integration_times.to(h0.device)
        h_traj = odeint(self.ode_func, h0, times, method=self.ode_method)
        h_latent = h_traj[-1]

        h_local = self.local_solver(h_latent, batch.edge_index, batch.region_mask)
        if use_flux:
            external_flux = getattr(batch, "external_flux", None)
            h_flux, flux_edges, flux_residual_pre, flux_residual_post = self.flux_layer(
                h_local,
                batch.pos,
                batch.edge_index,
                external_flux=external_flux,
            )
        else:
            h_flux = h_local
            flux_dim = self.flux_layer.flux_dim
            flux_edges = torch.zeros(batch.edge_index.size(1), flux_dim, device=h_flux.device, dtype=h_flux.dtype)
            flux_residual_pre = torch.zeros(h_flux.size(0), flux_dim, device=h_flux.device, dtype=h_flux.dtype)
            flux_residual_post = flux_residual_pre

        pred = self.decoder(h_flux)
        aux = {
            "latent": h_flux,
            "flux_edges": flux_edges,
            "flux_residual": flux_residual_post,
            # Keep backward compatibility with training/eval hooks.
            "flux_residual_pre_norm": flux_residual_pre,
            "flux_residual_post_norm": flux_residual_post if compute_flux_metrics else flux_residual_post,
        }
        return pred, aux
